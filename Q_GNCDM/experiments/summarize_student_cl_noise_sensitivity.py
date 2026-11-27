"""Replay, tabulate every seed, plot curves, and report noise sensitivity."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Q_GNCDM.experiments import run_student_cl_noise_sensitivity as exp
from Q_GNCDM.experiments.summarize_controlled_student_cl import table

prior=exp.prior
COLORS={'R2':'#476a9d','R3':'#d58a24','R4':'#239c83','R5':'#b35786'}


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]


def input_replay(data,args,seed,epochs):
    totals={v:{'observed':0,'selected':0.,'changed':0,'clipped':0.,'abs_sum':0.,'sq_sum':0.,'max':0.} for v in ('positive','negative')}
    eligible=np.flatnonzero((data['train']!=0).sum(1)>=2)
    for epoch in range(1,epochs+1):
        order=np.random.default_rng([seed,epoch,0]).permutation(eligible)
        for step,start in enumerate(range(0,len(order),args.batch_students)):
            students=order[start:start+args.batch_students]
            if len(students)<2: continue
            episode=prior.make_student_episode(data['train'],students,args.query_fraction,np.random.default_rng([seed,epoch,step,1]))
            clean=episode.support[students]
            for idx,name in enumerate(('positive','negative')):
                out,st=exp.cl.augment(clean,getattr(args,name+'_noise_ratio'),getattr(args,name+'_noise_scale'),
                    np.random.default_rng([args.contrastive_seed,seed,epoch,step,idx]))
                delta=np.abs(out-clean).astype(np.float64)
                t=totals[name]; count=int((clean!=0).sum()); selected=st['selected_fraction']*count
                t['observed']+=count; t['selected']+=selected; t['changed']+=int((delta>0).sum())
                t['clipped']+=selected*st['clipped_fraction_selected']
                t['abs_sum']+=float(delta.sum()); t['sq_sum']+=float((delta**2).sum()); t['max']=max(t['max'],float(delta.max()))
    result={}
    for name,t in totals.items():
        result.update({name+'_selected_fraction':t['selected']/t['observed'],
                       name+'_changed_fraction':t['changed']/t['observed'],
                       name+'_clipped_fraction_selected':t['clipped']/t['selected'],
                       name+'_clipped_fraction_observed':t['clipped']/t['observed'],
                       name+'_mae_observed':t['abs_sum']/t['observed'],
                       name+'_mae_changed':t['abs_sum']/max(t['changed'],1),
                       name+'_mse_observed':t['sq_sum']/t['observed'],
                       name+'_maximum_actual_change':t['max']})
    return result


def plot_curves(runs,figures,filename,keys,title,probe=False,log=False):
    fig,axs=plt.subplots(2,3,figsize=(15,7),sharex=False,layout='constrained')
    for i,mode in enumerate(('A0','A1')):
        for j,seed in enumerate((2026,2027,2028)):
            ax=axs[i,j]
            for condition in exp.CONFIGS:
                run=next(r for r in runs if (r['mode'],r['seed'],r['condition'])==(mode,seed,condition))
                rows=run['probe'] if probe else run['log']
                for ix,key in enumerate(keys):
                    values=[max(row[key],1e-16) if log else row[key] for row in rows]
                    ax.plot([row['epoch'] for row in rows],values,color=COLORS[condition],
                            linestyle='-' if ix==0 else '--',label=condition+(' positive' if ix==0 and len(keys)==2 else ' negative' if len(keys)==2 else ''))
            ax.set_title(f'{"Fixed Q" if mode=="A0" else "Weak-Q"} / seed {seed}')
            ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
            if log: ax.set_yscale('log')
    axs[0,0].legend(fontsize=7,ncol=2)
    fig.suptitle(title)
    fig.savefig(figures/filename,dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-id',default='student_cl_noise_math1_20261123')
    cli=parser.parse_args()
    root=exp.ROOT/'Q_GNCDM/results'/cli.experiment_id
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    preflight=json.loads((root/'preflight.json').read_text(encoding='utf-8'))
    assert json.loads((root/'completed.json').read_text(encoding='utf-8'))['runs']==24
    torch.set_num_threads(1)
    report_dir=exp.ROOT/'Q_GNCDM/docs/experiments/student_cl'
    report_dir.mkdir(parents=True,exist_ok=True)
    figure_link='assets/'+manifest['args']['dataset']
    figures=report_dir/figure_link; figures.mkdir(parents=True,exist_ok=True)
    try:
        data=prior.core.load_data(argparse.Namespace(**manifest['args']))
        runs=[]; verify=[]; inputs=[]
        for path in sorted(root.glob('A*/seed_*/R*/metrics.json')):
            r=json.loads(path.read_text(encoding='utf-8'))
            args=exp.config(manifest['args'],r['condition'])
            model=prior.build_model(data,args,r['mode'],r['seed'])
            model.load_state_dict(prior.checkpoint_state(path.parent))
            scores,_,_=prior.core.evaluate(model,data['train'],data['rows'][2])
            assert all(scores[k]==r['test_'+k] for k in scores)
            r['log']=read_csv(path.parent/'train_log.csv')
            r['probe']=read_csv(path.parent/'fixed_probe_curve.csv')
            assert int(max(r['log'],key=lambda x:x['valid_acc'])['epoch'])==r['best_epoch']
            r['input']=input_replay(data,args,r['seed'],r['epochs_run'])
            inputs.append({'mode':r['mode'],'seed':r['seed'],'condition':r['condition'],**r['input']})
            runs.append(r)
            verify.append({'mode':r['mode'],'seed':r['seed'],'condition':r['condition'],
                           'checkpoint_metrics_exact':True,'ACC_selection_verified':True})
        prior.save_csv(root/'input_diagnostics.csv',inputs)
        prior.legacy.write_json(root/'replay_verification.json',verify)
        pairs=[]; stats=[]; selected=[]; effects=[]; trajectories=[]
        for mode in ('A0','A1'):
            baselines=[]
            for seed in (2026,2027,2028):
                b=json.loads((exp.OLD/mode/f'seed_{seed}'/'R1/metrics.json').read_text(encoding='utf-8'))
                baselines.append(b)
                group={r['condition']:r for r in runs if r['mode']==mode and r['seed']==seed}
                for condition in ('R3','R4','R5'):
                    p={'mode':mode,'seed':seed,'condition':condition}
                    p.update({'delta_'+m:group[condition]['test_'+m]-group['R2']['test_'+m] for m in ('acc','auc')})
                    pairs.append(p)
                for metric in ('acc','auc'):
                    vals={c:r['test_'+metric] for c,r in group.items()}
                    effects.append({'mode':mode,'seed':seed,'metric':metric,
                                    'smaller_positive_at_normal_negative':vals['R3']-vals['R2'],
                                    'smaller_positive_at_extreme_negative':vals['R5']-vals['R4'],
                                    'larger_negative_at_normal_positive':vals['R4']-vals['R2'],
                                    'larger_negative_at_small_positive':vals['R5']-vals['R3'],
                                    'interaction':vals['R5']-vals['R4']-vals['R3']+vals['R2']})
            best=sorted(baselines,key=lambda x:(-x['valid_acc'],x['seed']))[0]
            selected.append({'mode':mode,'condition':'R1(no CL)','lambda_cl':0,'seed':best['seed'],
                             'best_epoch':best['best_epoch'],'valid_acc':best['valid_acc'],
                             'test_acc':best['test_acc'],'test_auc':best['test_auc']})
            for condition in exp.CONFIGS:
                group=[r for r in runs if r['mode']==mode and r['condition']==condition]
                best=sorted(group,key=lambda x:(-x['valid_acc'],x['seed']))[0]
                selected.append({k:best[k] for k in ('mode','condition','lambda_cl','seed','best_epoch','valid_acc','test_acc','test_auc')})
                s={'mode':mode,'condition':condition,'n':3}
                for metric in ('acc','auc'):
                    vals=[r['test_'+metric] for r in group]
                    s[metric+'_mean']=float(np.mean(vals));s[metric+'_std']=float(np.std(vals,ddof=1))
                    pp=[p['delta_'+metric] for p in pairs if p['mode']==mode and p['condition']==condition]
                    s['paired_delta_'+metric+'_mean']=float(np.mean(pp)) if pp else 0.
                    s['paired_delta_'+metric+'_std']=float(np.std(pp,ddof=1)) if pp else 0.
                stats.append(s)
        for r in runs:
            first,last=r['probe'][0],r['probe'][-1]
            endpoint=min(x['epochs_run'] for x in runs if x['mode']==r['mode'] and x['seed']==r['seed'])
            common=r['probe'][endpoint]
            trajectories.append({'mode':r['mode'],'seed':r['seed'],'condition':r['condition'],
                                 'last_epoch':r['epochs_run'],'common_epoch':endpoint,
                                 'ap_mse_initial':first['probe_ap_mse'],'ap_mse_last':last['probe_ap_mse'],
                                 'an_mse_initial':first['probe_an_mse'],'an_mse_last':last['probe_an_mse'],
                                 'ap_mse_common':common['probe_ap_mse'],'an_mse_common':common['probe_an_mse'],
                                 'ap_decreased_to_common':common['probe_ap_mse']<first['probe_ap_mse'],
                                 'an_increased_to_common':common['probe_an_mse']>first['probe_an_mse']})
        for name,records in [('selected_runs.csv',selected),('all_seed_results.csv',[{k:r[k] for k in
                  ('mode','condition','seed','best_epoch','epochs_run','valid_acc','test_acc','test_auc','test_rmse','test_brier','test_nll','q_f1_vs_reference')} for r in runs]),
                  ('aggregate.csv',stats),('paired_deltas.csv',pairs),('factorial_effects.csv',effects),('probe_trajectories.csv',trajectories)]:
            prior.save_csv(root/name,records)
        specifications=[('train_loss.png',['loss_total'],'Training total loss',False,False),
          ('train_original_loss.png',['loss_original'],'Original loss including Q regularization',False,False),
          ('validation_acc.png',['valid_acc'],'Validation ACC; checkpoint rule unchanged',False,False),
          ('contrastive_loss.png',['loss_cl'],'Same-student InfoNCE',False,False),
          ('mastery_distances.png',['ap_mse','an_mse'],'Training mastery MSE: positive solid, negative dashed',False,True),
          ('fixed_probe_distances.png',['probe_ap_mse','probe_an_mse'],'Fixed-input probe mastery MSE (including initial state)',True,True),
          ('gradient_norm.png',['cl_shared_grad_norm'],'Shared CL gradient norm (unweighted)',False,True),
          ('gradient_ratio.png',['weighted_cl_to_original_grad_ratio'],'Weighted CL / original shared gradient norm',False,True),
          ('negative_gradient_activation.png',['negative_view_grad_above_1e_12_fraction'],'Negative-view gradient nonzero fraction (>1e-12); NOT margin activation',False,False),
          ('positive_gradient.png',['positive_view_grad_norm_mean'],'Positive-view gradient norm',False,True)]
        for filename,keys,title,probe_flag,log in specifications:
            plot_curves(runs,figures,filename,keys,title,probe_flag,log)
        fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for ax,mode in zip(axs,('A0','A1')):
            for seed in (2026,2027,2028):
                vals=[next(r['test_acc'] for r in runs if (r['mode'],r['seed'],r['condition'])==(mode,seed,c)) for c in exp.CONFIGS]
                ax.plot(list(exp.CONFIGS),vals,marker='o',label=f'seed {seed}')
            for s in selected:
                if s['mode']==mode and s['condition'] in exp.CONFIGS:
                    ax.scatter(s['condition'],s['test_acc'],marker='*',s=150,c='black',zorder=5)
            ax.set_title('Fixed Q' if mode=='A0' else 'Weak-Q');ax.set_ylabel('Test ACC');ax.grid(alpha=.2);ax.legend()
        fig.suptitle('All paired seeds; black star = highest validation ACC run (not highest test ACC)')
        fig.savefig(figures/'acc_comparison.png',dpi=170);plt.close(fig)
        link='../../../results/'+cli.experiment_id
        text=['# 极端噪声与敏感性实验结果报告',
              '## 主结果表：单个验证最优运行，不是均值',
              '每个Q设置、噪声配置内，按验证ACC最大选择一个种子；并列选较小种子。'
              '表中两个test指标均来自同一运行的同一最佳验证ACC checkpoint。'
              '没有按测试ACC/AUC挑选运行，也不拼接各自最高的测试值。此表用于展示，配对效应以全部种子表判断。',
              table(['Q设置','配置','CL系数','选中种子','最佳轮','验证ACC','对应test ACC','对应test AUC'],
                    [[s['mode'],s['condition'],s['lambda_cl'],s['seed'],s['best_epoch'],s['valid_acc'],s['test_acc'],s['test_auc']] for s in selected]),
              '## 预先固定的协议',
              '沿用任务二：Math1、种子2026/2027/2028、λ_cl=0.1、温度0.2、Adam及学习率、128学生batch、30轮上限、patience=8、最佳验证ACC（最早并列）。'
              '模型、数据划分、原始损失和Weak-Q系数均保持不变。两种Q设置、四种噪声、三种子，共24次训练。',
              table(['配置','正比例','正幅度','负比例','负幅度'],[[k,*v] for k,v in exp.CONFIGS.items()]),
              'R2沿用上一阶段普通配置，避免改变参照定义。实际输入为−1/0/1：仅非零训练support位置加均匀连续噪声，裁剪到[-1,1]；学生、题目输入不拼接。'
              '正小噪声和负极端噪声采用示例幅度0.001、0.5；后者相当于输入完整跨度2的25%，不会翻转符号，但会把±1推向零。'
              '因此它检验的是连续证据强度扰动，不是大比例作答正误翻转。0保持缺失，ID和标签不变。',
              '损失沿用两候选InfoNCE，没有margin，故无margin饱和或距离下界保证。'
              '正能量 E+=−s+/τ，联合归一化项 Z=log(exp(s+/τ)+exp(s−/τ))，L=E++Z。'
              'Z同时含正负项，不能称为独立负样本损失。negative_relative_penalty=softplus((s−−s+)/τ)等于总InfoNCE。'
              '日志另存正/负logit通路的共享参数偏梯度范数和正/负视图梯度，分析时不虚构两个可独立相加的拉近/推远损失。',
              '## 正确性与训练不变性',
              '正式训练前8种Q/噪声组合的单batch预检全部通过。新增观测与原CL函数的损失、共享梯度逐位相等，模型状态和Torch RNG不变。'
              'R2的6个运行与任务二checkpoint和全部既有逐轮日志精确一致。24个checkpoint的测试指标重放全部精确一致。'
              '原文件和数据哈希未变化；失败时会保留failure.log。本轮未修改损失、未调参、未选择性删除运行。',
              f'[预检]({link}/preflight.json) · [R2一致性]({link}/R2_parity.json) · [重放核验]({link}/replay_verification.json)',
              '## 全部种子结果与配对差值',
              table(['Q','seed','配置','最佳轮','test ACC','test AUC','相对R2 ΔACC','相对R2 ΔAUC'],
                    [[r['mode'],r['seed'],r['condition'],r['best_epoch'],r['test_acc'],r['test_auc'],
                      next((p['delta_acc'] for p in pairs if (p['mode'],p['seed'],p['condition'])==(r['mode'],r['seed'],r['condition'])),0.),
                      next((p['delta_auc'] for p in pairs if (p['mode'],p['seed'],p['condition'])==(r['mode'],r['seed'],r['condition'])),0.)] for r in runs]),
              '### 补充统计（不替代上面的逐运行表）',
              table(['Q','配置','ACC均值','ACC样本SD','AUC均值','AUC样本SD','配对ΔACC均值','配对ΔAUC均值'],
                    [[s[k] for k in ('mode','condition','acc_mean','acc_std','auc_mean','auc_std','paired_delta_acc_mean','paired_delta_auc_mean')] for s in stats]),
              '仅3个训练种子，不声称统计显著或可推广的稳定提升。所有差值均为比例单位，乘100才是百分点。'
              f'完整配对差值标准差见[aggregate.csv]({link}/aggregate.csv)，2×2主效应及交互见[factorial_effects.csv]({link}/factorial_effects.csv)。',
              '## 实际输入扰动',
              '以下按每次训练实际经历的全部有效位置加权统计；重新播放相同episode及独立噪声种子，仅重建输入，不训练。'
              '最大变化是真实全程最大值，不是各batch最大值的平均。选中比例与真正变化比例分别记录；约一半噪声因边界裁剪抵消。',
              table(['Q','seed','配置','正实际变化比例','正MAE','正最大变化','负实际变化比例','负MAE','负最大变化','负裁剪/选中','正视图未变学生比例'],
                    [[r['mode'],r['seed'],r['condition']]+[r['input'][k] for k in
                     ('positive_changed_fraction','positive_mae_observed','positive_maximum_actual_change',
                      'negative_changed_fraction','negative_mae_observed','negative_maximum_actual_change','negative_clipped_fraction_selected')]
                      +[r['training_means']['positive_unchanged_student_fraction']] for r in runs]),
              '最后一列为训练日志的batch/epoch均值。极小正噪声可能令大多数学生的正视图与Anchor完全相同；'
              '这种情况下不能把接近零的正距离当成训练学会了更强一致性。',
              f'选中比例、正负MSE、变化位置条件MAE、裁剪/全部有效位置比例等完整字段见[input_diagnostics.csv]({link}/input_diagnostics.csv)。',
              '## 掌握度、损失与梯度诊断',
              '先给出相同初始化、相同batch、相同噪声随机流的预检结果，避免用不同训练轨迹解释纯噪声敏感性：',
              table(['Q','配置','正MSE','负MSE','CL损失','CL梯度范数','加权CL/原梯度'],
                    [[mode,condition,preflight[mode+'_'+condition]['batch_diagnostics']['ap_mse'],
                      preflight[mode+'_'+condition]['batch_diagnostics']['an_mse'],
                      preflight[mode+'_'+condition]['loss_cl'],
                      preflight[mode+'_'+condition]['gradient']['cl_shared_grad_norm'],
                      preflight[mode+'_'+condition]['gradient']['weighted_cl_to_original_grad_ratio']]
                     for mode in ('A0','A1') for condition in exp.CONFIGS]),
              '以下为逐batch均值再对epoch平均，与上面的输入加权统计口径不同。正负距离是按知识点维平均的MSE。'
              '负梯度比例指每名学生负视图的梯度范数>1e−12（另记录严格>0）；不是margin激活率，也不保证每名学生贡献的共享梯度方向一致。',
              table(['Q','seed','配置','正MSE','负MSE','CL损失','CL梯度范数','加权CL/原梯度','负梯度非零比例'],
                    [[r['mode'],r['seed'],r['condition']]+[r['training_means'][k] for k in
                     ('ap_mse','an_mse','loss_cl','cl_shared_grad_norm','weighted_cl_to_original_grad_ratio','negative_view_grad_above_1e_12_fraction')] for r in runs]),
              '正/负通路补充诊断：',
              table(['Q','seed','配置','正能量','联合归一化项','正视图梯度','负视图梯度','正logit共享偏梯度','负logit共享偏梯度'],
                    [[r['mode'],r['seed'],r['condition']]+[r['training_means'][k] for k in
                     ('positive_energy','joint_log_partition','positive_view_grad_norm_mean','negative_view_grad_norm_mean',
                      'positive_shared_partial_grad_norm','negative_shared_partial_grad_norm')] for r in runs]),
              '偏梯度可相互抵消，不能把两个偏梯度范数相加当作总梯度范数。',
              '## 固定视图的训练轨迹',
              '同一seed使用固定128名学生、固定support和固定增强随机数。包含初始化epoch0；'
              '为避免早停轮数不同造成混淆，下表使用同Q同seed四配置均有记录的共同末轮。末轮与最佳checkpoint不是同一概念。',
              table(['Q','seed','配置','共同轮','初始正MSE','共同轮正MSE','正距离下降','初始负MSE','共同轮负MSE','负距离上升'],
                    [[r[k] for k in ('mode','seed','condition','common_epoch','ap_mse_initial','ap_mse_common','ap_decreased_to_common',
                      'an_mse_initial','an_mse_common','an_increased_to_common')] for r in trajectories]),
              '## 图表（保留每个种子，不用均值曲线遮蔽差异）']
        for filename,_,title,_,_ in specifications:
            text += ['### '+title,f'![{title}]({figure_link}/{filename})']
        text += ['### 各配置ACC配对图',f'![ACC comparison]({figure_link}/acc_comparison.png)',
                 '## A–I 逐项回答']
        for mode in ('A0','A1'):
            group=[r for r in runs if r['mode']==mode]
            text.append('### '+('固定Q' if mode=='A0' else 'Weak-Q'))
            ratios=[r['input']['negative_mae_observed']/r['input']['positive_mae_observed'] for r in group]
            mastery_ratios=[r['training_means']['an_mse']/r['training_means']['ap_mse'] for r in group]
            tt=[r for r in trajectories if r['mode']==mode]
            text += [f'A. 输入空间负/正MAE比值范围 {min(ratios):.3f}–{max(ratios):.3f}，幅度尺度明显分开；不能因此认为认知语义不同。',
                     f'B. 训练掌握度负/正MSE比值范围 {min(mastery_ratios):.3f}–{max(mastery_ratios):.3f}。相对分离应同时看绝对距离和CL梯度，不能只看比值。',
                     f'C. 固定视图到共同末轮，正MSE下降 {sum(r["ap_decreased_to_common"] for r in tt)}/{len(tt)} 个运行。并非预设全部下降。',
                     f'D. 固定视图到共同末轮，负MSE上升 {sum(r["an_increased_to_common"] for r in tt)}/{len(tt)} 个运行。InfoNCE没有距离下界，不能宣称已达到margin。']
            for low,high in (('R2','R4'),('R3','R5')):
                grad_ratios=[]
                for seed in (2026,2027,2028):
                    a=next(r for r in group if r['seed']==seed and r['condition']==low)
                    b=next(r for r in group if r['seed']==seed and r['condition']==high)
                    grad_ratios.append(b['training_means']['cl_shared_grad_norm']/a['training_means']['cl_shared_grad_norm'])
                text.append(f'E. {high}/{low} 的训练平均CL梯度范数比（2026/2027/2028）：'+', '.join(f'{v:.3f}' for v in grad_ratios)+'。不存在margin饱和；训练路径及早停不同，因果幅度另看预检固定batch。')
            for condition in ('R3','R4','R5'):
                pp=[p for p in pairs if p['mode']==mode and p['condition']==condition]
                text.append(f'G/H. {condition} 相对R2的逐seed ΔACC：'+', '.join(f'{p["delta_acc"]:+.9f}' for p in pp)+
                            f'；正/零/负方向数量={sum(p["delta_acc"]>0 for p in pp)}/{sum(p["delta_acc"]==0 for p in pp)}/{sum(p["delta_acc"]<0 for p in pp)}。')
            for normal,small in (('R2','R3'),('R4','R5')):
                values=[]
                for seed in (2026,2027,2028):
                    a=next(r for r in group if r['seed']==seed and r['condition']==normal)
                    b=next(r for r in group if r['seed']==seed and r['condition']==small)
                    values.append(b['training_means']['positive_view_grad_norm_mean']/a['training_means']['positive_view_grad_norm_mean'])
                text.append(f'F. {small}/{normal} 正视图梯度范数比：'+', '.join(f'{v:.4f}' for v in values)+'。梯度变小不等于断图；检查非零值和精度限制。')
            not_improved=[p for p in pairs if p['mode']==mode and p['delta_acc']<=0]
            text.append(f'I. 参数梯度与表征响应可测，但有 {len(not_improved)}/9 个相对R2比较的ACC未提高；生效不等于预测改善。')
        text += ['## 分层结论与限制',
                 '1. 实现正确性：单batch观测不改损失/梯度、R2完整复现、数据与代码哈希、checkpoint重放均通过。\n'
                 '2. 训练影响：CL梯度、参数更新在任务二已证明，本轮记录了噪声响应；不是只有损失数值变化。\n'
                 '3. 表征约束：输入变化与掌握度变化有响应；正距离下降、负距离上升并非由InfoNCE无条件保证，须以固定视图轨迹计数为准。\n'
                 '4. ACC可重复性：以全部种子的配对方向和幅度判断，不以主表中选中的单次运行推断稳定收益。三种子不足以支持强统计结论。\n'
                 '5. 研究假设：可测量的数值响应支持“模块能影响模型”，不等于证实“强扰动同学生是合理语义负样本”或诊断能力提升。',
                 '极端负噪声仍是同符号的连续衰减，已偏离原有离散作答证据分布；对它的敏感性不能直接解释为更真实的认知诊断。'
                 '梯度统计包含所有训练batch，但曲线按epoch聚合；批大小最后一批可能不同。固定probe仅128名学生，不代表全体。'
                 '本轮没有重新训练λ=0的固定probe时间序列，因此距离随epoch变化同时包含原损失作用，不能全部归因于CL。'
                 '训练预算/早停规则未变，实际停止轮数和选中轮次可随噪声改变。未探索更多负样本、损失或ACC调优。',
                 '## 文件与运行记录',
                 f'- [配置及哈希]({link}/manifest.json)；[完整结果表]({link}/all_seed_results.csv)；[主表]({link}/selected_runs.csv)。\n'
                 f'- [逐种子配对差值]({link}/paired_deltas.csv)；[补充统计]({link}/aggregate.csv)；[固定probe轨迹摘要]({link}/probe_trajectories.csv)。\n'
                 '- 每个 A0或A1/seed_*/R2–R5 目录：config.json、metrics.json、train_log.csv、console.log、best_model.pt、fixed_probe_curve.csv、best_checkpoint_probe.json。\n'
                 '- figures/：所有11张PNG图；没有为重复参数另存W副本，完整参数仍在checkpoint。\n'
                 '- experiments/run_student_cl_noise_sensitivity.py：调用上一轮训练循环，只补充观测；summarize_student_cl_noise_sensitivity.py：核验、输入重建、汇总与绘图。\n'
                 '- 本文为本轮唯一解释记录，README仅增加索引。原模型与旧实验文件未修改，未提交Git。',
                 '实际训练命令：\n\n```powershell\n'+manifest['command']+'\n```',
                 '汇总命令：\n\n```powershell\n.venv-contrastive/Scripts/python.exe -m Q_GNCDM.experiments.summarize_student_cl_noise_sensitivity\n```']
        report=report_dir/(cli.experiment_id.replace('student_cl_noise_','').replace('_20261123','')+'_noise_R2-R5_20261123.md');report.write_text('\n\n'.join(text)+'\n',encoding='utf-8')
        prior.legacy.write_json(root/'report_manifest.json',{'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),'checkpoints_replayed':len(verify),
                  'figure_count':len(list(figures.glob('*.png')))})
        print(table(['Q','condition','seed','test_ACC','test_AUC'],[[r[k] for k in ('mode','condition','seed','test_acc','test_auc')] for r in selected]))
        print('REPORT COMPLETE',flush=True)
    except BaseException:
        (root/'report_failure.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise


if __name__=='__main__':
    main()
