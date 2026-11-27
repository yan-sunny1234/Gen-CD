"""Create self-contained Markdown and figures before removing temporary originals."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Q_GNCDM.experiments import run_student_cl_all_datasets as run
from Q_GNCDM.experiments.summarize_controlled_student_cl import table, matrix_md
from Q_GNCDM.experiments.summarize_student_cl_noise_sensitivity import plot_curves, read_csv

DEST=run.ROOT/'Q_GNCDM/docs/experiments/student_cl'


def publish_verified(name,seed=2028):
    """Merge a complete auxiliary group without touching any primary run.

    Handles the case where the primary started R0 before group publication.
    Only previously absent, fully verified run directories are made visible.
    """
    events=[]
    for mode in ('A0','A1'):
        source=run.WORK/f'_seed{seed}_helpers'/name/mode/f'seed_{seed}'
        target=run.WORK/name/mode/f'seed_{seed}'
        assert source.resolve().is_relative_to(run.WORK.resolve())
        assert target.resolve().is_relative_to(run.WORK.resolve())
        if not all((source/c/'verified.json').exists() for c in ('R0','R1','R2','R3','R4','R5')):continue
        if (target/'R0').exists():
            if not (target/'R0/verified.json').exists():continue
            a=run.prior.checkpoint_state(source/'R0');b=run.prior.checkpoint_state(target/'R0')
            assert all(torch.equal(v,b[k]) for k,v in a.items())
        target.mkdir(parents=True,exist_ok=True)
        for c in ('R0','R1','R2','R3','R4','R5'):
            if (target/c).exists():continue
            stage=target/('._publish_'+c)
            if stage.exists():continue
            shutil.copytree(source/c,stage)
            for f in (source/c).iterdir():
                if f.is_file():assert hashlib.sha256(f.read_bytes()).digest()==hashlib.sha256((stage/f.name).read_bytes()).digest()
            try:
                stage.rename(target/c)
                events.append({'mode':mode,'seed':seed,'condition':c,'status':'copied_verified_run_without_overwrite',
                               'publisher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
            except FileExistsError:
                events.append({'mode':mode,'condition':c,'status':'primary_started_during_copy; original retained'})
    if events:
        path=run.WORK/name/f'seed{seed}_merge_execution.json'
        old=json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
        run.prior.legacy.write_json(path,old+events)
        print(name,events,flush=True)


def selected_rows(rows,conditions):
    selected=[]
    for mode in ('A0','A1'):
        for condition in conditions:
            group=[r for r in rows if r['mode']==mode and r['condition']==condition]
            r=sorted(group,key=lambda x:(-x['valid_acc'],x['seed']))[0]
            selected.append([mode,condition,r['lambda_cl'],r['seed'],r['best_epoch'],r['valid_acc'],r['test_acc'],r['test_auc']])
    return table(['Q设置','条件','CL系数','验证选优种子','最佳轮','valid ACC','test ACC','test AUC'],selected)


def stats_table(rows,conditions,baseline):
    vals=[]
    for mode in ('A0','A1'):
        for c in conditions:
            group=[r for r in rows if r['mode']==mode and r['condition']==c]
            paired=[]
            for r in group:
                b=next(x for x in rows if x['mode']==mode and x['seed']==r['seed'] and x['condition']==baseline)
                paired.append([r['test_acc']-b['test_acc'],r['test_auc']-b['test_auc']])
            vals.append([mode,c,np.mean([r['test_acc'] for r in group]),np.std([r['test_acc'] for r in group],ddof=1),
                         np.mean([r['test_auc'] for r in group]),np.std([r['test_auc'] for r in group],ddof=1),
                         *np.mean(paired,axis=0),*np.std(paired,axis=0,ddof=1)])
    return table(['Q','条件','ACC均值','ACC样本SD','AUC均值','AUC样本SD','配对ΔACC均值','配对ΔAUC均值','配对ΔACC SD','配对ΔAUC SD'],vals)


def all_rows_table(rows,conditions,baseline):
    vals=[]
    for r in rows:
        if r['condition'] not in conditions:continue
        b=next(x for x in rows if x['mode']==r['mode'] and x['seed']==r['seed'] and x['condition']==baseline)
        vals.append([r['mode'],r['seed'],r['condition'],r['best_epoch'],r['epochs_run'],r['valid_acc'],r['test_acc'],r['test_auc'],
                     r['test_acc']-b['test_acc'],r['test_auc']-b['test_auc'],r['test_rmse'],r['test_brier'],r['test_nll'],r['q_f1_vs_reference']])
    return table(['Q','seed','条件','最佳轮','训练轮','valid ACC','test ACC','test AUC','ΔACC','ΔAUC','RMSE','Brier','NLL','Q-F1'],vals)


def report(name):
    folder=run.WORK/name
    complete=json.loads((folder/'completed.json').read_text(encoding='utf-8'))
    assert complete['status']=='passed' and complete['runs']==36
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    manifest['scheduling_records']=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(folder.glob('seed20*_*_execution.json'))]
    duplicates=[]
    for seed in (2027,2028):
        for mode in ('A0','A1'):
            for condition in ('R0','R1','R2','R3','R4','R5'):
                aux=run.WORK/f'_seed{seed}_helpers'/name/mode/f'seed_{seed}'/condition
                primary=folder/mode/f'seed_{seed}'/condition
                if (aux/'verified.json').exists() and (primary/'verified.json').exists():
                    a=run.prior.checkpoint_state(aux);b=run.prior.checkpoint_state(primary)
                    assert all(torch.equal(v,b[k]) for k,v in a.items()),'Scheduling duplicate mismatch'
                    duplicates.append({'mode':mode,'seed':seed,'condition':condition,'checkpoint_exact':True})
    manifest['scheduling_duplicate_state_checks']=duplicates
    checks=json.loads((folder/'preflight.json').read_text(encoding='utf-8'))
    parameters=json.loads((folder/'parameters.json').read_text(encoding='utf-8'))
    parity=json.loads((folder/'parity.json').read_text(encoding='utf-8'))
    rows=[]
    for p in sorted(folder.glob('A*/seed_*/R*/metrics.json')):
        r=json.loads(p.read_text(encoding='utf-8'))
        verification=json.loads((p.parent/'verified.json').read_text(encoding='utf-8'))
        assert verification['checkpoint_replay'] and verification['ACC_selector']
        r['log']=read_csv(p.parent/'train_log.csv')
        assert int(max(r['log'],key=lambda x:x['valid_acc'])['epoch'])==r['best_epoch']
        if r['condition'] in run.noise.CONFIGS:r['probe']=read_csv(p.parent/'fixed_probe_curve.csv')
        if 'training_means' not in r:
            r['training_means']={k:float(np.mean([x[k] for x in r['log']])) for k in r['log'][0]}
            r['training_means']['loss_original']=r['training_means']['loss_total']
            r['training_means']['weighted_loss_cl']=0.
        rows.append(r)
    assert len(rows)==36
    DEST.mkdir(parents=True,exist_ok=True)
    assets=DEST/'assets'/name;assets.mkdir(parents=True,exist_ok=True)
    nr=[r for r in rows if r['condition'] in run.noise.CONFIGS]
    specs=[('noise_train_loss.png',['loss_total'],'Training total loss',False,False),
           ('noise_validation_acc.png',['valid_acc'],'Validation ACC (selection metric)',False,False),
           ('noise_cl_loss.png',['loss_cl'],'Same-student InfoNCE',False,False),
           ('noise_mastery_distance.png',['ap_mse','an_mse'],'Mastery MSE: positive solid, negative dashed',False,True),
           ('noise_fixed_probe.png',['probe_ap_mse','probe_an_mse'],'Fixed input probe (initial state included)',True,True),
           ('noise_gradient.png',['cl_shared_grad_norm'],'CL shared gradient norm',False,True),
           ('noise_gradient_ratio.png',['weighted_cl_to_original_grad_ratio'],'Weighted CL / original gradient norm',False,True),
           ('noise_negative_gradient.png',['negative_view_grad_above_1e_12_fraction'],'Negative view gradient >1e-12 fraction; no margin',False,False)]
    for filename,keys,title,probe,log in specs:
        plot_curves(nr,assets,filename,keys,f'{name}: {title}',probe,log)
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,mode in zip(axs,('A0','A1')):
        for seed in (2026,2027,2028):
            vals=[next(r['test_acc'] for r in rows if (r['mode'],r['seed'],r['condition'])==(mode,seed,c)) for c in ('R1','R2','R3','R4','R5')]
            ax.plot(['R1(no CL)','R2','R3','R4','R5'],vals,'o-',label=f'seed {seed}')
        ax.set_title('Fixed Q' if mode=='A0' else 'Weak-Q');ax.set_ylabel('Test ACC');ax.grid(alpha=.2);ax.legend()
    fig.suptitle(name+': all paired seeds, no averaging')
    fig.savefig(assets/'acc_all_seeds.png',dpi=170);plt.close(fig)
    common=[
        '主表不是均值：每个Q/条件内按最高验证ACC选择种子，并列选较小种子；两个test指标来自同一checkpoint。'
        '严禁把不同种子的单次结果直接相减解释模块收益，收益看全部配对种子。A0=固定参考Q，A1=Weak-Q。',
        '## 数据与固定协议',
        f"数据集：{name}；学生={manifest['train_shape'][0]}，题目={manifest['train_shape'][1]}，概念={manifest['concepts']}；"
        f"train/valid/test交互={manifest['split_lengths']}。",
        ('本项为合成数据，不是真实教育数据。使用既有seed_2026生成数据，Q_noise_0.20，划分seed=2026，交互70/10/20；'
         'Q-F1对应生成真Q，另报告真实掌握度指标。' if name=='synthetic_weak_q' else
         '使用data目录现有训练/验证/测试文件；没有重新划分。Q-F1表示与参考Q一致性，不能自动解释为认知诊断真实性。'),
        'R0参考原循环（统一验证ACC选模），R1新增入口λ=0并完全跳过CL训练分支，R2 λ=0.1。'
        'R2同时用于受控与敏感性实验，没有重复训练一份R2。模型、原始损失、α=0.8、Adam及学习率、128学生batch、30轮上限、patience=8均不变。'
        '验证ACC严格提升才更新checkpoint，平分保留最早轮。种子2026/2027/2028；每个数据集36次训练。',
        table(['条件','正比例','正幅度','负比例','负幅度'],[[c,*v] for c,v in run.noise.CONFIGS.items()]),
        '两输入不拼接。只对学生support的非零±1位置施加均匀连续噪声并裁剪到[-1,1]；0缺失、ID、标签、query和原数组不变。'
        '负噪声不翻转正误，只降低输入证据绝对值；因此极端响应不能直接当成诊断能力改善。验证/测试不增强。',
        '损失沿用同学生一正一负InfoNCE，温度0.2，无margin。L=−正logit+logsumexp(正logit,负logit)。'
        '联合归一化项并不是独立负样本损失；负约束非零比例指负视图梯度范数>1e−12，不是margin激活率。',
        '## 配置、代码和核验依据',
        '以下完整配置/哈希和核验结果作为最终文档保留；用户要求清理新实验的checkpoint、原始日志及JSON/CSV副本。'
        '实际清理状态：自动审批以blocked by policy拒绝递归删除，原始产物仍暂存在Git忽略的scratch/student_cl_all_20261123。'
        '最终报告目录只包含Markdown和配图。若后续手工清理，便不能直接重放checkpoint，需按记录重跑。',
        '<details><summary>完整配置、版本及数据/源码哈希</summary>\n\n```json\n'+json.dumps(manifest,ensure_ascii=False,indent=2)+'\n```\n</details>',
        '8种Q/噪声设置预检全部通过；同初始化固定batch的CL梯度、一步更新测试通过；新诊断的损失/梯度与原实现精确一致。'
        '6组R0/R1的逐轮日志与checkpoint精确一致；36个checkpoint的全部测试指标已重放核验。',
        '```json\n'+json.dumps(complete,ensure_ascii=False,indent=2)+'\n```',
        '<details><summary>全部预检真实输出（含逐层梯度和单步参数更新差异）</summary>\n\n```json\n'+json.dumps(checks,ensure_ascii=False,indent=2)+'\n```\n</details>',
        '## 全部条件的初始固定batch敏感性',
        table(['Q','条件','CL损失','正MSE','负MSE','CL共享梯度','加权CL/原梯度'],
              [[m,c,checks[m+'_'+c]['loss_cl'],checks[m+'_'+c]['batch_diagnostics']['ap_mse'],checks[m+'_'+c]['batch_diagnostics']['an_mse'],
                checks[m+'_'+c]['gradient']['cl_shared_grad_norm'],checks[m+'_'+c]['gradient']['weighted_cl_to_original_grad_ratio']]
               for m in ('A0','A1') for c in run.noise.CONFIGS])]
    controls=['# '+name+'：R0/R1/R2受控对比学习实验',common[0],
              selected_rows(rows,('R0','R1','R2')),*common[1:],
              '## 全部逐种子结果（相对R1配对差值）',all_rows_table(rows,('R0','R1','R2'),'R1'),
              '## 补充统计：不替代主表',stats_table(rows,('R0','R1','R2'),'R1'),
              '## 训练损失与诊断',
              table(['Q','seed','条件','原损失(含Q)','CL损失','加权CL','总损失'],
                    [[r['mode'],r['seed'],r['condition']]+[r['training_means'][k] for k in ('loss_original','loss_cl','weighted_loss_cl','loss_total')]
                     for r in rows if r['condition'] in ('R0','R1','R2')]),
              '## W参数差异',
              '以下比较每个运行按验证ACC选出的最终参数；可能含不同最佳轮的影响。'
              'CL参与更新的因果证据来自上述同batch单步测试。有效PosLinear权重为|W|。',
              table(['Q','seed','参数','形状','原W均值','CL W均值','差L2','最大差','|W|差L2'],
                    [[r[k] for k in ('mode','seed','parameter','shape','baseline_mean','cl_mean','difference_l2','difference_max','effective_difference_l2')] for r in parameters]),
              '### 学生网络W数值示例',
              '为避免大数据集数十万权重淹没报告，只展示seed2026各学生层左上最多8×8子矩阵，明确不是完整矩阵。'
              '全部层、全部种子的变化范数已列在上表；用户要求清理完整checkpoint，但当前清理被自动审批拒绝，实际仍暂存scratch。']
    for p in parameters:
        if p['baseline_preview'] is not None:
            controls += [f"{p['mode']} {p['parameter']} R1：",matrix_md(np.asarray(p['baseline_preview'])),
                         'R2：',matrix_md(np.asarray(p['cl_preview']))]
    controls += ['## 逐种子方向与结论']
    for mode in ('A0','A1'):
        deltas=[]
        for seed in (2026,2027,2028):
            a=next(r for r in rows if (r['mode'],r['seed'],r['condition'])==(mode,seed,'R1'))
            b=next(r for r in rows if (r['mode'],r['seed'],r['condition'])==(mode,seed,'R2'))
            deltas.append((b['test_acc']-a['test_acc'],b['test_auc']-a['test_auc']))
        controls += [mode+'，R2−R1（2026/2027/2028）ACC：'+', '.join(f'{d[0]:+.9f}' for d in deltas)+
                     '；AUC：'+', '.join(f'{d[1]:+.9f}' for d in deltas)+'。']
    controls += ['实现和更新生效已核验；性能结论以配对方向和幅度为准。三种子不足以宣称统计显著，不能因权重不同就声称预测提高。',
                 f'![所有种子ACC](assets/{name}/acc_all_seeds.png)']
    noises=['# '+name+'：R2–R5噪声敏感性实验',common[0],
            selected_rows(rows,('R1','R2','R3','R4','R5')),*common[1:],
            '## 全部种子结果（相对R2配对差值）',all_rows_table(rows,('R2','R3','R4','R5'),'R2'),
            '## 补充统计：不替代主表',stats_table(rows,('R2','R3','R4','R5'),'R2'),
            '## 实际输入变化（有效位置加权；最大值为全程最大）',
            table(['Q','seed','条件','视图','选中比例','实际变化比例','MAE','MSE','全程最大变化','裁剪/选中'],
                  [[r['mode'],r['seed'],r['condition'],p]+[r['input_actual'][p][k] for k in
                   ('selected_fraction','changed_fraction','mae_observed','mse_observed','max_actual_change','clipped_fraction_selected')]
                   for r in nr for p in ('positive','negative')]),
            '## 训练平均诊断（先batch均值，再epoch均值）',
            table(['Q','seed','条件','正MSE','负MSE','CL损失','CL梯度','加权CL/原梯度','负视图梯度非零比例','正输入未变学生比例'],
                  [[r['mode'],r['seed'],r['condition']]+[r['training_means'][k] for k in
                   ('ap_mse','an_mse','loss_cl','cl_shared_grad_norm','weighted_cl_to_original_grad_ratio','negative_view_grad_above_1e_12_fraction',
                    'positive_unchanged_student_fraction')] for r in nr]),
            '正能量、联合归一化项及正负梯度通路：',
            table(['Q','seed','条件','正能量','联合归一化项','正视图梯度','负视图梯度','正共享偏梯度','负共享偏梯度'],
                  [[r['mode'],r['seed'],r['condition']]+[r['training_means'][k] for k in
                   ('positive_energy','joint_log_partition','positive_view_grad_norm_mean','negative_view_grad_norm_mean',
                    'positive_shared_partial_grad_norm','negative_shared_partial_grad_norm')] for r in nr])]
    trajectories=[]
    for r in nr:
        end=min(x['epochs_run'] for x in nr if x['mode']==r['mode'] and x['seed']==r['seed'])
        start,last=r['probe'][0],r['probe'][end]
        trajectories.append([r['mode'],r['seed'],r['condition'],end,start['probe_ap_mse'],last['probe_ap_mse'],
                             start['probe_an_mse'],last['probe_an_mse'],last['probe_ap_mse']<start['probe_ap_mse'],
                             last['probe_an_mse']>start['probe_an_mse']])
    noises += ['## 固定视图轨迹',
               '固定最多128名学生、support和噪声；从初始化到同Q同seed四配置共同末轮。'
               '不把较小输入噪声造成的初始距离小误判成学习收益。轨迹也包含原损失影响，本轮没有λ=0同probe训练轨迹。',
               table(['Q','seed','条件','共同轮','初始正MSE','末正MSE','初始负MSE','末负MSE','正下降','负上升'],trajectories),
               '## A–I分析与分层结论']
    for mode in ('A0','A1'):
        group=[r for r in nr if r['mode']==mode];tr=[t for t in trajectories if t[0]==mode]
        ratios=[r['input_actual']['negative']['mae_observed']/max(r['input_actual']['positive']['mae_observed'],1e-30) for r in group]
        noises += [f'### {mode}',f'A：负/正输入MAE比值范围{min(ratios):.4g}–{max(ratios):.4g}，应结合实际未变化比例判断。',
                   'B：掌握度绝对距离见上表；输入分离不保证表征或认知语义分离。',
                   f'C/D：共同末轮正距离下降{sum(t[-2] for t in tr)}/12，负距离上升{sum(t[-1] for t in tr)}/12；InfoNCE没有margin下界。']
        for low,high in (('R2','R4'),('R3','R5')):
            ratio=checks[mode+'_'+high]['gradient']['cl_shared_grad_norm']/max(checks[mode+'_'+low]['gradient']['cl_shared_grad_norm'],1e-30)
            noises.append(f'E：固定初始batch {high}/{low} CL共享梯度范数比={ratio:.6g}；无margin饱和机制。')
        for normal,small in (('R2','R3'),('R4','R5')):
            ratios=[]
            for seed in (2026,2027,2028):
                a=next(r for r in group if r['condition']==normal and r['seed']==seed)
                b=next(r for r in group if r['condition']==small and r['seed']==seed)
                ratios.append(b['training_means']['positive_view_grad_norm_mean']/max(a['training_means']['positive_view_grad_norm_mean'],1e-30))
            noises.append(f'F：{small}/{normal}正视图梯度范数比：'+', '.join(f'{x:.5g}' for x in ratios)+'。')
        for c in ('R3','R4','R5'):
            da=[];du=[]
            for seed in (2026,2027,2028):
                a=next(r for r in group if r['condition']=='R2' and r['seed']==seed)
                b=next(r for r in group if r['condition']==c and r['seed']==seed)
                da.append(b['test_acc']-a['test_acc']);du.append(b['test_auc']-a['test_auc'])
            noises.append(f'G/H/I：{c}相对R2逐seed ΔACC：'+', '.join(f'{x:+.9f}' for x in da)+
                          '；ΔAUC：'+', '.join(f'{x:+.9f}' for x in du)+
                          f'；ACC正/零/负={sum(x>0 for x in da)}/{sum(x==0 for x in da)}/{sum(x<0 for x in da)}。')
    noises += ['实现核验通过、共享梯度生效，与预期拉近/推远是否同时出现、ACC是否可重复提高，是不同层次的结论。'
               '三种子不足以作强统计推断；不以单个最优运行代替全部结果。不支持仅凭强噪声的数值响应宣称诊断真实性提高。',
               '## 图表']
    for filename,_,title,_,_ in specs:noises += ['### '+title,f'![{title}](assets/{name}/{filename})']
    noises += [f'![全部种子ACC](assets/{name}/acc_all_seeds.png)']
    if name=='synthetic_weak_q':
        keys=sorted({k for r in rows for k in r if k.startswith('mastery_')})
        extra=['## 合成真实掌握度指标',table(['Q','seed','条件']+keys,[[r['mode'],r['seed'],r['condition']]+[r.get(k) for k in keys] for r in rows])]
        torch.set_num_threads(1)
        data,cfg=run.load_dataset(name,manifest['args'])
        class_rows=[]
        for r in rows:
            if r['condition']!='R1':continue
            model=run.prior.build_model(data,cfg,r['mode'],r['seed'])
            model.load_state_dict(run.prior.checkpoint_state(folder/r['mode']/f"seed_{r['seed']}"/'R1'))
            _,_,probs=run.prior.core.evaluate(model,data['train'],data['rows'][2])
            class_rows.append([r['mode'],r['seed'],r['best_epoch'],float(data['rows'][2][:,2].mean()),
                               float((probs>.5).mean()),float(probs.min()),float(probs.max())])
        extra += ['### 基线分类行为核查',table(['Q','seed','最佳轮','测试答对比例','预测为答对比例','最小预测概率','最大预测概率'],class_rows),
                  '若预测为答对比例为0或1，ACC恒定是单类预测/阈值行为，不能解释为CL对全部表征完全无作用。'
                  '本轮仍保留预定的验证ACC选模与早停；未改用AUC优化这个现象。']
        controls+=extra;noises+=extra
    for suffix,content in [('control_R0-R2',controls),('noise_R2-R5',noises)]:
        if (folder/'report_generation_issue.txt').exists():
            content += ['## 报告生成修复记录',
                        '首次合成数据报告生成在写核验标记时出现NameError（prior未定义）；已改为run.prior并重新生成、核验报告。'
                        '该错误发生在训练完成后的报告阶段，不涉及模型、噪声、损失、实验指标或选模规则变更。']
        content += ['## 运行、文件用途及保留策略',
                    '运行脚本experiments/run_student_cl_all_datasets.py复用原训练；报告脚本report_student_cl_all_datasets.py生成本文及配图。'
                    '全部独立进程使用CPU单线程以保留原数值协议，进程间不共享模型或RNG。'
                    'run_student_cl_seed2028.py及run_student_cl_seed2027.py只拆分既定种子任务；完成六个条件及核验后才接入，未覆盖已开始的主进程任务。'
                    '若主进程已开始其中部分条件，仅合并尚不存在的已核验条件；调度重复不算新增实验条件，也不根据指标挑选重复结果。'
                    '接入状态及该调度脚本哈希在上面的scheduling_records中记录。',
                    '```powershell\n'+manifest['command']+'\n```',
                    '本目录仅放最终Markdown和assets配图。原始checkpoint、逐轮日志的清理被自动审批拒绝（blocked by policy），仍暂存于scratch/student_cl_all_20261123，未绕过拒绝执行删除。'
                    '已有Math1正式结果不删除。本文保存配置、哈希、逐种子指标、梯度/参数核验和结论。未提交Git。']
        path=DEST/f'{name}_{suffix}_20261123.md'
        path.write_text('\n\n'.join(content)+'\n',encoding='utf-8')
        links=re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8'))
        assert all((path.parent/p).exists() for p in links)
    run.prior.legacy.write_json(folder/'reports_verified.json',{'reports':2,'figures':9,'all_links_exist':True,
        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.glob(name+'_*20261123.md')}})
    print(name+': 2 self-contained reports, 9 figures verified',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=run.DATASETS)
    p.add_argument('--publish-verified',action='store_true',help='Publish complete auxiliary groups, no training or report generation')
    a=p.parse_args()
    if a.publish_verified:
        for d in ([a.dataset] if a.dataset else run.DATASETS):
            for seed in (2027,2028):publish_verified(d,seed)
        return
    for d in ([a.dataset] if a.dataset else run.DATASETS):report(d)


if __name__=='__main__':main()
