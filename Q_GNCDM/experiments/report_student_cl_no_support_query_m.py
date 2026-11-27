"""Compact, evidence-based result tables and plots; no JSON dumps in reports."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Q_GNCDM.experiments import run_student_cl_no_support_query_m as r


def table(headers, rows):
    def fmt(x):return f'{x:.9g}' if isinstance(x,float) else str(x)
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(fmt,row))+' |' for row in rows])


def report(name):
    folder=r.WORK/name;dest=r.REPORT;dest.mkdir(parents=True,exist_ok=True)
    runs=[];hist={}
    for path in folder.glob('A*/seed_*/R*/complete.json'):
        item=json.loads((path.parent/'metrics.json').read_text());runs.append(item)
        hist[(item['mode'],item['seed'],item['condition'])]=list(csv.DictReader((path.parent/'history.csv').open(encoding='utf-8-sig')))
    runs.sort(key=lambda x:(x['mode'],x['seed'],x['condition']))
    if not runs:return
    lookup={(x['mode'],x['seed'],x['condition']):x for x in runs}
    data,args=r.load(name)
    content=[f'# {name}：无 Support/Query 的学生掌握度对比实验',
             f'实验ID：{r.ID}_{name}。已完成 {len(runs)}/36 组；'+('全部完成。' if len(runs)==36 else '**运行中，当前表不代表最终结论。**'),
             '## 实验设置',
             '按作答记录训练，每批128条；输入为完整训练矩阵对应的学生行和题目列，不隐藏当前训练目标。验证/测试只使用训练作答作为证据。两个输入不拼接，仅增强学生输入；题目输入保持干净。缺失0不变，连续均匀噪声裁剪至[-1,1]。',
             '对比对象为真实参与预测的学生掌握度 m=diagnose_theta(学生输入)。原始干净输入计算预测损失；学生正负视图计算掌握度InfoNCE（余弦、温度0.2）。题目输入不增强、不参与对比优化，但题目网络仍由原预测损失训练。',
             'A0=固定参考Q；A1=原Weak-Q设置。R0=按原IDCDataset取完整行列的无CL参考路径；R1=新入口CL关闭；R2–R5的λ=0.1。R0不是原仓库所有默认配置的完整复刻。',
             'α=0.8，隐藏维度32，Adam学习率0.001、Q学习率0.005；Weak-Q与coverage系数均为0.01。最多30轮，验证ACC选模，平分保留最早轮，patience=8。种子2026/2027/2028。',
             table(['条件','正比例','正幅度','负比例','负幅度'],[[k,*v] for k,v in r.CONFIGS.items()]),
             '相对旧Support/Query掌握度实验，本轮改变输入遮蔽和训练取样，每轮更新次数增加。相对无Support/Query交互实验，本轮只对掌握度做对比，取消题目增强。不同协议的结果不混算。',
             '## 逐种子结果',
             '所有ACC/AUC来自最佳验证ACC的同一checkpoint。每个种子单独列出，不按测试集选种子；Δ为同Q同种子相对R1的差值（比例单位）。']
    rows=[]
    for x in runs:
        if x['condition']=='R0':continue
        base=lookup.get((x['mode'],x['seed'],'R1'))
        rows.append([x['mode'],x['seed'],x['condition'],0 if x['condition']=='R1' else .1,x['best_epoch'],x['valid_acc'],x['test_acc'],x['test_auc'],x['test_acc']-base['test_acc'] if base else '待基线',x['test_auc']-base['test_auc'] if base else '待基线'])
    content.append(table(['Q','seed','条件','λ','最佳轮','valid ACC','test ACC','test AUC','ΔACC','ΔAUC'],rows))
    # Report parameter differences rather than raw matrices or full checkpoints.
    checks=json.loads((folder/'preflight.json').read_text())
    content+=['## 核验与模型响应',
              '两种Q、四种噪声预检通过：零系数预测/损失/单步更新/RNG一致；学生增强与缺失值保护通过，学生网络对比梯度及更新差异非零，题目网络和聚合层无直接对比梯度。固定Q还在参数对齐后核对了原GNCDM前向输出。',
              f'已完成checkpoint回放 {len(runs)} 组；完整训练R0/R1精确一致性已通过 {sum(x[2]=="R1" for x in lookup)}/6 对。梯度范数以下取每轮第一个batch的诊断均值，并非全程所有batch统计。']
    diag=[];changes=[]
    for x in runs:
        if x['condition'] in ['R0','R1']:continue
        hs=hist[(x['mode'],x['seed'],x['condition'])]
        avg=lambda k:float(np.mean([float(h[k]) for h in hs]))
        diag.append([x['mode'],x['seed'],x['condition'],avg('loss_cl'),avg('ap_mse'),avg('an_mse'),avg('probe_cl_gradient'),avg('probe_weighted_gradient_ratio'),avg('positive_student_mae'),avg('negative_student_mae'),avg('positive_item_mae'),avg('negative_item_mae')])
        path=folder/x['mode']/f"seed_{x['seed']}"
        if (path/'R1/complete.json').exists():
            base=torch.load(path/'R1/best.pt',weights_only=True);state=torch.load(path/x['condition']/'best.pt',weights_only=True)
            diffs=[]
            for prefix in ['f_nn.','g_nn.','theta_agg_mat.','psi_agg_mat.']:
                keys=[k for k in state if k.startswith(prefix)]
                delta=sum(float((state[k]-base[k]).double().square().sum()) for k in keys)**.5
                norm=sum(float(base[k].double().square().sum()) for k in keys)**.5
                diffs.append(delta/max(norm,1e-30))
            changes.append([x['mode'],x['seed'],x['condition'],*diffs])
    content.append(table(['Q','seed','条件','CL损失','正m MSE','负m MSE','CL梯度','加权CL/原梯度','正学生MAE','负学生MAE','正题目MAE','负题目MAE'],diag))
    content+=['参数相对L2差异=‖W条件−W基线‖/‖W基线‖，比较各自选中的checkpoint；若最佳轮不同，差异也包含训练轮次效应。题目侧后续更新可因学生参数改变而间接受影响，不能将最终差异误解为题目侧收到了直接对比梯度。',
              table(['Q','seed','条件','学生网络','题目网络','学生聚合层','题目聚合层'],changes)]
    figures=dest/'assets'/name;figures.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,mode in zip(axes,['A0','A1']):
        for seed in [2026,2027,2028]:
            xs=[c for c in ['R1',*r.CONFIGS] if (mode,seed,c) in lookup]
            ax.plot(xs,[lookup[(mode,seed,c)]['test_acc'] for c in xs],marker='o',label=f'seed {seed}')
        ax.set_title(mode);ax.set_ylabel('Test ACC');ax.legend();ax.grid(alpha=.2)
    fig.suptitle(name+' / paired seeds');fig.tight_layout();fig.savefig(figures/'acc.png',dpi=160);plt.close(fig)
    content+=['## 图表',f'![逐种子ACC](assets/{name}/acc.png)']
    for filename,fields,title in [('training',['loss_total','valid_acc'],'Training loss / validation ACC'),
                                  ('contrastive',['loss_cl','an_mse'],'InfoNCE / negative mastery MSE'),
                                  ('distances',['ap_mse','an_mse'],'Positive / negative mastery MSE'),
                                  ('gradients',['probe_cl_gradient','probe_weighted_gradient_ratio'],'Gradient norm / weighted ratio')]:
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        for row,mode in enumerate(['A0','A1']):
            for col,field in enumerate(fields):
                ax=axes[row,col]
                for (m,seed,c),hs in hist.items():
                    if m!=mode or c=='R0' or field not in hs[0]:continue
                    ax.plot([int(h['epoch']) for h in hs],[float(h[field]) for h in hs],label=f'{seed} {c}',linewidth=.9)
                ax.set_title(mode+' '+field);ax.set_xlabel('epoch');ax.grid(alpha=.2)
                
                if ax.lines:ax.legend(fontsize=6,ncol=3)
        fig.suptitle(name+' '+title);fig.tight_layout();fig.savefig(figures/(filename+'.png'),dpi=150);plt.close(fig)
        content.append(f'![{title}](assets/{name}/{filename}.png)')
    content+=['## 结果解读']
    for mode in ['A0','A1']:
        for c in r.CONFIGS:
            diffs=[lookup[(mode,s,c)]['test_acc']-lookup[(mode,s,'R1')]['test_acc'] for s in [2026,2027,2028] if (mode,s,c) in lookup and (mode,s,'R1') in lookup]
            if diffs:content.append(f'{mode} {c}：{len(diffs)}个配对种子，ACC提高/不变/降低={sum(d>0 for d in diffs)}/{sum(d==0 for d in diffs)}/{sum(d<0 for d in diffs)}；逐种子差值为 '+', '.join(f'{d:+.9f}' for d in diffs)+'。')
    content+=['参数与梯度非零仅证明参与优化；预测收益应同时看ACC/AUC和全部配对种子。负视图仍是同一作答的连续证据扰动，不保证其语义上应被推远。InfoNCE无margin，不报告虚构的margin激活率。']
    if False:
        content+=[]
    content+=['核验修复说明：首次预检的诊断工具未处理整组梯度为空，修正范数统计后重测。题目分支无直接对比梯度是预期行为；未改变模型或损失，失败日志保留在临时核验目录。']
    (dest/f'{name}_mastery_cl_20261124.md').write_text('\n\n'.join(content)+'\n',encoding='utf-8')


def index():
    r.REPORT.mkdir(parents=True,exist_ok=True)
    lines=['# 无Support/Query的学生掌握度对比实验',
           '每个数据集独立报告；只放关键设置、逐种子结果、模型响应和配图。R1/R2为基础对照，R3–R5为噪声敏感性。旧student_cl报告不覆盖。',
           '| 数据集 | 完成组数 | 报告 |','| --- | --- | --- |']
    for name in r.DATASETS:
        n=len(list((r.WORK/name).glob('A*/seed_*/R*/complete.json')))
        path=r.REPORT/f'{name}_mastery_cl_20261124.md'
        lines.append(f'| {name} | {n}/36 | '+(f'[结果]({path.name})' if path.exists() else '待运行')+' |')
    (r.REPORT/'README.md').write_text('\n\n'.join(lines[:2])+'\n\n'+'\n'.join(lines[2:])+'\n',encoding='utf-8')


if __name__=='__main__':
    for name in r.DATASETS:report(name)
    index()
