"""Replay controlled checkpoints and generate the single experiment report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

from Q_GNCDM.experiments import run_simple_contrastive as core
from Q_GNCDM.experiments import run_controlled_student_cl as run


def table(headers, rows):
    def fmt(v):
        return (f'{v:.9g}' if isinstance(v, float) else str(v)).replace('|', '\\|')
    headers = [fmt(h) for h in headers]
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                       '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                      ['| ' + ' | '.join(fmt(v) for v in r) + ' |' for r in rows])


def matrix_md(array):
    return '```text\n' + np.array2string(array, precision=9, floatmode='unique',
                                       threshold=array.size + 1, max_line_width=240) + '\n```'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-id', default='student_cl_control_math1_20261123')
    cli = parser.parse_args()
    root = core.ROOT / 'Q_GNCDM/results' / cli.experiment_id
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    complete = json.loads((root / 'completed.json').read_text(encoding='utf-8'))
    assert complete['status'] == 'passed'
    args = argparse.Namespace(**manifest['args'])
    torch.set_num_threads(1)
    data = core.load_data(args)
    pre = json.loads((root / 'preflight.json').read_text(encoding='utf-8'))
    parity = json.loads((root / 'baseline_parity.json').read_text(encoding='utf-8'))
    report_dir = core.ROOT / 'Q_GNCDM/docs/experiments/student_cl'
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / (cli.experiment_id.replace('student_cl_control_', '').replace('_20261123', '') + '_control_R0-R2_20261123.md')
    rows, states, verification = [], {}, []
    try:
        for path in sorted(root.glob('A*/seed_*/R*/metrics.json')):
            saved = json.loads(path.read_text(encoding='utf-8'))
            mode, seed, condition = saved['mode'], saved['seed'], saved['condition']
            model = core.build_model(data, args, mode, seed)
            state = run.checkpoint_state(path.parent)
            model.load_state_dict(state)
            replay, _, _ = core.evaluate(model, data['train'], data['rows'][2])
            for metric, value in replay.items():
                assert value == saved['test_' + metric], (path, metric)
            exported = np.load(path.parent / 'weights.npz')
            for name in exported.files:
                if name.endswith('.effective'):
                    expected = state[name.removesuffix('.effective')].numpy().__abs__()
                else:
                    expected = state[name].numpy()
                np.testing.assert_array_equal(exported[name], expected)
            logs = list(csv.DictReader((path.parent / 'train_log.csv').open(encoding='utf-8-sig')))
            train_means = {key: float(np.mean([float(r[key]) for r in logs])) for key in logs[0]}
            if condition == 'R0':
                train_means['loss_original'] = train_means['loss_total']
                train_means['weighted_loss_cl'] = 0.
            # max returns the first tie, exactly the registered selector.
            assert saved['best_epoch'] == int(max(logs, key=lambda r: float(r['valid_acc']))['epoch'])
            saved['training_means'] = train_means
            rows.append(saved)
            states[(mode, seed, condition)] = state
            verification.append({'mode': mode, 'seed': seed, 'condition': condition,
                                 'metrics_exact_replay': True, 'weights_export_exact': True,
                                 'best_validation_acc_earliest_tie': True})
        for file, digest in manifest['code_sha256'].items():
            assert hashlib.sha256((core.ROOT / file).read_bytes()).hexdigest() == digest
        core.legacy.write_json(root / 'replay_verification.json', verification)
        aggregates, paired = [], []
        for mode in ('A0', 'A1'):
            for condition in ('R0', 'R1', 'R2'):
                group = [r for r in rows if r['mode'] == mode and r['condition'] == condition]
                item = {'mode': mode, 'condition': condition, 'n': len(group)}
                for metric in ('test_acc', 'test_auc', 'valid_acc', 'q_f1_vs_reference'):
                    vals = [r[metric] for r in group]
                    item[metric + '_mean'] = float(np.mean(vals))
                    item[metric + '_std'] = float(np.std(vals, ddof=1))
                aggregates.append(item)
            for seed in args.seeds:
                b = next(r for r in rows if (r['mode'], r['seed'], r['condition']) == (mode, seed, 'R1'))
                t = next(r for r in rows if (r['mode'], r['seed'], r['condition']) == (mode, seed, 'R2'))
                paired.append({'mode': mode, 'seed': seed, 'delta_acc': t['test_acc'] - b['test_acc'],
                               'delta_auc': t['test_auc'] - b['test_auc'],
                               'baseline_epoch': b['best_epoch'], 'treatment_epoch': t['best_epoch']})
        core.save_csv(root / 'aggregate.csv', aggregates)
        core.save_csv(root / 'paired_deltas.csv', paired)
        links = '../../../results/' + cli.experiment_id
        text = [f'# 同学生连续扰动 InfoNCE 受控实验：{cli.experiment_id}',
                '## 结论与适用范围',
                '本轮仅运行 Math1，固定 Q 与 Weak-Q 两种设置各做 R0/R1/R2，三个配对种子，共 18 次训练。'
                'R0/R1 的全部逐轮原始日志、最佳 checkpoint 张量和测试指标精确一致。'
                '新增 CL 确实产生非零共享梯度及不同的单步参数更新；这证明模块参与优化，不等于证明预测收益。'
                '没有扩展噪声搜索、其他损失、其他数据集或以测试结果调参。',
                'R0 是原有训练循环的参考副本，不是未经调整的历史 AUC 选模结果。'
                'R0/R1/R2 均使用用户预先指定的最佳验证 ACC，严格大于才更新（并列选最早）。'
                '参考副本只调整选模标量和精简保存；原模型、数据、原损失、评估函数保持不变。',
                table(['Q 设置', '条件', 'test ACC 均值', 'ACC 样本标准差', 'test AUC 均值', 'AUC 样本标准差'],
                      [[a['mode'], a['condition'], a['test_acc_mean'], a['test_acc_std'], a['test_auc_mean'], a['test_auc_std']] for a in aggregates]),
                '逐种子 R2−R1 差值（比例单位；乘 100 为百分点）：',
                table(['Q 设置', '种子', 'ΔACC', 'ΔAUC', 'R1 最佳轮', 'R2 最佳轮'],
                      [[r[k] for k in ('mode', 'seed', 'delta_acc', 'delta_auc', 'baseline_epoch', 'treatment_epoch')] for r in paired])]
        for mode in ('A0', 'A1'):
            pp = [r for r in paired if r['mode'] == mode]
            text.append(f"{mode} 平均 ΔACC={np.mean([r['delta_acc'] for r in pp]):+.9f}，"
                        f"平均 ΔAUC={np.mean([r['delta_auc'] for r in pp]):+.9f}。"
                        '仅三个训练种子，不能将微小差异解释为统计显著或可推广的收益。')
        text.append('固定Q的seed2027是本轮主要负向差异来源：R1选择第29轮，R2选择第19轮，'
                    'ΔAUC约−0.003989。验证ACC是离散指标，小幅轨迹变化可能改变最佳轮与早停；'
                    '本轮按预定规则保留该结果，没有改回AUC选模。Weak-Q三个种子的最佳轮分别相同，'
                    '平均ACC增量约0.0000198，仅相当于三个测试运行合计多判对1条，不能称为稳定收益。')
        text += ['## 协议与固定配置',
                 'A0：固定参考硬 Q；A1：可学习软 Q，MSE 弱监督与 coverage 保持原设置。'
                 '各 Q 设置内 R0=参考无 CL，R1=新增入口 λ=0，R2=新增入口 λ=0.1。'
                 '对应消融名称：固定 Q 的 R1/R2 为 A0/C0；Weak-Q 的 R1/R2 为 A1/S1。',
                 table(['配置', '值'], [[k, v] for k, v in vars(args).items()]),
                 '固定 Math1 原划分：58926/8418/16836 条 train/valid/test；学生 4209、题目 20、知识点 11。'
                 'Adam、每批128名学生、query比例0.25、最多30轮、patience=8。'
                 '主损失已包含 Weak-Q 和 coverage，不重复添加。ACC 仍为概率严格 >0.5，AUC 计算函数不变。',
                 '### 三视图和损失',
                 '两个输入不拼接。Anchor 是学生 support 行；只扰动其已观测 ±1 位置，习题输入保持原样。'
                 '三视图共用 diagnose_theta，Anchor 的融合掌握度同时进入原始预测。'
                 '新增分支不修改标签、query、学生 ID、mask 或原数组。验证/测试不增强。',
                 r'$L_{CL}=-\mathrm{mean}\log\frac{\exp(\cos(m,m^+)/0.2)}{\exp(\cos(m,m^+)/0.2)+\exp(\cos(m,m^-)/0.2)}$。',
                 '每个 Anchor 只有同一学生的一个正视图与一个强扰动负视图，无跨学生负对。'
                 '此 InfoNCE 约束相对相似性，没有绝对距离下界；margin、margin 激活比例均不适用，记录为 null，不能称其满足 margin。'
                 'cosine 使用概念维 L2 归一化；同时报告未归一化掌握度的逐元素 MSE。',
                 'positive/negative_noise_ratio 是每个有效位置的 Bernoulli 选中概率，并非保证每名学生恰好该比例。'
                 'uniform 噪声范围分别为 [-0.01,0.01] 与 [-0.1,0.1]；裁剪到 [-1,1]，不会翻转符号。'
                 '由于原值在边界，大约一半被选中位置被裁剪抵消：配置比例、实际改变比例必须分开报告。'
                 '短记录学生可能没有实际变化，这是本轮固定设置的局限，未偷偷强制至少修改一题。',
                 '## 自动测试的真实结果',
                 '[完整预检输出](' + links + '/preflight.json)、[完整训练零系数一致性](' + links + '/baseline_parity.json)、'
                 '[checkpoint 重放核验](' + links + '/replay_verification.json)。',
                 '增强测试覆盖形状/dtype、零比例/零幅度、缺失不变、原数组不变、取值合法、有限值、同种子复现及不同种子差异。'
                 '大样本选中比例与配置误差<0.01。模型测试使用同一学生列表、同一初始化与同一 Torch RNG 状态。'
                 'λ=0只调用一次掌握度生成，λ=0.1调用三次；增强分支不消耗 Torch RNG。'
                 '完整 R1 训练将对比分支替换为调用即报错的哨兵，18次训练中的6个 R1 均未触发。'
                 '单步预测、原损失、梯度与更新的 R0/R1 比较精确一致；完整训练也逐轮核验。',
                 table(['Q 设置', '原始损失', 'CL 损失', '总损失', 'CL 梯度范数', 'CL/原梯度', '0.1CL/原梯度'],
                       [[m, pre[m]['loss_original'], pre[m]['loss_cl'], pre[m]['loss_total'],
                         pre[m]['gradient']['cl_shared_grad_norm'], pre[m]['gradient']['cl_to_original_grad_ratio'],
                         pre[m]['gradient']['weighted_cl_to_original_grad_ratio']] for m in ('A0','A1')]),
                 '单步差异（固定 batch、同初始参数、同 Adam 初态、同 Dropout RNG）：',
                 table(['Q 设置','参数','梯度差 L2','一步参数差 L2','无CL更新 L2','有CL更新 L2'],
                       [[m,k,v['gradient_difference_l2'],v['update_difference_l2'],v['baseline_update_l2'],v['cl_update_l2']]
                        for m in ('A0','A1') for k,v in pre[m]['single_step_parameter_changes'].items()]),
                 '## 每个种子的评估结果',
                 table(['Q设置','seed','条件','最佳轮','训练轮数','valid ACC','test ACC','test AUC','Q-F1'],
                       [[r[k] for k in ('mode','seed','condition','best_epoch','epochs_run','valid_acc','test_acc','test_auc','q_f1_vs_reference')] for r in rows]),
                 '## 训练诊断',
                 '以下为每轮 batch 均值再对训练轮求均值，不是最后一轮或最佳轮，也不是按交互数加权。'
                 'R0/R1 未执行 CL，CL 损失为0；三视图距离和 CL 梯度没有测量，不能当作零距离解释。',
                 table(['Q','seed','条件','原损失(含Q)','CL','加权CL','总损失'],
                       [[r['mode'],r['seed'],r['condition']] + [r['training_means'][k] for k in
                         ('loss_original','loss_cl','weighted_loss_cl','loss_total')] for r in rows])]
        r2 = [r for r in rows if r['condition']=='R2']
        for title, keys in [
            ('掌握度距离与梯度', ['ap_mse','an_mse','ap_cos_distance','an_cos_distance','cl_shared_grad_norm',
                                'cl_to_original_grad_ratio','weighted_cl_to_original_grad_ratio']),
            ('实际输入变化', ['positive_selected_fraction','positive_changed_fraction','positive_input_mae_observed',
                             'negative_selected_fraction','negative_changed_fraction','negative_input_mae_observed']),
            ('裁剪与负视图区分', ['positive_clipped_fraction_selected','negative_clipped_fraction_selected',
                                 'negative_softmax_probability','negative_not_farther_fraction'])]:
            text += ['### '+title, table(['Q','seed']+keys, [[r['mode'],r['seed']]+[r['training_means'][k] for k in keys] for r in r2])]
        text += ['InfoNCE 接近 ln(2)=0.693147、负视图 softmax 概率接近0.5，说明余弦尺度上的两视图区分很弱。'
                 '损失数值约0.69并不代表有效梯度很强；需看梯度比值。'
                 '强扰动同学生是否是语义负样本仍未得到证明，不能由参数变化直接推出诊断更准确。',
                 '## 学到的参数矩阵 W',
                 '所有层的原始 weight/bias、q_logits 及学生侧 PosLinear 的有效 |W| 均保存到各运行目录的 weights.npz；'
                 'best_model.pt 保存完整状态。矩阵顺序是 PyTorch 的 [输出维,输入维]。'
                 '固定 Q 组的 q_logits 不参与学习；可学习 Q 应解释为 sigmoid(q_logits)。'
                 'θ 本身是输入生成的掌握度，不是独立参数矩阵。',
                 '[全部种子逐层参数差异](' + links + '/parameter_comparison.csv)。'
                 'R0与R1参数完全相同；以下展示seed2026的R1、R2及差矩阵，避免重复R0。'
                 '完整训练差异含训练轨迹与可能不同最佳轮的共同作用；因果生效证据以同batch单步测试为准。']
        parameter_rows = list(csv.DictReader((root / 'parameter_comparison.csv').open(encoding='utf-8-sig')))
        text += ['所有层的参数差异摘要（seed2026；其他种子见上述CSV）：',
                 table(['Q','参数','形状','R2-R1 L2','最大绝对差','有效|W|差L2'],
                       [[r['mode'],r['parameter'],r['shape'],r['difference_l2'],r['difference_max'],
                         r['effective_abs_weight_difference_l2']] for r in parameter_rows if r['seed']=='2026'])]
        for mode in ('A0','A1'):
            seed=2026
            for condition in ('R0','R1','R2'):
                text.append(f'[{mode}/seed_{seed}/{condition} 全部参数]({links}/{mode}/seed_{seed}/{condition}/weights.npz)')
            for name in ('f_nn.f_layer_1.weight','f_nn.f_layer_2.weight'):
                a=states[(mode,seed,'R1')][name].numpy()
                b=states[(mode,seed,'R2')][name].numpy()
                text += [f'### {mode}，seed2026，{name}，形状{a.shape}',
                         'R1 原始 W：',matrix_md(a),'R2 原始 W：',matrix_md(b),
                         'R2−R1 原始 W：',matrix_md(b-a),
                         '有效权重差 |W_R2|−|W_R1|：',matrix_md(np.abs(b)-np.abs(a))]
        text += ['## 文件用途、执行命令和状态',
                 table(['路径','用途'],[
                     ['experiments/controlled_student_cl.py','连续增强、两候选InfoNCE、梯度诊断；无新增网络'],
                     ['experiments/reference_student_cl_acc.py','原循环参考副本；ACC选模、精简保存'],
                     ['experiments/run_controlled_student_cl.py','预检、受控训练、零分支拦截、完整训练一致性、W导出'],
                     ['experiments/summarize_controlled_student_cl.py','checkpoint重放、汇总、生成本文'],
                     ['results/'+cli.experiment_id+'/manifest.json','参数、命令、Git版本、代码和数据哈希'],
                     ['results/.../A0或A1/seed_*/R0或R1或R2/','config、逐轮train_log、metrics、best_model、weights'],
                     ['results/.../summary.csv、aggregate.csv、paired_deltas.csv','逐运行结果、均值标准差、配对差值'],
                     ['results/.../preflight.json、baseline_parity.json、replay_verification.json、completed.json','自动测试及完成证据'],
                     ['docs/experiments/'+cli.experiment_id+'/结果报告.md','本轮唯一解释文档'],
                     ['README.md','新增本轮入口；不覆盖历史结论']]),
                 '未修改既有模型、既有训练入口、原损失、原评估函数、数据或旧结果。未提交Git。',
                 '实际训练命令（一个入口依次执行全部R0/R1/R2）：\n\n```powershell\n'+manifest['command']+'\n```',
                 '报告/重放命令：\n\n```powershell\n.venv-contrastive/Scripts/python.exe -m Q_GNCDM.experiments.summarize_controlled_student_cl\n```',
                 f'训练完成状态：{complete}。预检两种Q设置passed；完整训练{len(parity)}组R0/R1精确一致；{len(verification)}个checkpoint重放全部通过。',
                 '## 尚未解决的问题',
                 '- 仅Math1、3个种子；未验证其他数据集和5种子以上稳定性。\n'
                 '- 同学生强噪声负对缺少语义保证，裁剪使实际扰动比例减半，部分学生视图无变化。\n'
                 '- cosine归一化削弱绝对掌握度差异，本轮不切换损失来追求分数。\n'
                 '- 不同条件可能选择不同epoch，完整W差异不能独立归因于单步梯度强度。\n'
                 '- 30轮预算和patience沿用基线，不保证充分收敛。\n'
                 '- InfoNCE没有margin，不能提供margin激活率或负距离下界保证。\n'
                 '- 未证明认知诊断真实性改善；Q-F1只衡量与参考Q一致性。']
        report_file.write_text('\n\n'.join(text)+'\n', encoding='utf-8')
        core.legacy.write_json(root / 'report_manifest.json', {
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'report_sha256': hashlib.sha256(report_file.read_bytes()).hexdigest(),
            'command': [sys.executable, '-m', 'Q_GNCDM.experiments.summarize_controlled_student_cl', *sys.argv[1:]],
            'checkpoints_replayed': len(verification)})
        print(report_file)
        print(table(['mode','seed','delta_acc','delta_auc'], [[p[k] for k in ('mode','seed','delta_acc','delta_auc')] for p in paired]))
    except BaseException:
        (root / 'report_failure.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__=='__main__':
    main()
