# Markdown 文档清单（2026-10-29）

本清单覆盖仓库内 12 份原有 Markdown（项目根目录 4 份、`Q_GNCDM` 7 份、`GNCDM` 1 份）和本次新建的 2 份索引。路径和入站引用均按**整理后**状态记录；引用列不计本清单自身对其他文档的导航链接。`Q_GNCDM/results/` 下的文件本次只读，保留原位。这里的分类表示文档用途；“正式实验记录”不表示旧实验与当前 A0/A1/S1 协议可直接比较。结论和数值均以原文为准，本清单不替代实验报告。

| 分类 | 文件路径 | 主要内容 | 仍属当前实验？ | 与其他文档的重叠 | 被哪些文件引用（不计本清单） |
| --- | --- | --- | --- | --- | --- |
| 主线入口 | [README.md](../../README.md) | 原论文代码库英文概览、安装、引用 | 否；提供上游项目背景 | 与 `README.zh-CN.md` 为中英对应版本，非多余副本 | 未发现 |
| 主线入口 | [README.zh-CN.md](../../README.zh-CN.md) | 上游项目中文概览、安装、引用 | 否；提供上游项目背景 | 与 `README.md` 为翻译关系 | 未发现 |
| 主线入口 | [Q_GNCDM/README.md](../README.md) | Weak-Q／简单对比主线导航、主要结果及清理说明 | 是，唯一 Q_GNCDM 主入口 | 摘要复述 `实验结果.MD` 的部分指标；详细证据只在后者 | 根目录中英 README、`实验结果.MD`、归档 `INDEX.md` 和 `result6_9.md` |
| 主线入口 | [GNCDM/README.md](../../GNCDM/README.md) | 上游 GNCDM 子项目说明 | 否；上游模块入口 | 与根目录上游 README 有项目背景重叠，作用不同 | 未发现 |
| 正式实验记录 | [实验结果.MD](../../实验结果.MD) | 当前 A0/A1/S1、Math1／人工数据、权重与噪声搜索的协议、结果、核验 | 是，当前正式记录 | 与 Q_GNCDM 入口有结论摘要重叠；与第一版方案有模式定义重叠，均非全文重复 | `Q_GNCDM/README.md`、归档 `INDEX.md` |
| 历史方案 | [对比学习simple.MD](archive/2026-10-29/对比学习simple.MD) | 第一版简单对比的正负样本、损失、默认值和验收标准 | 否；是当前 S1 的设计来源，但后续执行及选参以正式结果为准 | 与 `实验结果.MD` 的框架说明重叠；独有原始假设与验收标准 | `实验结果.MD`、归档 `INDEX.md` |
| 正式实验记录 | [result6_9.md](archive/2026-10-29/result6_9.md) | 2026-06-09 旧合成人工数据 Weak-Q lambda sweep 的 Top 20 排名 | 否；单种子探索，不能作当前无偏测试结论 | 与旧 `weak_q_lambda_sweep/summary.csv` 数值重叠；Markdown 中无全文副本 | 归档 `INDEX.md` |
| 正式实验记录 | [math1_real_experiment_report.md](../results/math1_real/math1_real_experiment_report.md) | 旧 Math1 expert-Q 代表配置探针 | 否；旧训练协议 | 与 `math1_lambda_grid_report.md` 共用 fixed/default/free 基线，其他分析独立 | `Q_GNCDM/experiments/run_math1_real_experiment.py`（生成路径） |
| 正式实验记录 | [math1_lambda_grid_report.md](../results/math1_lambda_grid/math1_lambda_grid_report.md) | 旧 Math1 90 组 lambda 网格、验证集选择、排序 | 否；旧训练协议 | 与 `math1_real_experiment_report.md` 共用三项基线，搜索细节独立 | `Q_GNCDM/experiments/sweep_math1_real_lambdas.py`（生成路径） |
| 正式实验记录 | [math1_noisy_q_experiment_report.md](../results/math1_noisy_q/math1_noisy_q_experiment_report.md) | 旧 Math1 人工 Q 噪声鲁棒性实验 | 否；旧训练协议 | clean-Q 基线与 `math1_real_experiment_report.md` 重叠，噪声实验独立 | `Q_GNCDM/experiments/run_math1_noisy_q_experiment.py`（生成路径） |
| 正式实验记录 | [experiment_results.md](../results/weak_q_synthetic/experiment_results.md) | 旧人工数据 seed 2026、Q 噪声 0.10 的 A–E 结果 | 否；旧协议的单种子记录 | 与 `FIGURE_NAMING_RULES.md` 的 A–E 标签说明同题，但数值表独立 | `Q_GNCDM/results/weak_q_synthetic/figures/FIGURE_NAMING_RULES.md`（正文路径） |
| 历史方案 | [FIGURE_NAMING_RULES.md](../results/weak_q_synthetic/figures/FIGURE_NAMING_RULES.md) | 旧 Weak-Q 图目录命名与 A–E 图示规则 | 否；旧实验图示约定 | 提及 `experiment_results.md` 的实验标签，非结果副本 | 未发现 |

## 本次新建的索引文档

| 分类 | 文件路径 | 主要内容 | 仍属当前实验？ | 与其他文档的重叠 | 被哪些文件引用 |
| --- | --- | --- | --- | --- | --- |
| 主线入口（辅助索引） | [DOCUMENT_INVENTORY.md](DOCUMENT_INVENTORY.md) | 全仓 Markdown 的用途、重复关系及引用盘点 | 是；当前文档导航，不是实验结论 | 与 `Q_GNCDM/README.md` 的导航部分少量重叠 | `Q_GNCDM/README.md`、归档 `INDEX.md` |
| 主线入口（归档索引） | [archive/2026-10-29/INDEX.md](archive/2026-10-29/INDEX.md) | 本次两份归档文档的原路径、原因和恢复提示 | 否；历史文档导航 | 与本清单的归档决定重叠，提供逐份恢复路径 | `Q_GNCDM/README.md`、本清单 |

## 整理决定

- 保留当前正式结果和 Q_GNCDM 入口；根目录中英 README 与 `GNCDM/README.md` 继续承担上游项目说明。
- 第一版方案已由实际执行记录补充，旧 Top 20 报告的排序又不适合用作当前结论；两份原文移至 [2026-10-29 归档](archive/2026-10-29/INDEX.md)，保留原文件名和原始数值。
- 五份 `Q_GNCDM/results/` Markdown 本次不移动、不编辑。它们各有独立实验或图示用途，不认定为可删除的重复副本。
- 待确认：无。若后续决定统一存放旧结果报告，需另行授权，因为本次 `results/` 只读。
