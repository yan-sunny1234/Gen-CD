# Q_GNCDM 实验导航

当前工作分支：`simple_contristive`。本目录的主线是 Weak-Q G-NCDM，以及在相同实验协议下加入的最简学生表征对比学习。详细结果见项目根目录的 [实验结果.MD](../实验结果.MD)。

Math1 输入位于本目录的 `data/math1/`：[训练集](data/math1/math1_train_0.8_0.2.csv)、[验证集](data/math1/math1_valid_0.8_0.2.csv)、[测试集](data/math1/math1_test_0.8_0.2.csv)和[专家 Q 矩阵](data/math1/math1_Q_matrix.npy)；从项目根目录运行实验时使用 `--data-dir Q_GNCDM/data/math1`。

## 数据集

| 数据目录 | 学生 | 题目 | 知识点 | 去重后作答 |
| --- | ---: | ---: | ---: | ---: |
| [math1](data/math1/) | 4,209 | 20 | 11 | 84,180 |
| [math2](data/math2/) | 3,911 | 16 | 16 | 62,576 |
| [eedi_task34](data/eedi_task34/) | 4,918 | 948 | 25 | 1,382,727 |
| [assist2012](data/assist2012/) | 4,672 | 984 | 98 | 236,439 |
| [junyi2015](data/junyi2015/) | 5,000 | 500 | 39 | 531,967 |
| [synthetic_weak_q/seed_2026](data/synthetic_weak_q/seed_2026/)（合成） | 200 | 30 | 8 | 6,000 |

Math2 保留 16 道二值客观题；Eedi 使用题目元数据二级学科标签构建 Q；ASSISTments 2012–2013 和 Junyi 2015 是为当前密集矩阵模型选取的高频子集，分别按技能标签和练习主题构建 Q。后三者的 Q 是参考标签，**不是专家验证的真 Q**。Eedi 未使用官方 Task 4 冷启动划分，不能跨数据集直接相减指标当作模块收益。四个新增真实数据集都采用 `train.csv`、`valid.csv`、`test.csv`、`Q_matrix.npy`、兼容现有诊断函数的 `Q_true.npy` 和 `manifest.json`；[Math2](experiments/prepare_math2.py)、[Eedi](experiments/prepare_eedi_task34.py)、[ASSIST](experiments/prepare_assist2012.py)、[Junyi](experiments/prepare_junyi2015.py) 的预处理脚本可用于重建，处理后的数据已被 Git 忽略。

新数据的统一入口是 [run_prepared_dataset.py](experiments/run_prepared_dataset.py)。例如从项目根目录运行 `.\.venv-contrastive\Scripts\python.exe -m Q_GNCDM.experiments.run_prepared_dataset --dataset assist2012 --check-only` 检查加载；正式 A1/S1 对照使用 `--dataset assist2012 --experiment-id assist2012_cl_v1 --modes A1 S1`，结果进入 `results/assist2012_cl_v1/`。其他数据只需替换数据集名与实验 ID。默认训练种子为 2026/2027/2028，仅按验证 AUC 选模；如需固定 Q 与 Weak-Q 对照，另设实验 ID 并使用 `--modes A0 A1`。

全部 Markdown 的用途、重复关系和引用位置见[文档清单](docs/DOCUMENT_INVENTORY.md)；不再作为主线说明的第一版方案和旧排序报告见[归档索引](docs/archive/2026-10-29/INDEX.md)。

## 目前真正有收益的部分

| 模块／设置 | Math1 三训练种子结果 | 判断 |
| --- | --- | --- |
| 固定专家 Q（A0） | Test AUC 0.831658；ACC 0.739942 | 同协议基线 |
| Weak-Q 整体配置（A1） | Test AUC 0.837271；ACC 0.744021；相对 A0 为 **+0.005613 AUC、+0.004079 ACC** | 目前最明确的预测收益；RMSE −0.003169，NLL −0.007307 |
| 简单对比 S1，`lambda_cl=0.01` | 相对 A1：AUC −0.000029；ACC +0.000376 | 未显示一致收益 |
| 简单对比 S1，`lambda_cl=0.05`、实际翻转率 0.10 | 相对 A1：AUC +0.000083；ACC +0.000475 | 验证集选出的后续候选，增益太小，尚不能称为有效模块 |

表中比较来自同一个 Math1 数据划分、训练种子 `2026/2027/2028` 的 [主实验汇总](results/simple_contrastive_math1_v1/aggregate.csv)、[配对差值](results/simple_contrastive_math1_v1/paired_comparisons.csv) 和 [噪声搜索汇总](results/simple_contrastive_noise_sweep_math1_v1/sweep_aggregate.csv)。这只是三个训练随机种子，不是三个独立数据集。真实 Math1 没有隐藏的真 Q，Q-F1 只表示与专家 Q 的一致性；A1 的参考 Q-F1 为 0.445，A0 的 1.0 是固定专家 Q 的定义结果。Weak-Q 的预测增益不能归因到某一个独立正则项，因为尚无逐项消融。

旧 `math1_real`、`math1_noisy_q` 与当前 A0/A1/S1 的训练协议不同，不能跨表计算收益。旧噪声 Q 实验的 Weak-Q 预测有时改善，但没有证明 Q 结构修复；合成数据中的 mastery 指标也提醒我们，预测好不代表诊断结构好。

## 当前实验入口和应保留的结果

| 路径 | 用途 |
| --- | --- |
| [models/weak_q_gncdm.py](models/weak_q_gncdm.py)、[simple_contrastive.py](simple_contrastive.py) | Weak-Q 模型与最简作答翻转／InfoNCE |
| [experiments/run_simple_contrastive.py](experiments/run_simple_contrastive.py) | A0/A1/S1 统一训练入口 |
| [experiments/run_prepared_dataset.py](experiments/run_prepared_dataset.py) | Math2、Eedi 等已预处理真实数据的统一训练入口；每个实验 ID 只运行一个基线与改动的配对 |
| [experiments/run_weak_q_synthetic.py](experiments/run_weak_q_synthetic.py) | 当前训练和核验仍调用其中的数据、划分与指标函数；必须留在原路径 |
| [experiments/verify_simple_contrastive.py](experiments/verify_simple_contrastive.py)、[tests/test_simple_contrastive.py](tests/test_simple_contrastive.py) | 重放 checkpoint 与功能测试 |
| [results/simple_contrastive_math1_v1](results/simple_contrastive_math1_v1) | 三种子 A0/A1/S1 主基线；保留完整 checkpoint、预测和配置 |
| [results/simple_contrastive_math1_lambda_0p05_v1](results/simple_contrastive_math1_lambda_0p05_v1) | 当前候选 S1 的三种子完整结果 |
| [results/simple_contrastive_lambda_sweep_math1_v1](results/simple_contrastive_lambda_sweep_math1_v1)、[results/simple_contrastive_noise_sweep_math1_v1](results/simple_contrastive_noise_sweep_math1_v1) | 验证集选参、完整汇总及 A1 配对差值 |

其余 `simple_contrastive_math1_*` 的 0.001 权重与 0.20/0.30 噪声运行、`simple_contrastive_synthetic_v1` 仍留在 `results/`，用于展示参数敏感性和失败条件。名义噪声率 0.05 与 0.10 都实际翻转每人 1/10 条作答，结果完全重复；0.05 运行暂留原位，以保持已有汇总文件的路径有效，不能算第二个独立条件。

`experiments/` 中的 `run_math1_real_experiment.py`、`run_math1_noisy_q_experiment.py`、`sweep_math1_real_lambdas.py`、`sweep_weak_q_lambdas.py`、`summarize_weak_q_lambda_sweep.py` 和绘图脚本属于历史探索入口。它们仍能复查早期协议，其中部分依赖脚本相互导入或按自身位置定位项目；本次没有搬动代码。`config/weak_q_gncdm_default.json` 是参数参考文件，不是当前 runner 自动加载的配置。

## 2026-10-29 清理记录

原 `results/` 有 18,985 个文件、203.33 MiB；整理后为 953 个文件、36.25 MiB。原始大网格、两套烟测结果和 `tmp/` 共 18,084 个文件、169.01 MiB 已移动到 [本地归档](../local_experiment_archive/2026-10-29/README.md)，没有永久删除。归档保留原目录结构；`local_experiment_archive/` 被 Git 忽略，迁移或备份项目时需要单独带走。

| 原目录 | 原位保留内容 | 归档内容 |
| --- | --- | --- |
| `results/weak_q_lambda_sweep` | 原始 `summary.csv` | 全部 1,000 个原始组合及旧排序／选择文件 |
| `results/math1_lambda_grid` | 汇总 CSV、报告、选择 JSON 和两个验证集所选运行 | 全部 90 个原始组合；两个所选运行在原位保留副本 |
| `results/math1_noisy_q_smoke`、`results/simple_contrastive_smoke_v1`、`tmp/` | 无 | 全部移入归档，仅作流程历史记录 |

旧 `weak_q_lambda_sweep` 中的 `summary_sorted_by_valid_acc.csv` 与按测试 ACC 排序的文件哈希相同；名为 `best_by_valid_acc.json` 的配置验证 ACC 只有 0.641667，而 `summary.csv` 真正的最大验证 ACC 为 0.668333。旧排序／选择文件存在命名与内容不符的问题，已从活跃结果中撤下，归档保留原貌。该旧实验应只作探索记录。

早期网格的汇总表仍带原始 `result_dir` 字段；旧 synthetic 网格的所有逐次文件、Math1 网格的未选中组合现在位于归档中的相同相对路径。Math1 网格的两个验证集所选运行和当前三种子主实验可直接在原路径复查。旧大网格多数文件在 Git 中已跟踪，因此本次移动会表现为大量未暂存的删除记录；原始内容仍保存在本地归档，也可从 Git 历史恢复。本次未提交或暂存变更。
