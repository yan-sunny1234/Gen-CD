# Math1：掌握融合系数 α 的端点消融

实验 ID：alpha_endpoints_math1_20261110；日期：2026-11-10。状态：9 次训练完成并核验。

## 设置

本轮比较当前 Weak-Q、无对比学习的 α=0、0.8、1，唯一实验改动为 α。每个条件独立重新训练，不是训练后切换权重。沿用 Math1 原 train/valid/test，训练种子 2026、2027、2028；最多 60 epoch、patience=12，每个种子按验证 AUC 选模，测试指标只作报告。

Q 初始化及先验为 0.55/0.45，lambda_q=lambda_coverage=0.01，lambda_sparse=lambda_binary=lambda_cl=0；embedding_dim=32，学生 batch=128，query_fraction=0.25，lr=0.001、q_lr=0.005。全部配置和数据／代码哈希见 [manifest.json](../../results/alpha_endpoints_math1_20261110/manifest.json)。

- α=0：掌握完全来自隐式分支；Q 仍可学习，仍用于预测端题目掩码，不能称为去掉 Weak-Q。
- α=0.8：20% 隐式＋80% 显式，为当前参照。
- α=1：掌握完全来自显式分支；题目编码器和预测头仍然训练，并非整个模型没有神经网络。

训练 query 从学生、题目两侧证据中移除。未增加对比损失，以免同时改变两个因素。

## 最终结果

以下为三训练种子的均值±样本标准差，增量相对于同种子 α=0.8；指标为 0–1 数值，不是百分数。

| α | Test AUC | Test ACC | ΔAUC | ΔACC | Test NLL ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.831996 ± 0.005920 | 0.741150 ± 0.003755 | −0.005839 | −0.005583 | 0.481665 |
| 0.8 | 0.837836 ± 0.000637 | 0.746733 ± 0.001146 | 0 | 0 | 0.474120 |
| 1 | 0.837854 ± 0.000794 | 0.746476 ± 0.001567 | +0.000018 | −0.000257 | 0.473918 |

**直接比较两个端点：α=1 比 α=0，AUC 高 0.005858，ACC 高 0.005326（约 0.533 个百分点）。**

| 训练种子 | α=0 Test AUC | α=0.8 Test AUC | α=1 Test AUC |
| --- | ---: | ---: | ---: |
| 2026 | 0.825258 | 0.837572 | 0.837496 |
| 2027 | 0.834367 | 0.838563 | 0.838765 |
| 2028 | 0.836364 | 0.837372 | 0.837302 |

验证 AUC 均值分别为 0.834510、0.842993、0.843158；验证 ACC 均值分别为 0.743803、0.750574、0.751287。完整 AUC、ACC、NLL、Brier、Q 密度、掌握有效秩及逐种子数值见 [comparison.csv](../../results/alpha_endpoints_math1_20261110/comparison.csv) 和 [per_seed.csv](../../results/alpha_endpoints_math1_20261110/per_seed.csv)。

## 解释及边界

1. 相同 60 轮训练预算下，仅隐式分支的三种子 AUC 均低于融合及仅显式分支，波动也更大；2026 种子下降尤其明显。
2. α=1 与 α=0.8 的预测接近。AUC 的微小平均提升仅来自一个种子，另外两个种子略降；ACC 均值略降，不能称 α=1 稳定优于融合。
3. 本轮没有显示 20% 隐式分支带来明确 AUC 增量，但不能据此断言隐式分支在所有数据集、其他 Q 设置或加入 CL 后都无用。
4. α=0 的最佳轮次为 60、59、60；α=0.8 为 39、46、60；α=1 为 40、46、60。尤其 α=0 接近训练上限，尚不能把有限预算结果当作充分收敛后的能力上限。
5. 这里只比较 Math1 上的 Weak-Q、无 CL 协议，不外推五套数据，也不通过测试结果选择一个“最优 α”。诊断结构是否更准确仍需独立真值验证；有效秩不是选模指标。

## 核验与文件用途

9 个验证集所选模型均在内存中重新实例化、加载并重算指标，全部通过；训练结束再次核对数据和代码哈希。重跑 α=0.8 的三个种子与历史同协议 Weak-Q 基线的验证／测试指标及 best_epoch 一致（容差 1e-10）。

依照本次要求，不保存 checkpoint、逐轮曲线、预测数组或中间日志；最优模型只在内存中保留至核验结束。代价是不能直接加载本轮端点 checkpoint，复查需要按配置重跑。

| 文件 | 用途 |
| --- | --- |
| [run_alpha_endpoints.py](../../experiments/run_alpha_endpoints.py) | 三个 α、三个种子的运行与最终汇总入口 |
| [run_test_sim_con.py](../../experiments/run_test_sim_con.py) | 原训练函数增加 directory=None 的内存模式；原有落盘模式不变，训练目标和循环未改 |
| [manifest.json](../../results/alpha_endpoints_math1_20261110/manifest.json) | 协议、基线、唯一改动、种子、配置、数据及代码哈希 |
| [per_seed.csv](../../results/alpha_endpoints_math1_20261110/per_seed.csv) | 9 次正式训练的最终指标，不是逐轮日志 |
| [comparison.csv](../../results/alpha_endpoints_math1_20261110/comparison.csv) | 三种 α 的均值、标准差和相对 α=0.8 的配对差值 |
| [verification.json](../../results/alpha_endpoints_math1_20261110/verification.json) | 内存重放、哈希和历史基线复现核验 |
| 本文 | 本轮唯一解释记录；README 仅链接入口 |

重新运行命令（需使用新的实验 ID／输出目录，现有脚本拒绝覆盖正式结果）：
~~~powershell
.\.venv-contrastive\Scripts\python.exe -m Q_GNCDM.experiments.run_alpha_endpoints
~~~

本轮没有删除旧结果，没有提交 Git。

