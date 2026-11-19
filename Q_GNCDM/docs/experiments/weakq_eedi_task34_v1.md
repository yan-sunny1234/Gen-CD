# weakq_eedi_task34_v1

- **问题**：固定参考 Q（A0）与完整 Weak-Q 设置（A1）在 Eedi Task 3/4 数据上的同协议预测差异。
- **唯一改动**：A1 将 Q 设为可学习的软矩阵，并启用弱 Q 约束；不加入对比学习。该对照不能拆分各个 Q 约束的单独贡献。
- **数据**：`data/eedi_task34/`，4,918 名学生、948 道题、25 个二级学科标签；按学生作答划分，划分种子 2026；训练／验证／测试为 965,661／136,111／280,955 条。这不是官方 Task 4 冷启动划分；Q 来自题目元数据，不是专家验证的真 Q。
- **训练**：种子 2026、2027、2028；最多 30 epoch，耐心值 8；模型按验证 AUC 选取（无 AUC 时退回验证 NLL）。两模式共用训练设置，A1 的 `lambda_q=0.01`、`lambda_coverage=0.01`；完整设置及数据／代码哈希见 [manifest](../../results/weakq_eedi_task34_v1/manifest.json)。

| 模式 | 测试 AUC（均值±标准差） | 测试 ACC（均值±标准差） | 测试 RMSE | 测试 NLL |
| --- | ---: | ---: | ---: | ---: |
| A0 固定 Q | 0.765181 ± 0.002697 | 0.704676 ± 0.001871 | 0.440763 | 0.569826 |
| A1 Weak-Q | 0.773602 ± 0.000130 | 0.708576 ± 0.000102 | 0.438154 | 0.564627 |
| A1−A0 | **+0.008421 AUC** | **+0.003900 ACC** | −0.002609 | −0.005198 |

三个训练种子的 AUC 配对差值均为正，本划分下预测收益一致。但 A1 与元数据 Q 的平均 F1 仅 0.133，预测收益**不能当作 Q 结构修复或知识掌握诊断改善**。逐种子指标见 [summary](../../results/weakq_eedi_task34_v1/summary.csv)，配对差值见 [paired_comparisons](../../results/weakq_eedi_task34_v1/paired_comparisons.csv)，checkpoint 重放见 [verification](../../results/weakq_eedi_task34_v1/verification.json)。
