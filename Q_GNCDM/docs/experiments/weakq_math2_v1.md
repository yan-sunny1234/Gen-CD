# weakq_math2_v1

- **问题**：固定参考 Q（A0）与完整 Weak-Q 设置（A1）在 Math2 上的同协议预测差异。
- **唯一改动**：A1 将 Q 设为可学习的软矩阵，并启用弱 Q 约束；不加入对比学习。该对照不能拆分各个 Q 约束的单独贡献。
- **数据**：`data/math2/`，按学生作答划分，划分种子 2026；训练／验证／测试为 43,021／7,822／11,733 条。Q 为原始 Math2 的专家标注，选用 16 道二值客观题。
- **训练**：种子 2026、2027、2028；最多 30 epoch，耐心值 8；模型按验证 AUC 选取（无 AUC 时退回验证 NLL）。两模式共用训练设置，A1 的 `lambda_q=0.01`、`lambda_coverage=0.01`，其余配置及数据／代码哈希见 [manifest](../../results/weakq_math2_v1/manifest.json)。

| 模式 | 测试 AUC（均值±标准差） | 测试 ACC（均值±标准差） | 测试 RMSE | 测试 NLL |
| --- | ---: | ---: | ---: | ---: |
| A0 固定 Q | 0.780227 ± 0.000475 | 0.705077 ± 0.000130 | 0.437826 | 0.565022 |
| A1 Weak-Q | 0.787282 ± 0.000193 | 0.708060 ± 0.001224 | 0.433909 | 0.556382 |
| A1−A0 | **+0.007055 AUC** | **+0.002983 ACC** | −0.003917 | −0.008640 |

三个训练种子的 AUC 配对差值均为正；预测指标在本划分下支持 Weak-Q 整体设置有效。A1 与专家 Q 的平均 F1 为 0.376，说明预测改善**不能解释为 Q 结构修复**。逐种子原始指标见 [summary](../../results/weakq_math2_v1/summary.csv)，配对差值见 [paired_comparisons](../../results/weakq_math2_v1/paired_comparisons.csv)，checkpoint 重放见 [verification](../../results/weakq_math2_v1/verification.json)。
