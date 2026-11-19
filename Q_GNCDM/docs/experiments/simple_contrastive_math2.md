# Math2：固定 Q 上的简易对比学习消融

**实验 ID**：`simple_contrastive_math2`。A0 固定参考 Q、不使用对比学习；C0 保持同一固定 Q，只加入作答翻转的学生表征 InfoNCE。两组使用相同的 Math2 训练／验证／测试划分、种子 2026/2027/2028、训练设置；`lambda_q=lambda_coverage=0`，且固定 Q 不参与学习。各训练最多 30 epoch、patience 8，checkpoint 按验证 AUC 选择。

**系数选择**：候选 `lambda_cl` 为 0.001、0.005、0.01、0.03、0.05、0.1、0.2、0.5、1.0，按三种子的平均验证 AUC 选出 **0.03**。完整十行对照及负结果见[总表](../../results/simple_contrastive_math2/coefficient_comparison.csv)，选参规则见[选择记录](../../results/simple_contrastive_math2/selection.json)；[A0 基线](../../results/simple_contrastive_math2/baseline_A0/)与每个 `lambda_...` 目录分别保留配置、逐种子指标、checkpoint 和核验。每个系数目录的 `summary.csv` 同时列出同种子 A0/C0 及配对差值，例如[0.03 的逐种子对照](../../results/simple_contrastive_math2/lambda_0p03/summary.csv)。

| 条件 | 平均验证 AUC | 平均测试 AUC | 平均测试 ACC | 相对 A0 测试 AUC | 相对 A0 测试 ACC |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 固定 Q | 0.783192 | 0.780227 | 0.705077 | — | — |
| C0，`lambda_cl=0.03` | 0.783237 | 0.780082 | 0.704821 | **−0.000145** | **−0.000256** |

**结论**：在本划分和搜索范围内，验证集选中的对比学习系数没有改善 Math2 测试预测，三个种子的测试 AUC 配对差值均为负。测试结果未用于选参；全部条件的 checkpoint 重放与固定 Q 核验通过。一个划分、三个训练种子不足以证明普遍无效。
