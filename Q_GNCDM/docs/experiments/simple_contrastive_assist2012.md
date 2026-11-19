# ASSIST2012：固定 Q 上的简易对比学习消融

**实验 ID**：`simple_contrastive_assist2012`。A0 固定技能参考 Q、不使用对比学习；C0 保持同一固定 Q，只加入作答翻转的学生表征 InfoNCE。两组使用相同的 ASSISTments 2012–2013 高频子集划分、种子 2026/2027/2028、训练设置；`lambda_q=lambda_coverage=0`，且 Q 不参与学习。各训练最多 30 epoch、patience 8，checkpoint 按验证 AUC 选择。

候选 `lambda_cl` 为 0.001、0.005、0.01、0.03、0.05、0.1、0.2、0.5、1.0，按三种子的平均验证 AUC 选出 **0.1**。完整候选与 A0 对照见[总表](../../results/simple_contrastive_assist2012/coefficient_comparison.csv)，选参规则见[选择记录](../../results/simple_contrastive_assist2012/selection.json)。[A0 基线](../../results/simple_contrastive_assist2012/baseline_A0/)只训练一次；每个系数目录的 `summary.csv` 都直接列出同种子 A0/C0 和配对差值，例如[0.1 的逐种子对照](../../results/simple_contrastive_assist2012/lambda_0p1/summary.csv)。

| 条件 | 平均验证 AUC | 平均测试 AUC | 平均测试 ACC | 相对 A0 测试 AUC | 相对 A0 测试 ACC |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 固定 Q | 0.686455 | 0.688912 | 0.751863 | — | — |
| C0，`lambda_cl=0.1` | 0.688809 | 0.691267 | 0.752117 | **+0.002355** | **+0.000254** |

**结论**：本划分下，对比学习在 ASSIST 子集上带来正向预测增量，三个种子的测试 AUC 配对差值均为正；ACC 变化较小。大系数 0.5、1.0 的验证 AUC 低于 A0，负结果保留在总表。测试指标没有参与选参；全部条件的 checkpoint 重放与固定 Q 核验通过。该数据是高频子集，结论不自动推广到完整 ASSISTments 数据。
