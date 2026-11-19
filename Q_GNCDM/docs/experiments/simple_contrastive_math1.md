# Math1：固定 Q 上的简易对比学习消融

**实验 ID**：`simple_contrastive_math1`。A0 固定专家 Q、不使用对比学习；C0 保持同一固定 Q，只加入作答翻转的学生表征 InfoNCE。两组使用相同的 Math1 训练／验证／测试划分、种子 2026/2027/2028、训练设置；`lambda_q=lambda_coverage=0`，且固定 Q 不参与学习。各训练最多 30 epoch、patience 8，checkpoint 按验证 AUC 选择。

**系数选择**：候选 `lambda_cl` 为 0.001、0.005、0.01、0.03、0.05、0.1、0.2、0.5、1.0，按三种子的平均验证 AUC 选出 **0.1**。[总表](../../results/simple_contrastive_math1/coefficient_comparison.csv)列出 A0 与所有候选的 AUC、ACC 和差值；[选择记录](../../results/simple_contrastive_math1/selection.json)保存选参规则。[A0 基线](../../results/simple_contrastive_math1/baseline_A0/)只训练、保存一次；每个系数目录的 `summary.csv` 都直接列出同种子 A0/C0 和配对差值，例如[0.1 的逐种子对照](../../results/simple_contrastive_math1/lambda_0p1/summary.csv)。

| 条件 | 平均验证 AUC | 平均测试 AUC | 平均测试 ACC | 相对 A0 测试 AUC | 相对 A0 测试 ACC |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 固定 Q | 0.835689 | 0.831658 | 0.739942 | — | — |
| C0，`lambda_cl=0.1` | 0.836680 | 0.831728 | 0.739012 | **+0.000071** | **−0.000931** |

**结论**：验证 AUC 上升，但测试 AUC 增量很小，三个种子仅 1 个 AUC 配对差值为正，ACC 下降；不能宣称预测效果整体改善。测试指标未用于选参；全部条件的 checkpoint 重放与固定 Q 核验通过。只评估一个划分和三个训练种子。
