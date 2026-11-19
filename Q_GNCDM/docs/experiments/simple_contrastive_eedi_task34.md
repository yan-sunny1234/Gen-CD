# Eedi Task 3/4：固定 Q 上的简易对比学习消融

**实验 ID**：`simple_contrastive_eedi_task34`。A0 固定元数据参考 Q、不使用对比学习；C0 保持同一固定 Q，只加入作答翻转的学生表征 InfoNCE。两组使用相同的 Eedi 作答划分、种子 2026/2027/2028、训练设置；`lambda_q=lambda_coverage=0`，且 Q 不参与学习。此处为 seen-student 协议，**不是** Eedi 官方冷启动评测。各训练最多 30 epoch、patience 8，checkpoint 按验证 AUC 选择。

候选 `lambda_cl` 为 0.001、0.005、0.01、0.03、0.05、0.1、0.2、0.5、1.0。九个 C0 候选中平均验证 AUC 最高的是边界值 **0.001**，但其 0.764553 **低于 A0 的 0.764649**；若允许选择“不加对比”，验证集应保留 A0。完整候选见[总表](../../results/simple_contrastive_eedi_task34/coefficient_comparison.csv)，[选择记录](../../results/simple_contrastive_eedi_task34/selection.json)保存规则。[A0 基线](../../results/simple_contrastive_eedi_task34/baseline_A0/)只训练一次；每个系数目录的 `summary.csv` 都直接列出同种子 A0/C0 和配对差值，例如[0.001 的逐种子对照](../../results/simple_contrastive_eedi_task34/lambda_0p001/summary.csv)。

| 条件 | 平均验证 AUC | 平均测试 AUC | 平均测试 ACC | 相对 A0 测试 AUC | 相对 A0 测试 ACC |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 固定 Q | 0.764649 | 0.765181 | 0.704676 | — | — |
| C0，`lambda_cl=0.001` | 0.764553 | 0.765092 | 0.704583 | **−0.000089** | **−0.000094** |

**结论**：本划分下没有观察到对比学习的预测收益，最好的 C0 候选在三个种子上的测试 AUC 配对差值均为负，较大系数下降更明显。0.001 是已测正系数的下边界，不能称为全局最优；把 A0 视作 `lambda_cl=0` 时，验证集选择 A0。测试指标没有参与选参；全部条件的 checkpoint 重放与固定 Q 核验通过。
