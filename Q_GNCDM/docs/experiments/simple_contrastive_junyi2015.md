# Junyi2015：固定 Q 上的简易对比学习消融

**实验 ID**：`simple_contrastive_junyi2015`。A0 固定主题参考 Q、不使用对比学习；C0 保持同一固定 Q，只加入作答翻转的学生表征 InfoNCE。两组使用相同的 Junyi 高频子集划分、种子 2026/2027/2028、训练设置；`lambda_q=lambda_coverage=0`，且 Q 不参与学习。各训练最多 30 epoch、patience 8，checkpoint 按验证 AUC 选择。

候选 `lambda_cl` 为 0.001、0.005、0.01、0.03、0.05、0.1、0.2、0.5、1.0。九个 C0 候选中平均验证 AUC 最高的是 **0.01**，但其 0.791293 **低于 A0 的 0.791320**；若允许选择“不加对比”，验证集应保留 A0。完整候选见[总表](../../results/simple_contrastive_junyi2015/coefficient_comparison.csv)，[选择记录](../../results/simple_contrastive_junyi2015/selection.json)保存规则。[A0 基线](../../results/simple_contrastive_junyi2015/baseline_A0/)只训练一次；每个系数目录的 `summary.csv` 都直接列出同种子 A0/C0 和差值，例如[0.01 的逐种子对照](../../results/simple_contrastive_junyi2015/lambda_0p01/summary.csv)。

| 条件 | 平均验证 AUC | 平均测试 AUC | 平均测试 ACC | 相对 A0 测试 AUC | 相对 A0 测试 ACC |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 固定 Q | 0.791320 | 0.797409 | 0.754223 | — | — |
| C0，`lambda_cl=0.01` | 0.791293 | 0.797410 | 0.754115 | **+0.000001** | **−0.000108** |

**结论**：对比学习没有带来可信的验证集或测试集收益。测试指标没有参与选参；全部条件的 checkpoint 重放与固定 Q 核验通过。Junyi 的 Q 来自主题标签，不是独立专家验证的真 Q。
