# Math1 Artificial Q-Noise Robustness Experiment

本实验将 Math1 expert Q 作为 pseudo truth，人工构造 noisy Q；训练时只给 noisy Q，评价时比较 learned Q 是否回到 expert Q。

固定使用阶段一选出的 lambda，不重新调参：

- prediction-best: lambda_q=0.2, lambda_sparse=0.1, lambda_binary=0.8
- balanced-best: lambda_q=0.8, lambda_sparse=0.1, lambda_binary=0.8

## Clean-Q Baseline Reference

| mode | test_acc | test_auc | q_hat_f1_vs_true | q_hat_hamming_vs_true |
| --- | --- | --- | --- | --- |
| fixed-expert-q | 0.730221 | 0.815131 | 1.000000 | 0.000000 |
| free-q | 0.736576 | 0.827655 | 0.424242 | 0.518182 |

## All Noisy-Q Runs

| noise_rate | mode | lambda_q | lambda_sparse | lambda_binary | valid_acc | test_acc | test_auc | q_noise_f1_vs_true | q_noise_hamming_vs_true | q_hat_f1_vs_true | q_hat_auc_vs_true | q_hat_hamming_vs_true | deleted_edge_recovery | added_edge_suppression | q_change_from_noise | q_hat_density | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.050000 | fixed-noisy-q | 0.000000 | 0.000000 | 0.000000 | 0.556427 | 0.575671 | 0.748571 | 0.977778 | 0.013636 | 0.977778 | 0.986001 | 0.013636 | 0.000000 | 0.000000 | 0.000000 | 0.309091 | 1.000000 |
| 0.050000 | weak-balanced-best | 0.800000 | 0.100000 | 0.800000 | 0.556427 | 0.575671 | 0.601093 | 0.977778 | 0.013636 | 0.977778 | 0.984977 | 0.013636 | 0.000000 | 0.000000 | 0.000000 | 0.309091 | 1.000000 |
| 0.050000 | weak-prediction-best | 0.200000 | 0.100000 | 0.800000 | 0.556427 | 0.575671 | 0.555794 | 0.977778 | 0.013636 | 0.977778 | 0.980685 | 0.013636 | 0.000000 | 0.000000 | 0.000000 | 0.309091 | 1.000000 |

## Noise 0.05 Sorted By Q-F1

| noise_rate | mode | lambda_q | lambda_sparse | lambda_binary | valid_acc | test_acc | test_auc | q_noise_f1_vs_true | q_noise_hamming_vs_true | q_hat_f1_vs_true | q_hat_auc_vs_true | q_hat_hamming_vs_true | deleted_edge_recovery | added_edge_suppression | q_change_from_noise | q_hat_density | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.050000 | fixed-noisy-q | 0.000000 | 0.000000 | 0.000000 | 0.556427 | 0.575671 | 0.748571 | 0.977778 | 0.013636 | 0.977778 | 0.986001 | 0.013636 | 0.000000 | 0.000000 | 0.000000 | 0.309091 | 1.000000 |
| 0.050000 | weak-prediction-best | 0.200000 | 0.100000 | 0.800000 | 0.556427 | 0.575671 | 0.555794 | 0.977778 | 0.013636 | 0.977778 | 0.980685 | 0.013636 | 0.000000 | 0.000000 | 0.000000 | 0.309091 | 1.000000 |
| 0.050000 | weak-balanced-best | 0.800000 | 0.100000 | 0.800000 | 0.556427 | 0.575671 | 0.601093 | 0.977778 | 0.013636 | 0.977778 | 0.984977 | 0.013636 | 0.000000 | 0.000000 | 0.000000 | 0.309091 | 1.000000 |

## Interpretation Guide

- `fixed-noisy-q` 表示直接使用 noisy Q，不学习修正；它给出噪声下限。
- `deleted_edge_recovery` 越高，说明被噪声删掉的 expert edge 越多被恢复。
- `added_edge_suppression` 越高，说明噪声额外添加的假 edge 越多被抑制。
- 若 weak-q 的 Q-F1 高于 fixed-noisy-q，且 test 指标不下降，说明模型有一定 Q 噪声修正能力。
