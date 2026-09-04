# Weak-Q Experiment Results

实验设置：`seed=2026`，`noise_rate=0.10`，`mode=weak-q`，默认 `epochs=100`。每次实验结果保存在：

```text
Q_GNCDM/results/weak_q_synthetic/seed_2026/noise_0.10/weak-q_<run_tag>/
```

| run_tag | q_noise_f1 | q_hat_f1 | q_change | deleted_recovery | added_suppression | response_auc | mastery_pearson |
| ------- | ---------: | -------: | -------: | ---------------: | ----------------: | -----------: | --------------: |
| rec_only | 0.947368 | 0.473430 | 105 | 0.500000 | 0.000000 | 0.683533 | 0.328759 |
| mse_lq001 | 0.947368 | 0.473430 | 105 | 0.500000 | 0.000000 | 0.683467 | 0.329012 |
| softbce_lq001 | 0.947368 | 0.475728 | 104 | 0.500000 | 0.000000 | 0.683386 | 0.329035 |
| mse_lq0001 | 0.947368 | 0.473430 | 105 | 0.500000 | 0.000000 | 0.683525 | 0.328785 |
| mse_late_binary | 0.947368 | 0.762712 | 30 | 0.000000 | 1.000000 | 0.689387 | 0.525339 |

说明：`q_change` 来自该 run 的 `best_epoch` 对应行，和最终保存的 best model 对齐。
