# Weak-Q Figure Naming Rules

这个目录用于保存 Weak-Q G-NCDM 实验图。目标是让新人只看文件夹名，就能知道这一组图在比较什么。

## Folder Meaning

- `baseline/`: 固定基线实验，例如 `fixed-noisy-q`、`oracle-q`、`free-q`。
- `init_compare/`: 比较 Q 初始化方式，例如 `init30-70`、`init45-55`、`init48-52`。
- `loss_compare/`: 比较 `loss_weak_q` 形式，例如 `mse`、`soft_bce`、`smooth_l1`。
- `lambda_q_compare/`: 比较 `lambda_q` 大小。
- `q_lr_compare/`: 比较 `q_logits` 单独学习率 `q_lr`。
- `reg_schedule/`: 比较 sparse/binary 正则和 warmup 策略。
- `best_runs/`: 放最终准备写论文或汇报使用的最佳结果图。

## Current A-E Ablation Results

当前 A-E 实验使用：

```text
seed=2026
noise_rate=0.10
mode=weak-q
```

汇总表保存在：

```text
Q_GNCDM/results/weak_q_synthetic/experiment_results.md
```

每个 run 的原始结果文件夹为：

```text
Q_GNCDM/results/weak_q_synthetic/seed_2026/noise_0.10/weak-q_<run_tag>/
```

其中包含：

- `metrics.json`: 最终汇总指标。
- `train_log.csv`: 每个 epoch 的 loss、Q 变化、验证指标。
- `Q_soft.npy`: 训练后的连续 Q。
- `Q_hat.npy`: 按阈值二值化后的 Q。
- `config.yaml`: 本次实验参数。
- `model.pt`: 保存的 best model。

当前正式 run_tag：

- `rec_only`: 只用 response loss，检查 Q 是否能动。
- `mse_lq001`: MSE weak prior，`lambda_q=0.01`。
- `softbce_lq001`: Soft BCE weak prior，`lambda_q=0.01`。
- `mse_lq0001`: MSE weak prior，`lambda_q=0.001`。
- `mse_late_binary`: MSE weak prior，并在后期加入 sparse/binary。

如果后续为这批 A-E 实验生成图片，建议放到：

```text
figures/best_runs/weakq_ablation_noise0.10/
```

历史平铺 PNG 已整理到 `baseline/` 下；新实验图建议按上面分类存放。

历史基线图当前放置规则：

- `baseline/summary/`: 跨模式总览曲线，例如 `q_hat_f1_curve.png`、`response_auc_curve.png`。
- `baseline/fixed-noisy-q/`: 固定噪声 Q 基线图。
- `baseline/free-q/`: 不使用弱监督 Q 的自由学习 Q 图。
- `baseline/oracle-q/`: 使用真实 Q 的上界基线图。
- `baseline/weak-q/`: 旧版 Weak-Q 默认配置图。

## Recommended Path Format

推荐使用：

```text
figures/<experiment_type>/<fixed_conditions>/<changed_parameter>/
```

例子：

```text
figures/loss_compare/init45-55_lq1e-2_qlr5e-3/mse/
figures/loss_compare/init45-55_lq1e-2_qlr5e-3/soft_bce/

figures/lambda_q_compare/init45-55_loss-mse_qlr5e-3/lq1e-3/
figures/lambda_q_compare/init45-55_loss-mse_qlr5e-3/lq1e-2/

figures/reg_schedule/init45-55_loss-mse_lq1e-2/reg-none/
figures/reg_schedule/init45-55_loss-mse_lq1e-2/warm60_sparse1e-5_bin1e-4/
```

## Parameter Abbreviations

- `init45-55`: `q_init_low=0.45`, `q_init_high=0.55`
- `prior45-55`: `q_prior_low=0.45`, `q_prior_high=0.55`
- `loss-mse`: `weak_q_loss_type=mse`
- `loss-soft_bce`: `weak_q_loss_type=soft_bce`
- `lq1e-2`: `lambda_q=0.01`
- `qlr5e-3`: `q_lr=0.005`
- `sparse1e-5`: `lambda_sparse=0.00001`
- `bin1e-4`: `lambda_binary=0.0001`
- `warm60`: `q_reg_warmup_fraction=0.6`
- `reg-none`: `lambda_sparse=0` 且 `lambda_binary=0`

## Selection Rule

筛选最佳实验时，不要只看 `q_change_from_noise`。优先选择：

```text
q_hat_f1_vs_true 提升
response_auc 不明显下降
q_change_from_noise 适中
q_soft_delta_from_init 不为 0
```
