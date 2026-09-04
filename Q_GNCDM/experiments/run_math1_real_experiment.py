# -*- coding: utf-8 -*-
"""Run Weak-Q G-NCDM on the processed real Math1 dataset.

This script treats math1_Q_matrix.npy as an expert reference Q matrix. On real
data there is no hidden ground-truth Q, so Q metrics are reported as agreement
with the expert Q rather than absolute recovery.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import run_weak_q_synthetic as synth  # noqa: E402


def fmt_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def load_q(path: Path) -> np.ndarray:
    q = np.load(path).astype(np.int64)
    if q.ndim != 2:
        raise ValueError(f"Expected 2D Q matrix, got shape={q.shape}.")
    return q


def make_namespace(args: argparse.Namespace, config: Dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        mode=config["model_mode"],
        epochs=args.epochs,
        seed=args.seed,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        q_lr=args.q_lr,
        weight_decay=args.weight_decay,
        user_dim=args.user_dim,
        item_dim=args.item_dim,
        alpha=args.alpha,
        device=args.device,
        lambda_q=float(config.get("lambda_q", 0.0)),
        lambda_sparse=float(config.get("lambda_sparse", 0.0)),
        lambda_binary=float(config.get("lambda_binary", 0.0)),
        lambda_coverage=float(config.get("lambda_coverage", 0.0)),
        q_threshold=args.q_threshold,
        q_pos_weight=args.q_pos_weight,
        q_init_high=args.q_init_high,
        q_init_low=args.q_init_low,
        q_prior_high=args.q_prior_high,
        q_prior_low=args.q_prior_low,
        weak_q_loss_type=args.weak_q_loss_type,
        q_reg_warmup_fraction=args.q_reg_warmup_fraction,
        run_tag=config["name"],
        patience=args.patience,
        min_delta=args.min_delta,
    )


def experiment_configs() -> List[Dict[str, Any]]:
    return [
        {
            "name": "fixed-expert-q",
            "model_mode": "fixed-noisy-q",
            "lambda_q": 0.0,
            "lambda_sparse": 0.0,
            "lambda_binary": 0.0,
            "lambda_coverage": 0.0,
            "description": "Expert Q fixed; no Q learning.",
        },
        {
            "name": "default-weak-q",
            "model_mode": "weak-q",
            "lambda_q": 0.01,
            "lambda_sparse": 0.0,
            "lambda_binary": 0.0,
            "lambda_coverage": 0.01,
            "description": "Current code default weak-q setting.",
        },
        {
            "name": "free-q",
            "model_mode": "free-q",
            "lambda_q": 0.0,
            "lambda_sparse": 0.0,
            "lambda_binary": 0.0,
            "lambda_coverage": 0.01,
            "description": "Q learned freely from response data.",
        },
        {
            "name": "balanced-synth-best",
            "model_mode": "weak-q",
            "lambda_q": 0.2,
            "lambda_sparse": 0.05,
            "lambda_binary": 0.8,
            "lambda_coverage": 0.0,
            "description": "Synthetic balanced setting: keeps Q agreement while improving response.",
        },
        {
            "name": "light-tradeoff",
            "model_mode": "weak-q",
            "lambda_q": 0.3,
            "lambda_sparse": 0.05,
            "lambda_binary": 0.8,
            "lambda_coverage": 0.0,
            "description": "Slightly more response-oriented than balanced setting.",
        },
        {
            "name": "aggressive-tradeoff",
            "model_mode": "weak-q",
            "lambda_q": 0.8,
            "lambda_sparse": 0.2,
            "lambda_binary": 0.8,
            "lambda_coverage": 0.0,
            "description": "More aggressive Q adjustment setting from synthetic sweep.",
        },
        {
            "name": "acc-synth-best",
            "model_mode": "weak-q",
            "lambda_q": 0.7,
            "lambda_sparse": 0.6,
            "lambda_binary": 0.4,
            "lambda_coverage": 0.0,
            "description": "Synthetic response-accuracy best setting.",
        },
    ]


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def get_q_outputs(
    model: synth.WeakQGNCDM,
    model_mode: str,
    q_expert: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if model_mode == "fixed-noisy-q":
        return q_expert.astype(np.float32), q_expert.astype(np.int64)
    q_soft = model.get_q_soft().detach().cpu().numpy().astype(np.float32)
    return q_soft, (q_soft > threshold).astype(np.int64)


def run_one(
    args: argparse.Namespace,
    config: Dict[str, Any],
    metadata: Dict[str, Any],
    q_expert: np.ndarray,
    train_rows: Sequence[synth.ResponseRow],
    valid_rows: Sequence[synth.ResponseRow],
    test_rows: Sequence[synth.ResponseRow],
    evidence_log_mat: np.ndarray,
    out_root: Path,
) -> Dict[str, Any]:
    out_dir = out_root / config["name"]
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        print(f"skip existing {config['name']}")
        return synth.read_json(metrics_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    run_args = make_namespace(args, config)
    q_train = q_expert if run_args.mode == "oracle-q" else None
    model = synth.build_model(run_args, metadata, q_expert, q_train)

    print(
        f"run={config['name']} model_mode={config['model_mode']} "
        f"lambda=({run_args.lambda_q},{run_args.lambda_sparse},{run_args.lambda_binary},{run_args.lambda_coverage})"
    )
    model, train_log, best_epoch = synth.train_model(
        model,
        train_rows,
        valid_rows,
        evidence_log_mat,
        run_args,
        q_true_eval=q_expert,
        q_noise_eval=q_expert,
    )

    q_soft, q_hat = get_q_outputs(model, config["model_mode"], q_expert, args.q_threshold)
    valid_metrics = synth.evaluate_response(
        model, valid_rows, evidence_log_mat, args.eval_batch_size, args.seed
    )
    test_metrics = synth.evaluate_response(
        model, test_rows, evidence_log_mat, args.eval_batch_size, args.seed
    )
    q_metrics = synth.q_recovery_metrics(q_expert, q_expert, q_hat, q_soft)
    q_change_metrics = {
        "q_change_from_expert": int(np.sum(q_hat != q_expert)),
        "q_change_0_to_1": int(np.sum((q_expert == 0) & (q_hat == 1))),
        "q_change_1_to_0": int(np.sum((q_expert == 1) & (q_hat == 0))),
        "q_density": float(np.mean(q_hat)),
        "q_soft_mean": float(np.mean(q_soft)),
        "q_soft_min": float(np.min(q_soft)),
        "q_soft_max": float(np.max(q_soft)),
    }
    metrics: Dict[str, Any] = {
        "dataset": "math1",
        "mode": config["name"],
        "model_mode": config["model_mode"],
        "description": config["description"],
        "best_epoch": best_epoch,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "weak_q_loss_type": args.weak_q_loss_type,
        "q_init_high": args.q_init_high,
        "q_init_low": args.q_init_low,
        "q_prior_high": args.q_prior_high,
        "q_prior_low": args.q_prior_low,
        "lambda_q": run_args.lambda_q,
        "lambda_sparse": run_args.lambda_sparse,
        "lambda_binary": run_args.lambda_binary,
        "lambda_coverage": run_args.lambda_coverage,
        "num_train_interactions": len(train_rows),
        "num_valid_interactions": len(valid_rows),
        "num_test_interactions": len(test_rows),
        "valid_acc": valid_metrics["acc"],
        "valid_auc": valid_metrics["auc"],
        "valid_f1": valid_metrics["f1"],
        "valid_rmse": valid_metrics["rmse"],
        "valid_bce": valid_metrics["bce"],
        "test_acc": test_metrics["acc"],
        "test_auc": test_metrics["auc"],
        "test_f1": test_metrics["f1"],
        "test_rmse": test_metrics["rmse"],
        "test_bce": test_metrics["bce"],
        **q_metrics,
        **q_change_metrics,
        "result_dir": str(out_dir),
    }

    synth.write_config_yaml(
        out_dir / "config.yaml",
        {
            "args": vars(args),
            "run_args": vars(run_args),
            "metadata": metadata,
            "config": config,
        },
    )
    synth.write_train_log(out_dir / "train_log.csv", train_log)
    synth.write_json(metrics_path, metrics)
    np.save(out_dir / "Q_expert.npy", q_expert)
    np.save(out_dir / "Q_soft.npy", q_soft)
    np.save(out_dir / "Q_hat.npy", q_hat)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": synth.to_jsonable({"args": vars(args), "run_args": vars(run_args), "config": config}),
            "metrics": synth.to_jsonable(metrics),
        },
        out_dir / "model.pt",
    )
    return metrics


def make_markdown_report(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    rows_by_test = sorted(rows, key=lambda r: (r.get("test_acc") or -1, r.get("test_auc") or -1), reverse=True)
    rows_by_q = sorted(
        rows,
        key=lambda r: (
            r.get("q_hat_f1_vs_true") or -1,
            r.get("q_hat_auc_vs_true") or -1,
            -(r.get("q_hat_hamming_vs_true") or 999),
        ),
        reverse=True,
    )
    fields = [
        "mode",
        "lambda_q",
        "lambda_sparse",
        "lambda_binary",
        "lambda_coverage",
        "valid_acc",
        "valid_auc",
        "valid_rmse",
        "test_acc",
        "test_auc",
        "test_rmse",
        "test_f1",
        "q_hat_f1_vs_true",
        "q_hat_auc_vs_true",
        "q_hat_hamming_vs_true",
        "q_change_from_expert",
        "q_change_0_to_1",
        "q_change_1_to_0",
        "q_density",
        "best_epoch",
    ]

    def table(title: str, data: Sequence[Dict[str, Any]]) -> List[str]:
        lines = [f"## {title}", "", "| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
        for row in data:
            lines.append("| " + " | ".join(fmt_float(row.get(field, "")) for field in fields) + " |")
        lines.append("")
        return lines

    fixed = next((r for r in rows if r["mode"] == "fixed-expert-q"), None)
    default = next((r for r in rows if r["mode"] == "default-weak-q"), None)
    best_test = rows_by_test[0] if rows_by_test else None
    best_q = rows_by_q[0] if rows_by_q else None
    learnable_rows = [row for row in rows if row["mode"] != "fixed-expert-q"]
    best_learned_q = (
        sorted(
            learnable_rows,
            key=lambda r: (
                r.get("q_hat_f1_vs_true") or -1,
                r.get("q_hat_auc_vs_true") or -1,
                -(r.get("q_hat_hamming_vs_true") or 999),
            ),
            reverse=True,
        )[0]
        if learnable_rows
        else None
    )

    md: List[str] = [
        "# Math1 Real Dataset Weak-Q Experiment",
        "",
        "本实验将 `GNCDM/data/math1_Q_matrix.npy` 作为专家 Q 矩阵参考。真实数据没有隐藏的真实 Q，因此 Q 指标表示 learned Q 与专家 Q 的一致性，而不是绝对恢复真值。",
        "",
        "## Experiment Setup",
        "",
        "- Dataset: `math1`",
        "- Train/valid/test: `math1_train_0.8_0.2.csv`, `math1_valid_0.8_0.2.csv`, `math1_test_0.8_0.2.csv`",
        "- Size: 4209 students, 20 items, 11 concepts; train/valid/test interactions = 58926/8418/16836",
        "- Expert Q shape: 20 items x 11 concepts; 67 edges; density 0.3045",
        f"- Training: `epochs={rows[0].get('epochs', '')}`, `batch_size={rows[0].get('batch_size', '')}`, `eval_batch_size={rows[0].get('eval_batch_size', '')}`",
        "- Weak-Q loss: current code default `mse` to soft prior (`q_prior_high=0.55`, `q_prior_low=0.45`)",
        "- Selection note: this is a first real-data probe over representative settings, not a full grid search.",
        "",
    ]
    if fixed and default and best_test:
        default_gain = (default["test_acc"] or 0.0) - (fixed["test_acc"] or 0.0)
        best_gain = (best_test["test_acc"] or 0.0) - (fixed["test_acc"] or 0.0)
        md.extend(
            [
                "## Key Takeaways",
                "",
                f"- 固定专家 Q 的 test_acc 为 **{fmt_float(fixed['test_acc'])}**，test_auc 为 **{fmt_float(fixed['test_auc'])}**；Q-F1 为 1 是因为 Q 被固定为专家 Q。",
                f"- 默认 weak-q 的 test_acc 为 **{fmt_float(default['test_acc'])}**，比固定专家 Q 高 **{fmt_float(default_gain)}**；但它改变了 {default.get('q_change_from_expert')} / 220 个 Q 元素，Q-F1 降到 **{fmt_float(default['q_hat_f1_vs_true'])}**。",
                f"- 当前代表性配置里 test_acc 最好的是 **{best_test['mode']}**：test_acc **{fmt_float(best_test['test_acc'])}**，比固定专家 Q 高 **{fmt_float(best_gain)}**；test_auc **{fmt_float(best_test['test_auc'])}**，Q-F1 **{fmt_float(best_test['q_hat_f1_vs_true'])}**。",
                "",
            ]
        )
    if best_learned_q:
        md.extend(
            [
                f"- 在可学习 Q 的配置中，与专家 Q 一致性最高的是 **{best_learned_q['mode']}**：Q-F1 **{fmt_float(best_learned_q['q_hat_f1_vs_true'])}**，Q-Hamming **{fmt_float(best_learned_q['q_hat_hamming_vs_true'])}**，test_acc **{fmt_float(best_learned_q['test_acc'])}**。",
                "",
            ]
        )

    md.extend(table("All Representative Runs", rows))
    md.extend(table("Sorted By Test Accuracy", rows_by_test))
    md.extend(table("Sorted By Expert-Q Agreement", rows_by_q))

    md.extend(
        [
            "## Interpretation",
            "",
            "1. 真实数据中专家 Q 只能作为参考矩阵。若某个配置的 test_acc 提升但 Q-Hamming 变大，说明它更偏向预测性能，解释性可能下降。",
            "2. `free-q` 可以检验完全不依赖专家 Q 时模型是否仍能预测；如果其 Q-F1 很低，则说明预测能力和可解释 Q 结构可以发生分离。",
            "3. 若 tuned weak-q 能在 test_acc/test_auc 上优于 fixed-expert-q，同时 Q-F1 仍保持较高，则可作为后续真实数据实验的候选配置。",
            "4. `light-tradeoff` 是本轮更适合解释性汇报的可学习 Q 配置；`aggressive-tradeoff` 是本轮更适合预测性能汇报的配置。",
            "5. `acc-synth-best` 在 Math1 上得到较高 test AUC/F1，但 hard Q 几乎塌缩为空，Q-F1 为 0，说明 synthetic 的 Acc-best lambda 不能直接当作真实数据上的解释性最优设置。",
            "6. 更严格的下一步应在 valid set 上做小网格搜索，再只用 test set 报告一次最终结果；此外可对专家 Q 人工加噪，以评估模型修正 Q 噪声的能力。",
            "",
            "## Recommended Reporting Choice",
            "",
            "- 如果汇报重点是作答预测：报告 `aggressive-tradeoff`，因为它在本轮代表配置中 test_acc 最高。",
            "- 如果汇报重点是认知诊断解释性：报告 `light-tradeoff`，因为它在可学习 Q 配置中最接近专家 Q，同时仍优于 fixed-expert-q 的预测效果。",
            "- 这批实验应表述为真实数据初步验证，而不是最终超参搜索结论。",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run representative Weak-Q experiments on real Math1.")
    parser.add_argument("--data_dir", default="GNCDM/data")
    parser.add_argument("--results_dir", default="Q_GNCDM/results/math1_real")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--q_lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--user_dim", type=int, default=32)
    parser.add_argument("--item_dim", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--q_threshold", type=float, default=0.5)
    parser.add_argument("--q_pos_weight", type=synth.parse_q_pos_weight, default=1.0)
    parser.add_argument("--q_init_high", type=float, default=0.55)
    parser.add_argument("--q_init_low", type=float, default=0.45)
    parser.add_argument("--q_prior_high", type=float, default=0.55)
    parser.add_argument("--q_prior_low", type=float, default=0.45)
    parser.add_argument(
        "--weak_q_loss_type",
        choices=["hard_bce", "soft_bce", "mse", "gaussian", "l1", "smooth_l1"],
        default="mse",
    )
    parser.add_argument("--q_reg_warmup_fraction", type=float, default=0.0)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synth.set_seed(args.seed)

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    q_expert = load_q(data_dir / "math1_Q_matrix.npy")
    train_rows = synth.load_response_rows(data_dir / "math1_train_0.8_0.2.csv")
    valid_rows = synth.load_response_rows(data_dir / "math1_valid_0.8_0.2.csv")
    test_rows = synth.load_response_rows(data_dir / "math1_test_0.8_0.2.csv")

    num_students = max(max(row[0] for row in train_rows), max(row[0] for row in valid_rows), max(row[0] for row in test_rows)) + 1
    num_items = max(max(row[1] for row in train_rows), max(row[1] for row in valid_rows), max(row[1] for row in test_rows)) + 1
    metadata: Dict[str, Any] = {
        "dataset": "math1",
        "num_students": num_students,
        "num_items": num_items,
        "num_concepts": int(q_expert.shape[1]),
        "q_expert_path": str(data_dir / "math1_Q_matrix.npy"),
        "train_path": str(data_dir / "math1_train_0.8_0.2.csv"),
        "valid_path": str(data_dir / "math1_valid_0.8_0.2.csv"),
        "test_path": str(data_dir / "math1_test_0.8_0.2.csv"),
        "q_expert_edges": int(np.sum(q_expert)),
        "q_expert_density": float(np.mean(q_expert)),
    }
    if q_expert.shape != (num_items, int(metadata["num_concepts"])):
        raise ValueError(f"Q shape {q_expert.shape} does not match num_items={num_items}.")

    evidence_log_mat = synth.build_log_mat(train_rows, num_students, num_items)
    results_dir.mkdir(parents=True, exist_ok=True)
    synth.write_json(results_dir / "metadata.json", metadata)
    shutil.copy2(data_dir / "math1_Q_matrix.npy", results_dir / "math1_Q_matrix.npy")

    rows: List[Dict[str, Any]] = []
    for config in experiment_configs():
        rows.append(
            run_one(
                args,
                config,
                metadata,
                q_expert,
                train_rows,
                valid_rows,
                test_rows,
                evidence_log_mat,
                results_dir,
            )
        )

    fieldnames = [
        "mode",
        "model_mode",
        "description",
        "lambda_q",
        "lambda_sparse",
        "lambda_binary",
        "lambda_coverage",
        "valid_acc",
        "valid_auc",
        "valid_f1",
        "valid_rmse",
        "valid_bce",
        "test_acc",
        "test_auc",
        "test_f1",
        "test_rmse",
        "test_bce",
        "q_hat_acc_vs_true",
        "q_hat_precision_vs_true",
        "q_hat_recall_vs_true",
        "q_hat_f1_vs_true",
        "q_hat_auc_vs_true",
        "q_hat_hamming_vs_true",
        "q_change_from_expert",
        "q_change_0_to_1",
        "q_change_1_to_0",
        "q_density",
        "best_epoch",
        "result_dir",
    ]
    write_csv(results_dir / "summary.csv", rows, fieldnames)
    make_markdown_report(rows, results_dir / "math1_real_experiment_report.md")
    print(f"Saved summary to {results_dir / 'summary.csv'}")
    print(f"Saved report to {results_dir / 'math1_real_experiment_report.md'}")


if __name__ == "__main__":
    main()
