# -*- coding: utf-8 -*-
"""Artificial-Q-noise robustness experiment for real Math1.

The original Math1 expert Q is treated as pseudo truth. For each noise rate,
the script trains with a perturbed Q and evaluates whether learned Q returns
toward the expert Q.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import run_math1_real_experiment as real_exp  # noqa: E402
import run_weak_q_synthetic as synth  # noqa: E402


FIELDNAMES = [
    "noise_rate",
    "mode",
    "model_mode",
    "lambda_q",
    "lambda_sparse",
    "lambda_binary",
    "lambda_coverage",
    "valid_acc",
    "valid_auc",
    "valid_f1",
    "valid_rmse",
    "test_acc",
    "test_auc",
    "test_f1",
    "test_rmse",
    "q_noise_f1_vs_true",
    "q_noise_hamming_vs_true",
    "q_hat_f1_vs_true",
    "q_hat_auc_vs_true",
    "q_hat_hamming_vs_true",
    "deleted_edge_recovery",
    "added_edge_suppression",
    "q_change_from_noise",
    "q_change_0_to_1",
    "q_change_1_to_0",
    "q_hat_density",
    "best_epoch",
    "result_dir",
]


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def parse_values(raw: str) -> List[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one noise rate is required.")
    return values


def rate_label(rate: float) -> str:
    return f"{rate:.2f}"


def load_best_lambda(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return {
        "lambda_q": float(data["lambda_q"]),
        "lambda_sparse": float(data["lambda_sparse"]),
        "lambda_binary": float(data["lambda_binary"]),
        "lambda_coverage": float(data.get("lambda_coverage", 0.0)),
    }


def positions_to_lists(array: np.ndarray) -> List[List[int]]:
    return [[int(x), int(y)] for x, y in array.tolist()]


def sample_deletions(q: np.ndarray, count: int, rng: np.random.Generator, preserve_item_coverage: bool) -> List[Tuple[int, int]]:
    candidates = [tuple(map(int, pos)) for pos in np.argwhere(q == 1)]
    rng.shuffle(candidates)
    selected: List[Tuple[int, int]] = []
    row_counts = q.sum(axis=1).astype(int)
    for item, concept in candidates:
        if len(selected) >= count:
            break
        if preserve_item_coverage and row_counts[item] <= 1:
            continue
        selected.append((item, concept))
        row_counts[item] -= 1
    return selected


def sample_additions(q: np.ndarray, count: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    candidates = [tuple(map(int, pos)) for pos in np.argwhere(q == 0)]
    rng.shuffle(candidates)
    return candidates[:count]


def make_noisy_q(
    q_true: np.ndarray,
    noise_rate: float,
    seed: int,
    preserve_item_coverage: bool = True,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Perturb Q by a target number of changed entries based on true edge count."""
    rng = np.random.default_rng(seed)
    q_noise = q_true.copy().astype(np.int64)
    true_edges = int(q_true.sum())
    target = max(1, int(round(float(noise_rate) * true_edges)))
    delete_count = target // 3
    add_count = target // 3
    flip_count = target - delete_count - add_count

    deleted: List[Tuple[int, int]] = []
    added: List[Tuple[int, int]] = []

    for item, concept in sample_deletions(q_noise, delete_count, rng, preserve_item_coverage):
        q_noise[item, concept] = 0
        deleted.append((item, concept))
    for item, concept in sample_additions(q_noise, add_count, rng):
        q_noise[item, concept] = 1
        added.append((item, concept))

    flip_candidates = [tuple(map(int, pos)) for pos in np.argwhere(np.ones_like(q_noise, dtype=bool))]
    rng.shuffle(flip_candidates)
    flips_done = 0
    for item, concept in flip_candidates:
        if flips_done >= flip_count:
            break
        if q_noise[item, concept] == 1:
            if preserve_item_coverage and q_noise[item].sum() <= 1:
                continue
            q_noise[item, concept] = 0
            deleted.append((item, concept))
        else:
            q_noise[item, concept] = 1
            added.append((item, concept))
        flips_done += 1

    deleted_edges = np.argwhere((q_true == 1) & (q_noise == 0))
    added_edges = np.argwhere((q_true == 0) & (q_noise == 1))
    info = {
        "noise_rate": float(noise_rate),
        "target_perturbations_by_true_edges": target,
        "actual_perturbations": int(np.sum(q_true != q_noise)),
        "actual_perturbation_ratio_by_entries": float(np.mean(q_true != q_noise)),
        "Q_true_num_edges": int(q_true.sum()),
        "Q_noise_num_edges": int(q_noise.sum()),
        "Q_true_density": float(q_true.mean()),
        "Q_noise_density": float(q_noise.mean()),
        "deleted_edges": positions_to_lists(deleted_edges),
        "added_edges": positions_to_lists(added_edges),
        "operation_counts_requested": {
            "delete": delete_count,
            "add": add_count,
            "flip": flip_count,
        },
        "preserve_item_coverage": preserve_item_coverage,
    }
    return q_noise, info


def load_math1(args: argparse.Namespace) -> tuple[Dict[str, Any], np.ndarray, list, list, list, np.ndarray]:
    data_dir = Path(args.data_dir)
    q_expert = real_exp.load_q(data_dir / "math1_Q_matrix.npy")
    train_rows = synth.load_response_rows(data_dir / "math1_train_0.8_0.2.csv")
    valid_rows = synth.load_response_rows(data_dir / "math1_valid_0.8_0.2.csv")
    test_rows = synth.load_response_rows(data_dir / "math1_test_0.8_0.2.csv")
    num_students = max(
        max(row[0] for row in train_rows),
        max(row[0] for row in valid_rows),
        max(row[0] for row in test_rows),
    ) + 1
    num_items = max(
        max(row[1] for row in train_rows),
        max(row[1] for row in valid_rows),
        max(row[1] for row in test_rows),
    ) + 1
    metadata: Dict[str, Any] = {
        "dataset": "math1",
        "num_students": num_students,
        "num_items": num_items,
        "num_concepts": int(q_expert.shape[1]),
        "q_expert_edges": int(q_expert.sum()),
        "q_expert_density": float(q_expert.mean()),
    }
    evidence_log_mat = synth.build_log_mat(train_rows, num_students, num_items)
    return metadata, q_expert, train_rows, valid_rows, test_rows, evidence_log_mat


def make_run_args(args: argparse.Namespace, config: Dict[str, Any]) -> argparse.Namespace:
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


def run_one(
    args: argparse.Namespace,
    metadata: Dict[str, Any],
    q_expert: np.ndarray,
    q_noise: np.ndarray,
    train_rows: Sequence[synth.ResponseRow],
    valid_rows: Sequence[synth.ResponseRow],
    test_rows: Sequence[synth.ResponseRow],
    evidence_log_mat: np.ndarray,
    config: Dict[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        return synth.read_json(metrics_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    run_args = make_run_args(args, config)
    model = synth.build_model(run_args, metadata, q_noise, q_train=None)
    print(f"run={config['name']} noise={config['noise_rate']} mode={config['model_mode']}")
    model, train_log, best_epoch = synth.train_model(
        model,
        train_rows,
        valid_rows,
        evidence_log_mat,
        run_args,
        q_true_eval=q_expert,
        q_noise_eval=q_noise,
    )
    if config["model_mode"] == "fixed-noisy-q":
        q_soft = q_noise.astype(np.float32)
        q_hat = q_noise.astype(np.int64)
    else:
        q_soft = model.get_q_soft().detach().cpu().numpy().astype(np.float32)
        q_hat = (q_soft > args.q_threshold).astype(np.int64)

    valid_metrics = synth.evaluate_response(model, valid_rows, evidence_log_mat, args.eval_batch_size, args.seed)
    test_metrics = synth.evaluate_response(model, test_rows, evidence_log_mat, args.eval_batch_size, args.seed)
    q_metrics = synth.q_recovery_metrics(q_expert, q_noise, q_hat, q_soft)

    metrics: Dict[str, Any] = {
        "dataset": "math1",
        "noise_rate": float(config["noise_rate"]),
        "mode": config["name"],
        "model_mode": config["model_mode"],
        "lambda_q": run_args.lambda_q,
        "lambda_sparse": run_args.lambda_sparse,
        "lambda_binary": run_args.lambda_binary,
        "lambda_coverage": run_args.lambda_coverage,
        "best_epoch": best_epoch,
        "valid_acc": valid_metrics["acc"],
        "valid_auc": valid_metrics["auc"],
        "valid_f1": valid_metrics["f1"],
        "valid_rmse": valid_metrics["rmse"],
        "test_acc": test_metrics["acc"],
        "test_auc": test_metrics["auc"],
        "test_f1": test_metrics["f1"],
        "test_rmse": test_metrics["rmse"],
        **q_metrics,
        "q_change_from_noise": int(np.sum(q_hat != q_noise)),
        "q_change_0_to_1": int(np.sum((q_noise == 0) & (q_hat == 1))),
        "q_change_1_to_0": int(np.sum((q_noise == 1) & (q_hat == 0))),
        "q_hat_density": float(q_hat.mean()),
        "result_dir": str(out_dir),
    }
    synth.write_config_yaml(out_dir / "config.yaml", {"args": vars(args), "config": config, "metadata": metadata})
    synth.write_train_log(out_dir / "train_log.csv", train_log)
    synth.write_json(metrics_path, metrics)
    np.save(out_dir / "Q_expert.npy", q_expert)
    np.save(out_dir / "Q_noise.npy", q_noise)
    np.save(out_dir / "Q_soft.npy", q_soft)
    np.save(out_dir / "Q_hat.npy", q_hat)
    torch.save({"model_state_dict": model.state_dict(), "metrics": synth.to_jsonable(metrics)}, out_dir / "model.pt")
    return metrics


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def make_report(results_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    baseline_dir = Path("Q_GNCDM/results/math1_real")
    baselines = []
    for name in ["fixed-expert-q", "free-q"]:
        path = baseline_dir / name / "metrics.json"
        if path.exists():
            baselines.append(synth.read_json(path))

    def table(title: str, data: Sequence[Dict[str, Any]]) -> List[str]:
        fields = [
            "noise_rate",
            "mode",
            "lambda_q",
            "lambda_sparse",
            "lambda_binary",
            "valid_acc",
            "test_acc",
            "test_auc",
            "q_noise_f1_vs_true",
            "q_noise_hamming_vs_true",
            "q_hat_f1_vs_true",
            "q_hat_auc_vs_true",
            "q_hat_hamming_vs_true",
            "deleted_edge_recovery",
            "added_edge_suppression",
            "q_change_from_noise",
            "q_hat_density",
            "best_epoch",
        ]
        lines = [f"## {title}", "", "| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
        for row in data:
            lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
        lines.append("")
        return lines

    md: List[str] = [
        "# Math1 Artificial Q-Noise Robustness Experiment",
        "",
        "本实验将 Math1 expert Q 作为 pseudo truth，人工构造 noisy Q；训练时只给 noisy Q，评价时比较 learned Q 是否回到 expert Q。",
        "",
        "固定使用阶段一选出的 lambda，不重新调参：",
        "",
        "- prediction-best: lambda_q=0.2, lambda_sparse=0.1, lambda_binary=0.8",
        "- balanced-best: lambda_q=0.8, lambda_sparse=0.1, lambda_binary=0.8",
        "",
    ]
    if baselines:
        md.extend(["## Clean-Q Baseline Reference", "", "| mode | test_acc | test_auc | q_hat_f1_vs_true | q_hat_hamming_vs_true |", "| --- | --- | --- | --- | --- |"])
        for row in baselines:
            md.append(
                "| "
                + " | ".join(
                    [
                        fmt(row.get("mode")),
                        fmt(row.get("test_acc")),
                        fmt(row.get("test_auc")),
                        fmt(row.get("q_hat_f1_vs_true")),
                        fmt(row.get("q_hat_hamming_vs_true")),
                    ]
                )
                + " |"
            )
        md.append("")

    sorted_rows = sorted(rows, key=lambda r: (float(r["noise_rate"]), str(r["mode"])))
    md.extend(table("All Noisy-Q Runs", sorted_rows))
    for rate in sorted({float(row["noise_rate"]) for row in rows}):
        group = [row for row in rows if abs(float(row["noise_rate"]) - rate) < 1e-9]
        by_q = sorted(group, key=lambda r: (r.get("q_hat_f1_vs_true") or -1, r.get("test_acc") or -1), reverse=True)
        md.extend(table(f"Noise {rate_label(rate)} Sorted By Q-F1", by_q))

    md.extend(
        [
            "## Interpretation Guide",
            "",
            "- `fixed-noisy-q` 表示直接使用 noisy Q，不学习修正；它给出噪声下限。",
            "- `deleted_edge_recovery` 越高，说明被噪声删掉的 expert edge 越多被恢复。",
            "- `added_edge_suppression` 越高，说明噪声额外添加的假 edge 越多被抑制。",
            "- 若 weak-q 的 Q-F1 高于 fixed-noisy-q，且 test 指标不下降，说明模型有一定 Q 噪声修正能力。",
            "",
        ]
    )
    (results_dir / "math1_noisy_q_experiment_report.md").write_text("\n".join(md), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Math1 artificial Q-noise experiment.")
    parser.add_argument("--data_dir", default="GNCDM/data")
    parser.add_argument("--results_dir", default="Q_GNCDM/results/math1_noisy_q")
    parser.add_argument("--noise_rates", default="0.05,0.10,0.20,0.30")
    parser.add_argument("--prediction_best_json", default="Q_GNCDM/results/math1_lambda_grid/valid_prediction_best.json")
    parser.add_argument("--balanced_best_json", default="Q_GNCDM/results/math1_lambda_grid/valid_balanced_best_qf1_ge_0p8.json")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--eval_batch_size", type=int, default=2048)
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
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metadata, q_expert, train_rows, valid_rows, test_rows, evidence_log_mat = load_math1(args)
    synth.write_json(results_dir / "metadata.json", metadata)
    shutil.copy2(Path(args.data_dir) / "math1_Q_matrix.npy", results_dir / "math1_Q_matrix.npy")

    pred_lambda = load_best_lambda(Path(args.prediction_best_json))
    bal_lambda = load_best_lambda(Path(args.balanced_best_json))
    rows: List[Dict[str, Any]] = []
    for rate in parse_values(args.noise_rates):
        q_noise, noise_info = make_noisy_q(q_expert, rate, seed=args.seed + int(round(rate * 1000)))
        rate_dir = results_dir / f"noise_{rate_label(rate)}"
        rate_dir.mkdir(parents=True, exist_ok=True)
        np.save(rate_dir / "Q_noise.npy", q_noise)
        synth.write_json(rate_dir / "noise_info.json", noise_info)
        configs = [
            {
                "name": "fixed-noisy-q",
                "model_mode": "fixed-noisy-q",
                "noise_rate": rate,
                "lambda_q": 0.0,
                "lambda_sparse": 0.0,
                "lambda_binary": 0.0,
                "lambda_coverage": 0.0,
            },
            {
                "name": "weak-prediction-best",
                "model_mode": "weak-q",
                "noise_rate": rate,
                **pred_lambda,
            },
            {
                "name": "weak-balanced-best",
                "model_mode": "weak-q",
                "noise_rate": rate,
                **bal_lambda,
            },
        ]
        for config in configs:
            row = run_one(
                args,
                metadata,
                q_expert,
                q_noise,
                train_rows,
                valid_rows,
                test_rows,
                evidence_log_mat,
                config,
                rate_dir / config["name"],
            )
            rows.append(row)

    write_csv(results_dir / "summary.csv", rows)
    make_report(results_dir, rows)
    print(f"Saved summary to {results_dir / 'summary.csv'}")
    print(f"Saved report to {results_dir / 'math1_noisy_q_experiment_report.md'}")


if __name__ == "__main__":
    main()
