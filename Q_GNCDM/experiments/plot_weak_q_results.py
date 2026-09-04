# -*- coding: utf-8 -*-
"""Plot basic Weak-Q synthetic experiment results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def collect_runs(results_root: Path) -> List[Dict[str, Any]]:
    runs = []
    for metrics_path in results_root.glob("seed_*/noise_*/*/metrics.json"):
        metrics = read_json(metrics_path)
        run_dir = metrics_path.parent
        runs.append(
            {
                "metrics": metrics,
                "run_dir": run_dir,
                "mode": metrics.get("mode", run_dir.name),
                "run_label": metrics.get("run_label", run_dir.name),
                "noise_rate": float(metrics.get("noise_rate", run_dir.parent.name.replace("noise_", ""))),
                "seed": run_dir.parent.parent.name,
            }
        )
    return sorted(runs, key=lambda row: (row["run_label"], row["noise_rate"]))


def plot_q_heatmaps(
    runs: List[Dict[str, Any]],
    figures_dir: Path,
    max_items: int,
    max_concepts: int,
) -> None:
    for run in runs:
        run_dir = run["run_dir"]
        q_true_path = run_dir / "Q_true.npy"
        q_noise_path = run_dir / "Q_noise.npy"
        q_hat_path = run_dir / "Q_hat.npy"
        if not (q_true_path.exists() and q_noise_path.exists() and q_hat_path.exists()):
            continue
        q_true = np.load(q_true_path)[:max_items, :max_concepts]
        q_noise = np.load(q_noise_path)[:max_items, :max_concepts]
        q_hat = np.load(q_hat_path)[:max_items, :max_concepts]
        fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
        for ax, arr, title in zip(
            axes,
            [q_true, q_noise, q_hat],
            ["Q_true", "Q_noise", "Q_hat"],
        ):
            im = ax.imshow(arr, aspect="auto", vmin=0, vmax=1, cmap="viridis")
            ax.set_title(title)
            ax.set_xlabel("concept")
            ax.set_ylabel("item")
        fig.colorbar(im, ax=axes, shrink=0.8)
        out = figures_dir / (
            f"q_heatmap_{run['seed']}_noise_{run['noise_rate']:.2f}_{run['run_label']}.png"
        )
        fig.savefig(out, dpi=160)
        plt.close(fig)


def plot_metric_curve(
    runs: List[Dict[str, Any]],
    figures_dir: Path,
    metric_name: str,
    output_name: str,
    ylabel: str,
) -> None:
    grouped = defaultdict(list)
    for run in runs:
        value = finite_or_none(run["metrics"].get(metric_name))
        if value is not None:
            grouped[run["run_label"]].append((run["noise_rate"], value))
    if not grouped:
        return
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for mode, pairs in sorted(grouped.items()):
        pairs = sorted(pairs)
        ax.plot([x for x, _ in pairs], [y for _, y in pairs], marker="o", label=mode)
    ax.set_xlabel("noise rate")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(figures_dir / output_name, dpi=160)
    plt.close(fig)


def plot_mastery_curves(runs: List[Dict[str, Any]], figures_dir: Path) -> None:
    for metric_name, ylabel, out_name in [
        ("mastery_pearson", "mastery pearson", "mastery_pearson_curve.png"),
        ("mastery_spearman", "mastery spearman", "mastery_spearman_curve.png"),
    ]:
        plot_metric_curve(runs, figures_dir, metric_name, out_name, ylabel)


def plot_q_soft_histograms(runs: List[Dict[str, Any]], figures_dir: Path) -> None:
    for run in runs:
        q_soft_path = run["run_dir"] / "Q_soft.npy"
        if not q_soft_path.exists():
            continue
        q_soft = np.load(q_soft_path).reshape(-1)
        fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
        ax.hist(q_soft, bins=30, range=(0, 1), color="#4C78A8", alpha=0.85)
        ax.set_xlabel("Q_soft value")
        ax.set_ylabel("count")
        ax.set_title(f"{run['run_label']} noise={run['noise_rate']:.2f}")
        out = figures_dir / (
            f"q_soft_hist_{run['seed']}_noise_{run['noise_rate']:.2f}_{run['run_label']}.png"
        )
        fig.savefig(out, dpi=160)
        plt.close(fig)


def read_train_log(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if value is None or value == "":
                    parsed[key] = None
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def has_metric(rows: List[Dict[str, Any]], metric: str) -> bool:
    return any(finite_or_none(row.get(metric)) is not None for row in rows)


def plot_epoch_group(
    rows: List[Dict[str, Any]],
    metrics: List[str],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if not rows:
        return
    available_metrics = [metric for metric in metrics if has_metric(rows, metric)]
    if not available_metrics:
        return
    epochs = [finite_or_none(row.get("epoch")) for row in rows]
    if any(epoch is None for epoch in epochs):
        return

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for metric in available_metrics:
        xs = []
        ys = []
        for epoch, row in zip(epochs, rows):
            value = finite_or_none(row.get(metric))
            if value is None:
                continue
            xs.append(epoch)
            ys.append(value)
        if xs:
            ax.plot(xs, ys, label=metric)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_epoch_curves(runs: List[Dict[str, Any]], figures_dir: Path) -> None:
    for run in runs:
        rows = read_train_log(run["run_dir"] / "train_log.csv")
        if not rows:
            continue
        prefix = f"{run['seed']}_noise_{run['noise_rate']:.2f}_{run['run_label']}"
        plot_epoch_group(
            rows,
            [
                "loss_total",
                "loss_rec",
                "loss_weak_q",
                "loss_sparse",
                "loss_binary",
                "loss_coverage",
            ],
            f"Loss components ({run['run_label']}, noise={run['noise_rate']:.2f})",
            "loss",
            figures_dir / f"epoch_loss_{prefix}.png",
        )
        plot_epoch_group(
            rows,
            ["valid_acc", "valid_auc", "valid_rmse"],
            f"Validation metrics ({run['run_label']}, noise={run['noise_rate']:.2f})",
            "metric",
            figures_dir / f"epoch_valid_{prefix}.png",
        )
        plot_epoch_group(
            rows,
            ["q_change_from_noise", "q_change_0_to_1", "q_change_1_to_0"],
            f"Q_hat changes from Q_noise ({run['run_label']}, noise={run['noise_rate']:.2f})",
            "count",
            figures_dir / f"epoch_q_change_{prefix}.png",
        )
        plot_epoch_group(
            rows,
            [
                "q_hat_f1_vs_true",
                "q_hat_auc_vs_true",
                "deleted_edge_recovery",
                "added_edge_suppression",
            ],
            f"Q recovery ({run['run_label']}, noise={run['noise_rate']:.2f})",
            "metric",
            figures_dir / f"epoch_q_recovery_{prefix}.png",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Weak-Q synthetic results.")
    parser.add_argument("--results_root", default="results/weak_q_synthetic")
    parser.add_argument("--max_items", type=int, default=30)
    parser.add_argument("--max_concepts", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    figures_dir = results_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(results_root)
    if not runs:
        raise SystemExit(f"No metrics.json files found under {results_root}.")

    plot_q_heatmaps(runs, figures_dir, args.max_items, args.max_concepts)
    plot_metric_curve(
        runs,
        figures_dir,
        "q_hat_f1_vs_true",
        "q_hat_f1_curve.png",
        "Q_hat F1 vs Q_true",
    )
    plot_metric_curve(
        runs,
        figures_dir,
        "response_auc",
        "response_auc_curve.png",
        "response AUC",
    )
    plot_mastery_curves(runs, figures_dir)
    plot_q_soft_histograms(runs, figures_dir)
    plot_epoch_curves(runs, figures_dir)
    print(f"Saved figures to {figures_dir}")


if __name__ == "__main__":
    main()
