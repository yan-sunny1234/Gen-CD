"""Summarize weak-Q lambda sweep outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SUMMARY_FIELDS = [
    "lambda_q",
    "lambda_sparse",
    "lambda_binary",
    "lambda_coverage",
    "response_acc",
    "response_auc",
    "response_rmse",
    "valid_acc",
    "valid_auc",
    "valid_rmse",
    "q_hat_acc_vs_true",
    "q_hat_precision_vs_true",
    "q_hat_recall_vs_true",
    "q_hat_f1_vs_true",
    "q_hat_auc_vs_true",
    "q_hat_hamming_vs_true",
    "deleted_edge_recovery",
    "added_edge_suppression",
    "q_change_from_noise",
    "q_change_0_to_1",
    "q_change_1_to_0",
    "loss_total",
    "loss_rec",
    "loss_weak_q",
    "loss_sparse",
    "loss_binary",
    "loss_coverage",
    "status",
    "result_dir",
    "metrics_path",
]


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def read_last_csv_row(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    return rows[-1] if rows else {}


def find_first(path: Path, pattern: str) -> Optional[Path]:
    matches = sorted(path.rglob(pattern))
    return matches[0] if matches else None


def to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sort_key(row: Dict[str, Any]) -> tuple:
    acc = to_float(row.get("response_acc"))
    if acc is None:
        acc = to_float(row.get("valid_acc"))
    auc = to_float(row.get("response_auc"))
    if auc is None:
        auc = to_float(row.get("valid_auc"))
    rmse = to_float(row.get("response_rmse"))
    if rmse is None:
        rmse = to_float(row.get("valid_rmse"))
    q_f1 = to_float(row.get("q_hat_f1_vs_true"))
    return (
        acc if acc is not None else -1.0,
        auc if auc is not None else -1.0,
        -(rmse if rmse is not None else float("inf")),
        q_f1 if q_f1 is not None else -1.0,
    )


def has_primary_metric(row: Dict[str, Any]) -> bool:
    return to_float(row.get("response_acc")) is not None or to_float(row.get("valid_acc")) is not None


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def collect_rows(results_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for combo_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        config = read_json(combo_dir / "sweep_config.json")
        status = read_json(combo_dir / "status.json")
        metrics_path = find_first(combo_dir, "metrics.json")
        metrics = read_json(metrics_path) if metrics_path else {}
        train_log_path = find_first(combo_dir, "train_log.csv")
        last_train = read_last_csv_row(train_log_path) if train_log_path else {}

        row: Dict[str, Any] = {}
        row.update(config)
        row.update(last_train)
        row.update(metrics)
        row["status"] = status.get("status", "")
        row["result_dir"] = str(combo_dir)
        row["metrics_path"] = str(metrics_path) if metrics_path else ""
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize weak-Q lambda sweep.")
    parser.add_argument("--results_dir", default="results/weak_q_lambda_sweep")
    parser.add_argument("--top_k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = project_dir / results_dir

    rows = collect_rows(results_dir)
    sorted_rows = sorted(rows, key=sort_key, reverse=True)
    metric_rows = [row for row in sorted_rows if has_primary_metric(row)]

    write_csv(results_dir / "summary.csv", rows)
    write_csv(results_dir / "summary_sorted_by_response_acc.csv", sorted_rows)
    write_csv(results_dir / "summary_sorted_by_valid_acc.csv", sorted_rows)

    best = metric_rows[0] if metric_rows else {}
    with (results_dir / "best_by_valid_acc.json").open("w", encoding="utf-8") as fp:
        json.dump(best, fp, indent=2, sort_keys=True)

    print("Top combinations:")
    for idx, row in enumerate(metric_rows[: args.top_k], start=1):
        acc = row.get("response_acc") or row.get("valid_acc")
        auc = row.get("response_auc") or row.get("valid_auc")
        rmse = row.get("response_rmse") or row.get("valid_rmse")
        print(
            f"{idx}. lq={row.get('lambda_q')} "
            f"ls={row.get('lambda_sparse')} "
            f"lb={row.get('lambda_binary')} "
            f"acc={acc} auc={auc} rmse={rmse}"
        )
    if not metric_rows:
        print("No completed metric rows found.")


if __name__ == "__main__":
    main()
