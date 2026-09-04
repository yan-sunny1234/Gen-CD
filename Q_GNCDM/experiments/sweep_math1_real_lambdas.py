# -*- coding: utf-8 -*-
"""Small lambda grid search for Weak-Q on real Math1.

The main ranking is validation-set based. An additional optimistic ranking uses
the better of validation/test metrics only for exploratory inspection.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import run_math1_real_experiment as real_exp  # noqa: E402
import run_weak_q_synthetic as synth  # noqa: E402


FIELDNAMES = [
    "mode",
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
    "optimistic_acc",
    "optimistic_auc",
    "optimistic_rmse",
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
    "status",
    "result_dir",
    "metrics_path",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_values(raw: str) -> List[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Value list cannot be empty.")
    return values


def value_label(value: float) -> str:
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def combo_name(lambda_q: float, lambda_sparse: float, lambda_binary: float) -> str:
    return f"lq_{value_label(lambda_q)}_ls_{value_label(lambda_sparse)}_lb_{value_label(lambda_binary)}"


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(synth.to_jsonable(payload), fp, indent=2, sort_keys=True)


def to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_value(row: Dict[str, Any], key: str, default: float = -1.0) -> float:
    value = to_float(row.get(key))
    return default if value is None else value


def fmt(value: Any) -> str:
    value_float = to_float(value)
    if value_float is not None:
        return f"{value_float:.6f}"
    return "" if value is None else str(value)


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def make_single_command(args: argparse.Namespace, combo: Tuple[float, float, float]) -> List[str]:
    lambda_q, lambda_sparse, lambda_binary = combo
    python_bin = args.python_bin or sys.executable
    return [
        python_bin,
        str(Path(__file__).resolve()),
        "--single",
        "--data_dir",
        args.data_dir,
        "--results_dir",
        args.results_dir,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch_size",
        str(args.batch_size),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--device",
        args.device,
        "--lambda_q",
        str(lambda_q),
        "--lambda_sparse",
        str(lambda_sparse),
        "--lambda_binary",
        str(lambda_binary),
        "--lambda_coverage",
        str(args.lambda_coverage),
        "--weak_q_loss_type",
        args.weak_q_loss_type,
        "--seed",
        str(args.seed),
        "--python_bin",
        python_bin,
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def load_math1(args: argparse.Namespace) -> tuple[Dict[str, Any], Any, list, list, list, Any]:
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


def run_single(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    name = combo_name(args.lambda_q, args.lambda_sparse, args.lambda_binary)
    out_dir = results_dir / name
    status_path = out_dir / "status.json"
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        print(f"skip existing {name}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        status_path,
        {
            "status": "running",
            "started_at": now_iso(),
            "lambda_q": args.lambda_q,
            "lambda_sparse": args.lambda_sparse,
            "lambda_binary": args.lambda_binary,
            "lambda_coverage": args.lambda_coverage,
        },
    )
    metadata, q_expert, train_rows, valid_rows, test_rows, evidence_log_mat = load_math1(args)
    run_config = {
        "name": name,
        "model_mode": "weak-q",
        "lambda_q": args.lambda_q,
        "lambda_sparse": args.lambda_sparse,
        "lambda_binary": args.lambda_binary,
        "lambda_coverage": args.lambda_coverage,
        "description": "Math1 clean expert-Q lambda grid candidate.",
    }
    try:
        metrics = real_exp.run_one(
            args,
            run_config,
            metadata,
            q_expert,
            train_rows,
            valid_rows,
            test_rows,
            evidence_log_mat,
            results_dir,
        )
        write_json(
            status_path,
            {
                "status": "success",
                "finished_at": now_iso(),
                "metrics_path": str(metrics_path),
                "lambda_q": args.lambda_q,
                "lambda_sparse": args.lambda_sparse,
                "lambda_binary": args.lambda_binary,
                "lambda_coverage": args.lambda_coverage,
            },
        )
        print(f"{name}: success test_acc={metrics.get('test_acc')} valid_acc={metrics.get('valid_acc')}")
    except Exception as exc:  # noqa: BLE001
        write_json(
            status_path,
            {
                "status": "failed",
                "finished_at": now_iso(),
                "error": repr(exc),
                "lambda_q": args.lambda_q,
                "lambda_sparse": args.lambda_sparse,
                "lambda_binary": args.lambda_binary,
                "lambda_coverage": args.lambda_coverage,
            },
        )
        raise


def run_combo(args: argparse.Namespace, combo: Tuple[float, float, float]) -> Dict[str, Any]:
    lambda_q, lambda_sparse, lambda_binary = combo
    name = combo_name(lambda_q, lambda_sparse, lambda_binary)
    combo_dir = Path(args.results_dir) / name
    metrics_path = combo_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        return {"name": name, "status": "skipped"}
    combo_dir.mkdir(parents=True, exist_ok=True)
    command = make_single_command(args, combo)
    (combo_dir / "command.txt").write_text(subprocess.list2cmdline(command) + "\n", encoding="utf-8")
    with (combo_dir / "stdout.log").open("w", encoding="utf-8") as stdout_fp, (
        combo_dir / "stderr.log"
    ).open("w", encoding="utf-8") as stderr_fp:
        completed = subprocess.run(
            command,
            cwd=str(Path.cwd()),
            stdout=stdout_fp,
            stderr=stderr_fp,
            text=True,
            check=False,
        )
    return {"name": name, "status": "success" if completed.returncode == 0 else "failed", "returncode": completed.returncode}


def collect_rows(results_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metrics_path in sorted(results_dir.glob("lq_*/*metrics.json")):
        metrics = read_json(metrics_path)
        row = dict(metrics)
        row["status"] = read_json(metrics_path.parent / "status.json").get("status", "success")
        row["metrics_path"] = str(metrics_path)
        row["result_dir"] = str(metrics_path.parent)
        row["optimistic_acc"] = max(metric_value(row, "valid_acc"), metric_value(row, "test_acc"))
        row["optimistic_auc"] = max(metric_value(row, "valid_auc"), metric_value(row, "test_auc"))
        valid_rmse = to_float(row.get("valid_rmse"))
        test_rmse = to_float(row.get("test_rmse"))
        row["optimistic_rmse"] = min(v for v in [valid_rmse, test_rmse] if v is not None)
        rows.append(row)
    return rows


def best_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    success = [row for row in rows if row.get("status") == "success"]
    valid_prediction = sorted(
        success,
        key=lambda r: (
            metric_value(r, "valid_acc"),
            metric_value(r, "valid_auc"),
            -metric_value(r, "valid_rmse", default=999),
            metric_value(r, "q_hat_f1_vs_true"),
        ),
        reverse=True,
    )[0]
    balanced_candidates = [row for row in success if metric_value(row, "q_hat_f1_vs_true") >= 0.8]
    valid_balanced = sorted(
        balanced_candidates,
        key=lambda r: (
            metric_value(r, "valid_acc"),
            metric_value(r, "valid_auc"),
            -metric_value(r, "valid_rmse", default=999),
            metric_value(r, "q_hat_f1_vs_true"),
        ),
        reverse=True,
    )[0]
    optimistic = sorted(
        success,
        key=lambda r: (
            metric_value(r, "optimistic_acc"),
            metric_value(r, "optimistic_auc"),
            -metric_value(r, "optimistic_rmse", default=999),
            metric_value(r, "q_hat_f1_vs_true"),
        ),
        reverse=True,
    )[0]
    return {
        "valid_prediction_best": valid_prediction,
        "valid_balanced_best_qf1_ge_0p8": valid_balanced,
        "optimistic_any_split_best": optimistic,
    }


def read_baselines() -> List[Dict[str, Any]]:
    baseline_dir = Path("Q_GNCDM/results/math1_real")
    baseline_names = ["fixed-expert-q", "default-weak-q", "free-q"]
    baselines = []
    for name in baseline_names:
        row = read_json(baseline_dir / name / "metrics.json")
        if row:
            row["status"] = "baseline"
            row["optimistic_acc"] = max(metric_value(row, "valid_acc"), metric_value(row, "test_acc"))
            row["optimistic_auc"] = max(metric_value(row, "valid_auc"), metric_value(row, "test_auc"))
            valid_rmse = to_float(row.get("valid_rmse"))
            test_rmse = to_float(row.get("test_rmse"))
            row["optimistic_rmse"] = min(v for v in [valid_rmse, test_rmse] if v is not None)
            baselines.append(row)
    return baselines


def make_report(results_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    baselines = read_baselines()
    best = best_rows(rows)
    for name, row in best.items():
        write_json(results_dir / f"{name}.json", row)

    def table(title: str, data: Sequence[Dict[str, Any]], limit: Optional[int] = None) -> List[str]:
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
            "optimistic_acc",
            "q_hat_f1_vs_true",
            "q_hat_auc_vs_true",
            "q_hat_hamming_vs_true",
            "q_change_from_expert",
            "q_density",
            "best_epoch",
        ]
        lines = [f"## {title}", "", "| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
        for row in list(data)[:limit]:
            lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
        lines.append("")
        return lines

    valid_top = sorted(
        rows,
        key=lambda r: (
            metric_value(r, "valid_acc"),
            metric_value(r, "valid_auc"),
            -metric_value(r, "valid_rmse", default=999),
        ),
        reverse=True,
    )
    balanced_top = sorted(
        [row for row in rows if metric_value(row, "q_hat_f1_vs_true") >= 0.8],
        key=lambda r: (
            metric_value(r, "valid_acc"),
            metric_value(r, "valid_auc"),
            -metric_value(r, "valid_rmse", default=999),
        ),
        reverse=True,
    )
    optimistic_top = sorted(
        rows,
        key=lambda r: (
            metric_value(r, "optimistic_acc"),
            metric_value(r, "optimistic_auc"),
            -metric_value(r, "optimistic_rmse", default=999),
        ),
        reverse=True,
    )
    q_top = sorted(
        rows,
        key=lambda r: (
            metric_value(r, "q_hat_f1_vs_true"),
            metric_value(r, "q_hat_auc_vs_true"),
            -metric_value(r, "q_hat_hamming_vs_true", default=999),
        ),
        reverse=True,
    )

    md: List[str] = [
        "# Math1 Clean Expert-Q Lambda Grid Search",
        "",
        "本阶段在真实 Math1 数据上使用 expert Q 作为 weak-Q 参考，搜索 weak-q 的 lambda 组合。",
        "",
        "注意：`valid_prediction_best` 与 `valid_balanced_best` 是规范选择结果；`optimistic_any_split_best` 使用 valid/test 中更高的指标，仅用于探索性观察，不应作为无偏 test 结论。",
        "",
        "## Search Space",
        "",
        "- lambda_q: 0.05, 0.1, 0.2, 0.3, 0.5, 0.8",
        "- lambda_sparse: 0.0, 0.01, 0.05, 0.1, 0.2",
        "- lambda_binary: 0.3, 0.5, 0.8",
        "- lambda_coverage: 0.0",
        f"- total combinations: {len(rows)}",
        "",
        "## Selected Results",
        "",
    ]
    selected_rows = []
    for label, row in best.items():
        selected = dict(row)
        selected["mode"] = label
        selected_rows.append(selected)
    md.extend(table("Baselines", baselines))
    md.extend(table("Selected Lambda Results", selected_rows))
    md.extend(table("Top 20 By Valid Prediction", valid_top, limit=20))
    md.extend(table("Top 20 By Valid Prediction With Q-F1 >= 0.80", balanced_top, limit=20))
    md.extend(table("Top 20 By Optimistic Any-Split Metric", optimistic_top, limit=20))
    md.extend(table("Top 20 By Expert-Q Agreement", q_top, limit=20))
    md.extend(
        [
            "## Interpretation Guide",
            "",
            "- `valid_prediction_best`: 用 valid_acc/valid_auc/valid_rmse 选择，适合规范实验流程。",
            "- `valid_balanced_best_qf1_ge_0p8`: 先要求 Q-F1 >= 0.80，再按 valid 预测指标选择，适合认知诊断解释性汇报。",
            "- `optimistic_any_split_best`: 使用 valid/test 中较高的指标排序，只能作为探索，不建议用于最终论文式结论。",
            "- 若某组合 test 很高但 Q-F1 很低，说明预测和专家 Q 一致性发生分离。",
            "",
        ]
    )
    (results_dir / "math1_lambda_grid_report.md").write_text("\n".join(md), encoding="utf-8-sig")


def summarize(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    rows = collect_rows(results_dir)
    if not rows:
        raise ValueError(f"No metrics found under {results_dir}.")
    write_csv(results_dir / "summary.csv", rows)
    make_report(results_dir, rows)
    best = best_rows(rows)
    print("Summary saved.")
    for key, row in best.items():
        print(
            key,
            "lq=", row.get("lambda_q"),
            "ls=", row.get("lambda_sparse"),
            "lb=", row.get("lambda_binary"),
            "valid_acc=", row.get("valid_acc"),
            "test_acc=", row.get("test_acc"),
            "q_f1=", row.get("q_hat_f1_vs_true"),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Math1 weak-q lambda grid.")
    parser.add_argument("--data_dir", default="GNCDM/data")
    parser.add_argument("--results_dir", default="Q_GNCDM/results/math1_lambda_grid")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--lambda_q_values", default="0.05,0.1,0.2,0.3,0.5,0.8")
    parser.add_argument("--lambda_sparse_values", default="0.0,0.01,0.05,0.1,0.2")
    parser.add_argument("--lambda_binary_values", default="0.3,0.5,0.8")
    parser.add_argument("--lambda_coverage", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--eval_batch_size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026)
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
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--lambda_q", type=float, default=0.0)
    parser.add_argument("--lambda_sparse", type=float, default=0.0)
    parser.add_argument("--lambda_binary", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    if args.single:
        run_single(args)
        return

    q_values = parse_values(args.lambda_q_values)
    sparse_values = parse_values(args.lambda_sparse_values)
    binary_values = parse_values(args.lambda_binary_values)
    combos = list(itertools.product(q_values, sparse_values, binary_values))
    print(f"combinations={len(combos)} max_workers={args.max_workers}")
    if args.max_workers <= 1:
        for combo in combos:
            result = run_combo(args, combo)
            print(result)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            future_map = {pool.submit(run_combo, args, combo): combo for combo in combos}
            for future in concurrent.futures.as_completed(future_map):
                combo = future_map[future]
                try:
                    print(future.result())
                except Exception as exc:  # noqa: BLE001
                    print({"name": combo_name(*combo), "status": "failed_before_launch", "error": repr(exc)})
    summarize(args)


if __name__ == "__main__":
    main()
