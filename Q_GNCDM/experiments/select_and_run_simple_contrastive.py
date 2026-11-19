"""Compare fixed-Q A0 with fixed-Q plus response-flip InfoNCE over CL weights."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from Q_GNCDM.experiments import run_prepared_dataset as prepared
from Q_GNCDM.experiments import run_simple_contrastive as core


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES = (0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0)


def coefficient_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def read_metrics(directory: Path, seeds: list[int], mode: str) -> list[dict]:
    return [json.loads((directory / f"seed_{seed}" / mode / "metrics.json").read_text(
        encoding="utf-8"
    )) for seed in seeds]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(
            key for row in rows for key in row
        )))
        writer.writeheader()
        writer.writerows(rows)


def compact_and_pair(group: Path, condition: str, mode: str, coefficient: float) -> None:
    run = group / condition
    rows = read_csv(run / "summary.csv")
    if mode == "C0":
        base = {int(row["seed"]): row for row in read_csv(group / "baseline_A0" / "summary.csv")}
        paired = []
        for row in rows:
            baseline = base[int(row["seed"])]
            for record, label, weight in ((baseline, "A0", 0.0), (row, "C0", coefficient)):
                output = dict(record)
                output["mode"] = label
                output["lambda_cl"] = weight
                output["paired_delta_valid_auc"] = (
                    float(record["valid_auc"]) - float(baseline["valid_auc"])
                )
                output["paired_delta_test_auc"] = (
                    float(record["test_auc"]) - float(baseline["test_auc"])
                )
                output["paired_delta_test_acc"] = (
                    float(record["test_acc"]) - float(baseline["test_acc"])
                )
                paired.append(output)
        rows = paired
    else:
        rows = [{**row, "lambda_cl": 0.0,
                 "paired_delta_valid_auc": 0.0,
                 "paired_delta_test_auc": 0.0,
                 "paired_delta_test_acc": 0.0} for row in rows]
    save_csv(run / "summary.csv", rows)
    for redundant in ("aggregate.csv", "completed.json"):
        (run / redundant).unlink()
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["baseline"] = "A0 fixed reference Q, no contrastive loss"
    manifest["single_change"] = (
        "C0 adds response-flip InfoNCE; Q remains fixed" if mode == "C0"
        else "reference baseline"
    )
    manifest["orchestrator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["artifact_policy"] = (
        "Per condition: manifest, summary, verification, per-seed metrics and checkpoint; "
        "candidate summary includes paired A0 rows copied from baseline_A0."
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--candidates", type=float, nargs="+", default=list(DEFAULT_CANDIDATES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if (any(not np.isfinite(value) or value <= 0 for value in args.candidates)
            or len(set(args.candidates)) != len(args.candidates)):
        parser.error("Candidates must be distinct, finite positive coefficients")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct")

    # Register a new fixed-Q contrastive mode without changing the historical runner.
    core.MODES["C0"] = {
        "label": "fixed-q-simple-contrastive",
        "learn_q": False,
        "contrastive": True,
    }
    group_name = f"simple_contrastive_{args.dataset}"
    group = ROOT / "Q_GNCDM" / "results" / group_name
    prepared.load_data(ROOT / "Q_GNCDM" / "data" / args.dataset)
    common = [
        "--dataset", args.dataset,
        "--result-group", group_name,
        "--seeds", *map(str, args.seeds),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--device", args.device,
        "--lambda-q", "0",
        "--lambda-coverage", "0",
    ]
    conditions = [("baseline_A0", "A0", 0.0)] + [
        (f"lambda_{coefficient_token(value)}", "C0", value)
        for value in args.candidates
    ]
    for condition, mode, coefficient in conditions:
        run = group / condition
        if run.exists():
            if (run / "verification.json").is_file() and (
                json.loads((run / "verification.json").read_text(encoding="utf-8"))
                .get("status") == "passed"
            ):
                print(f"Already completed {group_name}/{condition}", flush=True)
                continue
            raise FileExistsError(f"Incomplete condition needs inspection: {run}")
        print(f"Starting {group_name}/{condition}: {mode}, lambda_cl={coefficient:g}", flush=True)
        with open(os.devnull, "w", encoding="utf-8") as null, contextlib.redirect_stdout(null):
            prepared.main([
                *common,
                "--experiment-id", condition,
                "--modes", mode,
                "--lambda-cl", str(coefficient),
            ])
        compact_and_pair(group, condition, mode, coefficient)
        print(f"Finished {group_name}/{condition}", flush=True)

    baseline = read_metrics(group / "baseline_A0", args.seeds, "A0")
    rows = [{
        "condition": "baseline_A0",
        "mode": "A0",
        "lambda_cl": 0.0,
        "n_seeds": len(args.seeds),
        "valid_auc": float(np.mean([r["valid_auc"] for r in baseline])),
        "valid_acc": float(np.mean([r["valid_acc"] for r in baseline])),
        "test_auc": float(np.mean([r["test_auc"] for r in baseline])),
        "test_acc": float(np.mean([r["test_acc"] for r in baseline])),
        "test_rmse": float(np.mean([r["test_rmse"] for r in baseline])),
        "test_nll": float(np.mean([r["test_nll"] for r in baseline])),
        "delta_test_auc_vs_A0": 0.0,
        "delta_test_acc_vs_A0": 0.0,
        "q_f1_vs_reference": float(np.mean([r["q_f1_vs_reference"] for r in baseline])),
    }]
    for coefficient in args.candidates:
        condition = f"lambda_{coefficient_token(coefficient)}"
        treatment = read_metrics(group / condition, args.seeds, "C0")
        rows.append({
            "condition": condition,
            "mode": "C0",
            "lambda_cl": coefficient,
            "n_seeds": len(args.seeds),
            "valid_auc": float(np.mean([r["valid_auc"] for r in treatment])),
            "valid_acc": float(np.mean([r["valid_acc"] for r in treatment])),
            "test_auc": float(np.mean([r["test_auc"] for r in treatment])),
            "test_acc": float(np.mean([r["test_acc"] for r in treatment])),
            "test_rmse": float(np.mean([r["test_rmse"] for r in treatment])),
            "test_nll": float(np.mean([r["test_nll"] for r in treatment])),
            "delta_test_auc_vs_A0": float(np.mean([
                s["test_auc"] - a["test_auc"] for a, s in zip(baseline, treatment)
            ])),
            "delta_test_acc_vs_A0": float(np.mean([
                s["test_acc"] - a["test_acc"] for a, s in zip(baseline, treatment)
            ])),
            "q_f1_vs_reference": float(np.mean([
                r["q_f1_vs_reference"] for r in treatment
            ])),
        })
    selected = max(rows[1:], key=lambda row: (row["valid_auc"], -row["lambda_cl"]))
    save_csv(group / "coefficient_comparison.csv", rows)
    (group / "selection.json").write_text(json.dumps({
        "experiment_id": group_name,
        "dataset": args.dataset,
        "baseline": "A0 fixed reference Q, no contrastive loss",
        "single_change": "C0 adds response-flip InfoNCE while Q remains fixed",
        "split": "fixed prepared train/valid/test files",
        "seeds": args.seeds,
        "selection_metric": "mean validation AUC across seeds; lower coefficient on tie",
        "candidate_lambdas": args.candidates,
        "selected_lambda_cl": selected["lambda_cl"],
        "selected_result_dir": selected["condition"],
        "boundary_selected": selected["lambda_cl"] in (
            min(args.candidates), max(args.candidates)
        ),
        "note": "Test metrics are reported but excluded from coefficient selection.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {args.dataset}: lambda_cl={selected['lambda_cl']:g}", flush=True)


if __name__ == "__main__":
    main()
