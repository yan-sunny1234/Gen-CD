"""Combine completed simple-contrastive lambda runs without retraining."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Q_GNCDM.experiments.math1_data_paths import same_controlled_args, same_source_files


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_run(value):
    try:
        weight, path = value.split("=", 1)
        return float(weight), Path(path).resolve()
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use LAMBDA=RESULTS_DIR") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--lambda-run", type=parse_run, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline_root = args.baseline_results.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    roots = [baseline_root] + [path for _, path in args.lambda_run]
    manifests = [read_json(root / "manifest.json") for root in roots]
    for root in roots:
        assert read_json(root / "completed.json")["runs_completed"] > 0
        assert read_json(root / "verification.json")["status"] == "passed"
    assert all(same_source_files(manifests[0], manifest) for manifest in manifests)
    stable_keys = (
        "dataset", "data_dir", "seeds", "q_noise", "split_seed", "alpha",
        "embedding_dim", "epochs", "patience", "batch_students", "query_fraction",
        "positive_noise_rate", "temperature", "lr", "q_lr", "lambda_q",
        "lambda_coverage", "q_init_high", "q_init_low", "device", "threads",
    )
    reference_args = manifests[0]["args"]
    for manifest in manifests[1:]:
        assert same_controlled_args(reference_args, manifest["args"], stable_keys)

    baseline = {
        int(row["seed"]): row for row in read_csv(baseline_root / "summary.csv")
        if row["mode"] == "A1"
    }
    detail = []
    for seed, row in sorted(baseline.items()):
        detail.append({"variant": "A1", "lambda_cl": 0.0, "seed": seed, **row})
    treatment_by_lambda = {}
    for weight, root in sorted(args.lambda_run):
        rows = {
            int(row["seed"]): row for row in read_csv(root / "summary.csv")
            if row["mode"] == "S1"
        }
        if rows.keys() != baseline.keys():
            raise ValueError(f"Seed mismatch for lambda={weight}")
        treatment_by_lambda[weight] = rows
        for seed, row in sorted(rows.items()):
            detail.append({"variant": "S1", "lambda_cl": weight, "seed": seed, **row})
    write_csv(output / "sweep_summary.csv", detail)

    numeric_metrics = (
        "valid_auc", "test_auc", "test_acc", "test_rmse", "test_brier",
        "test_nll", "q_f1_vs_reference", "theta_std_mean", "best_epoch",
    )
    aggregate = []
    groups = [(0.0, baseline)] + sorted(treatment_by_lambda.items())
    for weight, rows in groups:
        record = {"variant": "A1" if weight == 0 else "S1",
                  "lambda_cl": weight, "n_seeds": len(rows)}
        for metric in numeric_metrics:
            values = np.asarray([float(row[metric]) for row in rows.values()])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1))
        aggregate.append(record)
    write_csv(output / "sweep_aggregate.csv", aggregate)

    paired = []
    for weight, rows in sorted(treatment_by_lambda.items()):
        for metric in numeric_metrics:
            values = np.asarray([
                float(rows[seed][metric]) - float(baseline[seed][metric])
                for seed in sorted(baseline)
            ])
            paired.append({
                "base": "A1", "treatment": "S1", "lambda_cl": weight,
                "metric": metric, "n_pairs": len(values),
                "mean_delta": float(values.mean()),
                "std_delta": float(values.std(ddof=1)),
                "positive_seed_count": int(np.sum(values > 0)),
                "negative_seed_count": int(np.sum(values < 0)),
            })
    write_csv(output / "paired_deltas_vs_A1.csv", paired)

    candidates = [row for row in aggregate if row["variant"] == "S1"]
    selected = max(candidates, key=lambda row: row["valid_auc_mean"])
    selection = {
        "selection_metric": "valid_auc_mean",
        "test_metrics_not_used_for_selection": True,
        "selected_lambda_cl": selected["lambda_cl"],
        "selected_valid_auc_mean": selected["valid_auc_mean"],
        "baseline_results": str(baseline_root),
        "lambda_results": {str(weight): str(path) for weight, path in args.lambda_run},
        "source_sha256": reference_sources,
    }
    (output / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
