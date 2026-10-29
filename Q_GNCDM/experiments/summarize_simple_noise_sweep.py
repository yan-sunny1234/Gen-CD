"""Combine completed simple-contrastive noise-rate runs without retraining."""

import argparse
import csv
import json
import sys
from collections import defaultdict
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
        rate, path = value.split("=", 1)
        return float(rate), Path(path).resolve()
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use RATE=RESULTS_DIR") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--noise-run", type=parse_run, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline_root = args.baseline_results.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    roots = [baseline_root] + [path for _, path in args.noise_run]
    manifests = [read_json(root / "manifest.json") for root in roots]
    for root in roots:
        if read_json(root / "completed.json")["runs_completed"] <= 0:
            raise ValueError(f"Incomplete result set: {root}")
        if read_json(root / "verification.json")["status"] != "passed":
            raise ValueError(f"Unverified result set: {root}")
    reference_code = manifests[0]["code_sha256"]
    if not all(same_source_files(manifests[0], manifest) for manifest in manifests):
        raise ValueError("Data hashes differ between runs")
    if not all(manifest["code_sha256"] == reference_code for manifest in manifests):
        raise ValueError("Code hashes differ between runs")
    stable_keys = (
        "dataset", "data_dir", "seeds", "q_noise", "split_seed", "alpha",
        "embedding_dim", "epochs", "patience", "batch_students", "query_fraction",
        "temperature", "lambda_cl", "lr", "q_lr", "lambda_q",
        "lambda_coverage", "q_init_high", "q_init_low", "device", "threads",
    )
    reference_args = manifests[1]["args"]
    for manifest in manifests[2:]:
        if not same_controlled_args(reference_args, manifest["args"], stable_keys):
            raise ValueError("Noise runs do not share the same controlled settings")

    baseline = {
        int(row["seed"]): row for row in read_csv(baseline_root / "summary.csv")
        if row["mode"] == "A1"
    }
    detail = []
    for seed, row in sorted(baseline.items()):
        detail.append({
            "variant": "A1", "nominal_noise_rate": 0.0,
            "effective_flip_fraction": 0.0, "seed": seed, **row,
        })

    treatment_by_rate = {}
    effective_by_rate = {}
    for rate, root in sorted(args.noise_run):
        rows = {
            int(row["seed"]): row for row in read_csv(root / "summary.csv")
            if row["mode"] == "S1"
        }
        if rows.keys() != baseline.keys():
            raise ValueError(f"Seed mismatch for noise rate={rate}")
        actual = np.asarray([
            float(row["mean_actual_flip_fraction"]) for row in rows.values()
        ])
        effective = float(actual.mean())
        treatment_by_rate[rate] = rows
        effective_by_rate[rate] = effective
        for seed, row in sorted(rows.items()):
            detail.append({
                "variant": "S1", "nominal_noise_rate": rate,
                "effective_flip_fraction": float(row["mean_actual_flip_fraction"]),
                "seed": seed, **row,
            })
    write_csv(output / "sweep_summary.csv", detail)

    numeric_metrics = (
        "valid_auc", "test_auc", "test_acc", "test_rmse", "test_brier",
        "test_nll", "q_f1_vs_reference", "theta_std_mean", "best_epoch",
        "mean_actual_flip_fraction",
    )
    aggregate = []
    groups = [(0.0, baseline)] + sorted(treatment_by_rate.items())
    for rate, rows in groups:
        record = {
            "variant": "A1" if rate == 0 else "S1",
            "nominal_noise_rate": rate,
            "n_seeds": len(rows),
        }
        for metric in numeric_metrics:
            if metric == "mean_actual_flip_fraction" and rate == 0:
                values = np.zeros(len(rows))
            else:
                values = np.asarray([float(row[metric]) for row in rows.values()])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1))
        aggregate.append(record)
    write_csv(output / "sweep_aggregate.csv", aggregate)

    paired = []
    for rate, rows in sorted(treatment_by_rate.items()):
        for metric in numeric_metrics[:-1]:
            values = np.asarray([
                float(rows[seed][metric]) - float(baseline[seed][metric])
                for seed in sorted(baseline)
            ])
            paired.append({
                "base": "A1", "treatment": "S1",
                "nominal_noise_rate": rate,
                "effective_flip_fraction": effective_by_rate[rate],
                "metric": metric, "n_pairs": len(values),
                "mean_delta": float(values.mean()),
                "std_delta": float(values.std(ddof=1)),
                "positive_seed_count": int(np.sum(values > 0)),
                "negative_seed_count": int(np.sum(values < 0)),
            })
    write_csv(output / "paired_deltas_vs_A1.csv", paired)

    effective_groups = defaultdict(list)
    for rate, effective in effective_by_rate.items():
        effective_groups[round(effective, 12)].append(rate)
    representatives = {}
    for effective, rates in effective_groups.items():
        # Prefer the nominal rate that accurately names the realized flip fraction.
        representatives[effective] = min(rates, key=lambda rate: (abs(rate - effective), rate))
    aggregate_by_rate = {
        float(row["nominal_noise_rate"]): row
        for row in aggregate if row["variant"] == "S1"
    }
    selected_rate = max(
        representatives.values(),
        key=lambda rate: aggregate_by_rate[rate]["valid_auc_mean"],
    )
    duplicate_groups = [
        {
            "effective_flip_fraction": effective,
            "nominal_noise_rates": sorted(rates),
            "representative_rate": representatives[effective],
        }
        for effective, rates in sorted(effective_groups.items()) if len(rates) > 1
    ]
    selection = {
        "selection_metric": "valid_auc_mean",
        "test_metrics_not_used_for_selection": True,
        "selection_unit": "unique_effective_flip_fraction",
        "selected_nominal_noise_rate": selected_rate,
        "selected_effective_flip_fraction": effective_by_rate[selected_rate],
        "selected_valid_auc_mean": aggregate_by_rate[selected_rate]["valid_auc_mean"],
        "equivalent_nominal_rate_groups": duplicate_groups,
        "baseline_results": str(baseline_root),
        "noise_results": {str(rate): str(path) for rate, path in args.noise_run},
        "source_sha256": reference_sources,
        "code_sha256": reference_code,
    }
    (output / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
