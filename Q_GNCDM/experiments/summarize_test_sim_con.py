"""Summarize completed test_sim_con conditions against matched baselines.

Only validation AUC selects a method. Test metrics are reported after selection.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "Q_GNCDM" / "results" / "test_sim_con"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows_for(directory: Path) -> dict[int, dict]:
    return {int(path.parent.name.removeprefix("seed_")): read(path)
            for path in sorted(directory.glob("seed_*/metrics.json"))}


def main():
    completed = []
    for directory in sorted(ROOT.iterdir()):
        if not directory.is_dir() or not (directory / "verification.json").exists():
            continue
        if read(directory / "verification.json").get("status") != "passed":
            continue
        manifest = read(directory / "manifest.json")
        aggregate = read(directory / "aggregate.json")
        args = manifest["args"]
        completed.append((directory, args, aggregate, rows_for(directory)))
    baselines = {(args["dataset"], args.get("split_id", "prepared"),
                  args["q_mode"], args["epochs"], args["alpha"]):
                 (directory, seeds)
                 for directory, args, _, seeds in completed if args["method"] == "baseline"}
    output = []
    for directory, args, aggregate, seeds in completed:
        base_entry = baselines.get((args["dataset"], args.get("split_id", "prepared"),
                                   args["q_mode"],
                                   args["epochs"], args["alpha"]))
        baseline = base_entry[1] if base_entry else {}
        paired = sorted(set(seeds) & set(baseline))
        row = {
            "experiment_id": directory.name,
            "dataset": args["dataset"],
            "split_id": args.get("split_id", "prepared"),
            "q_mode": args["q_mode"],
            "method": args["method"],
            "lambda_aux": args["lambda_aux"],
            "view_rate": args["view_rate"],
            "epochs": args["epochs"],
            "alpha": args["alpha"],
            "q_init_high": args["q_init_high"],
            "q_init_low": args["q_init_low"],
            "n_seeds": len(seeds),
            "n_paired": len(paired),
            "valid_auc_mean": aggregate["valid_auc_mean"],
            "valid_auc_std": aggregate["valid_auc_std"],
            "test_auc_mean": aggregate["test_auc_mean"],
            "test_auc_std": aggregate["test_auc_std"],
            "test_acc_mean": aggregate["test_acc_mean"],
            "test_acc_std": aggregate["test_acc_std"],
            "test_nll_mean": aggregate["test_nll_mean"],
            "test_brier_mean": aggregate["test_brier_mean"],
            "theta_effective_rank_mean": aggregate["theta_effective_rank_mean"],
            "q_hard_density_mean": aggregate["q_hard_density_mean"],
            "q_f1_vs_reference_mean": aggregate["q_f1_vs_reference_mean"],
            "best_epoch_mean": aggregate["best_epoch_mean"],
        }
        for metric in ("valid_auc", "test_auc", "test_acc", "test_nll", "test_brier"):
            deltas = [seeds[s][metric] - baseline[s][metric] for s in paired]
            row[f"delta_{metric}_vs_matched_baseline"] = float(np.mean(deltas)) if deltas else None
            row[f"sd_delta_{metric}_vs_matched_baseline"] = (
                float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None)
        row["positive_auc_seeds"] = sum(seeds[s]["test_auc"] > baseline[s]["test_auc"]
                                        for s in paired)
        output.append(row)
    if output:
        columns = list(output[0])
        with (ROOT / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(output)
    # Select a coefficient within each dataset/method by validation AUC only.
    best_in_family = {}
    for row in output:
        if row["n_seeds"] < 3 or row["n_paired"] < 3:
            continue
        key = (row["dataset"], row["split_id"], row["q_mode"], row["epochs"],
               row["alpha"], row["method"])
        if key not in best_in_family or row["valid_auc_mean"] > best_in_family[key]["valid_auc_mean"]:
            best_in_family[key] = row
    overview = []
    families = sorted({key[1:] for key in best_in_family})
    for split_id, q_mode, epochs, alpha, method in families:
        group = [row for key, row in best_in_family.items()
                 if key[1:] == (split_id, q_mode, epochs, alpha, method)]
        overview.append({
            "split_id": split_id, "q_mode": q_mode, "epochs": epochs,
            "alpha": alpha, "method": method,
            "n_datasets": len(group),
            "dataset_ids_selected_by_validation": ";".join(
                f"{r['dataset']}:{r['experiment_id']}" for r in sorted(group, key=lambda r: r["dataset"])),
            "n_validation_auc_positive": sum(
                r["delta_valid_auc_vs_matched_baseline"] > 0 for r in group),
            "n_test_auc_positive_report_only": sum(
                r["delta_test_auc_vs_matched_baseline"] > 0 for r in group),
            "n_test_acc_nonnegative_report_only": sum(
                r["delta_test_acc_vs_matched_baseline"] >= 0 for r in group),
            "mean_paired_delta_test_auc_report_only": float(np.mean([
                r["delta_test_auc_vs_matched_baseline"] for r in group])),
        })
    if overview:
        with (ROOT / "method_overview.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(overview[0]))
            writer.writeheader()
            writer.writerows(overview)
    selections = []
    for dataset, split_id, q_mode, epochs, alpha in sorted(baselines):
        candidates = [r for r in output if r["dataset"] == dataset and
                      r["split_id"] == split_id and
                      r["q_mode"] == q_mode and r["n_seeds"] >= 3 and
                      r["epochs"] == epochs and r["alpha"] == alpha]
        if not candidates:
            continue
        selected = max(candidates, key=lambda r: r["valid_auc_mean"])
        baseline_row = next(r for r in candidates if r["method"] == "baseline")
        selections.append({
            "dataset": dataset, "split_id": split_id,
            "q_mode": q_mode, "epochs": epochs,
            "alpha": alpha,
            "selection_metric": "mean validation AUC, 3 seeds, identical budget and alpha",
            "baseline_id": baseline_row["experiment_id"],
            "selected_id": selected["experiment_id"],
            "selected_method": selected["method"],
            "valid_auc_gain": selected["valid_auc_mean"] - baseline_row["valid_auc_mean"],
            "test_auc_gain_report_only": selected["test_auc_mean"] - baseline_row["test_auc_mean"],
            "test_acc_gain_report_only": selected["test_acc_mean"] - baseline_row["test_acc_mean"],
        })
    prepared_datasets = {row["dataset"] for row in output
                         if row["split_id"] == "prepared" and row["q_mode"] == "fixed"
                         and row["epochs"] == 60 and row["alpha"] == 0.8
                         and row["method"] == "baseline" and row["n_seeds"] >= 3}
    soft_candidates = {}
    for row in output:
        if (row["split_id"] == "prepared" and row["q_mode"] == "fixed"
                and row["epochs"] == 60 and row["alpha"] == 0.8
                and row["method"] == "fixed_soft_q" and row["n_seeds"] >= 3
                and row["n_paired"] >= 3):
            key = (row["q_init_high"], row["q_init_low"])
            soft_candidates.setdefault(key, {})[row["dataset"]] = row
    global_options = []
    for (high, low), by_dataset in sorted(soft_candidates.items()):
        if set(by_dataset) != prepared_datasets:
            continue
        group = [by_dataset[d] for d in sorted(prepared_datasets)]
        global_options.append({
            "q_init_high": high, "q_init_low": low,
            "mean_delta_valid_auc": float(np.mean([
                row["delta_valid_auc_vs_matched_baseline"] for row in group])),
            "mean_delta_test_auc_report_only": float(np.mean([
                row["delta_test_auc_vs_matched_baseline"] for row in group])),
            "dataset_ids": {row["dataset"]: row["experiment_id"] for row in group},
        })
    global_soft_q = {
        "selection_rule": "Among soft-Q settings completed on all five prepared datasets, maximize the equal-dataset mean paired validation AUC gain; test reports only.",
        "eligible_candidates": global_options,
        "selected": max(global_options, key=lambda row: row["mean_delta_valid_auc"])
        if global_options else None,
        "caveat": "Campaign is exploratory: later candidates were proposed after inspecting some earlier test results.",
    }
    (ROOT / "selection.json").write_text(
        json.dumps({"completed_conditions": len(output),
                    "rule": "Within each listed candidate set, validation only; test columns are reported after selection.",
                    "selections": selections, "global_soft_q": global_soft_q},
                   indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"Summarized {len(output)} conditions, {len(selections)} selections")


if __name__ == "__main__":
    main()
