"""Post-selection response-evidence masking evaluation for test_sim_con.

This script does not train or select models. It masks only the original training
evidence; validation and test labels remain absent from encoder inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Q_GNCDM.experiments import run_prepared_dataset as prepared
from Q_GNCDM.experiments import run_simple_contrastive as core
from Q_GNCDM.experiments import run_test_sim_con as experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mask-seed", type=int, default=1801)
    args = parser.parse_args()
    if (any(rate < 0 or rate > 1 for rate in args.ratios) or args.repeats < 1
            or 0.0 not in args.ratios):
        parser.error("invalid ratios or repeats")
    args.ratios = sorted(set(args.ratios))
    folder = ROOT / "Q_GNCDM/results/test_sim_con" / args.experiment_id
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if json.loads((folder / "verification.json").read_text(encoding="utf-8"))["status"] != "passed":
        raise ValueError("Training checkpoint verification has not passed")
    settings = argparse.Namespace(**manifest["args"])
    torch.set_num_threads(settings.threads)
    data = prepared.load_data(ROOT / "Q_GNCDM/data" / settings.dataset)
    if data["source_sha256"] != manifest["data_source_sha256"]:
        raise ValueError("Prepared data changed")
    records = []
    train = data["train"]
    for seed in settings.seeds:
        checkpoint = torch.load(folder / f"seed_{seed}/best_model.pt",
                                map_location=settings.device, weights_only=True)
        model = experiment.get_model(data, settings, seed)
        model.load_state_dict(checkpoint["model_state_dict"])
        full_theta = None
        for rate in args.ratios:
            for repeat in range(1 if rate == 0 else args.repeats):
                rng = np.random.default_rng([args.mask_seed, repeat, int(round(rate * 10000))])
                evidence = experiment.mask_observed(train, rate, rng) if rate else train
                metrics, theta, _ = core.evaluate(model, evidence, data["rows"][2])
                if rate == 0:
                    full_theta = theta
                records.append({
                    "experiment_id": args.experiment_id,
                    "dataset": settings.dataset,
                    "seed": seed,
                    "mask_rate_nominal": rate,
                    "repeat": repeat,
                    "actual_mask_rate": float(np.mean(evidence[train != 0] == 0)),
                    "test_auc": metrics["auc"],
                    "test_acc": metrics["acc"],
                    "test_nll": metrics["nll"],
                    "test_brier": metrics["brier"],
                    "theta_mae_vs_full": float(np.mean(np.abs(theta - full_theta))),
                    "theta_effective_rank": experiment.effective_rank(theta),
                })
    with (folder / "robustness.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (folder / "robustness_manifest.json").write_text(
        json.dumps({"mask_seed": args.mask_seed, "nominal_rates": args.ratios,
                    "repeats_at_positive_rate": args.repeats,
                    "selection_use": "none; post-selection test diagnostics only",
                    "student_and_item_evidence": "same masked training matrix",
                    "test_labels_in_evidence": False}, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} robustness rows: {folder}")


if __name__ == "__main__":
    main()
