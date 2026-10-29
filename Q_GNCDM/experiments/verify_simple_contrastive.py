"""Replay simple-contrastive checkpoints and validate the experiment artifacts."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Q_GNCDM.experiments.run_simple_contrastive import build_model, evaluate, load_data
from Q_GNCDM.experiments.math1_data_paths import data_dir_for_replay, source_path
from Q_GNCDM.experiments.run_weak_q_synthetic import write_json


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_dir.resolve()
    manifest = read_json(root / "manifest.json")
    completed = read_json(root / "completed.json")
    for filename, digest in manifest["source_sha256"].items():
        assert hashlib.sha256(source_path(filename).read_bytes()).hexdigest() == digest, filename
    for filename, digest in manifest["code_sha256"].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest, filename
    splits = np.load(root / "splits.npz")
    pairs = [{tuple(row[:2]) for row in splits[name]}
             for name in ("train", "valid", "test")]
    assert all(not pairs[a].intersection(pairs[b])
               for a, b in ((0, 1), (0, 2), (1, 2)))

    records = read_csv(root / "summary.csv")
    assert len(records) == completed["runs_completed"]
    run_args = argparse.Namespace(**manifest["args"])
    run_args.data_dir = data_dir_for_replay(run_args.dataset, run_args.data_dir)
    torch.set_num_threads(run_args.threads)
    data = load_data(run_args)
    replayed = 0
    for row in records:
        mode, seed = row["mode"], int(row["seed"])
        run = root / f"seed_{seed}" / mode
        checkpoint = torch.load(
            run / "best_model.pt", map_location=run_args.device, weights_only=True
        )
        model = build_model(data, run_args, mode, seed)
        model.load_state_dict(checkpoint["model_state_dict"])
        scores, theta, probabilities = evaluate(model, data["train"], data["rows"][2])
        saved = np.load(run / "test_predictions.npz")
        np.testing.assert_array_equal(saved["rows"], data["rows"][2])
        np.testing.assert_allclose(probabilities, saved["probabilities"], atol=1e-7, rtol=0)
        np.testing.assert_allclose(theta, np.load(run / "theta.npy"), atol=1e-7, rtol=0)
        assert abs(scores["auc"] - float(row["test_auc"])) < 1e-10
        history = read_csv(run / "train_log.csv")
        expected_coefficient = run_args.lambda_cl if mode == "S1" else 0.0
        assert all(float(epoch["lambda_cl_effective"]) == expected_coefficient
                   for epoch in history)
        if mode == "A0":
            np.testing.assert_array_equal(np.load(run / "Q_soft.npy"), data["q"])
        if mode == "S1":
            assert all(float(epoch["actual_flip_fraction"]) > 0 for epoch in history)
        replayed += 1
    result = {
        "status": "passed",
        "checkpoints_replayed": replayed,
        "source_and_code_hashes": "passed",
        "split_disjointness": "passed",
        "mode_coefficients": "passed",
        "fixed_q_unchanged": "passed",
    }
    write_json(root / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
