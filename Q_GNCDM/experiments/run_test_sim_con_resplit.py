"""Confirm a prespecified Q-softening contrast on a new within-student split.

The split is independent of the prepared split. It is never used to choose the
0.95/0.05 soft-Q values, which were chosen on the prepared Eedi validation set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Q_GNCDM.experiments import run_prepared_dataset as prepared
from Q_GNCDM.experiments import run_test_sim_con as trial
from Q_GNCDM.experiments import run_weak_q_synthetic as legacy


def make_resplit(data: dict, split_seed: int) -> tuple[dict, dict]:
    all_rows = np.concatenate(data["rows"], axis=0)
    students = all_rows[:, 0].astype(np.int64)
    rng = np.random.default_rng(split_seed)
    sorted_positions = np.argsort(students, kind="stable")
    counts = np.bincount(students, minlength=len(data["train"]))
    offsets = np.r_[0, np.cumsum(counts)]
    groups = [[], [], []]
    for student in range(len(data["train"])):
        positions = sorted_positions[offsets[student]:offsets[student + 1]]
        if len(positions) < 10:
            raise ValueError(f"Student {student} has fewer than ten responses")
        order = rng.permutation(positions)
        train_end = int(0.8 * len(order))
        valid_end = train_end + int(0.1 * len(order))
        for part, selected in enumerate((order[:train_end], order[train_end:valid_end],
                                          order[valid_end:])):
            groups[part].append(selected)
    index_groups = [np.sort(np.concatenate(group)) for group in groups]
    rows = [all_rows[index] for index in index_groups]
    pairs = [{(int(row[0]), int(row[1])) for row in part} for part in rows]
    if any(pairs[a] & pairs[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Resplit leaked a student-item pair")
    resplit = dict(data)
    resplit["rows"] = rows
    resplit["train"] = legacy.build_log_mat(rows[0], len(data["train"]), len(data["q"]))
    details = {
        "split_id": f"resplit{split_seed}",
        "algorithm": "For each student, seeded random permutation of all prepared rows; floor(80%) train, floor(10%) valid, remainder test; order restored within each part.",
        "split_seed": split_seed,
        "split_lengths": [len(part) for part in rows],
        "row_sha256": [hashlib.sha256(np.ascontiguousarray(part).tobytes()).hexdigest()
                       for part in rows],
        "leakage_check": "passed",
    }
    return resplit, details


def run_condition(data: dict, split: dict, template: dict, experiment_id: str) -> None:
    args = argparse.Namespace(**template["args"])
    args.experiment_id = experiment_id
    args.split_id = split["split_id"]
    folder = trial.ROOT / "Q_GNCDM/results/test_sim_con" / experiment_id
    folder.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(trial.__file__), Path(__file__), Path(prepared.__file__),
                    trial.ROOT / "Q_GNCDM/experiments/run_simple_contrastive.py",
                    trial.ROOT / "Q_GNCDM/experiments/run_weak_q_synthetic.py",
                    trial.ROOT / "Q_GNCDM/simple_contrastive.py",
                    trial.ROOT / "Q_GNCDM/models/weak_q_gncdm.py"]
    hashes = {str(path.relative_to(trial.ROOT)): trial.sha256(path) for path in source_paths}
    trial.write_json(folder / "manifest.json", {
        "experiment_id": experiment_id,
        "baseline": f"{split['split_id']} fixed binary Q, no auxiliary loss",
        "single_change": trial.METHODS[args.method],
        "selection_metric": "validation AUC within each training seed; 0.95/0.05 fixed before this split; test only reports",
        "data_source_sha256": data["source_sha256"],
        "code_sha256": hashes,
        "split": split,
        "args": vars(args),
    })
    results = []
    for seed in args.seeds:
        results.append(trial.train_one(data, args, seed, folder / f"seed_{seed}"))
        trial.write_csv(folder / "summary.csv", results)
    aggregate = {"experiment_id": experiment_id, "dataset": args.dataset,
                 "q_mode": args.q_mode, "method": args.method, "n_seeds": len(results)}
    for key, value in results[0].items():
        if isinstance(value, (int, float)) and key != "seed":
            values = [result[key] for result in results]
            aggregate[f"{key}_mean"] = float(np.mean(values))
            aggregate[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    trial.write_json(folder / "aggregate.json", aggregate)
    verification = trial.verify(data, args, results, folder, hashes)
    actual_split_hashes = [hashlib.sha256(np.ascontiguousarray(part).tobytes()).hexdigest()
                           for part in data["rows"]]
    if actual_split_hashes != split["row_sha256"]:
        raise ValueError("Resplit content differs from the manifest")
    verification["split_sha256"] = "passed"
    trial.write_json(folder / "verification.json", verification)
    print(f"Completed {folder}", flush=True)


def main() -> None:
    root = trial.ROOT / "Q_GNCDM/results/test_sim_con"
    data, split = make_resplit(prepared.load_data(trial.ROOT / "Q_GNCDM/data/eedi_task34"), 4242)
    for source_id, target_id in (
        ("eedi_task34__fixed_q__baseline", "eedi_task34__resplit4242__fixed_q__baseline"),
        ("eedi_task34__fixed_q__soft_0p95_0p05", "eedi_task34__resplit4242__fixed_q__soft_0p95_0p05"),
    ):
        manifest = json.loads((root / source_id / "manifest.json").read_text(encoding="utf-8"))
        run_condition(data, split, manifest, target_id)


if __name__ == "__main__":
    main()
