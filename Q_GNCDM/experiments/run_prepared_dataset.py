"""Run A0/A1/S1 on a prepared real dataset without changing historic runners."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Q_GNCDM.experiments import run_simple_contrastive as core
from Q_GNCDM.experiments import run_weak_q_synthetic as legacy


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), None)
        if header != ["user_id", "item_id", "score"]:
            raise ValueError(f"Expected user_id,item_id,score header: {path}")
        rows = np.loadtxt(handle, delimiter=",", dtype=np.int64, ndmin=2)
    if rows.shape[1] != 3 or len(rows) == 0:
        raise ValueError(f"Expected nonempty three-column interaction data: {path}")
    return rows


def load_data(folder: Path) -> dict:
    folder = Path(folder).resolve()
    paths = [folder / f"{part}.csv" for part in ("train", "valid", "test")]
    q_path = folder / "Q_matrix.npy"
    manifest_path = folder / "manifest.json"
    q_true_path = folder / "Q_true.npy"
    for path in [*paths, q_path, manifest_path, q_true_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    arrays = [load_rows(path) for path in paths]
    q = np.load(q_path).astype(np.float32)
    q_true = np.load(q_true_path)
    if q.ndim != 2 or not np.isin(q, [0, 1]).all():
        raise ValueError("Q must be a binary item-by-concept matrix")
    if not np.array_equal(q, q_true) or not q.any(axis=1).all() or not q.any(axis=0).all():
        raise ValueError("Q_true must match Q_matrix; every item and concept must be covered")

    combined = np.concatenate(arrays)
    n_students, n_items = int(combined[:, 0].max()) + 1, len(q)
    if (combined < 0).any() or (combined[:, 1] >= n_items).any():
        raise ValueError("Negative ID/score or item ID outside Q")
    if not np.isin(combined[:, 2], [0, 1]).all():
        raise ValueError("Scores must be binary")
    if not np.array_equal(np.unique(combined[:, 0]), np.arange(n_students)):
        raise ValueError("Student IDs must be contiguous and zero-based")
    if not np.array_equal(np.unique(combined[:, 1]), np.arange(n_items)):
        raise ValueError("Item IDs must be contiguous and zero-based")
    if not np.array_equal(np.unique(arrays[0][:, 1]), np.arange(n_items)):
        raise ValueError("Every item needs at least one training response")
    pair_codes = combined[:, 0] * n_items + combined[:, 1]
    if np.unique(pair_codes).size != len(combined):
        raise ValueError("Repeated student-item pair across or within splits")

    train = np.zeros((n_students, n_items), dtype=np.float32)
    train[arrays[0][:, 0], arrays[0][:, 1]] = 2 * arrays[0][:, 2] - 1
    if np.any(train[arrays[1][:, 0], arrays[1][:, 1]] != 0) or np.any(
        train[arrays[2][:, 0], arrays[2][:, 1]] != 0
    ):
        raise ValueError("Validation or test labels visible in training evidence")
    if ((train != 0).sum(axis=1) < 2).any():
        raise ValueError("Every student needs at least two training responses")

    with manifest_path.open("r", encoding="utf-8") as handle:
        preparation = json.load(handle)
    actual_hashes = {
        path.name: sha256(path) for path in [*paths, q_path, q_true_path]
    }
    if preparation.get("files_sha256") != actual_hashes:
        raise ValueError("Prepared files do not match their manifest hashes")
    return {
        "folder": folder,
        "rows": arrays,
        "train": train,
        "q": q,
        "preparation": preparation,
        "source_sha256": {**actual_hashes, manifest_path.name: sha256(manifest_path)},
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Folder name under Q_GNCDM/data")
    parser.add_argument("--data-dir", type=Path, help="Override the prepared dataset folder")
    parser.add_argument("--experiment-id", help="Unique ID under Q_GNCDM/results")
    parser.add_argument("--check-only", action="store_true", help="Validate data and one forward pass; write no files")
    parser.add_argument("--scratch-run", action="store_true", help="Write a trial run under ignored Q_GNCDM/scratch")
    parser.add_argument("--modes", nargs="+", choices=list(core.MODES), default=["A1", "S1"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-students", type=int, default=128)
    parser.add_argument("--query-fraction", type=float, default=0.25)
    parser.add_argument("--positive-noise-rate", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--lambda-cl", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--q-lr", type=float, default=0.005)
    parser.add_argument("--lambda-q", type=float, default=0.01)
    parser.add_argument("--lambda-coverage", type=float, default=0.01)
    parser.add_argument("--q-init-high", type=float, default=0.55)
    parser.add_argument("--q-init-low", type=float, default=0.45)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.dataset):
        parser.error("dataset must be a folder name without path separators")
    if not args.check_only and not args.experiment_id:
        parser.error("--experiment-id is required for a formal run")
    if args.check_only and args.scratch_run:
        parser.error("--check-only and --scratch-run cannot be combined")
    if args.experiment_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.experiment_id):
        parser.error("experiment-id must be a simple unique folder name")
    if len(args.modes) != len(set(args.modes)) or len(args.modes) > 2:
        parser.error("Use one mode or one baseline-treatment pair per experiment ID")
    if len(args.modes) == 2 and set(args.modes) not in ({"A0", "A1"}, {"A1", "S1"}):
        parser.error("The supported pairs are A0/A1 and A1/S1")
    if args.epochs <= 0 or min(args.batch_students, args.embedding_dim, args.threads) <= 0:
        parser.error("epochs, batch-students, embedding-dim, and threads must be positive")
    if not 0 < args.query_fraction < 1 or not 0 < args.positive_noise_rate < 1:
        parser.error("query-fraction and positive-noise-rate must be in (0,1)")
    if not 0 <= args.alpha <= 1 or args.temperature <= 0:
        parser.error("alpha or temperature is invalid")
    if min(args.lambda_cl, args.lambda_q, args.lambda_coverage, args.lr, args.q_lr) < 0:
        parser.error("loss weights and learning rates must be nonnegative")
    args.data_dir = str(args.data_dir or ROOT / "Q_GNCDM" / "data" / args.dataset)
    return args


def verify_results(data: dict, args, results: list[dict], destination: Path,
                   code_hashes: dict[str, str]) -> dict:
    for filename, digest in data["source_sha256"].items():
        if sha256(data["folder"] / filename) != digest:
            raise ValueError(f"Dataset changed during training: {filename}")
    for filename, digest in code_hashes.items():
        if sha256(ROOT / filename) != digest:
            raise ValueError(f"Code changed during training: {filename}")

    for result in results:
        mode, seed = result["mode"], result["seed"]
        run = destination / f"seed_{seed}" / mode
        checkpoint = torch.load(
            run / "best_model.pt", map_location=args.device, weights_only=True
        )
        model = core.build_model(data, args, mode, seed)
        model.load_state_dict(checkpoint["model_state_dict"])
        scores, theta, probabilities = core.evaluate(
            model, data["train"], data["rows"][2]
        )
        with np.load(run / "test_predictions.npz") as saved:
            np.testing.assert_array_equal(saved["rows"], data["rows"][2])
            np.testing.assert_allclose(
                saved["probabilities"], probabilities, atol=1e-7, rtol=0
            )
        np.testing.assert_allclose(theta, np.load(run / "theta.npy"), atol=1e-7, rtol=0)
        for metric, value in scores.items():
            if value is not None and not np.isclose(
                value, result[f"test_{metric}"], atol=1e-10, rtol=0
            ):
                raise ValueError(f"Replayed {metric} differs for {mode}, seed {seed}")
        if mode == "A0":
            np.testing.assert_array_equal(np.load(run / "Q_soft.npy"), data["q"])
    return {
        "status": "passed",
        "checkpoints_replayed": len(results),
        "source_and_code_hashes": "passed",
        "test_predictions_and_metrics": "passed",
        "split_disjointness": "passed at load time",
    }


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    data = load_data(args.data_dir)
    if data["preparation"].get("dataset") != args.dataset:
        raise ValueError("--dataset does not match the preparation manifest")
    if args.check_only:
        model = core.build_model(data, args, args.modes[0], args.seeds[0])
        core.evaluate(model, data["train"], data["rows"][1][:128])
        print(
            f"OK {args.dataset}: students={len(data['train'])}, "
            f"items={len(data['q'])}, concepts={data['q'].shape[1]}, "
            f"split_lengths={[len(part) for part in data['rows']]}"
        )
        return

    destination = ROOT / "Q_GNCDM" / (
        "scratch" if args.scratch_run else "results"
    ) / args.experiment_id
    destination.mkdir(parents=True, exist_ok=False)
    import scipy
    import sklearn

    source_files = [
        Path(__file__),
        Path(core.__file__),
        ROOT / "Q_GNCDM/simple_contrastive.py",
        ROOT / "Q_GNCDM/models/weak_q_gncdm.py",
        Path(legacy.__file__),
    ]
    if set(args.modes) == {"A0", "A1"}:
        baseline, single_change = "A0 fixed expert Q", "A1 learns weak Q"
    elif set(args.modes) == {"A1", "S1"}:
        baseline, single_change = "A1 weak Q", "S1 adds response-flip InfoNCE"
    else:
        baseline, single_change = args.modes[0], "single-arm run; no paired estimate"
    code_hashes = {
        str(path.relative_to(ROOT)): sha256(path) for path in source_files
    }
    legacy.write_json(destination / "manifest.json", {
        "experiment_id": args.experiment_id,
        "dataset": args.dataset,
        "baseline": baseline,
        "single_change": single_change,
        "selection_metric": "validation AUC (validation NLL fallback)",
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "command": subprocess.list2cmdline(sys.argv),
        "train_shape": data["train"].shape,
        "split_lengths": [len(part) for part in data["rows"]],
        "preparation": data["preparation"],
        "source_sha256": data["source_sha256"],
        "code_sha256": code_hashes,
        "args": vars(args),
        "modes": {mode: core.MODES[mode] for mode in args.modes},
    })
    results = []
    for seed in args.seeds:
        for mode in args.modes:
            result = core.train_one(data, args, mode, seed, destination / f"seed_{seed}" / mode)
            results.append(result)
            core.summarize(results, destination)
    legacy.write_json(destination / "verification.json", verify_results(
        data, args, results, destination, code_hashes
    ))
    legacy.write_json(destination / "completed.json", {"runs_completed": len(results)})
    print(f"Completed {len(results)} runs: {destination}", flush=True)


if __name__ == "__main__":
    main()
