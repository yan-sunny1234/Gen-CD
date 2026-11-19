"""Math1 alpha ablation. Selected model states stay in memory; no epoch artifacts."""
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import numpy as np
import torch
from Q_GNCDM.experiments import run_test_sim_con as runner

ROOT = runner.ROOT
ID = "alpha_endpoints_math1_20261110"
OUT = ROOT / "Q_GNCDM/results" / ID
SEEDS = [2026, 2027, 2028]
ALPHAS = [0.0, 0.8, 1.0]
SOURCE = ROOT / "Q_GNCDM/results/test_sim_con/math1__weak_q__baseline/manifest.json"

def run_job(alpha, seed, config):
    torch.set_num_threads(1)
    args = Namespace(**config)
    args.alpha = alpha
    args.experiment_id = f"{ID}_alpha_{alpha}"
    data = runner.prepared.load_data(ROOT / "Q_GNCDM/data/math1")
    result = runner.train_one(data, args, seed, None)
    return {"alpha": alpha, **result}

def summarize(rows):
    aggregate = []
    for alpha in ALPHAS:
        selected = sorted([r for r in rows if r["alpha"] == alpha], key=lambda r: r["seed"])
        if len(selected) != len(SEEDS):
            continue
        entry = {"dataset": "math1", "alpha": alpha, "n_seeds": len(selected)}
        for metric in ["valid_auc", "valid_acc", "test_auc", "test_acc", "test_nll",
                       "test_brier", "theta_effective_rank", "q_hard_density", "best_epoch"]:
            values = [r[metric] for r in selected]
            entry[metric + "_mean"] = float(np.mean(values))
            entry[metric + "_std"] = float(np.std(values, ddof=1))
        baseline = {r["seed"]: r for r in rows if r["alpha"] == 0.8}
        if len(baseline) == len(SEEDS):
            for metric in ["test_auc", "test_acc", "test_nll", "test_brier"]:
                delta = [r[metric] - baseline[r["seed"]][metric] for r in selected]
                entry["delta_vs_0p8_" + metric] = float(np.mean(delta))
                entry["delta_vs_0p8_" + metric + "_std"] = float(np.std(delta, ddof=1))
        aggregate.append(entry)
    return aggregate

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    config = source["args"]
    data = runner.prepared.load_data(ROOT / "Q_GNCDM/data/math1")
    files = [Path(__file__), Path(runner.__file__), Path(runner.core.__file__),
             Path(runner.prepared.__file__), ROOT / "Q_GNCDM/simple_contrastive.py",
             ROOT / "Q_GNCDM/models/weak_q_gncdm.py"]
    hashes = {str(p.relative_to(ROOT)): runner.sha256(p) for p in files}
    runner.write_json(OUT / "manifest.json", {
        "experiment_id": ID, "single_change": "alpha only; retrain each condition",
        "baseline": "Weak-Q alpha=0.8, no contrastive loss; retrained same protocol",
        "dataset": "math1", "alphas": ALPHAS, "seeds": SEEDS,
        "selection": "per-seed best validation AUC; test never selects",
        "config": config, "data_sha256": data["source_sha256"], "code_sha256": hashes,
        "retention": "final per-seed metrics, comparison, manifest, verification only; no checkpoints or epoch logs",
        "date": "2026-11-10"
    })
    rows = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_job, a, s, config) for a in ALPHAS for s in SEEDS]
        for future in as_completed(futures):
            rows.append(future.result())
            rows.sort(key=lambda r: (r["alpha"], r["seed"]))
            runner.write_csv(OUT / "per_seed.csv", rows)
            runner.write_csv(OUT / "comparison.csv", summarize(rows))
    for name, digest in hashes.items():
        if runner.sha256(ROOT / name) != digest:
            raise ValueError(f"Code changed: {name}")
    for name, digest in data["source_sha256"].items():
        if runner.prepared.sha256(data["folder"] / name) != digest:
            raise ValueError(f"Data changed: {name}")
    runner.write_json(OUT / "verification.json", {
        "status": "passed", "in_memory_replays": len(rows),
        "data_hashes": "passed", "code_hashes": "passed",
        "checkpoint_saved": False, "test_used_for_selection": False
    })
    print(json.dumps(summarize(rows), indent=2), flush=True)

if __name__ == "__main__":
    main()

