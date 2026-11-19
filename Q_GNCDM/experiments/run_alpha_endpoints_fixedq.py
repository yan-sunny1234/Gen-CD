"""Fixed binary Q alpha endpoints; no Weak-Q or contrastive loss."""
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import numpy as np
import torch
from Q_GNCDM.experiments import run_alpha_endpoints as shared

runner = shared.runner
ROOT = runner.ROOT
ID = "alpha_endpoints_fixedq_math1_20261110"
OUT = ROOT / "Q_GNCDM/results" / ID
SOURCE = ROOT / "Q_GNCDM/results/test_sim_con/math1__fixed_q__baseline_60e/manifest.json"

def run_job(alpha, seed, config):
    torch.set_num_threads(1)
    args = Namespace(**config)
    args.alpha = alpha
    args.experiment_id = f"{ID}_alpha_{alpha}"
    data = runner.prepared.load_data(ROOT / "Q_GNCDM/data/math1")
    model = runner.get_model(data, args, seed)
    assert not model.q_logits.requires_grad
    assert np.array_equal(model.get_q_soft().detach().cpu().numpy(), data["q"])
    assert model.lambda_q == model.lambda_coverage == 0
    del model
    return {"alpha": alpha, **runner.train_one(data, args, seed, None)}

def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    config = dict(source["args"])
    config.update(q_mode="fixed", method="baseline", lambda_aux=0.0,
                  lambda_q=0.0, lambda_coverage=0.0)
    data = runner.prepared.load_data(ROOT / "Q_GNCDM/data/math1")
    files = [Path(__file__), Path(shared.__file__), Path(runner.__file__),
             Path(runner.core.__file__), Path(runner.prepared.__file__),
             ROOT / "Q_GNCDM/simple_contrastive.py",
             ROOT / "Q_GNCDM/models/weak_q_gncdm.py"]
    hashes = {str(p.relative_to(ROOT)): runner.sha256(p) for p in files}
    OUT.mkdir(parents=True, exist_ok=False)
    runner.write_json(OUT / "manifest.json", {
        "experiment_id": ID, "single_change": "alpha only; independently retrained",
        "baseline": "fixed binary reference Q, no Weak-Q, no CL, alpha=0.8",
        "dataset": "math1", "alphas": shared.ALPHAS, "seeds": shared.SEEDS,
        "selection": "per-seed best validation AUC; test never selects",
        "config": config, "data_sha256": data["source_sha256"], "code_sha256": hashes,
        "retention": "final metrics and verification only; in-memory selected states",
        "date": "2026-11-10"
    })
    rows = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_job, a, s, config) for a in shared.ALPHAS for s in shared.SEEDS]
        for f in as_completed(futures):
            rows.append(f.result())
            rows.sort(key=lambda r: (r["alpha"], r["seed"]))
            runner.write_csv(OUT / "per_seed.csv", rows)
            runner.write_csv(OUT / "comparison.csv", shared.summarize(rows))
    reproduced = []
    for row in rows:
        if row["alpha"] == 0.8:
            old = json.loads((SOURCE.parent / f"seed_{row['seed']}/metrics.json").read_text(encoding="utf-8"))
            for metric in ["valid_auc", "valid_acc", "test_auc", "test_acc", "test_nll", "test_brier", "best_epoch"]:
                if not np.isclose(row[metric], old[metric], atol=1e-10, rtol=0):
                    raise ValueError(f"Historical baseline mismatch: {row['seed']} {metric}")
            reproduced.append(row["seed"])
    for name, digest in hashes.items():
        assert runner.sha256(ROOT / name) == digest, name
    for name, digest in data["source_sha256"].items():
        assert runner.prepared.sha256(data["folder"] / name) == digest, name
    runner.write_json(OUT / "verification.json", {
        "status": "passed", "in_memory_replays": len(rows), "fixed_binary_q": "passed",
        "data_hashes": "passed", "code_hashes": "passed",
        "historical_alpha_0p8_reproduced_seeds": reproduced,
        "checkpoint_saved": False, "test_used_for_selection": False
    })
    print(json.dumps(shared.summarize(rows), indent=2), flush=True)

if __name__ == "__main__":
    main()

