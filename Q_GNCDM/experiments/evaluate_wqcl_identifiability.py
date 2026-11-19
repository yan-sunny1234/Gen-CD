# -*- coding: utf-8 -*-
"""Point 6 (IPW-Q) screening evaluation: probe injectivity + near-duplicate
IDS proxy on trained checkpoints.

Reads best_model.pt files produced by run_wqcl_screen --point p6_train (and
optionally p10_e3a), rebuilds each model, and measures whether distinct
response vectors still map to distinct diagnoses:

  Sep(h)      : mean ||theta(R) - theta(R')||_2 / sqrt(h) over h-step response
                perturbations (flip h observed entries). An identifiable
                generator has Sep(h) clearly above 0 and growing with h.
  Collapse(h) : fraction of perturbed pairs whose theta distance < eps_sep.
  IDS-near    : paper Eq.40 style statistic over true near-duplicate student
                pairs (1 <= Hamming <= 2), adapted to real data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Q_GNCDM.experiments import build_wqcl_evidence as evi
from Q_GNCDM.experiments import run_weak_q_synthetic as legacy
from Q_GNCDM.experiments.run_wqcl_screen import build_model, load_data, parse_args


def load_checkpoint_model(condition, data, ckpt_path, args):
    args_copy = argparse.Namespace(**{**vars(args)})
    model = build_model(args.point, {"name": condition}, data, None, args_copy)
    stored = torch.load(ckpt_path, map_location=args.device, weights_only=True)
    model.load_state_dict(stored["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def diagnose_all(model, evidence):
    matrix = torch.as_tensor(evidence, dtype=torch.float32, device=model.device)
    return model.diagnose_theta(matrix).cpu().numpy()


def probe_sep(model, evidence, n_base=300, hs=(1, 2, 3), eps_sep=0.01, seed=2026):
    """h-step perturbation injectivity probes."""
    rng = np.random.default_rng(seed)
    n_students = evidence.shape[0]
    base_ids = rng.choice(n_students, size=min(n_base, n_students), replace=False)
    result = {}
    for h in hs:
        dists, collapses, n_used = [], 0, 0
        for sid in base_ids:
            observed = np.flatnonzero(evidence[sid] != 0)
            if len(observed) < h:
                continue
            flip = rng.choice(observed, size=h, replace=False)
            pert = evidence[sid].copy()
            pert[flip] *= -1.0
            theta_pair = diagnose_all(model, np.stack([evidence[sid], pert]))
            d = float(np.linalg.norm(theta_pair[0] - theta_pair[1]))
            dists.append(d / np.sqrt(h))
            collapses += int(d < eps_sep)
            n_used += 1
        if n_used:
            result[f"Sep_{h}step"] = float(np.mean(dists))
            result[f"Collapse_{h}step"] = collapses / n_used
            result[f"n_pairs_{h}step"] = n_used
    return result


def near_duplicate_ids(theta, evidence, max_hamming=2):
    """Paper Eq.40 flavoured statistic over near-duplicate student pairs."""
    pos = (evidence > 0).astype(np.int32)
    neg = (evidence < 0).astype(np.int32)
    hamming = pos @ neg.T + neg @ pos.T          # co-observed disagreements
    n = theta.shape[0]
    dists, pairs = [], 0
    for i in range(n):
        row = hamming[i, i + 1:]
        idx = np.flatnonzero((row >= 1) & (row <= max_hamming)) + i + 1
        for j in idx:
            d = float(np.abs(theta[i] - theta[j]).sum())
            dists.append(1.0 / (1.0 + d) ** 2)
            pairs += 1
    return {"n_near_duplicate_pairs": pairs,
            "IDS_near": float(np.mean(dists)) if dists else None}


def main(argv=None):
    args = argparse.ArgumentParser()
    args.add_argument("--screen-dir", required=True,
                      help="e.g. Q_GNCDM/results/wqcl_ten_screen_v1/p6_train/math1")
    args.add_argument("--dataset", default="math1", choices=["math1", "synthetic"])
    args.add_argument("--noise-rate", type=float, default=0.2)
    args.add_argument("--n-base", type=int, default=300)
    args.add_argument("--device", default="cpu")
    args = args.parse_args(argv)

    screen_dir = Path(args.screen_dir).resolve()
    inner = argparse.Namespace(point=args.point, dataset=args.dataset, data_dir=str(ROOT / "Q_GNCDM" / "data" / args.dataset), split_seed=2026,
                               noise_rate=args.noise_rate, alpha=0.8,
                               embedding_dim=32, batch_students=128,
                               q_init_high=0.55, q_init_low=0.45,
                               lambda_q=0.01, lambda_coverage=0.01, device=args.device,
                               e3a_n0=5.0, d5_f=0.1, d5_n=0.1)
    data = load_data(inner)
    evidence = data["train"]
    results = {}
    for cond_dir in sorted(p for p in screen_dir.iterdir() if p.is_dir()):
        ckpt = cond_dir / "best_model.pt"
        if not ckpt.is_file():
            continue
        condition = cond_dir.name
        model = load_checkpoint_model(condition, data, ckpt, inner)
        theta = diagnose_all(model, evidence)
        entry = probe_sep(model, evidence, n_base=args.n_base)
        entry.update(near_duplicate_ids(theta, evidence))
        entry["theta_cross_student_std_mean"] = float(theta.std(axis=0).mean())
        results[condition] = entry
        print(f"{condition}: Sep1={entry.get('Sep_1step'):.4f} "
              f"Sep3={entry.get('Sep_3step'):.4f} "
              f"IDS_near={entry.get('IDS_near')}", flush=True)
    legacy.write_json(screen_dir / "identifiability.json", results)
    print(f"saved -> {screen_dir / 'identifiability.json'}", flush=True)


if __name__ == "__main__":
    main()
