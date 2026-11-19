# -*- coding: utf-8 -*-
"""Point 10 (E3A) phase-1 evaluation: evidence-stratified prediction and
k-shot instant-diagnosis curves on trained checkpoints.

Reads best_model.pt files from a screening directory (p6_train or p10_e3a),
buckets test responses by each student's training support size, and reports
per-bucket AUC/ACC. The k-shot protocol subsamples k evidence entries per
student (hidden from BOTH student rows and item columns, matching the project
protocol) and evaluates only on test items outside the selected set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Q_GNCDM.experiments import build_wqcl_evidence as evi
from Q_GNCDM.experiments import run_weak_q_synthetic as legacy
from Q_GNCDM.experiments import run_simple_contrastive as core
from Q_GNCDM.experiments.run_wqcl_screen import build_model, load_data


def load_checkpoint_model(condition, data, ckpt_path, args):
    teacher = None
    if args.point == "p10_e3a" and condition == "e3a_shrink":
        teacher = evi.fit_rasch_teacher(data["rows"][0], len(data["train"]),
                                        len(data["q"]), seed=2026)
    model = build_model(args.point, {"name": condition}, data, teacher, args)
    stored = torch.load(ckpt_path, map_location=args.device, weights_only=True)
    model.load_state_dict(stored["model_state_dict"])
    model.eval()
    return model


def strata_metrics(model, evidence, test_rows, support_sizes):
    rows = test_rows
    sizes = support_sizes[rows[:, 0]]
    out_rows = []
    edges = [0, 5, 10, 15, 10 ** 9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (sizes >= lo) & (sizes < hi)
        if mask.sum() < 30:
            continue
        metrics, _, _ = core.evaluate(model, evidence, rows[mask])
        out_rows.append({"bucket": f"[{lo},{'inf' if hi == 10**9 else hi})",
                         "n": int(mask.sum()), **metrics})
    return out_rows


def kshot_curve(model, evidence, test_rows, ks, seed=2026):
    rng = np.random.default_rng(seed)
    n_students = evidence.shape[0]
    rows_out = []
    for k in ks:
        selected = np.zeros_like(evidence, dtype=bool)
        for sid in range(n_students):
            observed = np.flatnonzero(evidence[sid] != 0)
            if not len(observed):
                continue
            take = observed[rng.choice(len(observed), size=min(k, len(observed)),
                                       replace=False)]
            selected[sid, take] = True
        masked = np.where(selected, evidence, 0.0).astype(np.float32)
        keep = np.array([selected[r[0], r[1]] == 0 for r in test_rows])
        metrics, _, _ = core.evaluate(model, masked, test_rows[keep])
        rows_out.append({"k": k, "n_eval": int(keep.sum()), **metrics})
    return rows_out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-dir", required=True)
    parser.add_argument("--point", default="p6_train")
    parser.add_argument("--dataset", default="math1", choices=["math1", "synthetic"])
    parser.add_argument("--noise-rate", type=float, default=0.2)
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 5, 10])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    screen_dir = Path(args.screen_dir).resolve()
    inner = argparse.Namespace(point=args.point, dataset=args.dataset, data_dir=str(ROOT / "Q_GNCDM" / "data" / args.dataset), split_seed=2026,
                               noise_rate=args.noise_rate, alpha=0.8,
                               embedding_dim=32, batch_students=128,
                               q_init_high=0.55, q_init_low=0.45,
                               lambda_q=0.01, lambda_coverage=0.01, device=args.device,
                               e3a_n0=5.0, d5_f=0.1, d5_n=0.1)
    data = load_data(inner)
    evidence = data["train"]
    support_sizes = (evidence != 0).sum(axis=1)
    test_rows = data["rows"][2]

    output = {"strata": {}, "kshot": {}}
    for cond_dir in sorted(p for p in screen_dir.iterdir() if p.is_dir()):
        ckpt = cond_dir / "best_model.pt"
        if not ckpt.is_file():
            continue
        condition = cond_dir.name
        model = load_checkpoint_model(condition, data, ckpt, inner)
        output["strata"][condition] = strata_metrics(model, evidence, test_rows,
                                                     support_sizes)
        output["kshot"][condition] = kshot_curve(model, evidence, test_rows,
                                                 args.ks)
        print(f"{condition}: strata+{len(args.ks)} k-shot points done", flush=True)
    legacy.write_json(screen_dir / "evidence_strata.json", output)
    print(f"saved -> {screen_dir / 'evidence_strata.json'}", flush=True)


if __name__ == "__main__":
    main()
