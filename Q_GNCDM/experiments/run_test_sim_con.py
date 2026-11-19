"""One-factor checks for student contrastive learning and Weak-Q controls.

Run one named condition per invocation. The orchestrated comparison is built by
``--summarize`` from completed conditions; test metrics never choose a candidate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Q_GNCDM.experiments import run_prepared_dataset as prepared
from Q_GNCDM.experiments import run_simple_contrastive as core
from Q_GNCDM.experiments import run_weak_q_synthetic as legacy
from Q_GNCDM.simple_contrastive import (
    flip_observed_responses,
    make_student_episode,
    simple_infonce,
)


METHODS = {
    "baseline": "No auxiliary loss",
    "flip_infonce": "Original response-flip InfoNCE on final mastery",
    "mask_infonce": "Change only flip to Bernoulli response masking",
    "mask_project_theta": "Mask view, InfoNCE on an independent mastery projector",
    "mask_project_implicit": "Mask view, InfoNCE on an implicit-branch projector",
    "mask_augmented_bce": "Mask view receives supervised query BCE; no contrastive loss",
    "mask_prediction_consistency": "Mask view query prediction matches stop-gradient clean view",
    "mask_concept_consistency": "Mask view mastery matches clean view where both have Q-referenced evidence",
    "fixed_soft_q": "Fixed 0.55/0.45 Q in place of fixed binary Q",
    "weak_q_no_prior": "Learnable Q without the soft-prior MSE",
}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(legacy.to_jsonable(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    names = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(legacy.to_jsonable(rows))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask_observed(rows: np.ndarray, probability: float, rng: np.random.Generator) -> np.ndarray:
    """Independent masking; never change a known correct/incorrect label."""
    view = rows.copy()
    hide = (rows != 0) & (rng.random(rows.shape) < probability)
    view[hide] = 0
    return view


def effective_rank(theta: np.ndarray) -> float:
    centered = theta - theta.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    mass = singular * singular
    if not np.any(mass):
        return 0.0
    p = mass / mass.sum()
    return float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))))


def get_model(data: dict, args: argparse.Namespace, seed: int):
    mode = "A0" if args.q_mode == "fixed" else "A1"
    model = core.build_model(data, args, mode, seed)
    if args.method == "fixed_soft_q":
        # q_init_mode='weak' makes get_q_soft return sigmoid(q_logits).
        with torch.no_grad():
            model.q_logits.copy_(torch.logit(model.Q_prior))
        model.q_init_mode = "weak"
        model.learn_q = False
        model.q_logits.requires_grad_(False)
    if args.method == "weak_q_no_prior":
        model.lambda_q = 0.0
        model.use_weak_q_loss = False
    return model


def get_projector(args: argparse.Namespace, concepts: int, seed: int):
    if args.method not in {"mask_project_theta", "mask_project_implicit"}:
        return None
    # Keep backbone initialization and dropout stream identical to the baseline.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 9173)
        projector = nn.Sequential(nn.Linear(concepts, 64), nn.ReLU(), nn.Linear(64, 32))
    return projector.to(args.device)


def score_model(model, data):
    validation, theta_valid, _ = core.evaluate(model, data["train"], data["rows"][1])
    test, theta_test, _ = core.evaluate(model, data["train"], data["rows"][2])
    q = model.get_q_soft().detach().cpu().numpy()
    return {
        **{f"valid_{k}": v for k, v in validation.items()},
        **{f"test_{k}": v for k, v in test.items()},
        "theta_std_mean": float(theta_test.std(axis=0).mean()),
        "theta_effective_rank": effective_rank(theta_test),
        "q_soft_density": float(q.mean()),
        "q_hard_density": float((q > .5).mean()),
        "q_f1_vs_reference": legacy.q_recovery_metrics(data["q"], data["q"],
            (q > .5).astype(np.int64), q)["q_hat_f1_vs_true"],
        "valid_theta_effective_rank": effective_rank(theta_valid),
    }


def train_one(data: dict, args: argparse.Namespace, seed: int, directory: Path | None) -> dict:
    # None keeps the selected state in memory and writes no intermediate artifacts.
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=False)
    legacy.set_seed(seed)
    model = get_model(data, args, seed)
    projector = get_projector(args, data["q"].shape[1], seed)
    groups = [{"params": [p for n, p in model.named_parameters()
             if n != "q_logits" and p.requires_grad], "lr": args.lr}]
    if model.q_logits.requires_grad:
        groups.append({"params": [model.q_logits], "lr": args.q_lr})
    if projector is not None:
        groups.append({"params": list(projector.parameters()), "lr": args.lr})
    optimizer = torch.optim.Adam(groups)
    eligible = np.flatnonzero((data["train"] != 0).sum(axis=1) >= 2)
    best_auc, best_epoch, bad = -float("inf"), 0, 0
    best_model, best_projector = deepcopy(model.state_dict()), None
    records = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        if projector is not None:
            projector.train()
        order = np.random.default_rng([seed, epoch, 0]).permutation(eligible)
        batch_records = []
        for step, start in enumerate(range(0, len(order), args.batch_students)):
            students = order[start:start + args.batch_students]
            if len(students) < 2:
                continue
            ep = make_student_episode(data["train"], students, args.query_fraction,
                                      np.random.default_rng([seed, epoch, step, 1]))
            clean_np = ep.support[students]
            clean = torch.as_tensor(clean_np, dtype=torch.float32, device=args.device)
            theta_clean = model.diagnose_theta(clean)
            items, inverse = np.unique(ep.query_items, return_inverse=True)
            item_logs = torch.as_tensor(ep.support[:, items].T.copy(), dtype=torch.float32,
                                        device=args.device)
            psi = model.diagnose_psi(item_logs)[torch.as_tensor(inverse, device=args.device)]
            users = torch.as_tensor(ep.query_students, device=args.device)
            query_items = torch.as_tensor(ep.query_items, device=args.device)
            q_batch = model.get_q_soft()[query_items]
            pred_clean = model.predict_response(theta_clean[users], psi, q_batch)
            labels = torch.as_tensor(ep.query_labels, dtype=torch.float32,
                                     device=args.device).reshape(-1, 1)
            base = model.compute_loss(pred_clean, labels)["loss_total"]
            aux = theta_clean.sum() * 0.0
            changed = 0.0
            if args.method in {
                "flip_infonce", "mask_infonce", "mask_project_theta",
                "mask_project_implicit", "mask_augmented_bce",
                "mask_prediction_consistency", "mask_concept_consistency",
            }:
                rng = np.random.default_rng([seed, epoch, step, 2])
                if args.method == "flip_infonce":
                    view_np = flip_observed_responses(clean_np, args.view_rate, rng)
                else:
                    view_np = mask_observed(clean_np, args.view_rate, rng)
                changed = float(np.sum(view_np != clean_np) / max(np.sum(clean_np != 0), 1))
                view = torch.as_tensor(view_np, dtype=torch.float32, device=args.device)
                theta_view = model.diagnose_theta(view)
                if args.method in {"flip_infonce", "mask_infonce"}:
                    aux = simple_infonce(theta_clean, theta_view, args.temperature)
                elif args.method in {"mask_project_theta", "mask_project_implicit"}:
                    if args.method == "mask_project_implicit":
                        z_clean, z_view = model.f_nn(clean), model.f_nn(view)
                    else:
                        z_clean, z_view = theta_clean, theta_view
                    aux = simple_infonce(projector(z_clean), projector(z_view), args.temperature)
                elif args.method == "mask_concept_consistency":
                    reference_q = torch.as_tensor(data["q"], device=args.device)
                    count_clean = torch.abs(clean) @ reference_q
                    count_view = torch.abs(view) @ reference_q
                    count = torch.minimum(count_clean, count_view)
                    weight = (count / (count + args.evidence_c)).detach()
                    aux = torch.mean(weight * (theta_view - theta_clean.detach()).square())
                else:
                    pred_view = model.predict_response(theta_view[users], psi, q_batch)
                    if args.method == "mask_augmented_bce":
                        aux = F.binary_cross_entropy(pred_view, labels)
                    else:
                        aux = F.mse_loss(pred_view, pred_clean.detach())
            loss = base + args.lambda_aux * aux
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss: {args.experiment_id}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_records.append((float(base.detach()), float(aux.detach()), changed))
        valid, theta_val, _ = core.evaluate(model, data["train"], data["rows"][1])
        record = {
            "epoch": epoch,
            "train_base_loss": float(np.mean([r[0] for r in batch_records])),
            "train_aux_loss": float(np.mean([r[1] for r in batch_records])),
            "actual_view_change": float(np.mean([r[2] for r in batch_records])),
            "valid_auc": valid["auc"], "valid_acc": valid["acc"],
            "valid_nll": valid["nll"],
            "theta_std_mean": float(theta_val.std(axis=0).mean()),
        }
        records.append(record)
        score = valid["auc"] if valid["auc"] is not None else -valid["nll"]
        if score > best_auc:
            best_auc, best_epoch, bad = score, epoch, 0
            best_model = deepcopy(model.state_dict())
            best_projector = deepcopy(projector.state_dict()) if projector is not None else None
        else:
            bad += 1
        if args.patience and bad >= args.patience:
            break
    model.load_state_dict(best_model)
    if projector is not None:
        projector.load_state_dict(best_projector)
    if directory is not None:
        torch.save({"model_state_dict": best_model, "projector_state_dict": best_projector,
                    "seed": seed, "best_epoch": best_epoch}, directory / "best_model.pt")
    values = score_model(model, data)
    result = {"experiment_id": args.experiment_id, "dataset": args.dataset,
              "q_mode": args.q_mode, "method": args.method, "seed": seed,
              "best_epoch": best_epoch, "epochs_run": len(records),
              "seconds": time.perf_counter() - started,
              "lambda_aux": args.lambda_aux, "view_rate": args.view_rate,
              "mean_actual_view_change": float(np.mean([r["actual_view_change"] for r in records])),
              **values}
    if directory is not None:
        write_json(directory / "metrics.json", result)
        write_csv(directory / "validation_curve.csv", records)
    else:
        replay = get_model(data, args, seed)
        replay.load_state_dict(best_model)
        replay_values = score_model(replay, data)
        for key, value in values.items():
            if not np.isclose(value, replay_values[key], atol=1e-10, rtol=0):
                raise ValueError(f"In-memory replay mismatch: {key}")
        result["in_memory_replay"] = "passed"
    print(f"{args.experiment_id} seed={seed} epoch={best_epoch}/{len(records)} "
          f"val_auc={result['valid_auc']:.6f} test_auc={result['test_auc']:.6f}", flush=True)
    return result


def verify(data: dict, args: argparse.Namespace, rows: list[dict], folder: Path, hashes: dict) -> dict:
    for name, digest in data["source_sha256"].items():
        if prepared.sha256(data["folder"] / name) != digest:
            raise ValueError(f"Data changed: {name}")
    for name, digest in hashes.items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Code changed: {name}")
    for row in rows:
        seed = row["seed"]
        stored = torch.load(folder / f"seed_{seed}" / "best_model.pt",
                            map_location=args.device, weights_only=True)
        model = get_model(data, args, seed)
        model.load_state_dict(stored["model_state_dict"])
        test, _, _ = core.evaluate(model, data["train"], data["rows"][2])
        for metric, value in test.items():
            if not np.isclose(value, row[f"test_{metric}"], atol=1e-10, rtol=0):
                raise ValueError(f"Checkpoint metric mismatch: {seed} {metric}")
    return {"status": "passed", "replayed_checkpoints": len(rows),
            "data_hashes": "passed", "code_hashes": "passed"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        choices=["math1", "math2", "eedi_task34", "assist2012", "junyi2015"])
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--q-mode", required=True, choices=["fixed", "weak"])
    parser.add_argument("--method", required=True, choices=list(METHODS))
    parser.add_argument("--lambda-aux", type=float, default=0.0)
    parser.add_argument("--view-rate", type=float, default=0.1)
    parser.add_argument("--evidence-c", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--batch-students", type=int, default=128)
    parser.add_argument("--query-fraction", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--q-lr", type=float, default=0.005)
    parser.add_argument("--lambda-q", type=float, default=0.01)
    parser.add_argument("--lambda-coverage", type=float, default=0.01)
    parser.add_argument("--q-init-high", type=float, default=0.55)
    parser.add_argument("--q-init-low", type=float, default=0.45)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--scratch", action="store_true")
    args = parser.parse_args(argv)
    if args.lambda_aux < 0 or not 0 < args.view_rate < 1 or args.epochs <= 0:
        parser.error("invalid lambda, view rate, or epochs")
    if args.method in {"fixed_soft_q", "weak_q_no_prior"} and (
        (args.method == "fixed_soft_q" and args.q_mode != "fixed") or
        (args.method == "weak_q_no_prior" and args.q_mode != "weak")):
        parser.error("Q-control method and q-mode disagree")
    if args.method == "baseline" and args.lambda_aux:
        parser.error("baseline lambda must be zero")
    if args.method not in {"baseline", "fixed_soft_q", "weak_q_no_prior"} and args.lambda_aux <= 0:
        parser.error("auxiliary method needs positive lambda")
    if not all(ch.isalnum() or ch in "_-" for ch in args.experiment_id):
        parser.error("experiment-id contains invalid characters")
    torch.set_num_threads(args.threads)
    data = prepared.load_data(ROOT / "Q_GNCDM" / "data" / args.dataset)
    root = ROOT / "Q_GNCDM" / ("scratch" if args.scratch else "results") / "test_sim_con"
    folder = root / args.experiment_id
    folder.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(core.__file__), Path(prepared.__file__),
               ROOT / "Q_GNCDM/simple_contrastive.py",
               ROOT / "Q_GNCDM/models/weak_q_gncdm.py"]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    write_json(folder / "manifest.json", {
        "experiment_id": args.experiment_id,
        "baseline": f"{args.q_mode} Q, no auxiliary method, same epochs/split/seeds",
        "single_change": METHODS[args.method],
        "selection_metric": "mean validation AUC; test never selects method or coefficient",
        "data_source_sha256": data["source_sha256"], "code_sha256": hashes,
        "split_lengths": [len(x) for x in data["rows"]],
        "args": vars(args),
    })
    rows = []
    for seed in args.seeds:
        rows.append(train_one(data, args, seed, folder / f"seed_{seed}"))
        write_csv(folder / "summary.csv", rows)
    numeric = [key for key, value in rows[0].items() if isinstance(value, (int, float))
               and key != "seed"]
    aggregate = {"experiment_id": args.experiment_id, "dataset": args.dataset,
                 "q_mode": args.q_mode, "method": args.method, "n_seeds": len(rows)}
    for key in numeric:
        values = [row[key] for row in rows]
        aggregate[f"{key}_mean"] = float(np.mean(values))
        aggregate[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    write_json(folder / "aggregate.json", aggregate)
    write_json(folder / "verification.json", verify(data, args, rows, folder, hashes))
    print(f"Completed {folder}", flush=True)


if __name__ == "__main__":
    main()
