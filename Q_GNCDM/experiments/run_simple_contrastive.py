"""Run fixed-Q, Weak-Q, and minimal response-flip contrastive experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Q_GNCDM.experiments import run_weak_q_synthetic as legacy
from Q_GNCDM.models.weak_q_gncdm import WeakQGNCDM
from Q_GNCDM.simple_contrastive import (
    flip_observed_responses,
    make_student_episode,
    simple_infonce,
)


MODES = {
    "A0": {"label": "fixed-given-q", "learn_q": False, "contrastive": False},
    "A1": {"label": "weak-q", "learn_q": True, "contrastive": False},
    "S1": {"label": "weak-q-simple-contrastive", "learn_q": True, "contrastive": True},
}


def save_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(legacy.to_jsonable(rows))


def load_data(args):
    folder = Path(args.data_dir).resolve()
    if args.dataset == "math1":
        files = [folder / f"math1_{part}_0.8_0.2.csv"
                 for part in ("train", "valid", "test")]
        rows = [legacy.load_response_rows(path) for path in files]
        q_path = folder / "math1_Q_matrix.npy"
    else:
        files = [folder / "response.csv"]
        rows = legacy.split_rows(legacy.load_response_rows(files[0]), args.split_seed)
        q_path = folder / f"Q_noise_{args.q_noise:.2f}.npy"
    q = np.load(q_path).astype(np.float32)
    if q.ndim != 2 or not np.isin(q, [0, 1]).all():
        raise ValueError("Q must be a binary item-by-concept matrix")
    all_rows = [row for part in rows for row in part]
    n_students = max(row[0] for row in all_rows) + 1
    array_rows = [np.asarray(part, dtype=np.int64) for part in rows]
    pairs = [{(int(row[0]), int(row[1])) for row in part} for part in array_rows]
    if any(len(pair_set) != len(part) for pair_set, part in zip(pairs, array_rows)):
        raise ValueError("Repeated student-item pairs need a temporal protocol")
    if any(pairs[left] & pairs[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Train, validation, and test pairs overlap")
    train = legacy.build_log_mat(rows[0], n_students, len(q))
    source_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files + [q_path]
    }
    return {
        "folder": folder,
        "rows": array_rows,
        "train": train,
        "q": q,
        "source_sha256": source_hashes,
    }


def build_model(data, args, mode, seed):
    settings = MODES[mode]
    legacy.set_seed(seed)
    fixed = not settings["learn_q"]
    return WeakQGNCDM(
        num_students=len(data["train"]),
        num_items=len(data["q"]),
        num_concepts=data["q"].shape[1],
        user_dim=args.embedding_dim,
        item_dim=args.embedding_dim,
        alpha=args.alpha,
        Q_noise=data["q"],
        learn_q=not fixed,
        use_weak_q_loss=not fixed,
        q_init_mode="fixed" if fixed else "weak",
        q_init_high=args.q_init_high,
        q_init_low=args.q_init_low,
        q_prior_high=args.q_init_high,
        q_prior_low=args.q_init_low,
        weak_q_loss_type="mse",
        lambda_q=0.0 if fixed else args.lambda_q,
        lambda_sparse=0.0,
        lambda_binary=0.0,
        lambda_coverage=0.0 if fixed else args.lambda_coverage,
        q_pos_weight=None if fixed else 1.0,
        device=args.device,
    )


def build_optimizer(model, args):
    base = [parameter for name, parameter in model.named_parameters()
            if name != "q_logits" and parameter.requires_grad]
    groups = [{"params": base, "lr": args.lr}]
    if model.q_logits.requires_grad:
        groups.append({"params": [model.q_logits], "lr": args.q_lr})
    return torch.optim.Adam(groups)


def metrics_from_predictions(rows, probabilities):
    truth = rows[:, 2]
    errors = probabilities - truth
    return {
        "auc": legacy.safe_auc(truth, probabilities),
        "acc": float(np.mean((probabilities > 0.5) == truth)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "brier": float(np.mean(errors ** 2)),
        "nll": legacy.binary_bce(truth, probabilities),
    }


@torch.no_grad()
def evaluate(model, evidence, rows):
    if np.any(evidence[rows[:, 0], rows[:, 1]] != 0):
        raise ValueError("Evaluation labels are visible in the evidence matrix")
    model.eval()
    matrix = torch.as_tensor(evidence, dtype=torch.float32, device=model.device)
    theta = model.diagnose_theta(matrix)
    psi = model.diagnose_psi(matrix.T)
    q = model.get_q_soft()
    predictions = []
    for start in range(0, len(rows), 2048):
        block = rows[start:start + 2048]
        users = torch.as_tensor(block[:, 0], device=model.device)
        items = torch.as_tensor(block[:, 1], device=model.device)
        values = model.predict_response(theta[users], psi[items], q[items])
        predictions.append(values.flatten().cpu().numpy())
    probabilities = np.concatenate(predictions)
    return metrics_from_predictions(rows, probabilities), theta.cpu().numpy(), probabilities


def train_one(data, args, mode, seed, output):
    output.mkdir(parents=True, exist_ok=False)
    settings = MODES[mode]
    model = build_model(data, args, mode, seed)
    optimizer = build_optimizer(model, args)
    eligible = np.flatnonzero((data["train"] != 0).sum(axis=1) >= 2)
    if len(eligible) < 2:
        raise ValueError("At least two eligible students are required")
    legacy.write_json(output / "config.json", {
        "args": vars(args),
        "mode": mode,
        "settings": settings,
        "seed": seed,
        "source_sha256": data["source_sha256"],
        "protocol": (
            "seen-student; clean support for prediction; response-flipped support "
            "only for S1 InfoNCE; query hidden from student and item inputs"
        ),
    })

    best_state = deepcopy(model.state_dict())
    best_auc, best_epoch, bad_epochs = -float("inf"), 0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.default_rng([seed, epoch, 0]).permutation(eligible)
        epoch_rows = []
        for step, start in enumerate(range(0, len(order), args.batch_students)):
            students = order[start:start + args.batch_students]
            if len(students) < 2:
                continue
            episode = make_student_episode(
                data["train"], students, args.query_fraction,
                np.random.default_rng([seed, epoch, step, 1]),
            )
            clean_np = episode.support[students]
            clean = torch.as_tensor(clean_np, dtype=torch.float32, device=model.device)
            theta_clean = model.diagnose_theta(clean)

            item_ids, inverse = np.unique(episode.query_items, return_inverse=True)
            item_logs = torch.as_tensor(
                episode.support[:, item_ids].T.copy(),
                dtype=torch.float32,
                device=model.device,
            )
            psi = model.diagnose_psi(item_logs)[
                torch.as_tensor(inverse, device=model.device)
            ]
            query_students = torch.as_tensor(
                episode.query_students, device=model.device
            )
            query_items = torch.as_tensor(episode.query_items, device=model.device)
            prediction = model.predict_response(
                theta_clean[query_students],
                psi,
                model.get_q_soft()[query_items],
            )
            target = torch.as_tensor(
                episode.query_labels, dtype=torch.float32, device=model.device
            ).reshape(-1, 1)
            losses = model.compute_loss(prediction, target)

            if settings["contrastive"]:
                noisy_np = flip_observed_responses(
                    clean_np,
                    args.positive_noise_rate,
                    np.random.default_rng([seed, epoch, step, 2]),
                )
                theta_noisy = model.diagnose_theta(torch.as_tensor(
                    noisy_np, dtype=torch.float32, device=model.device
                ))
                loss_cl = simple_infonce(
                    theta_clean, theta_noisy, args.temperature
                )
                flip_fraction = float(
                    np.sum(noisy_np != clean_np) / np.maximum(np.sum(clean_np != 0), 1)
                )
                coefficient = args.lambda_cl
            else:
                loss_cl = theta_clean.sum() * 0.0
                flip_fraction = 0.0
                coefficient = 0.0
            total = losses["loss_total"] + coefficient * loss_cl
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            epoch_rows.append({
                "loss_total": float(total.detach().cpu()),
                "loss_pred": float(losses["loss_rec"].detach().cpu()),
                "loss_q_regularization": float(
                    (losses["loss_total"] - losses["loss_rec"]).detach().cpu()
                ),
                "loss_cl": float(loss_cl.detach().cpu()),
                "lambda_cl_effective": coefficient,
                "actual_flip_fraction": flip_fraction,
            })

        valid, theta_valid, _ = evaluate(model, data["train"], data["rows"][1])
        record = {
            "epoch": epoch,
            **{key: float(np.mean([row[key] for row in epoch_rows]))
               for key in epoch_rows[0]},
            **{f"valid_{key}": value for key, value in valid.items()},
            "theta_std_mean": float(theta_valid.std(axis=0).mean()),
        }
        history.append(record)
        score = valid["auc"] if valid["auc"] is not None else -valid["nll"]
        if score > best_auc:
            best_auc, best_epoch = score, epoch
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        save_csv(output / "train_log.csv", history)
        print(
            f"{args.dataset} {mode} seed={seed} epoch={epoch} "
            f"pred={record['loss_pred']:.4f} cl={record['loss_cl']:.4f} "
            f"valid_auc={valid['auc']}",
            flush=True,
        )
        if args.patience > 0 and bad_epochs >= args.patience:
            break

    model.load_state_dict(best_state)
    torch.save({
        "model_state_dict": best_state,
        "mode": mode,
        "seed": seed,
        "best_epoch": best_epoch,
    }, output / "best_model.pt")
    valid, _, _ = evaluate(model, data["train"], data["rows"][1])
    test, theta, probabilities = evaluate(model, data["train"], data["rows"][2])
    np.save(output / "theta.npy", theta)
    np.save(output / "Q_soft.npy", model.get_q_soft().detach().cpu().numpy())
    np.save(output / "Q_hard.npy", model.get_q_hard().cpu().numpy())
    np.savez_compressed(
        output / "test_predictions.npz",
        rows=data["rows"][2], probabilities=probabilities,
    )

    q_reference = (
        data["q"] if args.dataset == "math1"
        else np.load(data["folder"] / "Q_true.npy")
    )
    q_scores = legacy.q_recovery_metrics(
        q_reference,
        data["q"],
        model.get_q_hard().cpu().numpy(),
        model.get_q_soft().detach().cpu().numpy(),
    )
    result = {
        "dataset": args.dataset,
        "mode": mode,
        "mode_label": settings["label"],
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "seconds": time.perf_counter() - started,
        **{f"valid_{key}": value for key, value in valid.items()},
        **{f"test_{key}": value for key, value in test.items()},
        "theta_std_mean": float(theta.std(axis=0).mean()),
        "q_f1_vs_reference": q_scores["q_hat_f1_vs_true"],
        "mean_actual_flip_fraction": float(
            np.mean([row["actual_flip_fraction"] for row in history])
        ),
    }
    if args.dataset == "synthetic":
        mastery_true = np.load(data["folder"] / "mastery_true.npy")
        result.update(legacy.mastery_metrics(theta, mastery_true))
    legacy.write_json(output / "metrics.json", result)
    return result


def summarize(rows, destination):
    save_csv(destination / "summary.csv", rows)
    aggregate = []
    for mode in MODES:
        group = [row for row in rows if row["mode"] == mode]
        if not group:
            continue
        record = {"mode": mode, "n_seeds": len(group)}
        for key in group[0]:
            if key in {"dataset", "mode", "mode_label", "seed"}:
                continue
            values = [row[key] for row in group
                      if isinstance(row.get(key), (int, float))]
            if values:
                record[f"{key}_mean"] = float(np.mean(values))
                record[f"{key}_std"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else None
                )
        aggregate.append(record)
    save_csv(destination / "aggregate.csv", aggregate)

    comparisons = []
    for base, treatment in (("A0", "A1"), ("A1", "S1")):
        left = {row["seed"]: row for row in rows if row["mode"] == base}
        right = {row["seed"]: row for row in rows if row["mode"] == treatment}
        for metric in (
            "test_auc", "test_acc", "test_rmse", "test_nll",
            "q_f1_vs_reference", "mastery_spearman",
        ):
            deltas = [right[seed][metric] - left[seed][metric]
                      for seed in sorted(left.keys() & right.keys())
                      if left[seed].get(metric) is not None
                      and right[seed].get(metric) is not None]
            if deltas:
                comparisons.append({
                    "base": base,
                    "treatment": treatment,
                    "metric": metric,
                    "n_pairs": len(deltas),
                    "mean_delta": float(np.mean(deltas)),
                    "std_delta": (
                        float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None
                    ),
                })
    save_csv(destination / "paired_comparisons.csv", comparisons)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["math1", "synthetic"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--q-noise", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=2026)
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
    if args.epochs <= 0 or min(args.batch_students, args.embedding_dim, args.threads) <= 0:
        parser.error("epochs, batch-students, embedding-dim, and threads must be positive")
    if not 0 < args.query_fraction < 1 or not 0 < args.positive_noise_rate < 1:
        parser.error("query-fraction and positive-noise-rate must be in (0,1)")
    if not 0 <= args.alpha <= 1 or args.temperature <= 0:
        parser.error("alpha or temperature is invalid")
    if min(args.lambda_cl, args.lambda_q, args.lambda_coverage, args.lr, args.q_lr) < 0:
        parser.error("loss weights and learning rates must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    data = load_data(args)
    destination = Path(args.results_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    import scipy
    import sklearn
    source_files = [
        Path(__file__),
        ROOT / "Q_GNCDM/simple_contrastive.py",
        ROOT / "Q_GNCDM/models/weak_q_gncdm.py",
        ROOT / "Q_GNCDM/experiments/run_weak_q_synthetic.py",
    ]
    manifest = {
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
        "source_sha256": data["source_sha256"],
        "code_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
        "args": vars(args),
        "modes": MODES,
    }
    legacy.write_json(destination / "manifest.json", manifest)
    np.savez_compressed(
        destination / "splits.npz",
        train=data["rows"][0], valid=data["rows"][1], test=data["rows"][2],
    )
    results = []
    for seed in args.seeds:
        for mode in args.modes:
            output = destination / f"seed_{seed}" / mode
            results.append(train_one(data, args, mode, seed, output))
            summarize(results, destination)
    legacy.write_json(destination / "completed.json", {"runs_completed": len(results)})
    print(f"Completed {len(results)} runs: {destination}", flush=True)


if __name__ == "__main__":
    main()
