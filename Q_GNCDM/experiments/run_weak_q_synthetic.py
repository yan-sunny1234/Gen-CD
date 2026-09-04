# -*- coding: utf-8 -*-
"""Run the synthetic Weak-Q G-NCDM experiment.

This script is intentionally small and self-contained. It trains on synthetic
response.csv interactions, evaluates Q recovery / response prediction / mastery
recovery, and archives the run outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_recall_fscore_support,
    roc_auc_score,
)
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import WeakQGNCDM  # noqa: E402


ResponseRow = Tuple[int, int, int]


def rate_label(rate: float) -> str:
    return f"{rate:.2f}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(to_jsonable(payload), fp, indent=2)


def write_config_yaml(path: Path, payload: Dict[str, Any]) -> None:
    # JSON is valid YAML 1.2 and avoids adding a PyYAML dependency.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(to_jsonable(payload), fp, indent=2)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def parse_q_pos_weight(value: str) -> Any:
    lower = value.lower()
    if lower == "auto":
        return "auto"
    if lower in {"none", "null"}:
        return None
    return float(value)


def load_response_rows(path: Path) -> List[ResponseRow]:
    rows: List[ResponseRow] = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            rows.append(
                (
                    int(row["user_id"]),
                    int(row["item_id"]),
                    int(float(row["score"])),
                )
            )
    if not rows:
        raise ValueError(f"No interactions found in {path}.")
    return rows


def split_rows(
    rows: Sequence[ResponseRow],
    seed: int,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
) -> Tuple[List[ResponseRow], List[ResponseRow], List[ResponseRow]]:
    indices = np.arange(len(rows))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_train = int(len(rows) * train_ratio)
    n_valid = int(len(rows) * valid_ratio)
    train_idx = indices[:n_train]
    valid_idx = indices[n_train : n_train + n_valid]
    test_idx = indices[n_train + n_valid :]
    return (
        [rows[int(i)] for i in train_idx],
        [rows[int(i)] for i in valid_idx],
        [rows[int(i)] for i in test_idx],
    )


def build_log_mat(
    rows: Sequence[ResponseRow],
    num_students: int,
    num_items: int,
) -> np.ndarray:
    log_mat = np.zeros((num_students, num_items), dtype=np.float32)
    for user_id, item_id, score in rows:
        log_mat[user_id, item_id] = (float(score) - 0.5) * 2.0
    return log_mat


class InteractionDataset(Dataset):
    """Interaction triples with train-only evidence vectors."""

    def __init__(self, rows: Sequence[ResponseRow], evidence_log_mat: np.ndarray) -> None:
        self.rows = list(rows)
        self.evidence_log_mat = evidence_log_mat

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        user_id, item_id, score = self.rows[index]
        user_log = self.evidence_log_mat[user_id, :]
        item_log = self.evidence_log_mat[:, item_id]
        return (
            torch.tensor(user_log, dtype=torch.float32),
            torch.tensor(item_log, dtype=torch.float32),
            torch.tensor([user_id], dtype=torch.long),
            torch.tensor([item_id], dtype=torch.long),
            torch.tensor([score], dtype=torch.float32),
        )


def make_loader(
    rows: Sequence[ResponseRow],
    evidence_log_mat: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        InteractionDataset(rows, evidence_log_mat),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def safe_auc(y_true: Sequence[float], y_score: Sequence[float]) -> Optional[float]:
    y_arr = np.asarray(y_true)
    if np.unique(y_arr).size < 2:
        return None
    try:
        return float(roc_auc_score(y_arr, np.asarray(y_score)))
    except ValueError:
        return None


def binary_bce(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    prob = np.clip(y_prob.astype(np.float64), 1e-7, 1.0 - 1e-7)
    true = y_true.astype(np.float64)
    return float(-np.mean(true * np.log(prob) + (1.0 - true) * np.log(1.0 - prob)))


def evaluate_response(
    model: WeakQGNCDM,
    rows: Sequence[ResponseRow],
    evidence_log_mat: np.ndarray,
    batch_size: int,
    seed: int,
) -> Dict[str, Optional[float]]:
    if len(rows) == 0:
        return {"acc": None, "auc": None, "f1": None, "rmse": None, "bce": None}
    loader = make_loader(rows, evidence_log_mat, batch_size, shuffle=False, seed=seed)
    y_true: List[float] = []
    y_pred: List[float] = []
    model.eval()
    with torch.no_grad():
        for user_log, item_log, user_id, item_id, score in loader:
            pred = model(user_log, item_log, user_id, item_id)
            y_pred.extend(pred.detach().cpu().numpy().reshape(-1).tolist())
            y_true.extend(score.detach().cpu().numpy().reshape(-1).tolist())
    true_arr = np.asarray(y_true, dtype=np.float64)
    pred_arr = np.asarray(y_pred, dtype=np.float64)
    label_arr = (pred_arr > 0.5).astype(np.int64)
    return {
        "acc": float(accuracy_score(true_arr, label_arr)),
        "auc": safe_auc(true_arr, pred_arr),
        "f1": float(f1_score(true_arr, label_arr, zero_division=0)),
        "rmse": float(np.sqrt(mean_squared_error(true_arr, pred_arr))),
        "bce": binary_bce(true_arr, pred_arr),
    }


def average_losses(loss_sums: Dict[str, float], batch_count: int) -> Dict[str, float]:
    if batch_count == 0:
        return {key: 0.0 for key in loss_sums}
    return {key: value / batch_count for key, value in loss_sums.items()}


def train_model(
    model: WeakQGNCDM,
    train_rows: Sequence[ResponseRow],
    valid_rows: Sequence[ResponseRow],
    evidence_log_mat: np.ndarray,
    args: argparse.Namespace,
    q_true_eval: np.ndarray,
    q_noise_eval: np.ndarray,
) -> Tuple[WeakQGNCDM, List[Dict[str, Any]], int]:
    train_loader = make_loader(
        train_rows,
        evidence_log_mat,
        args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    base_params = []
    q_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name == "q_logits":
            q_params.append(param)
        else:
            base_params.append(param)
    param_groups = []
    if base_params:
        param_groups.append({"params": base_params, "lr": args.lr})
    if q_params:
        param_groups.append({"params": q_params, "lr": args.q_lr})
    optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)

    best_state = deepcopy(model.state_dict())
    best_epoch = -1
    best_score = -float("inf")
    bad_epochs = 0
    train_log: List[Dict[str, Any]] = []
    base_lambda_sparse = float(model.lambda_sparse)
    base_lambda_binary = float(model.lambda_binary)
    q_reg_warmup_epochs = int(args.epochs * args.q_reg_warmup_fraction)

    for epoch in range(1, args.epochs + 1):
        model.train()
        if epoch <= q_reg_warmup_epochs:
            model.lambda_sparse = 0.0
            model.lambda_binary = 0.0
        else:
            model.lambda_sparse = base_lambda_sparse
            model.lambda_binary = base_lambda_binary

        loss_sums = {
            "loss_total": 0.0,
            "loss_rec": 0.0,
            "loss_weak_q": 0.0,
            "loss_sparse": 0.0,
            "loss_binary": 0.0,
            "loss_coverage": 0.0,
        }
        batch_count = 0
        q_grad_abs_sum = 0.0
        q_grad_batches = 0
        for user_log, item_log, user_id, item_id, score in train_loader:
            y_pred, loss_dict = model.forward_with_loss(
                user_log, item_log, user_id, item_id, score
            )
            del y_pred
            optimizer.zero_grad()
            loss_dict["loss_total"].backward()
            if model.q_logits.grad is not None:
                q_grad_abs_sum += float(model.q_logits.grad.detach().abs().mean().cpu())
                q_grad_batches += 1
            optimizer.step()
            for key in loss_sums:
                loss_sums[key] += float(loss_dict[key].detach().cpu())
            batch_count += 1

        avg_loss = average_losses(loss_sums, batch_count)
        q_grad_abs_mean = (
            q_grad_abs_sum / q_grad_batches if q_grad_batches > 0 else None
        )
        valid_metrics = evaluate_response(
            model,
            valid_rows,
            evidence_log_mat,
            batch_size=args.eval_batch_size,
            seed=args.seed,
        )
        q_epoch_metrics = epoch_q_metrics(
            model,
            q_true=q_true_eval,
            q_noise=q_noise_eval,
            threshold=args.q_threshold,
        )
        row = {
            "epoch": epoch,
            **avg_loss,
            "valid_acc": valid_metrics["acc"],
            "valid_auc": valid_metrics["auc"],
            "valid_rmse": valid_metrics["rmse"],
            "q_logits_grad_abs_mean": q_grad_abs_mean,
            **q_epoch_metrics,
        }
        train_log.append(row)

        valid_auc = valid_metrics["auc"]
        score = valid_auc if valid_auc is not None else -float(valid_metrics["rmse"])
        if score > best_score + args.min_delta:
            best_score = score
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(
            f"epoch={epoch} "
            f"loss_total={avg_loss['loss_total']:.6f} "
            f"loss_rec={avg_loss['loss_rec']:.6f} "
            f"valid_auc={format_optional(valid_metrics['auc'])} "
            f"valid_rmse={format_optional(valid_metrics['rmse'])} "
            f"q_change={q_epoch_metrics['q_change_from_noise']} "
            f"q_delta={format_optional(q_epoch_metrics['q_soft_delta_from_init'])} "
            f"q_grad={format_optional(q_grad_abs_mean)}"
        )
        if args.patience > 0 and bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}.")
            break

    model.load_state_dict(best_state)
    return model, train_log, best_epoch


def format_optional(value: Optional[float]) -> str:
    return "nan" if value is None else f"{value:.6f}"


def make_run_label(mode: str, run_tag: str) -> str:
    tag = run_tag.strip()
    if not tag:
        return mode
    safe_tag = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in tag)
    return f"{mode}_{safe_tag}"


def write_train_log(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "epoch",
        "loss_total",
        "loss_rec",
        "loss_weak_q",
        "loss_sparse",
        "loss_binary",
        "loss_coverage",
        "valid_acc",
        "valid_auc",
        "valid_rmse",
        "q_logits_grad_abs_mean",
        "q_soft_min",
        "q_soft_max",
        "q_soft_mean",
        "q_soft_delta_from_init",
        "q_hat_acc_vs_true",
        "q_hat_precision_vs_true",
        "q_hat_recall_vs_true",
        "q_hat_f1_vs_true",
        "q_hat_auc_vs_true",
        "q_hat_hamming_vs_true",
        "deleted_edge_recovery",
        "added_edge_suppression",
        "q_change_from_noise",
        "q_change_0_to_1",
        "q_change_1_to_0",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def q_recovery_metrics(
    q_true: np.ndarray,
    q_noise: np.ndarray,
    q_hat: np.ndarray,
    q_score: np.ndarray,
) -> Dict[str, Optional[float]]:
    true_flat = q_true.reshape(-1).astype(np.int64)
    noise_flat = q_noise.reshape(-1).astype(np.int64)
    hat_flat = q_hat.reshape(-1).astype(np.int64)
    score_flat = q_score.reshape(-1).astype(np.float64)

    def one_set(prefix: str, pred_flat: np.ndarray, score: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_flat,
            pred_flat,
            average="binary",
            zero_division=0,
        )
        result = {
            f"{prefix}_acc_vs_true": float(accuracy_score(true_flat, pred_flat)),
            f"{prefix}_precision_vs_true": float(precision),
            f"{prefix}_recall_vs_true": float(recall),
            f"{prefix}_f1_vs_true": float(f1),
            f"{prefix}_hamming_vs_true": float(np.mean(pred_flat != true_flat)),
        }
        if score is not None:
            result[f"{prefix}_auc_vs_true"] = safe_auc(true_flat, score)
        return result

    metrics = {}
    metrics.update(one_set("q_noise", noise_flat, None))
    metrics.update(one_set("q_hat", hat_flat, score_flat))

    deleted_mask = (q_true == 1) & (q_noise == 0)
    added_mask = (q_true == 0) & (q_noise == 1)
    metrics["deleted_edge_recovery"] = (
        float(np.mean(q_hat[deleted_mask] == 1)) if np.any(deleted_mask) else None
    )
    metrics["added_edge_suppression"] = (
        float(np.mean(q_hat[added_mask] == 0)) if np.any(added_mask) else None
    )
    return metrics


def epoch_q_metrics(
    model: WeakQGNCDM,
    q_true: np.ndarray,
    q_noise: np.ndarray,
    threshold: float,
) -> Dict[str, Optional[float]]:
    q_soft = model.get_q_soft().detach().cpu().numpy()
    q_hat = (q_soft > threshold).astype(np.int64)
    q_init = model.Q_init_prob.detach().cpu().numpy()
    metrics = q_recovery_metrics(q_true, q_noise, q_hat, q_soft)
    return {
        "q_soft_min": float(np.min(q_soft)),
        "q_soft_max": float(np.max(q_soft)),
        "q_soft_mean": float(np.mean(q_soft)),
        "q_soft_delta_from_init": float(np.mean(np.abs(q_soft - q_init))),
        "q_hat_acc_vs_true": metrics["q_hat_acc_vs_true"],
        "q_hat_precision_vs_true": metrics["q_hat_precision_vs_true"],
        "q_hat_recall_vs_true": metrics["q_hat_recall_vs_true"],
        "q_hat_f1_vs_true": metrics["q_hat_f1_vs_true"],
        "q_hat_auc_vs_true": metrics["q_hat_auc_vs_true"],
        "q_hat_hamming_vs_true": metrics["q_hat_hamming_vs_true"],
        "deleted_edge_recovery": metrics["deleted_edge_recovery"],
        "added_edge_suppression": metrics["added_edge_suppression"],
        "q_change_from_noise": int(np.sum(q_hat != q_noise)),
        "q_change_0_to_1": int(np.sum((q_noise == 0) & (q_hat == 1))),
        "q_change_1_to_0": int(np.sum((q_noise == 1) & (q_hat == 0))),
    }


def compute_theta_hat(
    model: WeakQGNCDM,
    evidence_log_mat: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    theta_chunks: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, evidence_log_mat.shape[0], batch_size):
            batch = torch.tensor(
                evidence_log_mat[start : start + batch_size],
                dtype=torch.float32,
            )
            theta = model.diagnose_theta(batch)
            theta_chunks.append(theta.detach().cpu().numpy())
    return np.concatenate(theta_chunks, axis=0)


def mastery_metrics(theta_hat: np.ndarray, mastery_true: np.ndarray) -> Dict[str, Optional[float]]:
    pearsons: List[float] = []
    spearmans: List[float] = []
    for concept_id in range(mastery_true.shape[1]):
        pred = theta_hat[:, concept_id]
        true = mastery_true[:, concept_id]
        if np.std(pred) <= 1e-12 or np.std(true) <= 1e-12:
            continue
        pearson_result = pearsonr(true, pred)
        spearman_result = spearmanr(true, pred)
        pearson_value = (
            pearson_result.statistic
            if hasattr(pearson_result, "statistic")
            else pearson_result[0]
        )
        spearman_value = (
            spearman_result.statistic
            if hasattr(spearman_result, "statistic")
            else spearman_result[0]
        )
        if np.isfinite(pearson_value):
            pearsons.append(float(pearson_value))
        if np.isfinite(spearman_value):
            spearmans.append(float(spearman_value))
    in_range = (
        np.min(theta_hat) >= -1e-6
        and np.max(theta_hat) <= 1.0 + 1e-6
        and np.min(mastery_true) >= -1e-6
        and np.max(mastery_true) <= 1.0 + 1e-6
    )
    return {
        "mastery_pearson": float(np.mean(pearsons)) if pearsons else None,
        "mastery_spearman": float(np.mean(spearmans)) if spearmans else None,
        "mastery_mse": float(mean_squared_error(mastery_true.reshape(-1), theta_hat.reshape(-1)))
        if in_range
        else None,
    }


def build_model(
    args: argparse.Namespace,
    metadata: Dict[str, Any],
    q_noise: np.ndarray,
    q_train: Optional[np.ndarray],
) -> WeakQGNCDM:
    common_kwargs = {
        "num_students": int(metadata["num_students"]),
        "num_items": int(metadata["num_items"]),
        "num_concepts": int(metadata["num_concepts"]),
        "user_dim": args.user_dim,
        "item_dim": args.item_dim,
        "alpha": args.alpha,
        "monotonicity_assumption": True,
        "q_threshold": args.q_threshold,
        "device": args.device,
    }

    if args.mode == "fixed-noisy-q":
        return WeakQGNCDM(
            **common_kwargs,
            Q_noise=q_noise,
            learn_q=False,
            use_weak_q_loss=False,
            q_init_mode="fixed",
            lambda_q=0.0,
            lambda_sparse=0.0,
            lambda_binary=0.0,
            lambda_coverage=0.0,
            q_pos_weight=None,
        )
    if args.mode == "oracle-q":
        if q_train is None:
            raise ValueError("oracle-q requires Q_true as q_train.")
        return WeakQGNCDM(
            **common_kwargs,
            Q_noise=q_train,
            learn_q=False,
            use_weak_q_loss=False,
            q_init_mode="fixed",
            lambda_q=0.0,
            lambda_sparse=0.0,
            lambda_binary=0.0,
            lambda_coverage=0.0,
            q_pos_weight=None,
        )
    if args.mode == "free-q":
        return WeakQGNCDM(
            **common_kwargs,
            Q_noise=None,
            learn_q=True,
            use_weak_q_loss=False,
            q_init_mode="random",
            q_init_high=args.q_init_high,
            q_init_low=args.q_init_low,
            lambda_q=0.0,
            lambda_sparse=args.lambda_sparse,
            lambda_binary=args.lambda_binary,
            lambda_coverage=args.lambda_coverage,
            q_pos_weight=None,
        )
    if args.mode == "weak-q":
        return WeakQGNCDM(
            **common_kwargs,
            Q_noise=q_noise,
            learn_q=True,
            use_weak_q_loss=True,
            q_init_mode="weak",
            q_init_high=args.q_init_high,
            q_init_low=args.q_init_low,
            q_prior_high=args.q_prior_high,
            q_prior_low=args.q_prior_low,
            weak_q_loss_type=args.weak_q_loss_type,
            lambda_q=args.lambda_q,
            lambda_sparse=args.lambda_sparse,
            lambda_binary=args.lambda_binary,
            lambda_coverage=args.lambda_coverage,
            q_pos_weight=args.q_pos_weight,
        )
    raise ValueError(f"Unknown mode: {args.mode}")


def get_q_outputs(
    model: WeakQGNCDM,
    mode: str,
    q_noise: np.ndarray,
    q_true: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "fixed-noisy-q":
        return q_noise.astype(np.float32), q_noise.astype(np.int64)
    if mode == "oracle-q":
        return q_true.astype(np.float32), q_true.astype(np.int64)
    q_soft = model.get_q_soft().detach().cpu().numpy().astype(np.float32)
    q_hat = (q_soft > threshold).astype(np.int64)
    return q_soft, q_hat


def save_numpy_outputs(
    out_dir: Path,
    q_true: np.ndarray,
    q_noise: np.ndarray,
    q_soft: np.ndarray,
    q_hat: np.ndarray,
    theta_true: np.ndarray,
    mastery_true: np.ndarray,
    theta_hat: np.ndarray,
) -> None:
    np.save(out_dir / "Q_true.npy", q_true)
    np.save(out_dir / "Q_noise.npy", q_noise)
    np.save(out_dir / "Q_soft.npy", q_soft)
    np.save(out_dir / "Q_hat.npy", q_hat)
    np.save(out_dir / "theta_true.npy", theta_true)
    np.save(out_dir / "mastery_true.npy", mastery_true)
    np.save(out_dir / "theta_hat.npy", theta_hat)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Weak-Q synthetic experiment.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--noise_rate", type=float, required=True)
    parser.add_argument(
        "--mode",
        choices=["fixed-noisy-q", "weak-q", "oracle-q", "free-q"],
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--results_dir", default="results/weak_q_synthetic")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--q_lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--user_dim", type=int, default=32)
    parser.add_argument("--item_dim", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lambda_q", type=float, default=1e-2)
    parser.add_argument("--lambda_sparse", type=float, default=0.0)
    parser.add_argument("--lambda_binary", type=float, default=0.0)
    parser.add_argument("--lambda_coverage", type=float, default=1e-2)
    parser.add_argument("--q_threshold", type=float, default=0.5)
    parser.add_argument("--q_pos_weight", type=parse_q_pos_weight, default=1.0)
    parser.add_argument("--q_init_high", type=float, default=0.55)
    parser.add_argument("--q_init_low", type=float, default=0.45)
    parser.add_argument("--q_prior_high", type=float, default=0.55)
    parser.add_argument("--q_prior_low", type=float, default=0.45)
    parser.add_argument(
        "--weak_q_loss_type",
        choices=["hard_bce", "soft_bce", "mse", "gaussian", "l1", "smooth_l1"],
        default="mse",
    )
    parser.add_argument(
        "--q_reg_warmup_fraction",
        type=float,
        default=0.0,
        help="Fraction of epochs with lambda_sparse and lambda_binary set to 0.",
    )
    parser.add_argument("--run_tag", default="")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.q_reg_warmup_fraction <= 1.0:
        raise ValueError("--q_reg_warmup_fraction must be in [0, 1].")
    set_seed(args.seed)
    run_label = make_run_label(args.mode, args.run_tag)

    data_dir = Path(args.data_dir)
    metadata = read_json(data_dir / "metadata.json")
    num_students = int(metadata["num_students"])
    num_items = int(metadata["num_items"])

    q_noise_path = data_dir / f"Q_noise_{rate_label(args.noise_rate)}.npy"
    q_noise = np.load(q_noise_path).astype(np.int64)
    if q_noise.shape != (num_items, int(metadata["num_concepts"])):
        raise ValueError(f"Unexpected Q_noise shape: {q_noise.shape}")

    q_true_for_eval = np.load(data_dir / "Q_true.npy").astype(np.int64)
    q_train = None
    if args.mode == "oracle-q":
        # Q_true is allowed during training only for the oracle upper bound.
        q_train = q_true_for_eval

    rows = load_response_rows(data_dir / "response.csv")
    train_rows, valid_rows, test_rows = split_rows(rows, args.seed)
    evidence_log_mat = build_log_mat(train_rows, num_students, num_items)

    model = build_model(args, metadata, q_noise, q_train)
    print(
        f"run={run_label} mode={args.mode} noise={rate_label(args.noise_rate)} "
        f"train={len(train_rows)} valid={len(valid_rows)} test={len(test_rows)}"
    )

    model, train_log, best_epoch = train_model(
        model,
        train_rows,
        valid_rows,
        evidence_log_mat,
        args,
        q_true_eval=q_true_for_eval,
        q_noise_eval=q_noise,
    )

    # Evaluation-only truth files are loaded after training for non-oracle modes.
    q_true = q_true_for_eval
    theta_true = np.load(data_dir / "theta_true.npy")
    mastery_true = np.load(data_dir / "mastery_true.npy")

    q_soft, q_hat = get_q_outputs(
        model, args.mode, q_noise, q_true, threshold=args.q_threshold
    )
    theta_hat = compute_theta_hat(model, evidence_log_mat, args.eval_batch_size)
    response_metrics = evaluate_response(
        model, test_rows, evidence_log_mat, args.eval_batch_size, args.seed
    )
    q_metrics = q_recovery_metrics(q_true, q_noise, q_hat, q_soft)
    m_metrics = mastery_metrics(theta_hat, mastery_true)

    metrics: Dict[str, Any] = {
        "mode": args.mode,
        "run_tag": args.run_tag,
        "run_label": run_label,
        "noise_rate": float(args.noise_rate),
        "seed": args.seed,
        "best_epoch": best_epoch,
        "num_train_interactions": len(train_rows),
        "num_valid_interactions": len(valid_rows),
        "num_test_interactions": len(test_rows),
        **q_metrics,
        "response_acc": response_metrics["acc"],
        "response_auc": response_metrics["auc"],
        "response_f1": response_metrics["f1"],
        "response_rmse": response_metrics["rmse"],
        "response_bce": response_metrics["bce"],
        **m_metrics,
    }

    out_dir = (
        Path(args.results_dir)
        / f"seed_{args.seed}"
        / f"noise_{rate_label(args.noise_rate)}"
        / run_label
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "args": vars(args),
        "metadata": metadata,
        "q_noise_path": str(q_noise_path),
        "train_split": {
            "train": len(train_rows),
            "valid": len(valid_rows),
            "test": len(test_rows),
        },
    }
    write_config_yaml(out_dir / "config.yaml", config_payload)
    write_train_log(out_dir / "train_log.csv", train_log)
    write_json(out_dir / "metrics.json", metrics)
    save_numpy_outputs(
        out_dir,
        q_true=q_true,
        q_noise=q_noise,
        q_soft=q_soft,
        q_hat=q_hat,
        theta_true=theta_true,
        mastery_true=mastery_true,
        theta_hat=theta_hat,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": to_jsonable(config_payload),
            "metrics": to_jsonable(metrics),
        },
        out_dir / "model.pt",
    )

    # Also preserve the exact metadata file used for this synthetic dataset.
    shutil.copy2(data_dir / "metadata.json", out_dir / "metadata.json")

    print(f"Saved run to {out_dir}")
    print(
        "Final metrics: "
        f"q_hat_f1={format_optional(metrics['q_hat_f1_vs_true'])} "
        f"response_auc={format_optional(metrics['response_auc'])} "
        f"mastery_pearson={format_optional(metrics['mastery_pearson'])}"
    )


if __name__ == "__main__":
    main()
