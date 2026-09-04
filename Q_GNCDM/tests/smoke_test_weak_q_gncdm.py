# -*- coding: utf-8 -*-
"""Minimal forward/backward check for WeakQGNCDM.

Run from Q_GNCDM:

    python tests/smoke_test_weak_q_gncdm.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import WeakQGNCDM  # noqa: E402


def load_response_log(
    response_path: Path,
    num_students: int,
    num_items: int,
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    log_mat = np.zeros((num_students, num_items), dtype=np.float32)
    rows: List[Tuple[int, int, int]] = []
    with response_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            user_id = int(row["user_id"])
            item_id = int(row["item_id"])
            score = int(float(row["score"]))
            log_mat[user_id, item_id] = (score - 0.5) * 2.0
            rows.append((user_id, item_id, score))
    if not rows:
        raise ValueError(f"No observed responses found in {response_path}.")
    return log_mat, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test WeakQGNCDM.")
    parser.add_argument(
        "--data_dir",
        default="data/synthetic_weak_q/seed_2026",
        help="Directory containing response.csv and Q_noise files.",
    )
    parser.add_argument("--q_file", default="Q_noise_0.20.npy")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--user_dim", type=int, default=16)
    parser.add_argument("--item_dim", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--export_dir", default="tmp/weak_q_smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(2026)
    np.random.seed(2026)

    data_dir = Path(args.data_dir)
    q_noise = np.load(data_dir / args.q_file)
    num_items, num_concepts = q_noise.shape

    response_path = data_dir / "response.csv"
    # Infer num_students from the response file to keep the test data-driven.
    with response_path.open("r", encoding="utf-8", newline="") as fp:
        rows_for_count = list(csv.DictReader(fp))
    num_students = max(int(row["user_id"]) for row in rows_for_count) + 1

    log_mat, observed_rows = load_response_log(response_path, num_students, num_items)
    batch_rows = observed_rows[: args.batch_size]
    user_ids_np = np.asarray([row[0] for row in batch_rows], dtype=np.int64)
    item_ids_np = np.asarray([row[1] for row in batch_rows], dtype=np.int64)
    scores_np = np.asarray([row[2] for row in batch_rows], dtype=np.float32).reshape(-1, 1)

    user_log = torch.tensor(log_mat[user_ids_np], dtype=torch.float32)
    item_log = torch.tensor(log_mat[:, item_ids_np].T, dtype=torch.float32)
    user_id = torch.tensor(user_ids_np, dtype=torch.long).reshape(-1, 1)
    item_id = torch.tensor(item_ids_np, dtype=torch.long).reshape(-1, 1)
    score = torch.tensor(scores_np, dtype=torch.float32)

    model = WeakQGNCDM(
        num_students=num_students,
        num_items=num_items,
        num_concepts=num_concepts,
        user_dim=args.user_dim,
        item_dim=args.item_dim,
        alpha=args.alpha,
        Q_noise=q_noise,
        learn_q=True,
        use_weak_q_loss=True,
        q_threshold=0.5,
        device=args.device,
    )

    q_soft = model.get_q_soft().detach().cpu().numpy()
    expected_soft = np.where(q_noise > 0.5, 0.9, 0.1)
    if q_soft.shape != (num_items, num_concepts):
        raise AssertionError(f"Bad Q_soft shape: {q_soft.shape}")
    if not np.allclose(q_soft, expected_soft, atol=1e-6):
        raise AssertionError("Q_soft was not initialized from Q_noise as 0.9/0.1.")

    y_pred, loss_dict = model.forward_with_loss(user_log, item_log, user_id, item_id, score)
    if y_pred.shape != score.shape:
        raise AssertionError(f"Bad y_pred shape: {tuple(y_pred.shape)}")
    if not torch.isfinite(loss_dict["loss_total"]):
        raise AssertionError("loss_total is not finite.")

    model.zero_grad(set_to_none=True)
    loss_dict["loss_total"].backward()
    if model.q_logits.grad is None:
        raise AssertionError("q_logits did not receive gradients.")
    if not torch.isfinite(model.q_logits.grad).all():
        raise AssertionError("q_logits gradients contain non-finite values.")

    q_hard = model.get_q_hard()
    if q_hard.shape != (num_items, num_concepts):
        raise AssertionError(f"Bad Q_hard shape: {tuple(q_hard.shape)}")
    if set(q_hard.detach().cpu().numpy().reshape(-1).tolist()) - {0, 1}:
        raise AssertionError("Q_hard contains non-binary values.")

    export_dir = Path(args.export_dir)
    model.export_q_soft(export_dir / "Q_soft.npy")
    model.export_q_hard(export_dir / "Q_hard.npy")
    if not (export_dir / "Q_soft.npy").exists():
        raise AssertionError("Q_soft export failed.")
    if not (export_dir / "Q_hard.npy").exists():
        raise AssertionError("Q_hard export failed.")

    detached_losses = {
        key: float(value.detach().cpu()) for key, value in loss_dict.items()
    }
    print("WeakQGNCDM smoke test passed.")
    print(f"num_students={num_students} num_items={num_items} num_concepts={num_concepts}")
    print(f"batch_size={len(batch_rows)} y_pred_shape={tuple(y_pred.shape)}")
    print(f"Q_soft_shape={tuple(q_soft.shape)} Q_hard_values=[0, 1]")
    print(f"q_logits_grad_norm={float(model.q_logits.grad.norm().detach().cpu()):.6f}")
    print(f"export_dir={export_dir}")
    print("losses:")
    for key, value in detached_losses.items():
        print(f"  {key}={value:.6f}")


if __name__ == "__main__":
    main()
