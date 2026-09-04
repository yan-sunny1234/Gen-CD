# -*- coding: utf-8 -*-
"""Weak-Q Enhanced G-NCDM.

This module is intentionally independent from the original GNCDM package. It
copies the small G-NCDM building blocks needed for the weak-Q experiment, then
replaces the fixed expert Q matrix with a differentiable soft Q matrix:

    Q_soft = sigmoid(S)

Training uses Q_soft in both G-NCDM Q positions:
1. explicit learner mastery generation;
2. the item-concept mask in the IRF.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ArrayLike = Union[np.ndarray, torch.Tensor]


class PosLinear(nn.Linear):
    """Linear layer with non-negative effective weights.

    This follows the original G-NCDM implementation and keeps monotonicity for
    the learner-side diagnostic and aggregation layers.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


def _safe_logit(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = prob.clamp(min=eps, max=1.0 - eps)
    return torch.log(prob / (1.0 - prob))


def _as_float_q(
    q_matrix: ArrayLike,
    num_items: int,
    num_concepts: int,
    device: torch.device,
) -> torch.Tensor:
    q_tensor = torch.as_tensor(q_matrix, dtype=torch.float32, device=device)
    expected_shape = (num_items, num_concepts)
    if tuple(q_tensor.shape) != expected_shape:
        raise ValueError(
            f"Q matrix shape must be {expected_shape}, got {tuple(q_tensor.shape)}."
        )
    return q_tensor


class WeakQGNCDM(nn.Module):
    """G-NCDM with learnable soft item-concept matrix.

    Parameters mirror the original G-NCDM where possible. The new Q-related
    controls are deliberately simple so the next training-script round can map
    them to experiment modes:

    - fixed-noisy-q/oracle-q: q_init_mode="fixed", learn_q=False
    - weak-q: learn_q=True, use_weak_q_loss=True
    - free-q: q_init_mode="random", learn_q=True, use_weak_q_loss=False
    """

    def __init__(
        self,
        num_students: int,
        num_items: int,
        num_concepts: int,
        user_dim: int,
        item_dim: int,
        alpha: float = 0.99,
        Q_noise: Optional[ArrayLike] = None,
        monotonicity_assumption: bool = True,
        learn_q: bool = True,
        use_weak_q_loss: bool = True,
        q_init_mode: str = "weak",
        q_init_high: float = 0.9,
        q_init_low: float = 0.1,
        q_prior_high: Optional[float] = None,
        q_prior_low: Optional[float] = None,
        weak_q_loss_type: str = "hard_bce",
        lambda_q: float = 1.0,
        lambda_sparse: float = 1e-4,
        lambda_binary: float = 1e-3,
        lambda_coverage: float = 1e-2,
        q_threshold: float = 0.5,
        q_pos_weight: Union[str, float, None] = "auto",
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        super().__init__()
        if user_dim != item_dim:
            raise ValueError("WeakQGNCDM currently expects user_dim == item_dim.")
        if not 0.0 < q_init_low < q_init_high < 1.0:
            raise ValueError("Require 0 < q_init_low < q_init_high < 1.")
        if q_init_mode not in {"weak", "random", "fixed"}:
            raise ValueError('q_init_mode must be "weak", "random", or "fixed".')
        valid_loss_types = {"hard_bce", "soft_bce", "mse", "gaussian", "l1", "smooth_l1"}
        if weak_q_loss_type not in valid_loss_types:
            raise ValueError(
                f"weak_q_loss_type must be one of {sorted(valid_loss_types)}, "
                f"got {weak_q_loss_type}."
            )
        prior_high = q_init_high if q_prior_high is None else float(q_prior_high)
        prior_low = q_init_low if q_prior_low is None else float(q_prior_low)
        if not 0.0 < prior_low < prior_high < 1.0:
            raise ValueError("Require 0 < q_prior_low < q_prior_high < 1.")

        self.n_user = int(num_students)
        self.n_item = int(num_items)
        self.n_know = int(num_concepts)
        self.user_dim = int(user_dim)
        self.item_dim = int(item_dim)
        self.alpha = float(alpha)
        self.learn_q = bool(learn_q)
        self.use_weak_q_loss = bool(use_weak_q_loss)
        self.q_threshold = float(q_threshold)
        self.lambda_q = float(lambda_q)
        self.lambda_sparse = float(lambda_sparse)
        self.lambda_binary = float(lambda_binary)
        self.lambda_coverage = float(lambda_coverage)
        self.device = torch.device(device)
        self.q_init_mode = q_init_mode
        self.weak_q_loss_type = weak_q_loss_type

        if Q_noise is None:
            q_target = torch.zeros(
                (self.n_item, self.n_know), dtype=torch.float32, device=self.device
            )
            has_q_target = False
        else:
            q_target = _as_float_q(Q_noise, self.n_item, self.n_know, self.device)
            has_q_target = True
        self.has_q_target = has_q_target
        self.register_buffer("Q_noise", q_target)

        if q_init_mode == "fixed" and not learn_q:
            q_init_prob = self.Q_noise.float()
        elif q_init_mode in {"weak", "fixed"}:
            if Q_noise is None:
                raise ValueError(
                    'Q_noise is required when q_init_mode is "weak" or "fixed".'
                )
            q_init_prob = torch.where(
                self.Q_noise > 0.5,
                torch.full_like(self.Q_noise, float(q_init_high)),
                torch.full_like(self.Q_noise, float(q_init_low)),
            )
        else:
            q_init_prob = torch.empty(
                (self.n_item, self.n_know), dtype=torch.float32, device=self.device
            ).uniform_(float(q_init_low), float(q_init_high))

        if has_q_target:
            q_prior = torch.where(
                self.Q_noise > 0.5,
                torch.full_like(self.Q_noise, prior_high),
                torch.full_like(self.Q_noise, prior_low),
            )
        else:
            q_prior = torch.zeros_like(q_target)
        self.register_buffer("Q_init_prob", q_init_prob.detach().clone())
        self.register_buffer("Q_prior", q_prior)
        self.q_logits = nn.Parameter(_safe_logit(q_init_prob), requires_grad=learn_q)

        pos_weight_tensor = self._build_q_pos_weight(q_pos_weight)
        if pos_weight_tensor is None:
            self.q_pos_weight = None
        else:
            self.register_buffer("q_pos_weight", pos_weight_tensor)

        self.itf = self.ncd_func
        f_linear = nn.Linear if not monotonicity_assumption else PosLinear

        self.f_nn = nn.Sequential(
            OrderedDict(
                [
                    ("f_layer_1", f_linear(self.n_item, self.n_know)),
                    ("f_activate_1", nn.Sigmoid()),
                    ("f_layer_2", f_linear(self.n_know, self.n_know)),
                    ("f_activate_2", nn.Sigmoid()),
                ]
            )
        ).to(self.device)

        self.g_nn = nn.Sequential(
            OrderedDict(
                [
                    ("g_layer_1", nn.Linear(self.n_user, self.n_know)),
                    ("g_activate_1", nn.Sigmoid()),
                    ("g_layer_2", nn.Linear(self.n_know, self.n_know)),
                    ("g_activate_2", nn.Sigmoid()),
                    ("g_layer_3", nn.Linear(self.n_know, self.n_know)),
                    ("g_activate_3", nn.Sigmoid()),
                ]
            )
        ).to(self.device)

        self.theta_agg_mat = f_linear(self.n_know, self.user_dim).to(self.device)
        self.psi_agg_mat = nn.Linear(self.n_know, self.item_dim).to(self.device)

        self.ncd = nn.Sequential(
            OrderedDict(
                [
                    ("pred_layer_1", nn.Linear(self.user_dim, 64)),
                    ("pred_activate_1", nn.Sigmoid()),
                    ("pred_dropout_1", nn.Dropout(p=0.5)),
                    ("pred_layer_2", nn.Linear(64, 32)),
                    ("pred_activate_2", nn.Sigmoid()),
                    ("pred_dropout_2", nn.Dropout(p=0.5)),
                    ("pred_layer_3", nn.Linear(32, 1)),
                    ("pred_activate_3", nn.Sigmoid()),
                ]
            )
        ).to(self.device)

        self.register_buffer("Theta_buf", torch.zeros((self.n_user, self.n_know)))
        self.register_buffer("Psi_buf", torch.zeros((self.n_item, self.n_know)))

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def _build_q_pos_weight(
        self, q_pos_weight: Union[str, float, None]
    ) -> Optional[torch.Tensor]:
        if q_pos_weight is None:
            return None
        if isinstance(q_pos_weight, str):
            if q_pos_weight != "auto":
                raise ValueError('q_pos_weight must be "auto", a number, or None.')
            if not self.has_q_target:
                return None
            num_pos = torch.sum(self.Q_noise > 0.5).float()
            num_total = torch.tensor(float(self.Q_noise.numel()), device=self.device)
            num_neg = num_total - num_pos
            value = torch.where(num_pos > 0, num_neg / num_pos.clamp_min(1.0), 1.0)
            return value.reshape(1)
        return torch.tensor([float(q_pos_weight)], dtype=torch.float32, device=self.device)

    def get_q_soft(self) -> torch.Tensor:
        """Return differentiable Q_soft = sigmoid(S)."""
        if self.q_init_mode == "fixed" and not self.learn_q:
            return self.Q_noise.float()
        return torch.sigmoid(self.q_logits)

    def get_q_hard(self, threshold: Optional[float] = None) -> torch.Tensor:
        """Return a detached binary Q matrix for evaluation/export."""
        threshold_value = self.q_threshold if threshold is None else float(threshold)
        return (self.get_q_soft().detach() > threshold_value).to(torch.int64)

    def export_q_soft(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.get_q_soft().detach().cpu().numpy())

    def export_q_hard(self, path: Union[str, Path], threshold: Optional[float] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.get_q_hard(threshold=threshold).cpu().numpy())

    def ncd_func(self, theta: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        return self.ncd(theta - psi)

    def diagnose_theta(self, user_log: torch.Tensor) -> torch.Tensor:
        """Generate learner knowledge proficiency from response vectors."""
        user_log = user_log.to(self.device).float()
        q_soft = self.get_q_soft()
        theta_imp = self.f_nn(user_log)
        theta_exp = torch.sigmoid(user_log @ q_soft / (self.n_know**0.5))
        return theta_imp * (1.0 - self.alpha) + theta_exp * self.alpha

    def diagnose_psi(self, item_log: torch.Tensor) -> torch.Tensor:
        """Generate item features from item response vectors."""
        item_log = item_log.to(self.device).float()
        return self.g_nn(item_log)

    def diagnose_theta_psi(
        self, user_log: torch.Tensor, item_log: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        theta = self.diagnose_theta(user_log)
        psi = self.diagnose_psi(item_log)
        return theta, psi

    def predict_response(
        self, theta: torch.Tensor, psi: torch.Tensor, q_batch: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct response probability using the soft-Q item mask."""
        theta_agg = self.theta_agg_mat(theta * q_batch)
        psi_agg = self.psi_agg_mat(psi * q_batch)
        return self.itf(theta_agg, psi_agg)

    def _flat_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return ids.to(self.device).long().reshape(-1)

    def forward(
        self,
        user_log: torch.Tensor,
        item_log: torch.Tensor,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
    ) -> torch.Tensor:
        del user_id  # Kept for API compatibility with the original G-NCDM.
        theta, psi = self.diagnose_theta_psi(user_log, item_log)
        q_batch = self.get_q_soft()[self._flat_ids(item_id)]
        return self.predict_response(theta, psi, q_batch)

    def forward_using_buf(self, user_id: torch.Tensor, item_id: torch.Tensor) -> torch.Tensor:
        user_idx = self._flat_ids(user_id)
        item_idx = self._flat_ids(item_id)
        theta = self.Theta_buf[user_idx]
        psi = self.Psi_buf[item_idx]
        q_batch = self.get_q_soft()[item_idx]
        return self.predict_response(theta, psi, q_batch)

    @torch.no_grad()
    def update_Theta_buf(self, theta_new: torch.Tensor, user_id: torch.Tensor) -> None:
        self.Theta_buf[self._flat_ids(user_id)] = theta_new.detach().to(self.device)

    @torch.no_grad()
    def update_Psi_buf(self, psi_new: torch.Tensor, item_id: torch.Tensor) -> None:
        self.Psi_buf[self._flat_ids(item_id)] = psi_new.detach().to(self.device)

    def get_Theta_buf(self) -> torch.Tensor:
        return self.Theta_buf.detach().cpu()

    def get_Psi_buf(self) -> torch.Tensor:
        return self.Psi_buf.detach().cpu()

    def compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return total loss and every component needed by training logs."""
        y_true = y_true.to(self.device).float().reshape_as(y_pred)
        loss_rec = F.binary_cross_entropy(y_pred, y_true)

        q_soft = self.get_q_soft()
        zero = torch.zeros((), dtype=loss_rec.dtype, device=self.device)

        if self.use_weak_q_loss and self.has_q_target:
            if self.weak_q_loss_type == "hard_bce":
                loss_weak_q = F.binary_cross_entropy_with_logits(
                    self.q_logits,
                    self.Q_noise.float(),
                    pos_weight=self.q_pos_weight,
                )
            elif self.weak_q_loss_type == "soft_bce":
                loss_weak_q = F.binary_cross_entropy_with_logits(
                    self.q_logits,
                    self.Q_prior,
                    pos_weight=self.q_pos_weight,
                )
            elif self.weak_q_loss_type in {"mse", "gaussian"}:
                loss_weak_q = F.mse_loss(q_soft, self.Q_prior)
            elif self.weak_q_loss_type == "smooth_l1":
                loss_weak_q = F.smooth_l1_loss(q_soft, self.Q_prior)
            else:
                loss_weak_q = F.l1_loss(q_soft, self.Q_prior)
        else:
            loss_weak_q = zero

        loss_sparse = torch.mean(q_soft)
        loss_binary = torch.mean(q_soft * (1.0 - q_soft))
        loss_coverage = torch.mean(F.relu(1.0 - torch.sum(q_soft, dim=1)))

        loss_total = (
            loss_rec
            + self.lambda_q * loss_weak_q
            + self.lambda_sparse * loss_sparse
            + self.lambda_binary * loss_binary
            + self.lambda_coverage * loss_coverage
        )
        return {
            "loss_total": loss_total,
            "loss_rec": loss_rec,
            "loss_weak_q": loss_weak_q,
            "loss_sparse": loss_sparse,
            "loss_binary": loss_binary,
            "loss_coverage": loss_coverage,
        }

    def forward_with_loss(
        self,
        user_log: torch.Tensor,
        item_log: torch.Tensor,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        y_true: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        y_pred = self.forward(user_log, item_log, user_id, item_id)
        return y_pred, self.compute_loss(y_pred, y_true)
