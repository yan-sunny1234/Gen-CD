# -*- coding: utf-8 -*-
"""Model variants and losses for the wqcl ten-point screening round.

Every variant subclasses WeakQGNCDM (or operates on it through free
functions) so the base forward path stays byte-identical with the historic
model. Nothing in the five hash-pinned shared library files is modified.

Screening-grade implementations of:
  - TRQ  (point 8): trust-region Q perturbation consistency
  - ECWQ (point 7): edge-confidence weighted weak prior
  - SACA (point 9): single-anchor concept attribution (row-mass preserving,
    exactly equivalent to the baseline at zero logits)
  - D5   (posterior Q): asymmetric noise-channel variational Q
  - E3A  (point 10): evidence-adaptive shrinkage toward an analytic anchor
  - free helpers for D1/D3 (relation NCE on theta / residual coordinates)
  - free helpers for D4 (pre-sigmoid edit effects with Q-only aux gradient)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Q_GNCDM.models.weak_q_gncdm import WeakQGNCDM  # noqa: E402


# ---------------------------------------------------------------------------
# free forward helpers (never modify the model in place)
# ---------------------------------------------------------------------------

def diagnose_theta_with_q(model: WeakQGNCDM, user_log: torch.Tensor,
                          Q: torch.Tensor) -> torch.Tensor:
    """Explicit-branch diagnose with an externally supplied Q matrix."""
    user_log = user_log.to(model.device).float()
    theta_imp = model.f_nn(user_log)
    theta_exp = torch.sigmoid(user_log @ Q / (model.n_know ** 0.5))
    return theta_imp * (1.0 - model.alpha) + theta_exp * model.alpha


def response_logit(model: WeakQGNCDM, theta: torch.Tensor, psi: torch.Tensor,
                   q_batch: torch.Tensor) -> torch.Tensor:
    """Pre-sigmoid response logit. model.ncd ends with Sigmoid; strip it."""
    theta_agg = model.theta_agg_mat(theta * q_batch)
    psi_agg = model.psi_agg_mat(psi * q_batch)
    return model.ncd[:-1](theta_agg - psi_agg)


# ---------------------------------------------------------------------------
# point 8: TRQ
# ---------------------------------------------------------------------------

def trq_perturb(model: WeakQGNCDM, mode: str, delta: float,
                loss_pred: Optional[torch.Tensor] = None,
                lo: float = 0.05, hi: float = 0.95) -> torch.Tensor:
    """Sample a trust-region perturbation of the current soft Q."""
    Q = model.get_q_soft()
    if mode == "random":
        U = torch.empty_like(Q).uniform_(-delta, delta)
    elif mode == "adv":
        if loss_pred is None:
            raise ValueError("adversarial TRQ needs the prediction loss")
        g = torch.autograd.grad(loss_pred, model.q_logits, retain_graph=True)[0]
        U = delta * torch.sign(g)
    else:
        raise ValueError(mode)
    return torch.clamp(Q + U, lo, hi)


def trq_prediction(model: WeakQGNCDM, clean: torch.Tensor, users: torch.Tensor,
                   query_items: torch.Tensor, psi: torch.Tensor,
                   Q_tilde: torch.Tensor, path: str = "both") -> torch.Tensor:
    """Forward pass under a perturbed Q. path in {both, exp, mask}."""
    if path in ("both", "exp"):
        theta_t = diagnose_theta_with_q(model, clean, Q_tilde)
    else:
        theta_t = model.diagnose_theta(clean)
    if path in ("both", "mask"):
        q_batch = Q_tilde[query_items]
    else:
        q_batch = model.get_q_soft()[query_items]
    return model.predict_response(theta_t[users], psi, q_batch)


# ---------------------------------------------------------------------------
# point 7: ECWQ
# ---------------------------------------------------------------------------

class ECWQModel(WeakQGNCDM):
    """WeakQGNCDM whose weak-prior loss is weighted per edge by confidence."""

    def __init__(self, *args, edge_conf: Optional[np.ndarray] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if edge_conf is not None:
            conf = torch.as_tensor(edge_conf, dtype=torch.float32,
                                   device=self.device)
            if conf.shape != (self.n_item, self.n_know):
                raise ValueError("edge_conf shape mismatch")
            self.register_buffer("q_conf", conf)
        else:
            self.q_conf = None

    def compute_loss(self, y_pred, y_true) -> Dict[str, torch.Tensor]:
        y_true = y_true.to(self.device).float().reshape_as(y_pred)
        loss_rec = F.binary_cross_entropy(y_pred, y_true)
        q_soft = self.get_q_soft()
        zero = torch.zeros((), dtype=loss_rec.dtype, device=self.device)

        if self.use_weak_q_loss and self.has_q_target:
            if self.weak_q_loss_type in {"mse", "gaussian"}:
                elem = (q_soft - self.Q_prior) ** 2
            elif self.weak_q_loss_type == "smooth_l1":
                elem = F.smooth_l1_loss(q_soft, self.Q_prior, reduction="none")
            elif self.weak_q_loss_type == "l1":
                elem = (q_soft - self.Q_prior).abs()
            elif self.weak_q_loss_type == "hard_bce":
                elem = F.binary_cross_entropy_with_logits(
                    self.q_logits, self.Q_noise.float(),
                    pos_weight=self.q_pos_weight, reduction="none")
            elif self.weak_q_loss_type == "soft_bce":
                elem = F.binary_cross_entropy_with_logits(
                    self.q_logits, self.Q_prior,
                    pos_weight=self.q_pos_weight, reduction="none")
            else:
                raise ValueError(self.weak_q_loss_type)
            if self.q_conf is not None:
                loss_weak_q = (elem * self.q_conf).sum() / self.q_conf.sum()
            else:
                loss_weak_q = elem.mean()
        else:
            loss_weak_q = zero

        loss_sparse = torch.mean(q_soft)
        loss_binary = torch.mean(q_soft * (1.0 - q_soft))
        loss_coverage = torch.mean(F.relu(1.0 - torch.sum(q_soft, dim=1)))
        loss_total = (loss_rec + self.lambda_q * loss_weak_q
                      + self.lambda_sparse * loss_sparse
                      + self.lambda_binary * loss_binary
                      + self.lambda_coverage * loss_coverage)
        return {"loss_total": loss_total, "loss_rec": loss_rec,
                "loss_weak_q": loss_weak_q, "loss_sparse": loss_sparse,
                "loss_binary": loss_binary, "loss_coverage": loss_coverage}


# ---------------------------------------------------------------------------
# point 9: SACA
# ---------------------------------------------------------------------------

class SACAModel(WeakQGNCDM):
    """Attribution inside the explicit branch.

    g_jk = A_jk * |S_j| with A = masked_softmax(attn_logits) over the
    reference support S_j. At zero logits A is uniform, g == 1 on the
    support and theta_exp == sigmoid(R @ Q_soft / sqrt(K)) exactly, i.e. the
    model degrades to its baseline. The IRF mask still uses the learnable
    Q_soft; attribution only reroutes evidence inside the explicit branch.
    """

    def __init__(self, *args, reference_q: Optional[np.ndarray] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if reference_q is None:
            reference_q = np.ones((self.n_item, self.n_know), dtype=np.float32)
        ref = torch.as_tensor(reference_q, dtype=torch.float32,
                              device=self.device)
        support = (ref > 0.5).float()
        self.register_buffer("attn_mask", support)
        self.attn_logits = nn.Parameter(torch.zeros(self.n_item, self.n_know))

    def gain_matrix(self) -> torch.Tensor:
        """Full-matrix multiplicative gain. At zero logits it is exactly 1
        everywhere, so diagnose_theta reproduces the baseline bit-for-bit
        (the off-support background of the soft Q is preserved); inside the
        support the gain A*|S_j| redistributes attribution."""
        masked = self.attn_logits.masked_fill(self.attn_mask == 0, -1e9)
        A = torch.softmax(masked, dim=1)
        row_size = self.attn_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        g_support = A * row_size                      # ==1 at zero logits
        return 1.0 + self.attn_mask * (g_support - 1.0)

    def diagnose_theta(self, user_log: torch.Tensor) -> torch.Tensor:
        user_log = user_log.to(self.device).float()
        Q_eff = self.get_q_soft() * self.gain_matrix()
        theta_imp = self.f_nn(user_log)
        theta_exp = torch.sigmoid(user_log @ Q_eff / (self.n_know ** 0.5))
        return theta_imp * (1.0 - self.alpha) + theta_exp * self.alpha

    def attribution_regularizer(self, weight: float = 1e-3) -> torch.Tensor:
        """Mild pull keeping gains from collapsing to a single concept."""
        g = self.gain_matrix()
        return weight * torch.mean((g - 1.0) ** 2 * self.attn_mask)


# ---------------------------------------------------------------------------
# D5: asymmetric noise-channel variational Q
# ---------------------------------------------------------------------------

class PosteriorQModel(WeakQGNCDM):
    """Variational Q with asymmetric annotation channel.

    pi = sigmoid(q_logits) initialised at 0.95/0.05 from the reference Q.
    Training samples one global binary Z per batch (Concrete straight-through,
    T=1) used by BOTH the explicit branch and the IRF mask; eval uses the
    mean pi forward (reported as an approximation, never mixed into training
    semantics). The uniform MSE weak prior is replaced by the analytic
    annotation term L_A plus a Bernoulli KL against a fixed neutral prior.
    """

    def __init__(self, *args, noise_f: float = 0.10, noise_n: float = 0.10,
                 **kwargs):
        kwargs["q_init_high"] = 0.95                  # force the strong start;
        kwargs["q_init_low"] = 0.05                   # args must not override
        kwargs.setdefault("weak_q_loss_type", "mse")
        super().__init__(*args, **kwargs)
        self.use_weak_q_loss = False          # replaced by L_A + KL
        self.register_buffer("noise_f", torch.tensor(float(noise_f)))
        self.register_buffer("noise_n", torch.tensor(float(noise_n)))
        self._Z: Optional[torch.Tensor] = None
        self._use_pi = False
        self.last_sample_diff = 0.0

    def begin_batch_sample(self) -> None:
        a = self.q_logits
        u = torch.rand_like(a).clamp(1e-6, 1.0 - 1e-6)
        gumbel = torch.log(u) - torch.log(1.0 - u)
        z_h = ((a + gumbel) > 0).float()
        z_s = torch.sigmoid(a + gumbel)                      # T = 1
        self._Z = z_h - z_s.detach() + z_s                   # straight-through
        self.last_sample_diff = float(
            (z_h != (torch.sigmoid(a) > 0.5).float()).float().mean())

    def end_batch_sample(self) -> None:
        self._Z = None

    def use_mean_pi(self, flag: bool = True) -> None:
        self._use_pi = flag

    def get_q_soft(self) -> torch.Tensor:
        if self._use_pi:
            return torch.sigmoid(self.q_logits)
        if self._Z is not None:
            return self._Z
        return torch.sigmoid(self.q_logits)

    def annotation_losses(self) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-7
        pi = torch.sigmoid(self.q_logits).clamp(eps, 1 - eps)
        A = self.Q_noise.float()
        f, n = self.noise_f.clamp(eps, 1 - eps), self.noise_n.clamp(eps, 1 - eps)
        term = pi * (A * torch.log(1 - n) + (1 - A) * torch.log(n)) \
             + (1 - pi) * (A * torch.log(f) + (1 - A) * torch.log(1 - f))
        L_A = -term.sum()
        kl = (pi * torch.log(pi / 0.5) + (1 - pi) * torch.log((1 - pi) / 0.5)).sum()
        return L_A, kl


# ---------------------------------------------------------------------------
# point 10: E3A
# ---------------------------------------------------------------------------

class E3AModel(WeakQGNCDM):
    """Evidence-adaptive shrinkage toward an analytic ability anchor.

    theta_exp_i = anchor_i + beta_i * (sigmoid(R Q / sqrt K)_i - anchor_i)
    alpha_i     = alpha_max * n_total_i / (n_total_i + n0)
    beta_i      = n_min_i / (n_min_i + n0)   (n_min = weakest concept evidence)

    n_eff -> infinity recovers the baseline exactly; n_eff -> 0 collapses to
    the anchor for every student (the "empty learner" point of the paper).
    """

    def __init__(self, *args, anchor_theta: Optional[np.ndarray] = None,
                 n0: float = 5.0, ref_q_for_counts: Optional[np.ndarray] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if anchor_theta is None:
            anchor_theta = np.full((self.n_user,), 0.5, dtype=np.float32)
        anchor = torch.as_tensor(anchor_theta, dtype=torch.float32,
                                 device=self.device).reshape(-1, 1)
        self.register_buffer("anchor", anchor)
        self.n0 = float(n0)
        if ref_q_for_counts is None:
            ref_q_for_counts = np.ones((self.n_item, self.n_know), dtype=np.float32)
        self.register_buffer("ref_q_counts",
                             torch.as_tensor(ref_q_for_counts, dtype=torch.float32))

    def diagnose_theta_indexed(self, user_log: torch.Tensor,
                               user_global_ids: torch.Tensor) -> torch.Tensor:
        user_log = user_log.to(self.device).float()
        Q = self.get_q_soft()
        theta_imp = self.f_nn(user_log)
        theta_exp = torch.sigmoid(user_log @ Q / (self.n_know ** 0.5))
        anchor = self.anchor[self._flat_ids(user_global_ids)]            # (B,1)
        m = (user_log != 0).float()
        n_min = (m @ self.ref_q_counts).clamp(min=0.0).min(dim=1, keepdim=True).values
        beta = n_min / (n_min + self.n0)
        theta_exp = anchor + beta * (theta_exp - anchor)
        return theta_imp * (1.0 - self.alpha) + theta_exp * self.alpha

    def diagnose_theta(self, user_log: torch.Tensor) -> torch.Tensor:
        """Full-matrix path (evaluation): identity index mapping."""
        idx = torch.arange(user_log.shape[0], device=self.device)
        return self.diagnose_theta_indexed(user_log, idx)


# ---------------------------------------------------------------------------
# D1/D3: relation NCE on concept coordinates
# ---------------------------------------------------------------------------

def relation_nce(theta: torch.Tensor, concept_pairs: Dict[int, Dict[str, torch.Tensor]],
                 scale_k: Optional[Dict[int, float]] = None, tau: float = 0.2
                 ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Vectorised distance-kernel InfoNCE over (student, concept) pairs.

    concept_pairs[k] = {"pos": LongTensor (L,2), "neg": LongTensor (M,2)} with
    local batch row indices; built by build_relation_pairs_batch. Every
    positive (i, u) forms a one-vs-rest classification against all negatives
    of the same concept. theta carries gradient.
    """
    zero = theta.sum() * 0.0
    terms: List[torch.Tensor] = []
    n_pos_total = 0
    for k, pair in concept_pairs.items():
        pos, neg = pair.get("pos"), pair.get("neg")
        if pos is None or neg is None or len(pos) == 0 or len(neg) == 0:
            continue
        s = float((scale_k or {}).get(k, 1.0))
        idx_i, idx_u = pos[:, 0], pos[:, 1]
        d_pos = theta[idx_i, k] - theta[idx_u, k]
        scores_pos = -(d_pos ** 2) / (s ** 2 + 1e-6)              # (L,)
        d_neg = theta[pos[:, 0], k] - theta[neg[:, 0], k]          # anchor vs each neg
        scores_neg = -(d_neg ** 2) / (s ** 2 + 1e-6)               # (M,)

        L, M = scores_pos.shape[0], scores_neg.shape[0]
        logits = torch.zeros(L, 1 + M, device=theta.device, dtype=theta.dtype)
        logits[:, 0] = scores_pos
        # negative score for positive row r uses ITS anchor (pos[r,0])
        logits[:, 1:] = scores_neg.unsqueeze(0).expand(L, M)
        target = torch.zeros(L, dtype=torch.long, device=theta.device)
        terms.append(F.cross_entropy(logits, target))
        n_pos_total += L
    if not terms:
        return zero, {"n_pos_pairs": 0}
    return torch.stack(terms).mean(), {"n_pos_pairs": n_pos_total}


# ---------------------------------------------------------------------------
# D3: exact decomposition helpers
# ---------------------------------------------------------------------------

def residual_coordinates(user_log: torch.Tensor, Q: torch.Tensor,
                         p_hat_rows: torch.Tensor, kappa: float = 1.0
                         ) -> torch.Tensor:
    """z_ik = sum_j q_jk (y_ij - p_hat_ij) / (kappa + sum_j q_jk).

    user_log entries are +-1/0; convert back to y in {0,1} via (R+1)/2 on
    observed entries. p_hat_rows must be detached. Q carries gradient.
    """
    y01 = (user_log + 1.0) / 2.0
    m = (user_log != 0).float()
    resid = m * (y01 - p_hat_rows)
    num = resid @ Q
    den = kappa + m @ Q
    return num / den


def decomposition_check(user_log: torch.Tensor, Q: torch.Tensor,
                        p_hat_rows: torch.Tensor) -> Dict[str, float]:
    """Numerically verify G + U == R Q / sqrt(K)."""
    m = (user_log != 0).float()
    y01 = (user_log + 1.0) / 2.0
    U = (2.0 / math.sqrt(Q.shape[1])) * (m * (y01 - p_hat_rows)) @ Q
    G = (1.0 / math.sqrt(Q.shape[1])) * (m * (2.0 * p_hat_rows - 1.0)) @ Q
    direct = user_log @ Q / math.sqrt(Q.shape[1])
    err = float((U + G - direct).abs().max())
    return {"max_abs_decomposition_error": err}


# ---------------------------------------------------------------------------
# D4: edit-effect loss with Q-only auxiliary gradient
# ---------------------------------------------------------------------------

def edit_effect_loss(model: WeakQGNCDM, clean_np: np.ndarray,
                     episode, Q_ref: np.ndarray,
                     teacher_p_hat: Optional[np.ndarray],
                     temperature: float = 1.0,
                     max_edits: int = 8) -> Tuple[torch.Tensor, Dict]:
    """Single-edit effect ranking on hidden query probes.

    Selects up to `max_edits` students in the batch that (a) have a wrong
    (=-1) support entry and (b) have at least two hidden queries with
    reference-concept sets disjoint / overlapping the edited item's concepts.
    Returns (loss, stats); loss carries gradient through theta (and hence Q).
    The caller is responsible for adding only the Q-part of the gradient.
    """
    device = model.device
    ref = torch.as_tensor(Q_ref, dtype=torch.float32, device=device)
    edits: List[Tuple[int, int, int, int]] = []      # (local_i, item_j, probe_pos, probe_neg)
    students = episode.student_ids
    for li in range(len(students)):
        if len(edits) >= max_edits:
            break
        rows_i = np.flatnonzero(clean_np[li] == -1)
        if not len(rows_i):
            continue
        q_mask = episode.query_students == li
        q_items = episode.query_items[q_mask]
        if len(q_items) < 2:
            continue
        j_star = int(rows_i[0])
        src_concepts = set(np.flatnonzero(Q_ref[j_star] > 0.5))
        pos_probe = neg_probe = None
        for it in q_items:
            cs = set(np.flatnonzero(Q_ref[it] > 0.5))
            if cs & src_concepts and pos_probe is None and it != j_star:
                pos_probe = int(it)
            if not (cs & src_concepts) and neg_probe is None and it != j_star:
                neg_probe = int(it)
        if pos_probe is None or neg_probe is None:
            continue
        edits.append((li, j_star, pos_probe, neg_probe))
    stats = {"n_edits": len(edits)}
    if not edits:
        zero = torch.zeros((), device=device) * model.get_q_soft().sum()
        return zero, stats

    clean = torch.as_tensor(clean_np, dtype=torch.float32, device=device)
    plus_np = clean_np.copy()
    for (li, j_star, _, _) in edits:
        plus_np[li, j_star] = 1.0
    plus = torch.as_tensor(plus_np, dtype=torch.float32, device=device)

    # aux forwards use an eval-mode prediction head (no dropout noise); the
    # main BCE path keeps its usual behaviour.
    head_training = model.ncd.training
    model.ncd.eval()
    sign_flags: List[float] = []
    try:
        theta_minus = model.diagnose_theta(clean)
        theta_plus = model.diagnose_theta(plus)
        items_needed = sorted({p for (_, _, p, _) in edits} | {n for (_, _, _, n) in edits})
        item_ids, inverse = np.unique(np.asarray(items_needed), return_inverse=True)
        item_logs = torch.as_tensor(episode.support[:, item_ids].T.copy(),
                                    dtype=torch.float32, device=device)
        psi = model.diagnose_psi(item_logs)[torch.as_tensor(inverse, device=device)]
        pos_of = {int(it): k for k, it in enumerate(item_ids)}
        diffs = []
        for (li, j_star, lp, ln) in edits:
            q_l_p = model.get_q_soft()[pos_of[lp]].detach()
            q_l_n = model.get_q_soft()[pos_of[ln]].detach()
            d_pos = response_logit(model, theta_plus[li:li + 1], psi[pos_of[lp]:pos_of[lp] + 1],
                                   q_l_p.unsqueeze(0))
            d_neg = response_logit(model, theta_plus[li:li + 1], psi[pos_of[ln]:pos_of[ln] + 1],
                                   q_l_n.unsqueeze(0))
            b_pos = response_logit(model, theta_minus[li:li + 1], psi[pos_of[lp]:pos_of[lp] + 1],
                                   q_l_p.unsqueeze(0))
            b_neg = response_logit(model, theta_minus[li:li + 1], psi[pos_of[ln]:pos_of[ln] + 1],
                                   q_l_n.unsqueeze(0))
            effect_pos = (d_pos - b_pos).reshape(-1)[0]
            effect_neg = (d_neg - b_neg).reshape(-1)[0]
            diffs.append(effect_pos - effect_neg)
            stats.setdefault("d_pos_list", []).append(float(effect_pos))
            stats.setdefault("d_neg_list", []).append(float(effect_neg))
            sign_flags.append(1.0 if float(effect_pos - effect_neg) > 0 else 0.0)
        loss = F.softplus(torch.stack(diffs).reshape(-1) / temperature).mean()
    finally:
        if head_training:
            model.ncd.train()
    stats["sign_correct_frac"] = float(np.mean(sign_flags)) if sign_flags else 0.0
    return loss, stats
