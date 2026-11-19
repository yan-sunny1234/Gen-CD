# -*- coding: utf-8 -*-
"""Train-only evidence builders for the wqcl ten-point screening round.

Everything here reads TRAIN data only. Valid/test labels never enter any
statistic produced by this module. The cross-fitted Rasch teacher, edge
confidence statistics, concept relation pairs, and anchors defined here are
shared by innovation points D1/D3 (relation pairs), 7 (edge confidence) and
9 (attribution anchors).

This is a stage-1 screening implementation: statistics are simple, clearly
labelled approximations, and every function records its own coverage so that
"not enough evidence" is visible instead of silently producing labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# basic matrices
# ---------------------------------------------------------------------------

def build_observation(rows, n_students: int, n_items: int):
    """rows: iterable of (u, i, score). Returns y in {0,1} and mask m in {0,1}."""
    y = np.zeros((n_students, n_items), dtype=np.float32)
    m = np.zeros((n_students, n_items), dtype=np.float32)
    for u, i, s in rows:
        y[u, i] = float(s)
        m[u, i] = 1.0
    return y, m


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


# ---------------------------------------------------------------------------
# cross-fitted Rasch teacher (screening-grade alternating fit)
# ---------------------------------------------------------------------------

@dataclass
class RaschTeacher:
    """Cross-fitted 1-D Rasch teacher p_hat = sigmoid(a_i - b_j).

    b used for student i comes from folds != fold(i); a_i is fitted on the
    student's own training responses with that out-of-fold b. The teacher is
    a screening tool for removing obvious difficulty effects, not a
    calibrated measurement model.
    """

    a: np.ndarray                     # (N,)
    b_by_fold: np.ndarray             # (F, J)
    fold: np.ndarray                  # (N,)
    p_hat: np.ndarray                 # (N, J) using out-of-fold b
    fit_log: Dict = field(default_factory=dict)

    def b_for_student(self, fold_id: int) -> np.ndarray:
        others = [f for f in range(self.b_by_fold.shape[0]) if f != fold_id]
        return self.b_by_fold[others].mean(axis=0)


def _fit_a_given_b(y: np.ndarray, m: np.ndarray, b: np.ndarray, reg: float = 0.5,
                   n_iter: int = 20) -> np.ndarray:
    a = np.zeros(y.shape[0], dtype=np.float64)
    for _ in range(n_iter):
        z = a[:, None] - b[None, :]
        p = _sigmoid(z)
        g = (m * (p - y)).sum(axis=1) + reg * a
        h = (m * p * (1 - p)).sum(axis=1) + reg
        a -= g / np.maximum(h, 1e-6)
    return a


def _fit_b_given_a(y: np.ndarray, m: np.ndarray, a: np.ndarray, reg: float = 0.5,
                   n_iter: int = 20) -> np.ndarray:
    b = np.zeros(y.shape[1], dtype=np.float64)
    for _ in range(n_iter):
        z = a[:, None] - b[None, :]
        p = _sigmoid(z)
        g = (m * (p - y)).sum(axis=0) + reg * b
        h = (m * p * (1 - p)).sum(axis=0) + reg
        b += g / np.maximum(h, 1e-6)
    return b


def fit_rasch_teacher(rows, n_students: int, n_items: int, n_folds: int = 3,
                      seed: int = 2026, rounds: int = 4) -> RaschTeacher:
    y, m = build_observation(rows, n_students, n_items)
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, n_folds, size=n_students)
    y64, m64 = y.astype(np.float64), m.astype(np.float64)

    # difficulty initialisation from global correct rates (per fold later)
    obs = m64.sum(axis=0)
    corr = y64.sum(axis=0)
    rate = np.clip(corr / np.maximum(obs, 1.0), 1e-3, 1.0 - 1e-3)
    b0 = np.log(1.0 / rate - 1.0)

    b_by_fold = np.zeros((n_folds, n_items), dtype=np.float64)
    a_by_fold = np.zeros((n_folds, n_students), dtype=np.float64)
    for f in range(n_folds):
        idx = np.flatnonzero(fold != f)
        bf, af = b0.copy(), np.zeros(len(idx), dtype=np.float64)
        for _ in range(rounds):
            af = _fit_a_given_b(y64[idx], m64[idx], bf)
            bf = _fit_b_given_a(y64[idx], m64[idx], af)
        b_by_fold[f], a_by_fold[f] = bf, 0.0  # a stored positionally later
        a_by_fold[f, :len(idx)] = af

    # a_i with out-of-fold difficulty; p_hat accordingly
    a = np.zeros(n_students, dtype=np.float64)
    p_hat = np.zeros((n_students, n_items), dtype=np.float64)
    for f in range(n_folds):
        idx = np.flatnonzero(fold == f)
        if not len(idx):
            continue
        b_out = b_by_fold[[g for g in range(n_folds) if g != f]].mean(axis=0)
        a[idx] = _fit_a_given_b(y64[idx], m64[idx], b_out)
        p_hat[idx] = _sigmoid(a[idx][:, None] - b_out[None, :])
    return RaschTeacher(a=a, b_by_fold=b_by_fold, fold=fold, p_hat=p_hat,
                        fit_log={"n_folds": n_folds, "rounds": rounds,
                                 "mean_abs_b": float(np.abs(b_by_fold).mean())})


# ---------------------------------------------------------------------------
# edge confidence statistics (point 7)
# ---------------------------------------------------------------------------

def single_concept_anchor_items(q: np.ndarray) -> Dict[int, List[int]]:
    """concept -> items whose reference row has exactly one concept (that one)."""
    anchors: Dict[int, List[int]] = {}
    row_sum = q.sum(axis=1)
    for j in range(q.shape[0]):
        if row_sum[j] == 1:
            k = int(np.argmax(q[j]))
            anchors.setdefault(k, []).append(j)
    return anchors


def edge_confidence_c0(q: np.ndarray) -> np.ndarray:
    """Specificity prior: 1/row_concept_count, broadcast over the row."""
    row_sum = np.maximum(q.sum(axis=1, keepdims=True), 1.0)
    return (np.ones_like(q, dtype=np.float64) / row_sum).astype(np.float64)


def edge_confidence_c1(rows, q: np.ndarray, teacher: RaschTeacher,
                       n_students: int, n_items: int,
                       min_group: int = 8, shrink: int = 20,
                       use_raw_y: bool = False) -> Tuple[np.ndarray, Dict]:
    """Anchor-residual discrimination confidence.

    For edge (j, k): among students who answered item j and have at least one
    observed single-concept anchor of concept k, does the anchor-calibrated
    concept proxy g_ik rank their outcome on j? Confidence = shrunk |2AUC-1|.
    use_raw_y=True compares raw correctness within item j (item difficulty
    cancels because the comparison is inside a single item).
    """
    y, m = build_observation(rows, n_students, n_items)
    anchors = single_concept_anchor_items(q)
    outcome = y if use_raw_y else m * (y - teacher.p_hat.astype(np.float32))
    c = np.full(q.shape, 0.5, dtype=np.float64)
    stats = {"edges_evaluated": 0, "edges_skipped_lowdata": 0,
             "anchor_items_total": int(sum(len(v) for v in anchors.values()))}
    for k, anchor_items in anchors.items():
        if not anchor_items:
            continue
        mask_anchor = np.zeros((n_students, n_items), dtype=np.float32)
        mask_anchor[:, anchor_items] = 1.0
        n_anchor_obs = (mask_anchor * m).sum(axis=1)
        denom = np.maximum(n_anchor_obs, 1.0)
        g_ik = (mask_anchor * outcome if not use_raw_y
                else mask_anchor * y).sum(axis=1) / denom      # concept proxy
        has_proxy = n_anchor_obs >= 1
        for j in range(n_items):
            answered = np.flatnonzero(m[:, j] > 0)
            pool = answered[has_proxy[answered]]
            if len(pool) < 2 * min_group:
                stats["edges_skipped_lowdata"] += 1
                continue
            gj = g_ik[pool]
            rj = outcome[pool, j]
            med = np.median(gj)
            hi, lo = gj >= med, gj < med
            n_hi, n_lo = int(hi.sum()), int(lo.sum())
            if min(n_hi, n_lo) < min_group:
                stats["edges_skipped_lowdata"] += 1
                continue
            order = np.argsort(np.concatenate([rj[lo], rj[hi]]))
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, len(order) + 1)
            auc = (ranks[len(lo):].sum() - n_hi * (n_hi + 1) / 2.0) / (n_hi * n_lo)
            raw = abs(2.0 * auc - 1.0)
            n = n_hi + n_lo
            c[j, k] = 0.5 + 0.5 * raw * n / (n + shrink)
            stats["edges_evaluated"] += 1
    return c, stats


def edge_confidence_c2(rows, q: np.ndarray, teacher: RaschTeacher,
                       n_students: int, n_items: int,
                       min_co: int = 10) -> Tuple[np.ndarray, Dict]:
    """Item-pair residual-correlation confidence (de la Torre / Chiu flavoured).

    c2_jk = shrunk positive part of corr(residual_j, anchor-mean-residual_k)
            minus corr(residual_j, non-k mean residual), among co-answerers.
    """
    y, m = build_observation(rows, n_students, n_items)
    anchors = single_concept_anchor_items(q)
    residual = m * (y - teacher.p_hat.astype(np.float32))
    c = np.full(q.shape, 0.5, dtype=np.float64)
    stats = {"edges_evaluated": 0, "edges_skipped_lowdata": 0}

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() < 1e-8 or b.std() < 1e-8:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    for k in range(q.shape[1]):
        anchor_items = anchors.get(k, [])
        if not anchor_items:
            continue
        mask_anchor = np.zeros((n_students, n_items), dtype=np.float32)
        mask_anchor[:, anchor_items] = 1.0
        n_obs = (mask_anchor * m).sum(axis=1)
        anchor_mean = (mask_anchor * residual).sum(axis=1) / np.maximum(n_obs, 1.0)
        valid_anchor = n_obs >= 1
        for j in range(n_items):
            if q[j, k] == 0 and anchors.get(k):
                pass  # still compute: confidence is informative for 0-edges too
            both = np.flatnonzero((m[:, j] > 0) & valid_anchor)
            if len(both) < min_co:
                stats["edges_skipped_lowdata"] += 1
                continue
            r_j = residual[both, j]
            r_anchor = anchor_mean[both] - (
                (m[both, j] * residual[both, j]) / np.maximum(n_obs[both], 1.0) * 0
            )  # anchor mean already excludes j because j is (usually) not an anchor
            corr_pos = _corr(r_j, r_anchor)
            # non-k baseline: mean residual on items not anchored by k and not j
            nonk_items = [i2 for i2 in range(n_items)
                          if i2 != j and (len(anchor_items) == 0 or i2 not in anchor_items)]
            if nonk_items:
                mask_nonk = np.zeros((n_students, n_items), dtype=np.float32)
                mask_nonk[:, nonk_items] = 1.0
                n_nonk = (mask_nonk * m).sum(axis=1)
                nonk_mean = (mask_nonk * residual).sum(axis=1) / np.maximum(n_nonk, 1.0)
                both2 = np.flatnonzero((m[:, j] > 0) & (n_nonk >= 1))
                corr_neg = _corr(residual[both2, j], nonk_mean[both2]) if len(both2) >= min_co else 0.0
            else:
                corr_neg = 0.0
            c[j, k] = 0.5 + 0.5 * np.tanh(2.0 * (corr_pos - corr_neg))
            stats["edges_evaluated"] += 1
    return c, stats


def confidence_vs_noise_auc(c: np.ndarray, noise_info: Dict) -> Dict[str, float]:
    """Directional noise tracking (synthetic only).

    Deleted (missed) true edges should have HIGH confidence; added (false)
    edges should have LOW confidence. Reported separately because pooling
    the two directions cancels the signal.
    """
    deleted = {tuple(e) for e in noise_info.get("deleted_edges", [])}
    added = {tuple(e) for e in noise_info.get("added_edges", [])}
    flipped = {tuple(e[:2]) for e in noise_info.get("flipped_edges", [])}
    J, K = c.shape
    flat = c.reshape(-1)

    def _auc(pos_set, high_means_positive: bool) -> Tuple[float, int]:
        if not pos_set or len(pos_set) == J * K:
            return float("nan"), len(pos_set)
        labels = np.zeros(J * K, dtype=np.int64)
        for (j, k) in pos_set:
            labels[j * K + k] = 1
        n_pos = int(labels.sum())
        n_neg = labels.size - n_pos
        score = flat if high_means_positive else -flat
        order = np.argsort(score)
        lab = labels[order]
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, lab.size + 1)
        auc = (ranks[lab == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        return float(auc), n_pos

    del_set = deleted | flipped     # flipped edges were true edges too
    auc_del, n_del = _auc(del_set, high_means_positive=True)
    auc_add, n_add = _auc(added, high_means_positive=False)
    return {"auc_deleted_high_c": auc_del, "n_deleted": n_del,
            "auc_added_low_c": auc_add, "n_added": n_add}


# ---------------------------------------------------------------------------
# concept relation pairs (points D1 / D3)
# ---------------------------------------------------------------------------

@dataclass
class RelationPairs:
    positives: Dict[int, List[Tuple[int, int]]]
    negatives: Dict[int, List[Tuple[int, int]]]
    coverage: Dict[str, float]


def build_relation_pairs_batch(y: np.ndarray, m: np.ndarray, p_hat: np.ndarray,
                               q_ref: np.ndarray, a: np.ndarray,
                               delta_p: float = 0.10, delta_n: float = 0.35,
                               min_anchor_obs: int = 2, max_pos: int = 2,
                               max_neg: int = 6, seed: int = 0
                               ) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict]:
    """Vectorised per-batch pair construction with local row indices.

    Screening thresholds use point estimates of the anchor-residual
    difference (the normal-approximation interval is too wide when a
    student has only 2-4 anchor items); the evidence-count gate stays.
    y/m: (B, J) support matrices; p_hat: (B, J); a: (B,).
    """
    rng = np.random.default_rng(seed)
    residuals = m * (y - p_hat)
    anchors = single_concept_anchor_items(q_ref)
    pairs: Dict[int, Dict[str, np.ndarray]] = {}
    stats = {"concepts_with_pairs": 0, "n_pos": 0, "n_neg": 0}
    for k, anchor_items in anchors.items():
        if not anchor_items:
            continue
        mask_anchor = np.zeros(m.shape, dtype=np.float32)
        mask_anchor[:, anchor_items] = 1.0
        n_obs = (mask_anchor * m).sum(axis=1)
        elig = np.flatnonzero(n_obs >= min_anchor_obs)
        if len(elig) < 4:
            continue
        t = (mask_anchor[elig] * residuals[elig]).sum(axis=1) / n_obs[elig]
        med = np.median(a[elig])
        d = t[:, None] - t[None, :]
        same_ability = ((a[elig] > med)[:, None] == (a[elig] > med)[None, :])
        pos_mask = same_ability & (np.abs(d) <= delta_p)
        neg_mask = same_ability & (np.abs(d) >= delta_n)
        np.fill_diagonal(pos_mask, False)
        np.fill_diagonal(neg_mask, False)
        pos_idx = np.argwhere(pos_mask)
        neg_idx = np.argwhere(neg_mask)
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        if len(pos_idx) > max_pos * len(elig):
            pos_idx = pos_idx[rng.choice(len(pos_idx), max_pos * len(elig), replace=False)]
        if len(neg_idx) > max_neg * len(elig):
            neg_idx = neg_idx[rng.choice(len(neg_idx), max_neg * len(elig), replace=False)]
        # far-student lists per anchor row, then one negative per positive
        far: Dict[int, List[int]] = {}
        for r, c in neg_idx:
            far.setdefault(int(r), []).append(int(c))
        pos_rows, neg_rows = [], []
        for r in range(len(pos_idx)):
            anchor_row = int(pos_idx[r, 0])
            candidates = far.get(anchor_row)
            if not candidates:
                continue
            v_row = candidates[r % len(candidates)]
            pos_rows.append([elig[pos_idx[r, 0]], elig[pos_idx[r, 1]]])
            neg_rows.append([elig[anchor_row], elig[v_row]])
        if not pos_rows:
            continue
        pairs[k] = {"pos": np.asarray(pos_rows, dtype=np.int64),
                    "neg": np.asarray(neg_rows, dtype=np.int64)}
        stats["concepts_with_pairs"] += 1
        stats["n_pos"] += len(pos_rows)
        stats["n_neg"] += len(neg_rows)
    return pairs, stats


# ---------------------------------------------------------------------------
# misc helpers
# ---------------------------------------------------------------------------

def dump_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(str(type(o)))
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_default),
                    encoding="utf-8")
