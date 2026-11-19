# -*- coding: utf-8 -*-
"""Stage-1 screening runner for the ten wqcl innovation points (2026-11-08).

Single training seed, Math1 + synthetic, 40 epochs. Results are screening
evidence, NOT formal gains. Protocol otherwise follows the project standard:
episode query hiding on both sides, validation-AUC model selection, no test
selection. The five hash-pinned shared library files are imported, never
modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Q_GNCDM.experiments import build_wqcl_evidence as evi
from Q_GNCDM.experiments import run_simple_contrastive as core
from Q_GNCDM.experiments import run_weak_q_synthetic as legacy
from Q_GNCDM.experiments.wqcl_screen_lib import (
    E3AModel, ECWQModel, PosteriorQModel, SACAModel,
    decomposition_check, diagnose_theta_with_q, edit_effect_loss,
    relation_nce, residual_coordinates, response_logit, trq_perturb,
    trq_prediction,
)
from Q_GNCDM.models.weak_q_gncdm import WeakQGNCDM
from Q_GNCDM.simple_contrastive import make_student_episode

POINTS = ["p8_trq", "p6_train", "p7_ecwq", "p7_conf", "p9_saca", "p10_e3a",
          "d1_relation", "d2_item_struct", "d3_residual", "d4_edit_effect",
          "d5_posterior"]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_data(args):
    folder = Path(args.data_dir).resolve()
    if args.dataset == "math1":
        import argparse as _ap
        data = core.load_data(_ap.Namespace(dataset="math1", data_dir=str(folder)))
        data["reference_q"] = data["q"]
        data["eval_reference_q"] = data["q"]
        data["noise_info"] = None
        data["n_students"] = len(data["train"])
        return data
    metadata = legacy.read_json(folder / "metadata.json")
    rows = legacy.load_response_rows(folder / "response.csv")
    parts = legacy.split_rows(rows, args.split_seed)
    q_noise = np.load(folder / f"Q_noise_{legacy.rate_label(args.noise_rate)}.npy").astype(np.float32)
    q_true = np.load(folder / "Q_true.npy").astype(np.float32)
    all_rows = [r for part in parts for r in part]
    n_students = max(r[0] for r in all_rows) + 1
    return {
        "folder": folder,
        "rows": [np.asarray(p, dtype=np.int64) for p in parts],
        "train": legacy.build_log_mat(parts[0], n_students, len(q_noise)),
        "q": q_noise,
        "reference_q": q_noise,
        "eval_reference_q": q_true,
        "q_true": q_true,
        "n_students": n_students,
        "metadata": metadata,
        "noise_info": legacy.read_json(
            folder / f"noise_info_{legacy.rate_label(args.noise_rate)}.json"),
        "source_sha256": {},
    }


# ---------------------------------------------------------------------------
# model factory and conditions
# ---------------------------------------------------------------------------

def base_kwargs(data, args):
    return dict(
        num_students=len(data["train"]), num_items=len(data["q"]),
        num_concepts=data["q"].shape[1], user_dim=args.embedding_dim,
        item_dim=args.embedding_dim, alpha=args.alpha, Q_noise=data["q"],
        q_init_high=args.q_init_high, q_init_low=args.q_init_low,
        weak_q_loss_type="mse", lambda_q=args.lambda_q,
        lambda_sparse=0.0, lambda_binary=0.0,
        lambda_coverage=args.lambda_coverage,
        q_pos_weight=1.0, device=args.device,
    )


def build_model(point, condition, data, teacher, args):
    kw = base_kwargs(data, args)

    def merged(**over):
        """kw with overridden keys removed, then overrides applied (no dup)."""
        kk = {k: v for k, v in kw.items() if k not in over}
        return {**kk, **over}

    name = condition["name"]
    if point == "p6_train":
        if name == "fixed_binary":
            return WeakQGNCDM(**merged(learn_q=False, use_weak_q_loss=False,
                                       q_init_mode="fixed", lambda_q=0.0,
                                       lambda_coverage=0.0, q_pos_weight=None))
        if name == "fixed_soft":
            return WeakQGNCDM(**merged(learn_q=False, use_weak_q_loss=False,
                                       q_init_mode="weak", q_init_high=0.95,
                                       q_init_low=0.05, lambda_q=0.0,
                                       lambda_coverage=0.0, q_pos_weight=None))
        return WeakQGNCDM(**merged(learn_q=True, use_weak_q_loss=True,
                                   q_init_mode="weak"))
    if point == "p8_trq":
        return WeakQGNCDM(**merged(learn_q=True, use_weak_q_loss=True,
                                   q_init_mode="weak"))
    if point == "p7_ecwq":
        if name == "ecwq_c1":
            return ECWQModel(**merged(learn_q=True, use_weak_q_loss=True,
                                      q_init_mode="weak",
                                      edge_conf=condition["edge_conf"]))
        return WeakQGNCDM(**merged(learn_q=True, use_weak_q_loss=True,
                                   q_init_mode="weak"))
    if point == "p9_saca":
        return SACAModel(**merged(learn_q=True, use_weak_q_loss=True,
                                  q_init_mode="weak",
                                  reference_q=data["reference_q"]))
    if point == "p10_e3a":
        if name == "e3a_shrink":
            anchor = evi._sigmoid(teacher.a - teacher.a.mean()).astype(np.float32)
            return E3AModel(**merged(learn_q=True, use_weak_q_loss=True,
                                     q_init_mode="weak", anchor_theta=anchor,
                                     n0=args.e3a_n0,
                                     ref_q_for_counts=data["reference_q"]))
        return WeakQGNCDM(**merged(learn_q=True, use_weak_q_loss=True,
                                   q_init_mode="weak"))
    if point == "d5_posterior":
        if name == "d5_asym":
            return PosteriorQModel(**merged(learn_q=True, q_init_mode="weak",
                                            noise_f=args.d5_f, noise_n=args.d5_n,
                                            lambda_q=0.0, lambda_coverage=0.0))
        return WeakQGNCDM(**merged(learn_q=False, use_weak_q_loss=False,
                                   q_init_mode="fixed", lambda_q=0.0,
                                   lambda_coverage=0.0, q_pos_weight=None))
    return WeakQGNCDM(**merged(learn_q=True, use_weak_q_loss=True,
                               q_init_mode="weak"))


def conditions_for(point, data, teacher, args):
    if point == "p8_trq":
        return [{"name": "weakq_ctrl", "mode": None},
                {"name": "trq_random", "mode": "random"},
                {"name": "trq_adv", "mode": "adv"}]
    if point == "p6_train":
        return [{"name": "fixed_binary"}, {"name": "fixed_soft"}, {"name": "weakq"}]
    if point == "p7_conf":
        return [{"name": "conf_stats"}]
    if point == "p7_ecwq":
        # Screening gate failed for c1/c1b/c2 on synthetic (noise-edge AUC
        # well below 0.7); c0 (item specificity) is the only usable weight,
        # and on Math1 it is the ONLY one (no single-concept items at all).
        c0 = evi.edge_confidence_c0(data["reference_q"])
        return [{"name": "weakq_uniform_ctrl", "edge_conf": None},
                {"name": "ecwq_c0", "edge_conf": c0}]
    if point == "p9_saca":
        anchors = evi.single_concept_anchor_items(data["reference_q"])
        return [{"name": "saca_uniform_ctrl", "learn_attn": False},
                {"name": "saca_learned", "learn_attn": True,
                 "n_anchor_items": sum(len(v) for v in anchors.values())}]
    if point == "p10_e3a":
        return [{"name": "weakq_ctrl"}, {"name": "e3a_shrink"}]
    if point == "d1_relation":
        return [{"name": "weakq_ctrl"}, {"name": "d1_relation"}]
    if point == "d2_item_struct":
        if args.dataset == "math1":
            return [{"name": "d2_full"}]
        return [{"name": "d2_budget_only"}, {"name": "d2_full"}]
    if point == "d3_residual":
        return [{"name": "d3_theta_ctrl"}, {"name": "d3_residual"}]
    if point == "d4_edit_effect":
        return [{"name": "weakq_ctrl"}, {"name": "d4_effect"}]
    if point == "d5_posterior":
        return [{"name": "fixed_binary_ctrl"}, {"name": "d5_asym"}]
    raise ValueError(point)


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------

def save_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as h:
        w = csv.DictWriter(h, fieldnames=keys)
        w.writeheader()
        w.writerows(legacy.to_jsonable(rows))


def q_summary(model, data):
    q = model.get_q_soft().detach().cpu().numpy()
    hard = (q > 0.5).astype(np.int64)
    out = {"q_soft_density": float(q.mean()),
           "q_hard_density": float((q > .5).mean())}
    out["q_f1_vs_reference"] = float(
        legacy.q_recovery_metrics(data["eval_reference_q"], data["q"], hard, q)
        ["q_hat_f1_vs_true"])
    return out


def aggregate(batch_rows):
    keys = {k for r in batch_rows for k in r}
    return {k: float(np.mean([r.get(k, 0.0) for r in batch_rows])) for k in keys}


# ---------------------------------------------------------------------------
# point-specific helper losses
# ---------------------------------------------------------------------------

def saca_attraction(model, clean_np, y_full, m_full, p_hat_full, student_ids,
                    anchors, multi_mask_items, device):
    B, J = clean_np.shape
    K = model.n_know
    y = y_full[student_ids]
    m = m_full[student_ids]
    support = (clean_np != 0).astype(np.float32)
    m_eff, y_eff = m * support, y * support
    rho = m_eff * (y_eff - p_hat_full[student_ids])
    n_ik = np.zeros((B, K), dtype=np.float32)
    t_ik = np.zeros((B, K), dtype=np.float32)
    for k, items in anchors.items():
        if not items:
            continue
        mk = np.zeros((B, J), dtype=np.float32)
        mk[:, items] = 1.0
        nk = (mk * m_eff).sum(axis=1)
        n_ik[:, k] = nk
        t_ik[:, k] = (mk * rho).sum(axis=1) / np.maximum(nk, 1.0)
    obs = (m_eff > 0) & multi_mask_items[None, :]
    if obs.sum() == 0:
        zero = model.get_q_soft().sum() * 0.0
        return zero, {"n_terms": 0}
    g = model.gain_matrix()
    resid_t = torch.as_tensor(rho, dtype=torch.float32, device=device)
    t_t = torch.as_tensor(t_ik, dtype=torch.float32, device=device)
    obs_t = torch.as_tensor(obs.astype(np.float32), device=device)
    n_t = torch.as_tensor(n_ik, dtype=torch.float32, device=device)
    q_mask_t = model.attn_mask.unsqueeze(0)
    target = t_t[:, None, :].expand(B, J, K)
    resid_e = resid_t[:, :, None].expand(B, J, K)
    err = F.smooth_l1_loss(g * resid_e, g * target, reduction="none")
    valid = obs_t[:, :, None] * q_mask_t * (n_t[:, None, :] > 0).float()
    loss = (err * valid).sum() / valid.sum().clamp(min=1.0)
    return loss, {"n_terms": int(valid.sum().item())}


def attribution_correlation(model, data):
    disc = np.load(data["folder"] / "item_discrimination.npy").astype(np.float64)
    g = (model.gain_matrix() * model.attn_mask).detach().cpu().numpy()
    ref = data["reference_q"]
    xs, ys = [], []
    for j in range(ref.shape[0]):
        sup = np.flatnonzero(ref[j] > 0.5)
        if len(sup) < 2 or disc[j, sup].std() < 1e-6:
            continue
        xs.extend(disc[j, sup].tolist())
        ys.extend(g[j, sup].tolist())
    if len(xs) < 8 or np.std(ys) < 1e-6:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def budget_loss(model, ref_q_np):
    P0 = 0.05 + 0.90 * torch.as_tensor(ref_q_np, dtype=torch.float32,
                                       device=model.device)
    E = (model.get_q_soft() - P0).abs().sum(dim=1) / 0.90
    return torch.mean(F.relu(E - 1.0) ** 2)


def edit_amounts(model, ref_q_np):
    P0 = 0.05 + 0.90 * ref_q_np
    Q = model.get_q_soft().detach().cpu().numpy()
    return np.abs(Q - P0).sum(axis=1) / 0.90


def item_structure_infonce(model, head, clean_np, y_full, m_full, p_hat_full,
                           student_ids, ref_q_np, args):
    device = args.device
    B, J = clean_np.shape
    y, m = y_full[student_ids], m_full[student_ids]
    sup = (clean_np != 0).astype(np.float32)
    m_eff, y_eff = m * sup, y * sup
    rho = m_eff * (y_eff - p_hat_full[student_ids])
    rng = np.random.default_rng([args.seed, id(head) % 9973])
    perm = rng.permutation(B)
    U, V = perm[:B // 2], perm[B // 2:]
    z_views = []
    for half in (U, V):
        inp = np.concatenate([rho[half].T, m_eff[half].T], axis=1)
        if inp.shape[1] < 2 * args.batch_students:
            pad = np.zeros((J, 2 * args.batch_students - inp.shape[1]), dtype=np.float32)
            inp = np.concatenate([inp, pad], axis=1)
        z_views.append(F.normalize(
            head(torch.as_tensor(inp, dtype=torch.float32, device=device)), dim=1))
    z_q = F.normalize(model.get_q_soft(), dim=1)
    correct = y_eff.sum(axis=0) / np.maximum(m_eff.sum(axis=0), 1.0)
    labels = ref_q_np > 0.5
    losses = []
    for j in range(J):
        negs = [l for l in range(J) if l != j and not (labels[j] & labels[l]).any()
                and abs(correct[j] - correct[l]) <= 0.25]
        if len(negs) < 2:
            continue
        for v in z_views:
            pos = torch.cosine_similarity(v[j:j + 1], z_q[j:j + 1]).reshape(1)
            negs_sim = torch.stack([torch.cosine_similarity(v[j:j + 1], z_q[l:l + 1])
                                    for l in negs]).reshape(-1)
            logits = torch.cat([pos, negs_sim]) / args.temperature
            losses.append(F.cross_entropy(logits.unsqueeze(0),
                                          torch.zeros(1, dtype=torch.long,
                                                      device=device)))
    if not losses:
        zero = model.get_q_soft().sum() * 0.0 + sum(
            p.sum() for p in head.parameters()) * 0.0
        return zero, {"n_items": 0}
    return torch.stack(losses).mean(), {"n_items": len(losses)}


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train_condition(point, cond, data, teacher, args, out_dir):
    out_dir.mkdir(parents=True, exist_ok=False)
    legacy.set_seed(args.seed)
    model = build_model(point, cond, data, teacher, args)
    relevance_head = None
    if point == "d2_item_struct":
        K = data["q"].shape[1]
        relevance_head = nn.Sequential(
            nn.Linear(2 * args.batch_students, 64), nn.ReLU(),
            nn.Linear(64, K)).to(args.device)
    if point == "p9_saca" and not cond.get("learn_attn", True):
        model.attn_logits.requires_grad_(False)
    groups = [{"params": [p for n, p in model.named_parameters()
                          if n != "q_logits" and p.requires_grad], "lr": args.lr}]
    if model.q_logits.requires_grad:
        groups.append({"params": [model.q_logits], "lr": args.q_lr})
    if relevance_head is not None:
        groups.append({"params": list(relevance_head.parameters()), "lr": args.lr})
    optimizer = torch.optim.Adam(groups)

    eligible = np.flatnonzero((data["train"] != 0).sum(axis=1) >= 2)
    ref_q_np = data["reference_q"]
    y_full, m_full = evi.build_observation(data["rows"][0], len(data["train"]),
                                           len(data["q"]))
    p_hat_full = teacher.p_hat.astype(np.float32) if teacher is not None else None
    anchors = evi.single_concept_anchor_items(ref_q_np)
    multi_mask_items = (ref_q_np.sum(axis=1) >= 2)

    best_state, best_auc, best_epoch, bad = deepcopy(model.state_dict()), -math.inf, 0, 0
    history, started = [], time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.default_rng([args.seed, epoch, 0]).permutation(eligible)
        batch_rows: List[Dict] = []
        for step, start in enumerate(range(0, len(order), args.batch_students)):
            students = order[start:start + args.batch_students]
            if len(students) < 2:
                continue
            episode = make_student_episode(
                data["train"], students, args.query_fraction,
                np.random.default_rng([args.seed, epoch, step, 1]))
            clean_np = episode.support[students]
            clean = torch.as_tensor(clean_np, dtype=torch.float32, device=args.device)
            global_ids = torch.as_tensor(episode.student_ids, dtype=torch.long,
                                         device=args.device)
            sampling_d5 = point == "d5_posterior" and cond["name"].startswith("d5")
            if sampling_d5:
                model.begin_batch_sample()
            if point == "p10_e3a" and cond["name"] == "e3a_shrink":
                theta_clean = model.diagnose_theta_indexed(clean, global_ids)
            else:
                theta_clean = model.diagnose_theta(clean)
            item_ids, inverse = np.unique(episode.query_items, return_inverse=True)
            item_logs = torch.as_tensor(episode.support[:, item_ids].T.copy(),
                                        dtype=torch.float32, device=args.device)
            psi = model.diagnose_psi(item_logs)[torch.as_tensor(inverse, device=args.device)]
            users = torch.as_tensor(episode.query_students, device=args.device)
            query_items = torch.as_tensor(episode.query_items, device=args.device)
            Q_soft = model.get_q_soft()
            pred = model.predict_response(theta_clean[users], psi, Q_soft[query_items])
            labels = torch.as_tensor(episode.query_labels, dtype=torch.float32,
                                     device=args.device).reshape(-1, 1)
            losses = model.compute_loss(pred, labels)
            total = losses["loss_total"]
            aux: Dict = {"loss_aux": 0.0}
            inject_q_grad: Optional[torch.Tensor] = None

            if point == "p8_trq" and cond.get("mode"):
                Q_tilde = trq_perturb(model, cond["mode"], args.trq_delta,
                                      loss_pred=losses["loss_rec"])
                pred_t = trq_prediction(model, clean, users, query_items, psi,
                                        Q_tilde, path=args.trq_path)
                l_cons = F.binary_cross_entropy(pred_t, pred.detach())
                total = total + args.lambda_trq * l_cons
                aux = {"loss_aux": float(l_cons.detach())}

            elif point in ("d1_relation", "d3_residual") and not cond["name"].endswith("ctrl"):
                a_batch = teacher.a[episode.student_ids]
                p_hat_batch = p_hat_full[episode.student_ids]
                if point == "d1_relation":
                    pairs, _ = evi.build_relation_pairs_batch(
                        (clean_np + 1) / 2 * (clean_np != 0),
                        (clean_np != 0).astype(np.float32),
                        p_hat_batch, ref_q_np, a_batch,
                        delta_p=args.delta_p, delta_n=args.delta_n,
                        seed=args.seed + epoch)
                    coords = {"theta": theta_clean}
                    pick = "theta"
                else:
                    half = max(1, clean_np.shape[1] // 2)
                    cleanA = clean.clone(); cleanA[:, half:] = 0.0
                    cleanB = clean.clone(); cleanB[:, :half] = 0.0
                    pA = torch.as_tensor(p_hat_batch, dtype=torch.float32,
                                         device=args.device)
                    colsA = np.arange(clean_np.shape[1]) < half
                    colsB = ~colsA

                    def _pairs(cols, off):
                        sub = np.flatnonzero(cols)
                        return evi.build_relation_pairs_batch(
                            (clean_np[:, sub] + 1) / 2 * (clean_np[:, sub] != 0),
                            (clean_np[:, sub] != 0).astype(np.float32),
                            p_hat_batch[:, sub], ref_q_np[sub], a_batch,
                            delta_p=args.delta_p, delta_n=args.delta_n,
                            seed=args.seed + epoch + off)[0]
                    pairsA = _pairs(colsA, 1)
                    pairsB = _pairs(colsB, 2)

                    def _t(pairs):
                        return {k: {kk: torch.as_tensor(vv, device=args.device)
                                    for kk, vv in v.items()}
                                for k, v in pairs.items()}
                    coordA = residual_coordinates(cleanA, Q_soft, pA)
                    coordB = residual_coordinates(cleanB, Q_soft, pA)
                    thetaA = model.diagnose_theta(cleanA)
                    thetaB = model.diagnose_theta(cleanB)
                    l1, s1 = relation_nce(coordA, _t(pairsB), tau=args.temperature)
                    l2, s2 = relation_nce(coordB, _t(pairsA), tau=args.temperature)
                    l3, s3 = relation_nce(thetaA, _t(pairsB), tau=args.temperature)
                    l4, s4 = relation_nce(thetaB, _t(pairsA), tau=args.temperature)
                    if cond["name"] == "d3_residual":
                        l_rel, n_pos = 0.5 * (l1 + l2), int(s1["n_pos_pairs"] + s2["n_pos_pairs"])
                    else:
                        l_rel, n_pos = 0.5 * (l3 + l4), int(s3["n_pos_pairs"] + s4["n_pos_pairs"])
                    if n_pos == 0:
                        l_rel = theta_clean.sum() * 0.0
                    total = total + args.lambda_cl * l_rel
                    aux = {"loss_aux": float(l_rel.detach()), "n_pos_pairs": n_pos}
                    batch_rows.append({"loss_aux": aux["loss_aux"],
                                       "n_pos_pairs": n_pos,
                                       "loss_pred": float(losses["loss_rec"].detach())})
                    optimizer.zero_grad(set_to_none=True)
                    total.backward()
                    optimizer.step()
                    if sampling_d5:
                        model.end_batch_sample()
                    continue

                if point == "d1_relation":
                    pairs_t = {k: {kk: torch.as_tensor(vv, device=args.device)
                                   for kk, vv in v.items()}
                               for k, v in pairs.items()}
                    l_rel, rstats = relation_nce(coords[pick], pairs_t,
                                                 tau=args.temperature)
                    n_pos = int(rstats["n_pos_pairs"])
                    total = total + args.lambda_cl * l_rel
                    aux = {"loss_aux": float(l_rel.detach()), "n_pos_pairs": n_pos}

            elif point == "p9_saca" and cond["name"] == "saca_learned":
                l_attr, attr_stats = saca_attraction(
                    model, clean_np, y_full, m_full, p_hat_full,
                    episode.student_ids, anchors, multi_mask_items, args.device)
                total = total + args.lambda_cl * l_attr \
                    + model.attribution_regularizer(args.saca_reg)
                aux = {"loss_aux": float(l_attr.detach()),
                       "n_attr_terms": attr_stats["n_terms"]}

            elif point == "d2_item_struct":
                l_budget = budget_loss(model, ref_q_np)
                total = total + args.lambda_budget * l_budget
                aux = {"loss_aux": float(l_budget.detach())}
                if cond["name"] == "d2_full":
                    l_rq, _ = item_structure_infonce(
                        model, relevance_head, clean_np, y_full, m_full,
                        p_hat_full, episode.student_ids, ref_q_np, args)
                    total = total + args.lambda_cl * l_rq
                    aux["loss_rq"] = float(l_rq.detach())

            if not torch.isfinite(total):
                raise FloatingPointError("nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            total.backward()

            if point == "d5_posterior" and cond["name"].startswith("d5"):
                L_A, KL = model.annotation_losses()
                scale = len(data["q"]) * data["q"].shape[1]
                total_vi = args.lambda_vi * (L_A + KL) / scale
                total_vi.backward()
                model.end_batch_sample()
                aux = {"loss_aux": float(total_vi.detach()),
                       "sample_diff": model.last_sample_diff}

            if point == "d4_edit_effect" and cond["name"] == "d4_effect":
                l_eff, eff_stats = edit_effect_loss(
                    model, clean_np, episode, ref_q_np,
                    p_hat_full[episode.student_ids] if p_hat_full is not None else None,
                    temperature=args.temperature, max_edits=args.d4_max_edits)
                if l_eff.requires_grad:
                    g = torch.autograd.grad(l_eff, model.q_logits,
                                            allow_unused=True)[0]
                    if g is not None:
                        inject_q_grad = args.lambda_cl * g
                aux = {"loss_aux": float(l_eff.detach()) if l_eff.requires_grad else 0.0,
                       "n_edits": eff_stats["n_edits"],
                       "sign_correct_frac": eff_stats.get("sign_correct_frac", 0.0)}

            if inject_q_grad is not None:
                if model.q_logits.grad is None:
                    model.q_logits.grad = inject_q_grad.clone()
                else:
                    model.q_logits.grad += inject_q_grad
            optimizer.step()
            batch_rows.append({"loss_pred": float(losses["loss_rec"].detach()), **aux})

        if not batch_rows:
            continue
        valid, _, _ = core.evaluate(model, data["train"], data["rows"][1])
        record = {"epoch": epoch, **aggregate(batch_rows),
                  **{f"valid_{k}": v for k, v in valid.items()}}
        history.append(record)
        score = valid["auc"] if valid["auc"] is not None else -valid["nll"]
        if score > best_auc:
            best_auc, best_epoch, bad = score, epoch, 0
            best_state = deepcopy(model.state_dict())
        else:
            bad += 1
        save_csv(out_dir / "train_log.csv", history)
        print(f"{args.dataset} {point} {cond['name']} epoch={epoch} "
              f"pred={record['loss_pred']:.4f} aux={record['loss_aux']:.4f} "
              f"valid_auc={valid['auc']}", flush=True)
        if args.patience and bad >= args.patience:
            break

    model.load_state_dict(best_state)
    torch.save({"model_state_dict": best_state, "condition": cond["name"],
                "point": point, "seed": args.seed, "best_epoch": best_epoch},
               out_dir / "best_model.pt")
    valid, _, _ = core.evaluate(model, data["train"], data["rows"][1])
    test, theta, _ = core.evaluate(model, data["train"], data["rows"][2])
    np.save(out_dir / "theta.npy", theta)
    np.save(out_dir / "Q_soft.npy", model.get_q_soft().detach().cpu().numpy())
    result = {"dataset": args.dataset, "point": point, "condition": cond["name"],
              "seed": args.seed, "best_epoch": best_epoch,
              "epochs_run": len(history),
              "seconds": time.perf_counter() - started,
              **{f"valid_{k}": v for k, v in valid.items()},
              **{f"test_{k}": v for k, v in test.items()},
              **q_summary(model, data)}
    if point == "p9_saca" and data.get("q_true") is not None:
        corr = attribution_correlation(model, data)
        if corr is not None:
            result["attribution_corr"] = corr
    if point == "d5_posterior" and cond["name"].startswith("d5"):
        pi = torch.sigmoid(model.q_logits).detach().cpu().numpy()
        result["q_edge_entropy_mean"] = float(
            -(pi * np.log(pi + 1e-9) + (1 - pi) * np.log(1 - pi + 1e-9)).mean())
    if point == "d2_item_struct":
        E = edit_amounts(model, ref_q_np)
        result["mean_edit_amount"] = float(E.mean())
        result["over_budget_frac"] = float((E > 1.0).mean())
    legacy.write_json(out_dir / "metrics.json", result)
    print(f"== done {point}/{cond['name']}: test_auc={result['test_auc']} "
          f"test_acc={result['test_acc']} q_f1={result['q_f1_vs_reference']}",
          flush=True)
    return result


# ---------------------------------------------------------------------------
# mechanism checks
# ---------------------------------------------------------------------------

def run_mech_check(point, data, teacher, args, out_dir):
    checks: Dict = {}
    device = args.device
    torch.manual_seed(args.seed)
    J, K = len(data["q"]), data["q"].shape[1]
    if point == "p8_trq":
        model = build_model(point, {"name": "trq_random"}, data, teacher, args)
        x = (torch.rand(4, J, device=device) > 0.5).float() * 2 - 1
        Q0 = model.get_q_soft()
        Qt = trq_perturb(model, "random", args.trq_delta)
        t0 = diagnose_theta_with_q(model, x, Q0)
        t1 = diagnose_theta_with_q(model, x, Qt)
        checks["perturbation_max_abs_q"] = float((Qt - Q0).abs().max())
        checks["theta_changes_under_perturb"] = float((t1 - t0).abs().max())
        checks["trust_region_respected"] = bool(
            (Qt >= 0.05 - 1e-5).all() and (Qt <= 0.95 + 1e-5).all())
    elif point in ("p7_ecwq", "p7_conf"):
        variants = {"c1": dict(use_raw_y=False), "c1b": dict(use_raw_y=True),
                    "c2": None}
        for name, fn in (("c1", evi.edge_confidence_c1),
                         ("c1b", evi.edge_confidence_c1),
                         ("c2", evi.edge_confidence_c2)):
            kwargs = variants[name] or {}
            c, stats = fn(data["rows"][0], data["reference_q"], teacher,
                          len(data["train"]), J, **kwargs)
            entry: Dict = {"std": float(c.std()), "stats": stats}
            if data.get("noise_info"):
                entry["noise_auc"] = evi.confidence_vs_noise_auc(c, data["noise_info"])
            checks[name] = entry
        c0 = evi.edge_confidence_c0(data["reference_q"])
        entry0: Dict = {"std": float(c0.std())}
        if data.get("noise_info"):
            entry0["noise_auc"] = evi.confidence_vs_noise_auc(c0, data["noise_info"])
        checks["c0"] = entry0
    elif point == "p9_saca":
        model = build_model("p9_saca", {"name": "saca_uniform_ctrl",
                                        "learn_attn": False}, data, teacher, args)
        base = WeakQGNCDM(**base_kwargs(data, args), learn_q=True,
                          use_weak_q_loss=True, q_init_mode="weak")
        base.load_state_dict(model.state_dict(), strict=False)
        x = (torch.rand(3, J, device=device) > 0.5).float() * 2 - 1
        with torch.no_grad():
            d = (model.diagnose_theta(x) - base.diagnose_theta(x)).abs().max()
        g = model.gain_matrix() * model.attn_mask
        checks["uniform_equals_baseline_max_diff"] = float(d)
        checks["gain_row_mass_ok"] = bool(torch.allclose(
            g.sum(dim=1), model.attn_mask.sum(dim=1), atol=1e-4))
        anchors = evi.single_concept_anchor_items(data["reference_q"])
        checks["anchor_items_total"] = int(sum(len(v) for v in anchors.values()))
        checks["concepts_with_anchors"] = len(anchors)
    elif point == "p10_e3a":
        model = build_model("p10_e3a", {"name": "e3a_shrink"}, data, teacher, args)
        base = WeakQGNCDM(**base_kwargs(data, args), learn_q=True,
                          use_weak_q_loss=True, q_init_mode="weak")
        base.load_state_dict(model.state_dict(), strict=False)
        x = (torch.rand(3, J, device=device) > 0.5).float() * 2 - 1
        x[x == 0] = -1.0
        idx = torch.arange(3, device=device)
        with torch.no_grad():
            model.n0 = 1e9
            d_shrink = (model.diagnose_theta_indexed(x, idx)
                        - base.diagnose_theta(x)).abs().max()
            model.n0 = 1e-9
            d_free = (model.diagnose_theta_indexed(x, idx)
                      - base.diagnose_theta(x)).abs().max()
        model.n0 = args.e3a_n0
        checks["tiny_n0_equals_baseline_diff"] = float(d_free)
        checks["huge_n0_shrinkage_active_diff"] = float(d_shrink)
    elif point == "d3_residual":
        model = build_model("d3_residual", {"name": "d3_residual"}, data, teacher, args)
        x = (torch.rand(5, J, device=device) > 0.5).float() * 2 - 1
        checks.update(decomposition_check(x, model.get_q_soft(),
                                          torch.rand(5, J, device=device)))
    elif point == "d4_edit_effect":
        model = build_model("d4_effect", {"name": "d4_effect"}, data, teacher, args)
        model.eval()
        checks["ncd_last_is_sigmoid"] = isinstance(model.ncd[-1], nn.Sigmoid)
        x = (torch.rand(1, J, device=device) > 0.5).float() * 2 - 1
        x[x == 0] = -1.0
        theta = model.diagnose_theta(x)
        item_logs = torch.as_tensor(data["train"].T.copy(),
                                    dtype=torch.float32, device=device)
        psi = model.diagnose_psi(item_logs)
        lg = response_logit(model, theta, psi, model.get_q_soft())
        checks["logit_finite"] = bool(torch.isfinite(lg).all())
        rebuilt = model.ncd(model.theta_agg_mat(theta * model.get_q_soft())
                            - model.psi_agg_mat(psi * model.get_q_soft()))
        checks["sigmoid_roundtrip_max_err"] = float((torch.sigmoid(lg) - rebuilt).abs().max())
    elif point == "d4_edit_effect":
        model = build_model("d4_effect", {"name": "d4_effect"}, data, teacher, args)
        model.eval()
        checks["ncd_last_is_sigmoid"] = isinstance(model.ncd[-1], nn.Sigmoid)
        x = (torch.rand(1, J, device=device) > 0.5).float() * 2 - 1
        x[x == 0] = -1.0
        theta = model.diagnose_theta(x)
        item_logs = torch.as_tensor(data["train"].T.copy(),
                                    dtype=torch.float32, device=device)
        psi = model.diagnose_psi(item_logs)
        lg = response_logit(model, theta, psi, model.get_q_soft())
        checks["logit_finite"] = bool(torch.isfinite(lg).all())
        rebuilt = model.ncd(model.theta_agg_mat(theta * model.get_q_soft())
                            - model.psi_agg_mat(psi * model.get_q_soft()))
        checks["sigmoid_roundtrip_max_err"] = float((torch.sigmoid(lg) - rebuilt).abs().max())
    elif point == "d5_posterior":
        model = build_model("d5_posterior", {"name": "d5_asym"}, data, teacher, args)
        model.begin_batch_sample()
        z1 = model.get_q_soft().detach().clone()
        model.begin_batch_sample()
        z2 = model.get_q_soft().detach().clone()
        model.end_batch_sample()
        checks["sample_diff_between_batches"] = float((z1 - z2).abs().mean())
        checks["Z_is_binary"] = bool(((z1 == 0) | (z1 == 1)).all().item())
        L_A, KL = model.annotation_losses()
        checks["L_A_finite"] = bool(torch.isfinite(L_A))
        checks["KL_finite"] = bool(torch.isfinite(KL))
        pi = torch.sigmoid(model.q_logits).detach()
        checks["pi_init_matches_reference"] = float(
            (pi - torch.where(model.Q_noise > 0.5, 0.95, 0.05)).abs().max())
    elif point == "d1_relation":
        y, m = evi.build_observation(data["rows"][0][:2000], len(data["train"]), J)
        pairs, stats = evi.build_relation_pairs_batch(
            y[:128], m[:128], teacher.p_hat[:128], data["reference_q"],
            teacher.a[:128], seed=args.seed)
        checks["pair_stats_sample_batch"] = stats
        model = build_model("d1_relation", {"name": "d1_relation"}, data, teacher, args)
        theta = model.diagnose_theta(torch.as_tensor(y[:128] * 2 - 1,
                                                     dtype=torch.float32, device=device))
        pairs_t = {k: {kk: torch.as_tensor(vv, device=device)
                       for kk, vv in v.items()} for k, v in pairs.items()}
        l, s = relation_nce(theta, pairs_t)
        l.backward()
        checks["relation_loss_finite"] = bool(torch.isfinite(l))
        checks["q_grad_mean_abs"] = float(model.q_logits.grad.abs().mean()) \
            if model.q_logits.grad is not None else 0.0
        checks["n_pos_pairs"] = s["n_pos_pairs"]
    elif point == "d2_item_struct":
        model = build_model("d2_item_struct", {"name": "d2_full"}, data, teacher, args)
        l0 = budget_loss(model, data["reference_q"])
        with torch.no_grad():
            model.q_logits.add_(2.0)
        l1 = budget_loss(model, data["reference_q"])
        checks["budget_loss_at_init"] = float(l0)
        checks["budget_loss_grows_when_Q_moves"] = float(l1)
    legacy.write_json(out_dir / "mech_check.json", checks)
    print(f"mech-check {point}: {json.dumps(checks, default=str)[:400]}", flush=True)
    return checks


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--point", required=True, choices=POINTS)
    p.add_argument("--dataset", default="math1", choices=["math1", "synthetic"])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--results-root", default="Q_GNCDM/results/wqcl_ten_screen_v1")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--split-seed", type=int, default=2026)
    p.add_argument("--noise-rate", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--embedding-dim", type=int, default=32)
    p.add_argument("--batch-students", type=int, default=128)
    p.add_argument("--query-fraction", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--q-lr", type=float, default=5e-3)
    p.add_argument("--lambda-q", type=float, default=0.01)
    p.add_argument("--lambda-coverage", type=float, default=0.01)
    p.add_argument("--q-init-high", type=float, default=0.55)
    p.add_argument("--q-init-low", type=float, default=0.45)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--lambda-cl", type=float, default=0.01)
    p.add_argument("--lambda-trq", type=float, default=0.05)
    p.add_argument("--trq-delta", type=float, default=0.1)
    p.add_argument("--trq-path", default="both", choices=["both", "exp", "mask"])
    p.add_argument("--delta-p", type=float, default=0.25)
    p.add_argument("--delta-n", type=float, default=0.5)
    p.add_argument("--lambda-budget", type=float, default=0.01)
    p.add_argument("--lambda-vi", type=float, default=1.0)
    p.add_argument("--d5-f", type=float, default=0.10)
    p.add_argument("--d5-n", type=float, default=0.10)
    p.add_argument("--saca-reg", type=float, default=1e-3)
    p.add_argument("--e3a-n0", type=float, default=5.0)
    p.add_argument("--d4-max-edits", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--mech-check-only", action="store_true")
    p.add_argument("--skip-mech-check", action="store_true")
    args = p.parse_args(argv)
    if args.data_dir is None:
        if args.dataset == "math1":
            args.data_dir = str(ROOT / "Q_GNCDM" / "data" / "math1")
        else:
            args.data_dir = str(ROOT / "Q_GNCDM" / "data" / "synthetic_weak_q" / "seed_2026")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    data = load_data(args)
    need_teacher = args.point in {"p7_ecwq", "p7_conf", "p9_saca", "p10_e3a",
                                  "d1_relation", "d3_residual", "d4_edit_effect",
                                  "d2_item_struct"}
    teacher = None
    if need_teacher:
        t0 = time.perf_counter()
        teacher = evi.fit_rasch_teacher(data["rows"][0], len(data["train"]),
                                        len(data["q"]), seed=args.seed)
        print(f"rasch teacher fitted in {time.perf_counter() - t0:.1f}s", flush=True)
    out_root = ROOT / args.results_root / args.point / args.dataset
    if args.point == "p7_conf":
        if out_root.exists():
            import shutil
            shutil.rmtree(out_root)
        out_root.mkdir(parents=True)
        run_mech_check("p7_conf", data, teacher, args, out_root)
        return
    if args.point in ("p9_saca", "d1_relation", "d3_residual") and args.dataset == "math1":
        anchors = evi.single_concept_anchor_items(data["reference_q"])
        if sum(len(v) for v in anchors.values()) == 0:
            out_root.mkdir(parents=True, exist_ok=True)
            legacy.write_json(out_root / "skipped.json", {
                "reason": (f"{args.point} needs single-concept anchor items; "
                           "Math1 reference Q has none (every item spans >=2 concepts)"),
                "n_anchor_items": 0})
            print(f"{args.point} skipped: no single-concept anchors on Math1", flush=True)
            return
    if not args.skip_mech_check:
        chk_dir = out_root / "mech_check"
        if chk_dir.exists():
            import shutil
            shutil.rmtree(chk_dir)
        chk_dir.mkdir(parents=True)
        try:
            run_mech_check(args.point, data, teacher, args, chk_dir)
        except Exception as exc:                                  # noqa: BLE001
            legacy.write_json(chk_dir / "mech_check.json", {"error": repr(exc)})
            print(f"mech-check FAILED for {args.point}: {exc!r}", flush=True)
            if args.mech_check_only:
                raise
    if args.mech_check_only:
        return
    results = []
    for cond in conditions_for(args.point, data, teacher, args):
        out_dir = out_root / cond["name"]
        if out_dir.exists():
            import shutil
            shutil.rmtree(out_dir)
        try:
            results.append(train_condition(args.point, cond, data, teacher,
                                           args, out_dir))
        except Exception as exc:                                  # noqa: BLE001
            out_dir.mkdir(parents=True, exist_ok=True)
            legacy.write_json(out_dir / "failed.json", {"error": repr(exc)})
            print(f"CONDITION FAILED {args.point}/{cond['name']}: {exc!r}",
                  flush=True)
    legacy.write_json(out_root / "summary.json", results)


if __name__ == "__main__":
    main()
