"""Continuous observed-only views and same-student, two-candidate InfoNCE.

No model layers or global RNG state are created/modified here.
"""
import numpy as np
import torch
from torch.nn import functional as F


def augment(x, ratio, scale, rng, distribution="uniform"):
    if not 0 <= ratio <= 1 or not 0 <= scale < 1:
        raise ValueError("ratio in [0,1], scale in [0,1) required")
    if distribution != "uniform":
        raise ValueError("Only the preregistered uniform distribution is supported")
    if x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("Expected finite float32 input")
    mask = x != 0
    selected = np.zeros(x.shape, dtype=bool)
    raw = x.copy()
    if ratio > 0 and scale > 0:
        selected = (rng.random(x.shape) < ratio) & mask
        noise = rng.uniform(-scale, scale, x.shape).astype(np.float32)
        raw[selected] += noise[selected]
    clipped = selected & ((raw < -1) | (raw > 1))
    out = np.clip(raw, -1, 1).astype(np.float32)
    delta = np.abs(out - x)
    count = max(int(mask.sum()), 1)
    stats = {
        "selected_fraction": float(selected.sum() / count),
        "changed_fraction": float((delta > 0).sum() / count),
        "clipped_fraction_selected": float(clipped.sum() / max(int(selected.sum()), 1)),
        "input_mae_observed": float(delta.sum() / count),
        "input_mae_all": float(delta.mean()),
        "input_max_change": float(delta.max()),
    }
    return out, stats


def same_student_infonce(anchor, positive, negative, temperature):
    if temperature <= 0 or not (anchor.shape == positive.shape == negative.shape):
        raise ValueError("Invalid temperature or mismatched views")
    if anchor.ndim != 2:
        raise ValueError("Expected student by concept tensors")
    a, p, n = [F.normalize(t, dim=1) for t in (anchor, positive, negative)]
    sim_pos, sim_neg = (a * p).sum(1), (a * n).sum(1)
    logits = torch.stack((sim_pos, sim_neg), dim=1) / temperature
    loss = F.cross_entropy(logits, torch.zeros(len(a), device=a.device, dtype=torch.long))
    return loss, {
        "ap_mse": float((anchor - positive).square().mean().detach()),
        "an_mse": float((anchor - negative).square().mean().detach()),
        "ap_cos_distance": float((1 - sim_pos).mean().detach()),
        "an_cos_distance": float((1 - sim_neg).mean().detach()),
        "negative_softmax_probability": float(logits.softmax(1)[:, 1].mean().detach()),
        "negative_not_farther_fraction": float((sim_neg >= sim_pos).float().mean().detach()),
    }


def contrastive_branch(model, clean_np, anchor, args, seed, epoch, step):
    # Independent streams: no Python/NumPy-global/Torch RNG consumption.
    rng_p = np.random.default_rng([args.contrastive_seed, seed, epoch, step, 0])
    rng_n = np.random.default_rng([args.contrastive_seed, seed, epoch, step, 1])
    pos, ps = augment(clean_np, args.positive_noise_ratio, args.positive_noise_scale,
                      rng_p, args.noise_distribution)
    neg, ns = augment(clean_np, args.negative_noise_ratio, args.negative_noise_scale,
                      rng_n, args.noise_distribution)
    p = model.diagnose_theta(torch.as_tensor(pos, device=anchor.device))
    n = model.diagnose_theta(torch.as_tensor(neg, device=anchor.device))
    loss, stats = same_student_infonce(anchor, p, n, args.temperature)
    stats.update({"positive_" + k: v for k, v in ps.items()})
    stats.update({"negative_" + k: v for k, v in ns.items()})
    return loss, stats


def shared_parameters(model):
    return [(n, p) for n, p in model.named_parameters()
            if p.requires_grad and (n.startswith("f_nn.") or n == "q_logits")]


def gradient_diagnostics(original, contrastive, model, coefficient):
    params = shared_parameters(model)
    base = torch.autograd.grad(original, [p for _, p in params], retain_graph=True,
                               allow_unused=True)
    cl = torch.autograd.grad(contrastive, [p for _, p in params], retain_graph=True,
                             allow_unused=True)
    def norm(grads):
        return float(sum(g.detach().double().square().sum() for g in grads if g is not None).sqrt())
    b, c = norm(base), norm(cl)
    per_layer = {}
    for (name, _), g, h in zip(params, base, cl):
        per_layer[name] = {"original_grad_norm": 0.0 if g is None else float(g.norm()),
                           "cl_grad_norm": 0.0 if h is None else float(h.norm()),
                           "cl_finite": h is not None and bool(torch.isfinite(h).all())}
    if not all(v["cl_finite"] for v in per_layer.values()):
        raise FloatingPointError("Missing or nonfinite shared CL gradient")
    return {"original_shared_grad_norm": b, "cl_shared_grad_norm": c,
            "cl_to_original_grad_ratio": c / max(b, 1e-30),
            "weighted_cl_to_original_grad_ratio": coefficient * c / max(b, 1e-30)}, per_layer
