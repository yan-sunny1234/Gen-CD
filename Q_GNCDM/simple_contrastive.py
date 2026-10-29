"""Minimal response-flip augmentation and in-batch contrastive loss."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class StudentEpisode:
    student_ids: np.ndarray
    support: np.ndarray
    query_students: np.ndarray
    query_items: np.ndarray
    query_labels: np.ndarray


def make_student_episode(train, student_ids, query_fraction, rng):
    """Hide every sampled query from the shared student/item evidence matrix."""
    ids = np.asarray(student_ids, dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("student_ids must be unique within an episode")
    if not 0 < query_fraction < 1:
        raise ValueError("query_fraction must be in (0, 1)")
    support = np.asarray(train).copy()
    local_students, query_items, query_labels = [], [], []
    for local_id, student_id in enumerate(ids):
        observed = np.flatnonzero(train[student_id])
        if len(observed) < 2:
            raise ValueError("episode students need at least two observed responses")
        count = min(
            len(observed) - 1,
            max(1, int(round(len(observed) * query_fraction))),
        )
        held_out = rng.choice(observed, count, replace=False)
        local_students.extend([local_id] * count)
        query_items.extend(held_out.tolist())
        query_labels.extend(((train[student_id, held_out] + 1.0) / 2.0).tolist())
        support[student_id, held_out] = 0.0
    return StudentEpisode(
        student_ids=ids,
        support=support,
        query_students=np.asarray(local_students, dtype=np.int64),
        query_items=np.asarray(query_items, dtype=np.int64),
        query_labels=np.asarray(query_labels, dtype=np.float32),
    )


def flip_observed_responses(responses, noise_rate, rng):
    """Flip approximately noise_rate of each row's +/-1 entries; keep zeros."""
    if not 0 < noise_rate < 1:
        raise ValueError("noise_rate must be in (0, 1)")
    source = np.asarray(responses)
    noisy = source.copy()
    for row_id, row in enumerate(source):
        observed = np.flatnonzero(row)
        if not len(observed):
            continue
        count = min(len(observed), max(1, int(round(len(observed) * noise_rate))))
        flipped = rng.choice(observed, count, replace=False)
        noisy[row_id, flipped] *= -1.0
    return noisy


def simple_infonce(theta_clean, theta_noisy, temperature=0.2):
    """Symmetric InfoNCE: the diagonal is positive, other students are negative."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if theta_clean.shape != theta_noisy.shape or theta_clean.ndim != 2:
        raise ValueError("theta tensors must have the same two-dimensional shape")
    if len(theta_clean) < 2:
        raise ValueError("InfoNCE needs at least two students")
    clean = F.normalize(theta_clean, dim=1)
    noisy = F.normalize(theta_noisy, dim=1)
    logits = clean @ noisy.T / temperature
    labels = torch.arange(len(clean), device=clean.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )
