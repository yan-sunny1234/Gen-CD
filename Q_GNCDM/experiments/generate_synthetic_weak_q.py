# -*- coding: utf-8 -*-
"""Generate a synthetic weak-Q cognitive diagnosis dataset.

The generated hidden variables are meant for simulation and evaluation only.
Weak-Q G-NCDM training should only consume response.csv and a selected
Q_noise_*.npy file.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


DEFAULT_NOISE_RATES = [0.05, 0.10, 0.20, 0.30, 0.40]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def rate_label(rate: float) -> str:
    """Use fixed labels such as 0.05 and 0.10 in output filenames."""
    return f"{rate:.2f}"


def positions_to_lists(positions: np.ndarray) -> List[List[int]]:
    return [[int(row), int(col)] for row, col in positions.tolist()]


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def validate_generation_args(args: argparse.Namespace) -> None:
    if args.num_students <= 0:
        raise ValueError("--num_students must be positive.")
    if args.num_items <= 0:
        raise ValueError("--num_items must be positive.")
    if args.num_concepts <= 0:
        raise ValueError("--num_concepts must be positive.")
    if args.min_concepts_per_item < 1:
        raise ValueError("--min_concepts_per_item must be at least 1.")
    if args.max_concepts_per_item < args.min_concepts_per_item:
        raise ValueError("--max_concepts_per_item must be >= --min_concepts_per_item.")
    if args.max_concepts_per_item > args.num_concepts:
        raise ValueError("--max_concepts_per_item cannot exceed --num_concepts.")
    if args.min_items_per_concept < 1:
        raise ValueError("--min_items_per_concept must be at least 1.")
    required_capacity = args.num_concepts * args.min_items_per_concept
    total_capacity = args.num_items * args.max_concepts_per_item
    if required_capacity > total_capacity:
        raise ValueError(
            "Cannot cover each concept with the requested minimum count: "
            f"need {required_capacity} edges but only {total_capacity} item-concept "
            "slots are available."
        )
    for rate in args.noise_rates:
        if rate < 0:
            raise ValueError("--noise_rates cannot contain negative values.")
    if not 0.0 <= args.observation_rate <= 1.0:
        raise ValueError("--observation_rate must be between 0 and 1.")
    if args.min_student_observations < 0:
        raise ValueError("--min_student_observations must be non-negative.")
    if args.min_item_observations < 0:
        raise ValueError("--min_item_observations must be non-negative.")
    if args.min_student_observations > args.num_items:
        raise ValueError("--min_student_observations cannot exceed --num_items.")
    if args.min_item_observations > args.num_students:
        raise ValueError("--min_item_observations cannot exceed --num_students.")


def generate_q_true(
    num_items: int,
    num_concepts: int,
    min_concepts_per_item: int,
    max_concepts_per_item: int,
    min_items_per_concept: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a binary Q matrix with item and concept coverage constraints."""
    q_true = np.zeros((num_items, num_concepts), dtype=np.int64)
    row_counts = np.zeros(num_items, dtype=np.int64)
    concept_counts = np.zeros(num_concepts, dtype=np.int64)

    # First, make sure every concept receives the requested minimum coverage.
    concept_sequence = np.repeat(np.arange(num_concepts), min_items_per_concept)
    rng.shuffle(concept_sequence)
    for concept_id in concept_sequence:
        candidates = np.where(
            (row_counts < max_concepts_per_item) & (q_true[:, concept_id] == 0)
        )[0]
        if candidates.size == 0:
            raise RuntimeError(
                "Failed to assign concept coverage. Try increasing "
                "--num_items or --max_concepts_per_item."
            )
        min_load = row_counts[candidates].min()
        least_loaded = candidates[row_counts[candidates] == min_load]
        item_id = int(rng.choice(least_loaded))
        q_true[item_id, concept_id] = 1
        row_counts[item_id] += 1
        concept_counts[concept_id] += 1

    # Next, make sure every item has at least the minimum number of concepts.
    for item_id in range(num_items):
        while row_counts[item_id] < min_concepts_per_item:
            candidates = np.where(q_true[item_id] == 0)[0]
            min_coverage = concept_counts[candidates].min()
            least_covered = candidates[concept_counts[candidates] == min_coverage]
            concept_id = int(rng.choice(least_covered))
            q_true[item_id, concept_id] = 1
            row_counts[item_id] += 1
            concept_counts[concept_id] += 1

    # Finally, randomly add extra concepts up to each item's sampled target size.
    for item_id in range(num_items):
        low = int(max(row_counts[item_id], min_concepts_per_item))
        target_size = int(rng.integers(low, max_concepts_per_item + 1))
        while row_counts[item_id] < target_size:
            candidates = np.where(q_true[item_id] == 0)[0]
            min_coverage = concept_counts[candidates].min()
            least_covered = candidates[concept_counts[candidates] == min_coverage]
            concept_id = int(rng.choice(least_covered))
            q_true[item_id, concept_id] = 1
            row_counts[item_id] += 1
            concept_counts[concept_id] += 1

    return q_true


def generate_responses(
    q_true: np.ndarray,
    theta_true: np.ndarray,
    item_discrimination: np.ndarray,
    item_difficulty: np.ndarray,
    response_sampling: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate probabilities and binary responses with Q-masked MIRT."""
    effective_discrimination = q_true * item_discrimination
    logits = theta_true @ effective_discrimination.T - item_difficulty.reshape(1, -1)
    response_prob = sigmoid(logits)

    if response_sampling == "bernoulli":
        response = rng.binomial(1, response_prob).astype(np.int64)
    elif response_sampling == "threshold":
        response = (response_prob > 0.5).astype(np.int64)
    else:
        raise ValueError(f"Unknown response_sampling: {response_sampling}")

    return response_prob, response


def generate_observation_mask(
    num_students: int,
    num_items: int,
    observation_rate: float,
    min_student_observations: int,
    min_item_observations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample sparse observed responses while preventing empty rows/columns.

    The full response matrix is still generated from Q-masked MIRT, but only
    entries selected by this mask are written to response.csv. In the original
    G-NCDM data loader, missing entries become 0 in log_mat, while observed
    wrong/correct responses become -1/+1.
    """
    total_pairs = num_students * num_items
    requested_observations = int(round(observation_rate * total_pairs))
    minimum_required = max(
        num_students * min_student_observations,
        num_items * min_item_observations,
    )
    target_observations = max(requested_observations, minimum_required)
    if target_observations > total_pairs:
        raise ValueError(
            "Observation constraints are infeasible: requested at least "
            f"{target_observations} observations but only {total_pairs} "
            "student-item pairs exist."
        )

    observation_mask = np.zeros((num_students, num_items), dtype=bool)
    student_counts = np.zeros(num_students, dtype=np.int64)
    item_counts = np.zeros(num_items, dtype=np.int64)

    # First satisfy per-student minima, preferring items with fewer observations.
    for user_id in rng.permutation(num_students):
        while student_counts[user_id] < min_student_observations:
            candidates = np.where(~observation_mask[user_id])[0]
            min_count = item_counts[candidates].min()
            least_observed_items = candidates[item_counts[candidates] == min_count]
            item_id = int(rng.choice(least_observed_items))
            observation_mask[user_id, item_id] = True
            student_counts[user_id] += 1
            item_counts[item_id] += 1

    # Then satisfy per-item minima, preferring students with fewer observations.
    for item_id in rng.permutation(num_items):
        while item_counts[item_id] < min_item_observations:
            candidates = np.where(~observation_mask[:, item_id])[0]
            min_count = student_counts[candidates].min()
            least_observed_students = candidates[student_counts[candidates] == min_count]
            user_id = int(rng.choice(least_observed_students))
            observation_mask[user_id, item_id] = True
            student_counts[user_id] += 1
            item_counts[item_id] += 1

    current_observations = int(observation_mask.sum())
    remaining = target_observations - current_observations
    if remaining > 0:
        candidates = np.argwhere(~observation_mask)
        chosen = rng.choice(candidates.shape[0], size=remaining, replace=False)
        selected_pairs = candidates[chosen]
        observation_mask[selected_pairs[:, 0], selected_pairs[:, 1]] = True

    return observation_mask


def save_response_csv(
    path: Path,
    response: np.ndarray,
    observation_mask: np.ndarray,
) -> None:
    """Save observed response logs in the original G-NCDM triplet format."""
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["user_id", "item_id", "score"])
        num_students, num_items = response.shape
        for user_id in range(num_students):
            for item_id in range(num_items):
                if observation_mask[user_id, item_id]:
                    writer.writerow([user_id, item_id, int(response[user_id, item_id])])


def split_mixed_counts(target_count: int, weights: Sequence[float]) -> Tuple[int, int, int]:
    """Split a target perturbation count into delete/add/flip operation counts."""
    weights_arr = np.asarray(weights, dtype=np.float64)
    if weights_arr.shape != (3,) or np.any(weights_arr < 0) or weights_arr.sum() <= 0:
        raise ValueError("--mixed_noise_weights must contain three non-negative values.")

    raw = target_count * weights_arr / weights_arr.sum()
    counts = np.floor(raw).astype(np.int64)
    remainder = int(target_count - counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[idx] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def apply_delete_operations(
    q_noise: np.ndarray,
    changed: np.ndarray,
    count: int,
    rng: np.random.Generator,
    preserve_item_coverage: bool,
) -> List[List[int]]:
    deleted = []
    for _ in range(count):
        mask = (q_noise == 1) & (~changed)
        if preserve_item_coverage:
            mask &= q_noise.sum(axis=1).reshape(-1, 1) > 1
        candidates = np.argwhere(mask)
        if candidates.size == 0:
            break
        item_id, concept_id = candidates[int(rng.integers(candidates.shape[0]))]
        q_noise[item_id, concept_id] = 0
        changed[item_id, concept_id] = True
        deleted.append([int(item_id), int(concept_id)])
    return deleted


def apply_add_operations(
    q_noise: np.ndarray,
    changed: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> List[List[int]]:
    added = []
    for _ in range(count):
        mask = (q_noise == 0) & (~changed)
        candidates = np.argwhere(mask)
        if candidates.size == 0:
            break
        item_id, concept_id = candidates[int(rng.integers(candidates.shape[0]))]
        q_noise[item_id, concept_id] = 1
        changed[item_id, concept_id] = True
        added.append([int(item_id), int(concept_id)])
    return added


def apply_flip_operations(
    q_noise: np.ndarray,
    changed: np.ndarray,
    count: int,
    rng: np.random.Generator,
    preserve_item_coverage: bool,
) -> List[Dict[str, Any]]:
    flipped = []
    for _ in range(count):
        mask = ~changed
        if preserve_item_coverage:
            removable_one = (q_noise == 1) & (q_noise.sum(axis=1).reshape(-1, 1) <= 1)
            mask &= ~removable_one
        candidates = np.argwhere(mask)
        if candidates.size == 0:
            break
        item_id, concept_id = candidates[int(rng.integers(candidates.shape[0]))]
        before = int(q_noise[item_id, concept_id])
        after = 1 - before
        q_noise[item_id, concept_id] = after
        changed[item_id, concept_id] = True
        flipped.append(
            {
                "position": [int(item_id), int(concept_id)],
                "from": before,
                "to": after,
            }
        )
    return flipped


def perturb_q_true(
    q_true: np.ndarray,
    noise_rate: float,
    noise_mode: str,
    mixed_noise_weights: Sequence[float],
    preserve_item_coverage: bool,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Create one noisy Q matrix and a detailed perturbation record."""
    q_noise = q_true.copy()
    changed = np.zeros_like(q_true, dtype=bool)
    q_true_edges = int(q_true.sum())
    target_count = int(round(noise_rate * q_true_edges))

    if noise_mode == "delete":
        delete_count, add_count, flip_count = target_count, 0, 0
    elif noise_mode == "add":
        delete_count, add_count, flip_count = 0, target_count, 0
    elif noise_mode == "flip":
        delete_count, add_count, flip_count = 0, 0, target_count
    elif noise_mode == "mixed":
        delete_count, add_count, flip_count = split_mixed_counts(
            target_count, mixed_noise_weights
        )
    else:
        raise ValueError(f"Unknown noise_mode: {noise_mode}")

    delete_operation_edges = apply_delete_operations(
        q_noise, changed, delete_count, rng, preserve_item_coverage
    )
    add_operation_edges = apply_add_operations(q_noise, changed, add_count, rng)
    flipped_positions = apply_flip_operations(
        q_noise, changed, flip_count, rng, preserve_item_coverage
    )

    deleted_edges = np.argwhere((q_true == 1) & (q_noise == 0))
    added_edges = np.argwhere((q_true == 0) & (q_noise == 1))
    actual_perturbations = int(np.sum(q_true != q_noise))
    total_entries = int(q_true.size)

    info = {
        "noise_rate": float(noise_rate),
        "noise_mode": noise_mode,
        "target_perturbations_by_true_edges": target_count,
        "actual_perturbations": actual_perturbations,
        "actual_perturbation_ratio_by_true_edges": (
            actual_perturbations / q_true_edges if q_true_edges > 0 else 0.0
        ),
        "actual_perturbation_ratio_by_entries": actual_perturbations / total_entries,
        "Q_true_num_edges": q_true_edges,
        "Q_noise_num_edges": int(q_noise.sum()),
        "Q_true_density": q_true_edges / total_entries,
        "Q_noise_density": float(q_noise.mean()),
        "deleted_edges": positions_to_lists(deleted_edges),
        "added_edges": positions_to_lists(added_edges),
        "flipped_positions": flipped_positions,
        "delete_operation_edges": delete_operation_edges,
        "add_operation_edges": add_operation_edges,
        "operation_counts_requested": {
            "delete": delete_count,
            "add": add_count,
            "flip": flip_count,
        },
        "operation_counts_applied": {
            "delete": len(delete_operation_edges),
            "add": len(add_operation_edges),
            "flip": len(flipped_positions),
        },
        "preserve_item_coverage": preserve_item_coverage,
    }
    return q_noise, info


def parse_mixed_noise_weights(value: str) -> Tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--mixed_noise_weights must be formatted as delete,add,flip."
        )
    try:
        weights = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Noise weights must be numeric.") from exc
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic Q-masked MIRT dataset with noisy Q matrices."
    )
    parser.add_argument("--num_students", type=int, default=2000)
    parser.add_argument("--num_items", type=int, default=100)
    parser.add_argument("--num_concepts", type=int, default=20)
    parser.add_argument("--min_concepts_per_item", type=int, default=1)
    parser.add_argument("--max_concepts_per_item", type=int, default=3)
    parser.add_argument("--min_items_per_concept", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_dir", default="data/synthetic_weak_q")
    parser.add_argument(
        "--response_sampling",
        choices=["bernoulli", "threshold"],
        default="bernoulli",
    )
    parser.add_argument(
        "--observation_rate",
        type=float,
        default=1.0,
        help=(
            "Fraction of student-item pairs written to response.csv. "
            "Use values below 1.0 to simulate sparse response logs."
        ),
    )
    parser.add_argument(
        "--min_student_observations",
        type=int,
        default=1,
        help="Minimum observed responses per student in response.csv.",
    )
    parser.add_argument(
        "--min_item_observations",
        type=int,
        default=1,
        help="Minimum observed responses per item in response.csv.",
    )
    parser.add_argument(
        "--noise_rates",
        type=float,
        nargs="+",
        default=DEFAULT_NOISE_RATES,
        help="Noise rates measured relative to the number of true Q edges.",
    )
    parser.add_argument(
        "--noise_mode",
        choices=["delete", "add", "flip", "mixed"],
        default="mixed",
        help="How to perturb Q_true. The default mixed mode uses all three types.",
    )
    parser.add_argument(
        "--mixed_noise_weights",
        type=parse_mixed_noise_weights,
        default=(1.0, 1.0, 1.0),
        help="Comma-separated delete,add,flip weights for mixed mode.",
    )
    parser.add_argument(
        "--allow_empty_noise_items",
        action="store_true",
        help="Allow deletion/flip noise to produce all-zero item rows in Q_noise.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_generation_args(args)

    rng = np.random.default_rng(args.seed)
    output_root = Path(args.output_dir)
    output_dir = output_root / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    q_true = generate_q_true(
        num_items=args.num_items,
        num_concepts=args.num_concepts,
        min_concepts_per_item=args.min_concepts_per_item,
        max_concepts_per_item=args.max_concepts_per_item,
        min_items_per_concept=args.min_items_per_concept,
        rng=rng,
    )

    theta_true = rng.normal(0.0, 1.0, size=(args.num_students, args.num_concepts))
    mastery_true = sigmoid(theta_true)
    item_discrimination = rng.lognormal(
        mean=0.0, sigma=0.35, size=(args.num_items, args.num_concepts)
    )
    item_difficulty = rng.normal(0.0, 1.0, size=args.num_items)
    response_prob, response = generate_responses(
        q_true=q_true,
        theta_true=theta_true,
        item_discrimination=item_discrimination,
        item_difficulty=item_difficulty,
        response_sampling=args.response_sampling,
        rng=rng,
    )
    observation_mask = generate_observation_mask(
        num_students=args.num_students,
        num_items=args.num_items,
        observation_rate=args.observation_rate,
        min_student_observations=args.min_student_observations,
        min_item_observations=args.min_item_observations,
        rng=rng,
    )
    observed_response_count = int(observation_mask.sum())
    observed_scores = response[observation_mask]
    observed_correct_rate = float(observed_scores.mean()) if observed_response_count else 0.0
    actual_observation_rate = observed_response_count / response.size

    np.save(output_dir / "Q_true.npy", q_true)
    np.save(output_dir / "theta_true.npy", theta_true)
    np.save(output_dir / "mastery_true.npy", mastery_true)
    np.save(output_dir / "item_discrimination.npy", item_discrimination)
    np.save(output_dir / "item_difficulty.npy", item_difficulty)
    np.save(output_dir / "response_prob.npy", response_prob)
    save_response_csv(output_dir / "response.csv", response, observation_mask)

    metadata = {
        "num_students": args.num_students,
        "num_items": args.num_items,
        "num_concepts": args.num_concepts,
        "seed": args.seed,
        "min_concepts_per_item": args.min_concepts_per_item,
        "max_concepts_per_item": args.max_concepts_per_item,
        "min_items_per_concept": args.min_items_per_concept,
        "noise_rates": [float(rate) for rate in args.noise_rates],
        "noise_mode": args.noise_mode,
        "mixed_noise_weights": list(args.mixed_noise_weights),
        "generator": "q_masked_mirt",
        "response_sampling": args.response_sampling,
        "response_observation": "dense" if actual_observation_rate == 1.0 else "sparse",
        "observation_rate_requested": args.observation_rate,
        "observation_rate_actual": actual_observation_rate,
        "min_student_observations": args.min_student_observations,
        "min_item_observations": args.min_item_observations,
        "observed_response_count": observed_response_count,
        "full_response_count": int(response.size),
        "observed_correct_rate": observed_correct_rate,
        "full_response_correct_rate": float(response.mean()),
        "response_columns": ["user_id", "item_id", "score"],
        "training_visible_files": ["response.csv", "Q_noise_<rate>.npy"],
        "hidden_or_evaluation_only_files": [
            "Q_true.npy",
            "theta_true.npy",
            "mastery_true.npy",
            "item_discrimination.npy",
            "item_difficulty.npy",
            "response_prob.npy",
        ],
    }
    write_json(output_dir / "metadata.json", metadata)

    preserve_item_coverage = not args.allow_empty_noise_items
    noise_summaries = []
    for noise_rate in args.noise_rates:
        q_noise, noise_info = perturb_q_true(
            q_true=q_true,
            noise_rate=float(noise_rate),
            noise_mode=args.noise_mode,
            mixed_noise_weights=args.mixed_noise_weights,
            preserve_item_coverage=preserve_item_coverage,
            rng=rng,
        )
        label = rate_label(float(noise_rate))
        np.save(output_dir / f"Q_noise_{label}.npy", q_noise)
        write_json(output_dir / f"noise_info_{label}.json", noise_info)
        noise_summaries.append(
            {
                "rate": label,
                "density": float(q_noise.mean()),
                "actual_perturbations": int(noise_info["actual_perturbations"]),
            }
        )

    print("Synthetic weak-Q dataset generated.")
    print(f"Output directory: {output_dir}")
    print(f"Students: {args.num_students}")
    print(f"Items: {args.num_items}")
    print(f"Concepts: {args.num_concepts}")
    print(f"Full student-item pairs: {args.num_students * args.num_items}")
    print(f"Observed responses: {observed_response_count}")
    print(f"Requested observation rate: {args.observation_rate:.6f}")
    print(f"Actual observation rate: {actual_observation_rate:.6f}")
    print(f"Min observations per student: {int(observation_mask.sum(axis=1).min())}")
    print(f"Min observations per item: {int(observation_mask.sum(axis=0).min())}")
    print(f"Observed correct rate: {observed_correct_rate:.6f}")
    print(f"Full matrix correct rate: {float(response.mean()):.6f}")
    print(f"Q_true density: {float(q_true.mean()):.6f}")
    print("Q_noise summary:")
    for summary in noise_summaries:
        print(
            "  "
            f"rate={summary['rate']} "
            f"density={summary['density']:.6f} "
            f"actual_perturbations={summary['actual_perturbations']}"
        )


if __name__ == "__main__":
    main()
