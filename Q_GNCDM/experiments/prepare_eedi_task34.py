"""Prepare local Eedi NeurIPS 2020 Task 3/4 data for Q_GNCDM.

The output is a seen-student interaction split.  It is not the competition's
official Task 4 cold-start split.  The metadata-derived Q is a reference Q,
not an expert-validated ground-truth Q.

Eedi licenses this data CC BY-NC-ND 4.0.  Do not distribute derived output.
See https://www.eedischool.com/projects/neurips-education-challenge .
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_SOURCE = Path(r"E:\Desktop\数据集\data_Eedi")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "eedi_task34"
SPLIT_SEED = 2026


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_subjects(path: Path) -> dict[int, dict]:
    subjects = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            subject_id = int(row["SubjectId"])
            parent = row["ParentId"]
            subjects[subject_id] = {
                "name": row["Name"],
                "parent": None if parent == "NULL" else int(parent),
                "level": int(row["Level"]),
            }
    return subjects


def level_two_ancestor(subject_id: int, subjects: dict[int, dict]) -> int | None:
    visited = set()
    while subject_id is not None:
        if subject_id in visited:
            raise ValueError(f"Cycle in subject hierarchy at {subject_id}")
        visited.add(subject_id)
        subject = subjects[subject_id]
        if subject["level"] == 2:
            return subject_id
        if subject["level"] < 2:
            return None
        subject_id = subject["parent"]
    return None


def load_question_concepts(path: Path, subjects: dict[int, dict]) -> dict[int, set[int]]:
    question_concepts = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            question_id = int(row["QuestionId"])
            raw_subjects = ast.literal_eval(row["SubjectId"])
            concepts = {
                concept
                for subject_id in raw_subjects
                if (concept := level_two_ancestor(int(subject_id), subjects)) is not None
            }
            if not concepts:
                raise ValueError(f"Question {question_id} has no Level-2 concept")
            if question_id in question_concepts:
                raise ValueError(f"Duplicate question metadata: {question_id}")
            question_concepts[question_id] = concepts
    return question_concepts


def load_responses(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    users, items, scores = [], [], []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            score = int(row["IsCorrect"])
            if score not in (0, 1):
                raise ValueError(f"Non-binary IsCorrect: {score}")
            users.append(int(row["UserId"]))
            items.append(int(row["QuestionId"]))
            scores.append(score)
    if not users:
        raise ValueError("No Eedi responses found")
    return tuple(np.asarray(values, dtype=np.int32) for values in (users, items, scores))


def split_by_student(user_ids: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    """Deterministic 70/10/20 split of each user's interactions."""
    student_order = np.argsort(user_ids, kind="stable")
    _, starts, counts = np.unique(
        user_ids[student_order], return_index=True, return_counts=True
    )
    split_parts: list[list[np.ndarray]] = [[], [], []]
    for student_id, start, count in zip(range(len(starts)), starts, counts):
        if count < 3:
            raise ValueError(f"Student {student_id} has only {count} responses")
        indices = student_order[start : start + count]
        shuffled = np.random.default_rng([seed, student_id]).permutation(indices)
        train_end = max(1, int(0.7 * count))
        valid_end = train_end + max(1, int(0.1 * count))
        if valid_end >= count:
            valid_end = count - 1
        for part, selected in zip(
            split_parts,
            (shuffled[:train_end], shuffled[train_end:valid_end], shuffled[valid_end:]),
        ):
            part.append(selected)
    return tuple(np.concatenate(part) for part in split_parts)


def write_split(path: Path, indices: np.ndarray, columns: tuple[np.ndarray, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("user_id", "item_id", "score"))
        writer.writerows(zip(*(column[indices] for column in columns)))


def prepare(source_dir: Path, output_dir: Path, split_seed: int) -> dict:
    source_files = {
        "responses": source_dir / "train_data" / "train_task_3_4.csv",
        "questions": source_dir / "metadata" / "question_metadata_task_3_4.csv",
        "subjects": source_dir / "metadata" / "subject_metadata.csv",
    }
    for path in source_files.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    subjects = load_subjects(source_files["subjects"])
    question_concepts = load_question_concepts(source_files["questions"], subjects)
    raw_users, raw_items, scores = load_responses(source_files["responses"])
    unique_users = np.unique(raw_users)
    unique_items = np.unique(raw_items)
    if set(unique_items.tolist()) != set(question_concepts):
        raise ValueError("Question metadata and response question IDs differ")
    user_ids = np.searchsorted(unique_users, raw_users).astype(np.int32)
    item_ids = np.searchsorted(unique_items, raw_items).astype(np.int32)
    pair_codes = user_ids.astype(np.int64) * len(unique_items) + item_ids
    if len(np.unique(pair_codes)) != len(pair_codes):
        raise ValueError("Repeated student-question pairs require a temporal protocol")

    concept_ids = sorted({concept for concepts in question_concepts.values() for concept in concepts})
    concept_to_index = {concept: index for index, concept in enumerate(concept_ids)}
    q = np.zeros((len(unique_items), len(concept_ids)), dtype=np.int8)
    for item_id, raw_item in enumerate(unique_items):
        for concept in question_concepts[int(raw_item)]:
            q[item_id, concept_to_index[concept]] = 1
    if not q.any(axis=1).all() or not q.any(axis=0).all():
        raise ValueError("Q has an empty question or concept")

    split_indices = split_by_student(user_ids, split_seed)
    if sum(len(indices) for indices in split_indices) != len(scores):
        raise AssertionError("Split did not cover every response")
    if len(np.unique(np.concatenate(split_indices))) != len(scores):
        raise AssertionError("Split indices overlap")

    output_dir.mkdir(parents=True, exist_ok=False)
    columns = (user_ids, item_ids, scores)
    split_names = ("train", "valid", "test")
    split_stats = {}
    for name, indices in zip(split_names, split_indices):
        write_split(output_dir / f"{name}.csv", indices, columns)
        split_stats[name] = {
            "responses": int(len(indices)),
            "students": int(len(np.unique(user_ids[indices]))),
            "correct_rate": float(np.mean(scores[indices])),
        }
    np.save(output_dir / "Q_matrix.npy", q, allow_pickle=False)
    np.save(output_dir / "Q_true.npy", q, allow_pickle=False)

    manifest = {
        "dataset": "eedi_task34",
        "source": {
            "name": "Eedi NeurIPS 2020 Education Challenge, Tasks 3/4 training data",
            "url": "https://www.eedischool.com/projects/neurips-education-challenge",
            "license": "CC BY-NC-ND 4.0; transformed data must not be redistributed",
            "files_sha256": {
                str(path.relative_to(source_dir)).replace("\\", "/"): sha256_file(path)
                for path in source_files.values()
            },
        },
        "preparation": {
            "script": "Q_GNCDM/experiments/prepare_eedi_task34.py",
            "split_seed": split_seed,
            "split_protocol": "per-student random 70/10/20 interaction split, seen students in all sets",
            "id_mapping": "0-based contiguous indices by ascending original UserId and QuestionId",
            "score": "IsCorrect (binary)",
            "q_reference": "Level-2 ancestor of each question's SubjectId tags; metadata-derived, not expert-validated",
            "q_true_file": "Exact duplicate of Q_matrix.npy for current evaluator compatibility; reference Q only",
            "official_competition_split": False,
            "derived_data_redistributable": False,
        },
        "shape": {
            "students": int(len(unique_users)),
            "items": int(len(unique_items)),
            "concepts": int(len(concept_ids)),
            "responses": int(len(scores)),
            "q_edges": int(q.sum()),
        },
        "concepts": [
            {"index": index, "source_subject_id": concept, "name": subjects[concept]["name"]}
            for index, concept in enumerate(concept_ids)
        ],
        "splits": split_stats,
        "files_sha256": {
            path.name: sha256_file(path)
            for path in [
                *(output_dir / f"{name}.csv" for name in split_names),
                output_dir / "Q_matrix.npy",
                output_dir / "Q_true.npy",
            ]
        },
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()
    manifest = prepare(args.source_dir, args.output_dir, args.split_seed)
    print(json.dumps({"shape": manifest["shape"], "splits": manifest["splits"]}, indent=2))


if __name__ == "__main__":
    main()
