"""Prepare the objective questions of Math2 for binary CD experiments.

The source archive is the Math 2015 research release. Its first 16 Math2
questions are objective; the remaining four have fractional scores and are
excluded rather than thresholded. Run this script once before experiments:

    python -m Q_GNCDM.experiments.prepare_math2

The output uses zero-based student/item IDs and the same CSV columns as the
existing Q_GNCDM response loader.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = Path(r"E:\Desktop\数据集\math2015.rar")
DEFAULT_OUTPUT = PROJECT / "data" / "math2"
SOURCE_PREFIX = "math2015/Math2/"
SPLIT_NAMES = ("train", "valid", "test")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_member(archive: Path, member: str) -> bytes:
    result = subprocess.run(
        ["tar", "-xOf", str(archive), member],
        capture_output=True,
        check=True,
    )
    if not result.stdout:
        raise ValueError(f"Empty archive member: {member}")
    return result.stdout


def parse_objective_items(description: bytes) -> list[int]:
    entries = [line.split() for line in description.decode("utf-8").splitlines()[1:]
               if line.strip()]
    if len(entries) != 20 or any(len(entry) != 3 for entry in entries):
        raise ValueError("Unexpected Math2 problem description")
    if [int(entry[0]) for entry in entries] != list(range(1, 21)):
        raise ValueError("Math2 questions are not numbered 1 through 20")
    objective = [int(entry[0]) - 1 for entry in entries if entry[1] == "Obj"]
    if objective != list(range(16)):
        raise ValueError("Expected the first 16 Math2 questions to be objective")
    return objective


def prepare(archive: Path, output: Path, seed: int) -> dict:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    members = {
        name: archive_member(archive, SOURCE_PREFIX + name)
        for name in ("data.txt", "q.txt", "problemdesc.txt")
    }
    terms = archive_member(archive, "math2015/TermsOfUse.txt")
    objective_items = parse_objective_items(members["problemdesc.txt"])
    responses = np.loadtxt(io.BytesIO(members["data.txt"]), dtype=np.float64)
    q_full = np.loadtxt(io.BytesIO(members["q.txt"]), dtype=np.int64)
    if responses.shape != (3911, 20) or q_full.shape != (20, 16):
        raise ValueError(f"Unexpected Math2 shapes: {responses.shape}, {q_full.shape}")
    responses = responses[:, objective_items]
    q = q_full[objective_items].astype(np.uint8)
    if not np.isin(responses, [0, 1]).all():
        raise ValueError("Objective responses must be exactly binary")
    if not np.isin(q, [0, 1]).all() or not q.any(axis=0).all() or not q.any(axis=1).all():
        raise ValueError("Expert Q must be binary and cover every question and skill")

    # With 16 questions, rounding the target 70/10/20 split gives 11/2/3
    # responses per student. A seeded permutation prevents label-based splits.
    per_student = (11, 2, 3)
    rng = np.random.default_rng(seed)
    rows: dict[str, list[tuple[int, int, int]]] = {name: [] for name in SPLIT_NAMES}
    for student in range(len(responses)):
        order = rng.permutation(len(objective_items))
        start = 0
        for name, count in zip(SPLIT_NAMES, per_student):
            for item in order[start:start + count]:
                rows[name].append((student, int(item), int(responses[student, item])))
            start += count

    output.mkdir(parents=True, exist_ok=False)
    outputs = []
    for name in SPLIT_NAMES:
        path = output / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("user_id", "item_id", "score"))
            writer.writerows(rows[name])
        outputs.append(path)
    for name in ("Q_matrix.npy", "Q_true.npy"):
        path = output / name
        np.save(path, q)
        outputs.append(path)

    manifest = {
        "dataset": "math2",
        "source": {
            "archive": str(archive.resolve()),
            "url": "http://staff.ustc.edu.cn/~qiliuql/data/math2015.rar",
            "archive_sha256": sha256_file(archive),
            "member_sha256": {SOURCE_PREFIX + name: sha256_bytes(data)
                              for name, data in members.items()},
            "terms_member_sha256": sha256_bytes(terms),
            "citation": "Runze Wu et al., Cognitive Modelling for Predicting Examinee Performance, IJCAI 2015.",
            "terms_text": terms.decode("utf-8").strip(),
        },
        "preparation": {
            "response_source": "normalized Math2/data.txt",
            "selected_original_question_numbers": [item + 1 for item in objective_items],
            "excluded_original_question_numbers": [17, 18, 19, 20],
            "exclusion_reason": "Subjective questions have fractional scores; no thresholding applied.",
            "student_ids": "Original row order, zero-based.",
            "item_ids": "Selected objective question order, zero-based.",
            "q_source": "Expert Math2/q.txt, selected question rows.",
            "q_true": "Exact duplicate of Q_matrix.npy for existing runner diagnostics.",
        },
        "split": {
            "method": "Independent seeded question permutation per student.",
            "seed": seed,
            "target_ratios": {"train": 0.7, "valid": 0.1, "test": 0.2},
            "per_student_counts": dict(zip(SPLIT_NAMES, per_student)),
            "counts": {name: len(rows[name]) for name in SPLIT_NAMES},
            "observed_ratios": {name: len(rows[name]) / responses.size
                                for name in SPLIT_NAMES},
        },
        "dimensions": {
            "students": int(responses.shape[0]),
            "items": int(responses.shape[1]),
            "concepts": int(q.shape[1]),
            "interactions": int(responses.size),
        },
        "files_sha256": {path.name: sha256_file(path) for path in outputs},
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    manifest = prepare(args.archive, args.output, args.seed)
    print(json.dumps({"output": str(args.output.resolve()),
                      "dimensions": manifest["dimensions"],
                      "split_counts": manifest["split"]["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
