"""Prepare a bounded ASSISTments 2012-13 subset for the dense G-NCDM runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SOURCE = Path(r"E:\Desktop\数据集\Assist.zip")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "assist2012"
SOURCE_MEMBER = "2012-2013-data-with-predictions-4-final.csv"
USECOLS = ["user_id", "problem_id", "skill_id", "correct", "original", "problem_type", "start_time"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def chunks(archive: Path):
    with zipfile.ZipFile(archive) as z, z.open(SOURCE_MEMBER) as stream:
        yield from pd.read_csv(stream, usecols=USECOLS, chunksize=200_000, low_memory=False)


def eligible(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("user_id", "problem_id", "skill_id", "correct", "original"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    mask = (
        frame["user_id"].notna()
        & frame["problem_id"].notna()
        & frame["skill_id"].notna()
        & (frame["original"] == 1)
        & frame["correct"].between(0, 1)
        & (frame["problem_type"] != "open_response")
        & (frame["skill_id"] % 1 == 0)
    )
    return frame.loc[mask]


def top_ids(counts: Counter, limit: int) -> list:
    return [key for key, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]]


def prepare(source: Path, output: Path, max_users: int, max_items: int,
            min_items_per_user: int, min_users_per_item: int, seed: int) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(output)
    user_counts, item_counts = Counter(), Counter()
    raw_rows = eligible_rows = 0
    for frame in chunks(source):
        raw_rows += len(frame)
        frame = eligible(frame)
        eligible_rows += len(frame)
        user_counts.update(frame.user_id.astype(np.int64).value_counts().to_dict())
        item_counts.update(frame.problem_id.astype(np.int64).value_counts().to_dict())
    chosen_users = set(top_ids(user_counts, max_users))
    chosen_items = set(top_ids(item_counts, max_items))

    first: dict[tuple[int, int], tuple[str, int]] = {}
    item_skills: dict[int, set[int]] = defaultdict(set)
    selected_rows = 0
    for frame in chunks(source):
        frame = eligible(frame)
        frame = frame[frame.user_id.isin(chosen_users) & frame.problem_id.isin(chosen_items)]
        selected_rows += len(frame)
        for user, item, skill, credit, timestamp in frame[
            ["user_id", "problem_id", "skill_id", "correct", "start_time"]
        ].itertuples(index=False, name=None):
            user, item, skill = int(user), int(item), int(skill)
            item_skills[item].add(skill)
            pair = (user, item)
            time_key = str(timestamp) if pd.notna(timestamp) else "9999"
            if pair not in first or time_key < first[pair][0]:
                first[pair] = (time_key, int(credit == 1))

    pair_counts = Counter(user for user, _ in first)
    kept_user_set = {user for user, count in pair_counts.items()
                     if count >= min_items_per_user}
    while True:
        item_counts = Counter(item for user, item in first if user in kept_user_set)
        kept_item_set = {item for item, count in item_counts.items()
                         if count >= min_users_per_item}
        user_counts = Counter(user for user, item in first if item in kept_item_set)
        updated_users = {user for user in kept_user_set
                         if user_counts[user] >= min_items_per_user}
        if updated_users == kept_user_set:
            break
        kept_user_set = updated_users
    kept_users = sorted(kept_user_set)
    kept_items = sorted(kept_item_set)
    if len(kept_users) < 2 or len(kept_items) < 2:
        raise ValueError("Selection left too few students or items")
    kept_concepts = sorted({skill for item in kept_items for skill in item_skills[item]})
    user_index = {user: index for index, user in enumerate(kept_users)}
    item_index = {item: index for index, item in enumerate(kept_items)}
    concept_index = {skill: index for index, skill in enumerate(kept_concepts)}
    q = np.zeros((len(kept_items), len(kept_concepts)), dtype=np.uint8)
    for item in kept_items:
        for skill in item_skills[item]:
            q[item_index[item], concept_index[skill]] = 1
    if not q.any(axis=0).all() or not q.any(axis=1).all():
        raise ValueError("Q has an empty item or concept")

    by_user: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (user, item), (_, score) in first.items():
        if user in kept_user_set and item in item_index:
            by_user[user].append((item_index[item], score))
    splits = {name: [] for name in ("train", "valid", "test")}
    for user in kept_users:
        responses = sorted(by_user[user])
        order = np.random.default_rng([seed, user]).permutation(len(responses))
        n_train = max(2, int(0.7 * len(responses)))
        n_valid = max(1, int(0.1 * len(responses)))
        n_valid = min(n_valid, len(responses) - n_train - 1)
        for name, positions in zip(splits, (
            order[:n_train], order[n_train:n_train + n_valid], order[n_train + n_valid:]
        )):
            splits[name].extend((user_index[user], *responses[int(pos)]) for pos in positions)
    if len({row[1] for row in splits["train"]}) != len(kept_items):
        raise ValueError("Some selected items have no training response")

    output.mkdir(parents=True, exist_ok=False)
    files = []
    for name, rows in splits.items():
        path = output / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("user_id", "item_id", "score"))
            writer.writerows(rows)
        files.append(path)
    for name in ("Q_matrix.npy", "Q_true.npy"):
        path = output / name
        np.save(path, q, allow_pickle=False)
        files.append(path)

    manifest = {
        "dataset": "assist2012",
        "source": {
            "archive": str(source.resolve()),
            "member": SOURCE_MEMBER,
            "official_url": "https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect",
            "archive_sha256": sha256_file(source),
            "citation": "Feng, Heffernan & Koedinger (2009), User Modeling and User-Adapted Interaction.",
        },
        "preparation": {
            "subset": True,
            "raw_rows": raw_rows,
            "eligible_rows": eligible_rows,
            "selected_rows_before_pair_dedup": selected_rows,
            "selection": "Top users and problems by frequency of eligible rows; no score-based ranking.",
            "max_users": max_users,
            "max_items": max_items,
            "min_distinct_items_per_user": min_items_per_user,
            "min_distinct_users_per_item": min_users_per_item,
            "filters": "Main problems only (original=1), labeled skill_id, non-open-response, valid 0..1 credit.",
            "score": "1 iff correct=1; all partial credit below 1 maps to 0, as source documentation recommends.",
            "pair_dedup": "Earliest start_time per student-problem; first source row breaks ties.",
            "q_reference": "Union of observed source skill_id tags per selected problem; not expert-validated ground truth.",
            "q_true_file": "Exact copy of Q_matrix.npy for existing runner compatibility; reference Q only.",
            "id_mapping": "Ascending retained raw user_id and problem_id mapped to zero-based IDs.",
            "split": "Per-student seeded random 70/10/20 interaction split after pair dedup.",
            "split_seed": seed,
        },
        "raw_ids": {
            "users": kept_users,
            "items": kept_items,
            "concepts": kept_concepts,
        },
        "shape": {
            "students": len(kept_users),
            "items": len(kept_items),
            "concepts": len(kept_concepts),
            "responses": sum(map(len, splits.values())),
            "q_edges": int(q.sum()),
        },
        "splits": {name: len(rows) for name, rows in splits.items()},
        "files_sha256": {path.name: sha256_file(path) for path in files},
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-users", type=int, default=5000)
    parser.add_argument("--max-items", type=int, default=1000)
    parser.add_argument("--min-items-per-user", type=int, default=5)
    parser.add_argument("--min-users-per-item", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if min(args.max_users, args.max_items, args.min_items_per_user,
           args.min_users_per_item) < 2:
        parser.error("Selection limits must be at least two")
    result = prepare(args.source, args.output, args.max_users, args.max_items,
                     args.min_items_per_user, args.min_users_per_item, args.seed)
    print(json.dumps({"shape": result["shape"], "splits": result["splits"]}))


if __name__ == "__main__":
    main()
