"""Prepare a bounded 2015 Junyi subset with topic-derived Q for G-NCDM."""

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


DEFAULT_SOURCE = Path(r"E:\Desktop\数据集\junyi")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "junyi2015"
ARCHIVE_NAME = "junyi_ProblemLog_original.zip"
MEMBER_NAME = "junyi_ProblemLog_original.csv"
TABLE_NAME = "junyi_Exercise_table.csv"
USECOLS = ["user_id", "exercise", "correct", "time_done"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def chunks(archive: Path):
    with zipfile.ZipFile(archive) as z, z.open(MEMBER_NAME) as stream:
        yield from pd.read_csv(stream, usecols=USECOLS, chunksize=300_000, low_memory=False)


def eligible(frame: pd.DataFrame, known_exercises: set[str]) -> pd.DataFrame:
    correct = frame["correct"].astype(str).str.lower()
    mask = (
        frame["user_id"].notna()
        & frame["exercise"].isin(known_exercises)
        & frame["time_done"].notna()
        & correct.isin(("true", "false"))
    )
    return frame.loc[mask]


def top_ids(counts: Counter, limit: int) -> list:
    return [key for key, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]]


def prepare(source_dir: Path, output: Path, max_users: int, max_items: int,
            min_items_per_user: int, seed: int) -> dict:
    archive = source_dir / ARCHIVE_NAME
    table_path = source_dir / TABLE_NAME
    readme_path = source_dir / "README.txt"
    for path in (archive, table_path, readme_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(output)
    table = pd.read_csv(table_path, encoding="utf-8")
    table = table[table["topic"].notna() & table["name"].notna()]
    if (table.groupby("name")["topic"].nunique() > 1).any():
        raise ValueError("Conflicting topic labels for one exercise")
    duplicate_metadata_rows = int(table.duplicated("name").sum())
    table = table.drop_duplicates("name")
    exercise_topic = dict(zip(table["name"], table["topic"]))
    known_exercises = set(exercise_topic)

    user_counts, exercise_counts = Counter(), Counter()
    raw_rows = eligible_rows = 0
    for frame in chunks(archive):
        raw_rows += len(frame)
        frame = eligible(frame, known_exercises)
        eligible_rows += len(frame)
        user_counts.update(frame.user_id.astype(np.int64).value_counts().to_dict())
        exercise_counts.update(frame.exercise.value_counts().to_dict())
    chosen_users = set(top_ids(user_counts, max_users))
    chosen_exercises = set(top_ids(exercise_counts, max_items))

    first: dict[tuple[int, str], tuple[int, int]] = {}
    selected_rows = 0
    for frame in chunks(archive):
        frame = eligible(frame, known_exercises)
        frame = frame[frame.user_id.isin(chosen_users)
                      & frame.exercise.isin(chosen_exercises)]
        selected_rows += len(frame)
        for user, exercise, correct, timestamp in frame[
            ["user_id", "exercise", "correct", "time_done"]
        ].itertuples(index=False, name=None):
            pair = (int(user), exercise)
            time_key = int(timestamp)
            if pair not in first or time_key < first[pair][0]:
                first[pair] = (time_key, int(str(correct).lower() == "true"))

    pair_counts = Counter(user for user, _ in first)
    kept_users = sorted(user for user, count in pair_counts.items()
                        if count >= min_items_per_user)
    kept_user_set = set(kept_users)
    kept_exercises = sorted({exercise for user, exercise in first
                             if user in kept_user_set})
    if len(kept_users) < 2 or len(kept_exercises) < 2:
        raise ValueError("Selection left too few students or exercises")
    topics = sorted({exercise_topic[exercise] for exercise in kept_exercises})
    user_index = {user: index for index, user in enumerate(kept_users)}
    exercise_index = {name: index for index, name in enumerate(kept_exercises)}
    topic_index = {topic: index for index, topic in enumerate(topics)}
    q = np.zeros((len(kept_exercises), len(topics)), dtype=np.uint8)
    for exercise in kept_exercises:
        q[exercise_index[exercise], topic_index[exercise_topic[exercise]]] = 1
    if not q.any(axis=0).all() or not q.any(axis=1).all():
        raise ValueError("Q has an empty item or topic")

    by_user: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (user, exercise), (_, score) in first.items():
        if user in kept_user_set:
            by_user[user].append((exercise_index[exercise], score))
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
    if len({row[1] for row in splits["train"]}) != len(kept_exercises):
        raise ValueError("Some selected exercises have no training response")

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
        "dataset": "junyi2015",
        "source": {
            "directory": str(source_dir.resolve()),
            "archive": ARCHIVE_NAME,
            "member": MEMBER_NAME,
            "exercise_table": TABLE_NAME,
            "release_url": "https://base.ustc.edu.cn/data/JunyiAcademy_Math_Practicing_Log/",
            "archive_sha256": sha256_file(archive),
            "exercise_table_sha256": sha256_file(table_path),
            "readme_sha256": sha256_file(readme_path),
            "terms": "Noncommercial use; cite Chang, Hsu & Chen, EDM 2015, as stated in local README.txt.",
        },
        "preparation": {
            "subset": True,
            "raw_rows": raw_rows,
            "eligible_rows": eligible_rows,
            "duplicate_identical_metadata_rows_removed": duplicate_metadata_rows,
            "selected_rows_before_pair_dedup": selected_rows,
            "selection": "Top students and exercises by eligible attempt frequency; no score-based ranking.",
            "max_users": max_users,
            "max_items": max_items,
            "min_distinct_items_per_user": min_items_per_user,
            "filters": "Known exercise with nonempty topic, boolean correct, nonmissing timestamp.",
            "score": "First attempt correct field (true=1, false=0).",
            "pair_dedup": "Earliest time_done per student-exercise; first source row breaks ties.",
            "q_reference": "One-hot exercise topic from exercise metadata; topic is a proxy concept, not expert mastery ground truth.",
            "q_true_file": "Exact copy of Q_matrix.npy for existing runner compatibility; reference Q only.",
            "id_mapping": "Ascending retained source user_id and exercise name mapped to zero-based IDs.",
            "split": "Per-student seeded random 70/10/20 interaction split after pair dedup.",
            "split_seed": seed,
        },
        "raw_ids": {
            "users": kept_users,
            "items": kept_exercises,
            "concepts": topics,
        },
        "shape": {
            "students": len(kept_users),
            "items": len(kept_exercises),
            "concepts": len(topics),
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
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-users", type=int, default=5000)
    parser.add_argument("--max-items", type=int, default=500)
    parser.add_argument("--min-items-per-user", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if min(args.max_users, args.max_items, args.min_items_per_user) < 2:
        parser.error("Selection limits must be at least two")
    result = prepare(args.source_dir, args.output, args.max_users, args.max_items,
                     args.min_items_per_user, args.seed)
    print(json.dumps({"shape": result["shape"], "splits": result["splits"]}))


if __name__ == "__main__":
    main()
