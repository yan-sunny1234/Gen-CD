"""Compatibility for Math1 data now stored in Q_GNCDM/data/math1.

Historical manifests remain unchanged; only these four files may be relocated.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEGACY_DIR = (ROOT / "GNCDM" / "data").resolve()
INTERIM_DIR = (ROOT / "Q_GNCDM" / "data").resolve()
CURRENT_DIR = (INTERIM_DIR / "math1").resolve()
MATH1_DATA_DIRS = frozenset({LEGACY_DIR, INTERIM_DIR, CURRENT_DIR})
MATH1_FILENAMES = frozenset({
    "math1_train_0.8_0.2.csv",
    "math1_valid_0.8_0.2.csv",
    "math1_test_0.8_0.2.csv",
    "math1_Q_matrix.npy",
})


def project_path(value):
    """Resolve a recorded path relative to the project, not the caller's cwd."""
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def source_path(filename):
    """Find a recorded source file, redirecting only the four old Math1 paths."""
    path = project_path(filename)
    if path.parent in (LEGACY_DIR, INTERIM_DIR) and path.name in MATH1_FILENAMES:
        return CURRENT_DIR / path.name
    return path


def data_dir_for_replay(dataset, data_dir):
    """Use the new directory when replaying an old Math1 manifest."""
    if dataset == "math1" and project_path(data_dir) in (LEGACY_DIR, INTERIM_DIR):
        return str(CURRENT_DIR)
    return data_dir


def _math1_sources(manifest):
    if manifest["args"]["dataset"] != "math1":
        return None
    sources = manifest["source_sha256"]
    if len(sources) != len(MATH1_FILENAMES):
        return None
    by_name = {}
    for filename, digest in sources.items():
        path = project_path(filename)
        if path.parent not in MATH1_DATA_DIRS:
            return None
        if path.name not in MATH1_FILENAMES or path.name in by_name:
            return None
        by_name[path.name] = digest
    return by_name if set(by_name) == MATH1_FILENAMES else None


def same_source_files(reference, candidate):
    """Preserve exact comparison except for the known Math1 directory move."""
    if reference["source_sha256"] == candidate["source_sha256"]:
        return True
    left = _math1_sources(reference)
    right = _math1_sources(candidate)
    return left is not None and right is not None and left == right


def same_controlled_args(reference, candidate, keys):
    """Allow only the three known Math1 data locations among controlled settings."""
    for key in keys:
        if reference[key] == candidate[key]:
            continue
        if key != "data_dir" or reference["dataset"] != "math1":
            return False
        if candidate["dataset"] != "math1":
            return False
        if not {project_path(reference[key]), project_path(candidate[key])}.issubset(
            MATH1_DATA_DIRS
        ):
            return False
    return True
