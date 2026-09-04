"""Grid-search lambda weights for the weak-Q synthetic experiment.

This wrapper keeps each lambda combination in its own directory and delegates
the actual training to run_weak_q_synthetic.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_VALUES = "0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_values(raw: str) -> List[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("--values must contain at least one numeric value.")
    return values


def format_value(value: float) -> str:
    text = f"{value:.6g}"
    return text.replace("-", "m").replace(".", "p")


def combo_name(lambda_q: float, lambda_sparse: float, lambda_binary: float) -> str:
    return (
        f"lq_{format_value(lambda_q)}_"
        f"ls_{format_value(lambda_sparse)}_"
        f"lb_{format_value(lambda_binary)}"
    )


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)


def find_metrics(combo_dir: Path) -> List[Path]:
    return sorted(combo_dir.rglob("metrics.json"))


def build_command(
    args: argparse.Namespace,
    project_dir: Path,
    combo_dir: Path,
    run_tag: str,
    lambda_q: float,
    lambda_sparse: float,
    lambda_binary: float,
) -> List[str]:
    python_bin = args.python_bin or sys.executable
    script_path = project_dir / "experiments" / "run_weak_q_synthetic.py"
    command = [
        python_bin,
        str(script_path),
        "--data_dir",
        args.data_dir,
        "--noise_rate",
        str(args.noise),
        "--mode",
        "weak-q",
        "--epochs",
        str(args.epochs),
        "--results_dir",
        str(combo_dir),
        "--device",
        args.device,
        "--lambda_q",
        str(lambda_q),
        "--lambda_sparse",
        str(lambda_sparse),
        "--lambda_binary",
        str(lambda_binary),
        "--lambda_coverage",
        str(args.lambda_coverage),
        "--run_tag",
        run_tag,
    ]
    if args.patience is not None:
        command.extend(["--patience", str(args.patience)])
    return command


def command_text(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def run_one(
    args: argparse.Namespace,
    project_dir: Path,
    combo: Tuple[float, float, float],
) -> Dict[str, Any]:
    lambda_q, lambda_sparse, lambda_binary = combo
    run_tag = combo_name(lambda_q, lambda_sparse, lambda_binary)
    results_root = Path(args.results_dir)
    if not results_root.is_absolute():
        results_root = project_dir / results_root
    combo_dir = results_root / run_tag
    combo_dir.mkdir(parents=True, exist_ok=True)

    status_path = combo_dir / "status.json"
    config_path = combo_dir / "sweep_config.json"
    command_path = combo_dir / "command.txt"

    existing_metrics = find_metrics(combo_dir)
    if existing_metrics and not args.overwrite:
        status = {
            "status": "skipped",
            "reason": "metrics.json already exists",
            "result_dir": str(combo_dir),
            "metrics_paths": [str(path) for path in existing_metrics],
            "updated_at": now_iso(),
        }
        write_json(status_path, status)
        return status

    command = build_command(
        args,
        project_dir,
        combo_dir,
        run_tag,
        lambda_q,
        lambda_sparse,
        lambda_binary,
    )
    command_path.write_text(command_text(command) + "\n", encoding="utf-8")
    write_json(
        config_path,
        {
            "lambda_q": lambda_q,
            "lambda_sparse": lambda_sparse,
            "lambda_binary": lambda_binary,
            "lambda_coverage": args.lambda_coverage,
            "data_dir": args.data_dir,
            "noise": args.noise,
            "epochs": args.epochs,
            "device": args.device,
            "run_tag": run_tag,
            "combo_dir": str(combo_dir),
        },
    )

    if args.dry_run:
        status = {
            "status": "dry_run",
            "command": command_text(command),
            "result_dir": str(combo_dir),
            "updated_at": now_iso(),
        }
        write_json(status_path, status)
        print(command_text(command))
        return status

    status = {
        "status": "running",
        "started_at": now_iso(),
        "command": command_text(command),
        "result_dir": str(combo_dir),
    }
    write_json(status_path, status)

    stdout_path = combo_dir / "stdout.log"
    stderr_path = combo_dir / "stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout_fp, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_fp:
        completed = subprocess.run(
            command,
            cwd=str(project_dir),
            stdout=stdout_fp,
            stderr=stderr_fp,
            text=True,
            check=False,
        )

    metrics_paths = find_metrics(combo_dir)
    final_status = "success" if completed.returncode == 0 and metrics_paths else "failed"
    status.update(
        {
            "status": final_status,
            "finished_at": now_iso(),
            "returncode": completed.returncode,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "metrics_paths": [str(path) for path in metrics_paths],
        }
    )
    write_json(status_path, status)
    return status


def iter_combos(values: Iterable[float]) -> List[Tuple[float, float, float]]:
    return list(itertools.product(values, values, values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep weak-Q lambda weights.")
    parser.add_argument("--data_dir", default="data/synthetic_weak_q/seed_2026")
    parser.add_argument("--noise", type=float, default=0.2)
    parser.add_argument("--results_dir", default="results/weak_q_lambda_sweep")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--values", default=DEFAULT_VALUES)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lambda_coverage", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = parse_values(args.values)
    combos = iter_combos(values)
    project_dir = Path(__file__).resolve().parents[1]

    print(f"project_dir={project_dir}")
    print(f"num_combinations={len(combos)} max_workers={args.max_workers}")

    if args.max_workers <= 1:
        for combo in combos:
            status = run_one(args, project_dir, combo)
            print(f"{combo_name(*combo)}: {status['status']}")
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        future_to_combo = {
            pool.submit(run_one, args, project_dir, combo): combo for combo in combos
        }
        for future in concurrent.futures.as_completed(future_to_combo):
            combo = future_to_combo[future]
            try:
                status = future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"{combo_name(*combo)}: failed before launch: {exc}")
                continue
            print(f"{combo_name(*combo)}: {status['status']}")


if __name__ == "__main__":
    main()
