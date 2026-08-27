"""Run five independent trials of only the latest reliability-gated model.

Completed trials are skipped.  Interrupted attempts are preserved and a new
attempt directory is created on restart.  No RouteNet-Fermi baseline or older
Queue-aware model is launched by this file.

Compatible with Python 3.6 and TensorFlow 2.6.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCALABILITY_ROOT = HERE.parent
REPOSITORY_ROOT = SCALABILITY_ROOT.parent
DEFAULT_SOURCE_ROOT = REPOSITORY_ROOT.parent / "RouteNet-Fermi"
CONFIG_FILE = HERE / "experiment_config.json"
ENTRYPOINT = HERE / "train_reliability_gated_scalability.py"
AGGREGATION = "reliability_gated_distribution_aware"


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def timestamp():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def load_config():
    return load_json(CONFIG_FILE)


def parse_args(config):
    parser = argparse.ArgumentParser(
        description="Run five seeds of only the reliability-gated latest model."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--seeds", nargs="+", type=int, default=config["seeds"])
    parser.add_argument("--epochs", type=int, default=config["epochs"])
    parser.add_argument(
        "--steps-per-epoch", type=int, default=config["steps_per_epoch"]
    )
    parser.add_argument(
        "--validation-file-modulus",
        type=int,
        default=config["validation_file_modulus"],
    )
    parser.add_argument(
        "--shuffle-buffer", type=int, default=config["shuffle_buffer"]
    )
    parser.add_argument(
        "--learning-rate", type=float, default=config["learning_rate"]
    )
    parser.add_argument("--clipnorm", type=float, default=config["clipnorm"])
    parser.add_argument(
        "--lr-patience", type=int, default=config["lr_patience"]
    )
    parser.add_argument(
        "--reliability-tau", type=float, default=config["reliability_tau"]
    )
    parser.add_argument(
        "--encoder-hidden-dim",
        type=int,
        default=config["encoder_hidden_dim"],
    )
    parser.add_argument("--skip-environment-check", action="store_true")
    parser.add_argument("--skip-final-summary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def valid_completed_run(run_dir, seed, reliability_tau):
    result_file = run_dir / "results.json"
    topology_file = run_dir / "test_by_topology_size.csv"
    state_file = run_dir / "training_state.json"
    config_file = run_dir / "config.json"
    if not all(
        path.is_file()
        for path in (result_file, topology_file, state_file, config_file)
    ):
        return False
    try:
        result = load_json(result_file)
        state = load_json(state_file)
        run_config = load_json(config_file)
        if int(result.get("seed")) != int(seed):
            return False
        if result.get("aggregation") != AGGREGATION:
            return False
        if state.get("status") != "complete":
            return False
        if abs(float(run_config.get("reliability_tau")) - reliability_tau) > 1.0e-12:
            return False
        with topology_file.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 25:
            return False
        if any("topology_nodes" not in row for row in rows):
            return False
    except (OSError, TypeError, ValueError):
        return False
    return True


def completed_candidates(seed, reliability_tau):
    seed_root = HERE / "runs" / "seed_{}".format(seed)
    if not seed_root.is_dir():
        return []
    return [
        child
        for child in sorted(seed_root.iterdir())
        if child.is_dir() and valid_completed_run(child, seed, reliability_tau)
    ]


def next_attempt_dir(seed):
    seed_root = HERE / "runs" / "seed_{}".format(seed)
    seed_root.mkdir(parents=True, exist_ok=True)
    attempt = seed_root / "attempt_{}".format(timestamp())
    suffix = 2
    while attempt.exists():
        attempt = seed_root / "attempt_{}_{}".format(timestamp(), suffix)
        suffix += 1
    return attempt


def command_text(command):
    return " ".join(
        '"{}"'.format(value) if " " in str(value) else str(value)
        for value in command
    )


def build_command(args, seed, attempt_dir):
    return [
        str(args.python_bin),
        "-u",
        str(ENTRYPOINT),
        "--source-root",
        str(args.source_root.resolve()),
        "--output-root",
        str(attempt_dir.parent),
        "--run-name",
        attempt_dir.name,
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--steps-per-epoch",
        str(args.steps_per_epoch),
        "--validation-file-modulus",
        str(args.validation_file_modulus),
        "--shuffle-buffer",
        str(args.shuffle_buffer),
        "--learning-rate",
        str(args.learning_rate),
        "--clipnorm",
        str(args.clipnorm),
        "--lr-patience",
        str(args.lr_patience),
        "--reliability-tau",
        str(args.reliability_tau),
        "--encoder-hidden-dim",
        str(args.encoder_hidden_dim),
        "--fit-verbose",
        "2",
        "--evaluate-verbose",
        "2",
    ]


def snapshot_code(run_dir):
    snapshot = run_dir / "code_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    for source in (
        HERE / "reliability_gated_model.py",
        HERE / "train_reliability_gated_scalability.py",
        HERE / "run_five_seeds.py",
        HERE / "summarize_five_seeds.py",
        HERE / "experiment_config.json",
        HERE / "README.md",
    ):
        if source.is_file():
            shutil.copy2(str(source), str(snapshot / source.name))


def run_environment_check(args):
    command = [
        str(args.python_bin),
        "-u",
        str(SCALABILITY_ROOT / "check_runpod.py"),
        "--source-root",
        str(args.source_root.resolve()),
    ]
    print("Environment check: {}".format(command_text(command)), flush=True)
    subprocess.check_call(command)


def run_summary(args):
    command = [
        str(args.python_bin),
        "-u",
        str(HERE / "summarize_five_seeds.py"),
        "--seeds",
    ] + [str(seed) for seed in args.seeds]
    print("Final summary: {}".format(command_text(command)), flush=True)
    subprocess.check_call(command)


def main():
    config = load_config()
    args = parse_args(config)
    args.source_root = args.source_root.resolve()
    if args.reliability_tau <= 0.0:
        raise ValueError("--reliability-tau must be positive")
    for required in (
        args.source_root,
        ENTRYPOINT,
        SCALABILITY_ROOT / "train_scalability.py",
    ):
        if not Path(required).exists():
            raise FileNotFoundError("Required path not found: {}".format(required))

    print("Latest model only: {}".format(AGGREGATION), flush=True)
    print("Seeds: {}".format(args.seeds), flush=True)
    print("Reliability tau: {}".format(args.reliability_tau), flush=True)
    print("Dry run: {}".format(args.dry_run), flush=True)

    if args.dry_run:
        for seed in args.seeds:
            existing = completed_candidates(seed, args.reliability_tau)
            if existing:
                print("SKIP complete seed {}: {}".format(seed, existing[0]))
            else:
                fake = (
                    HERE
                    / "runs"
                    / "seed_{}".format(seed)
                    / "attempt_DRY_RUN"
                )
                print(command_text(build_command(args, seed, fake)))
        return

    if not args.skip_environment_check:
        run_environment_check(args)

    state_file = HERE / "five_seed_state.json"
    state = {
        "status": "running",
        "started_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "updated_at": None,
        "model": AGGREGATION,
        "requested_seeds": args.seeds,
        "current_seed": None,
        "current_run_dir": None,
        "completed": [],
        "skipped_existing": [],
    }
    atomic_write_json(state_file, state)
    environment = os.environ.copy()
    environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    environment["PYTHONUNBUFFERED"] = "1"

    try:
        for seed in args.seeds:
            existing = completed_candidates(seed, args.reliability_tau)
            if len(existing) > 1:
                raise RuntimeError(
                    "Multiple complete runs found for seed {}: {}".format(
                        seed, existing
                    )
                )
            if existing:
                print("SKIP complete seed {}: {}".format(seed, existing[0]), flush=True)
                state["skipped_existing"].append(
                    {"seed": seed, "run_dir": str(existing[0])}
                )
                state["updated_at"] = datetime.utcnow().replace(
                    microsecond=0
                ).isoformat() + "Z"
                atomic_write_json(state_file, state)
                continue

            attempt_dir = next_attempt_dir(seed)
            command = build_command(args, seed, attempt_dir)
            state["current_seed"] = seed
            state["current_run_dir"] = str(attempt_dir)
            state["updated_at"] = datetime.utcnow().replace(
                microsecond=0
            ).isoformat() + "Z"
            atomic_write_json(state_file, state)

            print("=" * 78, flush=True)
            print("START seed {} -> {}".format(seed, attempt_dir), flush=True)
            print(command_text(command), flush=True)
            subprocess.check_call(command, env=environment)

            if not valid_completed_run(attempt_dir, seed, args.reliability_tau):
                raise RuntimeError(
                    "Training returned successfully but artifacts are incomplete: {}".format(
                        attempt_dir
                    )
                )
            snapshot_code(attempt_dir)
            state["completed"].append(
                {"seed": seed, "run_dir": str(attempt_dir)}
            )
            state["updated_at"] = datetime.utcnow().replace(
                microsecond=0
            ).isoformat() + "Z"
            atomic_write_json(state_file, state)
            print("COMPLETE seed {}".format(seed), flush=True)

        if not args.skip_final_summary:
            run_summary(args)

        state["status"] = "complete"
        state["current_seed"] = None
        state["current_run_dir"] = None
        state["updated_at"] = datetime.utcnow().replace(
            microsecond=0
        ).isoformat() + "Z"
        atomic_write_json(state_file, state)
        print("All requested latest-model trials completed.", flush=True)
    except Exception as error:
        state["status"] = "failed"
        state["error"] = repr(error)
        state["updated_at"] = datetime.utcnow().replace(
            microsecond=0
        ).isoformat() + "Z"
        atomic_write_json(state_file, state)
        raise


if __name__ == "__main__":
    main()

