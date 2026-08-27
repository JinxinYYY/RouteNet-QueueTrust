"""Summarize only reliability-gated trials across independent seeds."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
AGGREGATION = "reliability_gated_distribution_aware"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1234, 2026, 3407, 5678, 9012]
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def find_trial(seed):
    seed_root = HERE / "runs" / "seed_{}".format(seed)
    candidates = []
    if seed_root.is_dir():
        for child in sorted(seed_root.iterdir()):
            result_file = child / "results.json"
            topology_file = child / "test_by_topology_size.csv"
            state_file = child / "training_state.json"
            if not all(path.is_file() for path in (result_file, topology_file, state_file)):
                continue
            result = load_json(result_file)
            state = load_json(state_file)
            if (
                int(result.get("seed")) == int(seed)
                and result.get("aggregation") == AGGREGATION
                and state.get("status") == "complete"
            ):
                candidates.append(child)
    if len(candidates) > 1:
        raise RuntimeError("Multiple complete trials for seed {}: {}".format(seed, candidates))
    return candidates[0] if candidates else None


def mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def save_figure(fig, path):
    fig.savefig(str(path.with_suffix(".png")), dpi=220, bbox_inches="tight")
    fig.savefig(str(path.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    trials = []
    missing = []
    for seed in args.seeds:
        run_dir = find_trial(seed)
        if run_dir is None:
            missing.append(seed)
            continue
        result = load_json(run_dir / "results.json")
        with (run_dir / "test_by_topology_size.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            topology = list(csv.DictReader(handle))
        if len(topology) != 25:
            raise ValueError("Expected 25 topology rows in {}".format(run_dir))
        trials.append({"seed": seed, "run_dir": run_dir, "result": result, "topology": topology})

    if missing and not args.allow_incomplete:
        raise RuntimeError("Missing complete trials for seeds: {}".format(missing))
    if not trials:
        raise RuntimeError("No complete trials found")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else HERE / "summaries" / "summary_{}".format(datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Summary directory is not empty: {}".format(output_dir))
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    topology_rows = []
    for trial in trials:
        result = trial["result"]
        overall_rows.append(
            {
                "seed": trial["seed"],
                "test_mape": float(result["test_mape"]),
                "best_validation_mape": float(result["best_validation_mape"]),
                "parameters": int(result["parameters"]),
                "run_dir": str(trial["run_dir"]),
            }
        )
        for row in trial["topology"]:
            topology_rows.append(
                {
                    "seed": trial["seed"],
                    "topology_nodes": int(row["topology_nodes"]),
                    "test_mape": float(row["test_mape"]),
                }
            )

    test_mean, test_std = mean_std([row["test_mape"] for row in overall_rows])
    val_mean, val_std = mean_std(
        [row["best_validation_mape"] for row in overall_rows]
    )
    overall_summary = {
        "model": AGGREGATION,
        "n": len(overall_rows),
        "seeds": [row["seed"] for row in overall_rows],
        "mean_test_mape": test_mean,
        "std_test_mape": test_std,
        "mean_best_validation_mape": val_mean,
        "std_best_validation_mape": val_std,
        "missing_seeds": missing,
    }

    topology_summary = []
    for nodes in sorted(set(row["topology_nodes"] for row in topology_rows)):
        selected = [row["test_mape"] for row in topology_rows if row["topology_nodes"] == nodes]
        value_mean, value_std = mean_std(selected)
        topology_summary.append(
            {
                "topology_nodes": nodes,
                "n": len(selected),
                "mean_test_mape": value_mean,
                "std_test_mape": value_std,
                "min_test_mape": min(selected),
                "max_test_mape": max(selected),
            }
        )

    write_csv(
        output_dir / "overall_by_seed.csv",
        overall_rows,
        ("seed", "test_mape", "best_validation_mape", "parameters", "run_dir"),
    )
    write_csv(
        output_dir / "topology_by_seed.csv",
        topology_rows,
        ("seed", "topology_nodes", "test_mape"),
    )
    write_csv(
        output_dir / "topology_statistics.csv",
        topology_summary,
        (
            "topology_nodes",
            "n",
            "mean_test_mape",
            "std_test_mape",
            "min_test_mape",
            "max_test_mape",
        ),
    )
    with (output_dir / "overall_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(overall_summary, handle, indent=2, ensure_ascii=False)

    fig, axis = plt.subplots(figsize=(8.2, 5.0))
    seeds = [row["seed"] for row in overall_rows]
    values = [row["test_mape"] for row in overall_rows]
    axis.scatter(seeds, values, s=65, color="#4C78A8", zorder=3)
    axis.axhline(test_mean, color="#E45756", linewidth=2.0, label="Mean")
    axis.fill_between(
        [min(seeds), max(seeds)],
        [max(0.0, test_mean - test_std)] * 2,
        [test_mean + test_std] * 2,
        color="#E45756",
        alpha=0.15,
        label="Mean +/- std",
    )
    axis.set_xticks(seeds)
    axis.set_xlabel("Seed")
    axis.set_ylabel("Overall Test MAPE (%)")
    axis.set_title("Reliability-Gated Distribution-Aware Model")
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    save_figure(fig, figures_dir / "01_overall_five_seeds")

    nodes = np.asarray([row["topology_nodes"] for row in topology_summary])
    means = np.asarray([row["mean_test_mape"] for row in topology_summary])
    stds = np.asarray([row["std_test_mape"] for row in topology_summary])
    fig, axis = plt.subplots(figsize=(12.0, 6.0))
    axis.plot(nodes, means, marker="o", linewidth=2.2, color="#4C78A8")
    axis.fill_between(nodes, np.maximum(0.0, means - stds), means + stds, color="#4C78A8", alpha=0.18)
    axis.set_xticks(nodes)
    axis.tick_params(axis="x", rotation=45)
    axis.set_xlabel("Number of nodes")
    axis.set_ylabel("Test MAPE (%)")
    axis.set_title("Per-Topology MAPE Across Seeds: Mean +/- Standard Deviation")
    axis.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, figures_dir / "02_topology_mean_std")

    small = [row for row in topology_summary if row["topology_nodes"] <= 100]
    small_nodes = np.asarray([row["topology_nodes"] for row in small])
    small_means = np.asarray([row["mean_test_mape"] for row in small])
    small_stds = np.asarray([row["std_test_mape"] for row in small])
    fig, axis = plt.subplots(figsize=(9.0, 5.2))
    axis.plot(small_nodes, small_means, marker="o", linewidth=2.2, color="#59A14F")
    axis.fill_between(
        small_nodes,
        np.maximum(0.0, small_means - small_stds),
        small_means + small_stds,
        color="#59A14F",
        alpha=0.2,
    )
    axis.set_xticks(small_nodes)
    axis.set_xlabel("Number of nodes")
    axis.set_ylabel("Test MAPE (%)")
    axis.set_title("Small-Topology Reliability Check")
    axis.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, figures_dir / "03_small_topology_mean_std")

    lines = [
        "# Reliability-gated five-seed summary",
        "",
        "- Model: `{}`".format(AGGREGATION),
        "- Completed seeds: `{}`".format(", ".join(str(seed) for seed in seeds)),
        "- Missing seeds: `{}`".format(missing if missing else "none"),
        "- Overall test MAPE: `{:.6f} +/- {:.6f}`".format(test_mean, test_std),
        "- Best validation MAPE: `{:.6f} +/- {:.6f}`".format(val_mean, val_std),
        "",
        "This summary contains only the latest reliability-gated model. It does not rerun or aggregate RouteNet-Fermi, Global Weighted, or the ungated Distribution-Aware model.",
        "",
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("Summary artifacts: {}".format(output_dir))


if __name__ == "__main__":
    main()

