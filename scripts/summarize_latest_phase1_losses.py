#!/usr/bin/env python3
"""Summarize loss CSVs from the latest Phase 1 training run.

By default, this script finds the newest logs/vjepa_drive/phase1-* directory
and summarizes all log_r*.csv files inside it.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize V-JEPA2 Phase 1 losses from latest timestamped run."
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("logs/vjepa_drive"),
        help="Root directory containing phase1-* run folders.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional explicit run dir (overrides latest timestamp lookup).",
    )
    parser.add_argument(
        "--show-per-rank",
        action="store_true",
        help="Also print per-rank metrics.",
    )
    return parser.parse_args()


def find_latest_run(logs_root: Path) -> Path:
    runs = [p for p in logs_root.glob("phase1-*") if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No phase1-* directories found under {logs_root}")
    return sorted(runs, key=lambda p: p.name)[-1]


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str) -> float:
    return float(value.strip())


def linear_slope(xs: list[float], ys: list[float]) -> float:
    """Return least-squares slope for y = a + b*x."""
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have equal length")
    if len(xs) < 2:
        return 0.0
    x_bar = mean(xs)
    y_bar = mean(ys)
    num = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
    den = sum((x - x_bar) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


def epoch_means(rows: list[dict[str, str]], key: str) -> dict[int, float]:
    by_epoch: dict[int, list[float]] = {}
    for row in rows:
        epoch = int(row["epoch"])
        by_epoch.setdefault(epoch, []).append(to_float(row[key]))
    return {epoch: mean(values) for epoch, values in sorted(by_epoch.items())}


def summarize_csv(csv_path: Path) -> dict[str, float | int | str]:
    rows = read_rows(csv_path)
    if not rows:
        raise ValueError(f"{csv_path} has no rows")

    last_row = rows[-1]
    last_epoch = int(last_row["epoch"])
    epoch_rows = [r for r in rows if int(r["epoch"]) == last_epoch]

    epoch_loss = epoch_means(rows, "loss")
    epoch_jepa = epoch_means(rows, "loss_jepa")
    epoch_traj = epoch_means(rows, "loss_traj")
    epochs = [float(e) for e in epoch_loss.keys()]

    return {
        "file": csv_path.name,
        "last_epoch": last_epoch,
        "final_iter_loss": to_float(last_row["loss"]),
        "final_iter_loss_jepa": to_float(last_row["loss_jepa"]),
        "final_iter_loss_traj": to_float(last_row["loss_traj"]),
        "last_epoch_avg_loss": mean(to_float(r["loss"]) for r in epoch_rows),
        "last_epoch_avg_loss_jepa": mean(to_float(r["loss_jepa"]) for r in epoch_rows),
        "last_epoch_avg_loss_traj": mean(to_float(r["loss_traj"]) for r in epoch_rows),
        "overall_avg_loss": mean(to_float(r["loss"]) for r in rows),
        "overall_avg_loss_jepa": mean(to_float(r["loss_jepa"]) for r in rows),
        "overall_avg_loss_traj": mean(to_float(r["loss_traj"]) for r in rows),
        "trend_slope_epoch_loss": linear_slope(epochs, list(epoch_loss.values())),
        "trend_slope_epoch_loss_jepa": linear_slope(epochs, list(epoch_jepa.values())),
        "trend_slope_epoch_loss_traj": linear_slope(epochs, list(epoch_traj.values())),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir if args.run_dir is not None else find_latest_run(args.logs_root)
    run_dir = run_dir.resolve()

    csv_files = sorted(run_dir.glob("log_r*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No log_r*.csv files found in {run_dir}")

    summaries = [summarize_csv(csv_path) for csv_path in csv_files]

    print(f"run_dir: {run_dir}")
    print("csv_files:")
    for csv_path in csv_files:
        print(f"  - {csv_path.name}")

    if args.show_per_rank:
        print("\nper_rank:")
        for stats in summaries:
            print(
                f"  {stats['file']}: "
                f"final_loss={stats['final_iter_loss']:.5f}, "
                f"final_jepa={stats['final_iter_loss_jepa']:.5f}, "
                f"final_traj={stats['final_iter_loss_traj']:.5f}, "
                f"last_epoch_avg={stats['last_epoch_avg_loss']:.5f}"
            )

    keys = [
        "final_iter_loss",
        "final_iter_loss_jepa",
        "final_iter_loss_traj",
        "last_epoch_avg_loss",
        "last_epoch_avg_loss_jepa",
        "last_epoch_avg_loss_traj",
        "overall_avg_loss",
        "overall_avg_loss_jepa",
        "overall_avg_loss_traj",
        "trend_slope_epoch_loss",
        "trend_slope_epoch_loss_jepa",
        "trend_slope_epoch_loss_traj",
    ]
    print("\nsummary_mean_across_ranks:")
    print(f"  rank_count: {len(summaries)}")
    print(f"  last_epoch: {int(mean(float(s['last_epoch']) for s in summaries))}")
    for key in keys:
        print(f"  {key}: {mean(float(s[key]) for s in summaries):.5f}")

    print("\nsummary_spread_across_ranks:")
    for key in keys:
        values = [float(s[key]) for s in summaries]
        std = pstdev(values) if len(values) > 1 else 0.0
        print(
            f"  {key}: min={min(values):.5f}, max={max(values):.5f}, std={std:.5f}"
        )


if __name__ == "__main__":
    main()
