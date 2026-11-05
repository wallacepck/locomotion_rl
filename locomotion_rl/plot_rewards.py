"""Graph the reward terms recorded during training (logs/<run>/metrics.csv, written by locomotion.py at every eval).

    PYTHONPATH=src/locomotion_rl venv/bin/python -m locomotion_rl.plot_rewards logs/<run> [logs/<run2> ...]
        [--terms pose energy tracking_lin_vel] [--per_step] [--out file.png] [--list]

One panel per term (first panel: the total eval reward), one line per run, x axis = environment steps. With no --terms every
recorded term is drawn. A term's value is its contribution to the eval episode reward, i.e. weight x raw term summed over the
episode and times ctrl_dt, averaged over the eval episodes. The terms add up to the UNCLIPPED reward; the total
panel is the clipped reward the policy actually trains on (each step is floored at 0), so it is higher than the sum by
whatever the clip removed. --per_step divides by the mean episode length, which makes runs with different survival comparable.

Runs trained before metrics.csv existed have no record and cannot be plotted.
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(run) -> dict[str, np.ndarray]:
    """Columns of a run's metrics.csv as float arrays (`run` is the run dir or the csv itself)."""
    path = Path(run)
    path = path if path.suffix == ".csv" else path / "metrics.csv"
    if not path.exists():
        raise SystemExit(f"{path} does not exist (the run was trained before metrics.csv was recorded)")
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def terms_of(data) -> list[str]:
    return [k.split("/", 1)[1] for k in data if k.startswith("term/")]


def plot(runs, terms=None, out=None, per_step=False):
    runs = [Path(r) for r in runs]
    data = [load(r) for r in runs]
    available = sorted(set.union(*(set(terms_of(d)) for d in data)))
    terms = terms or available
    missing = [t for t in terms if t not in available]
    if missing:
        raise SystemExit(f"unknown reward term(s) {missing}; recorded: {available}")
    panels = ["total"] + list(terms)
    cols = min(3, len(panels))
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.2 * rows), squeeze=False, sharex=True)
    for ax, name in zip(axes.flat, panels):
        for run, d in zip(runs, data):
            key = "episode_reward" if name == "total" else f"term/{name}"
            if key not in d:
                continue
            y = d[key] / np.maximum(d["avg_episode_length"], 1.0) if per_step else d[key]
            label = (run if run.is_dir() else run.parent).name
            ax.plot(d["steps"], y, label=label, linewidth=1.6)
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("# environment steps")
    axes[0, 0].set_ylabel("reward per step" if per_step else "reward per episode")
    if len(runs) > 1:
        axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    out = Path(out) if out else (runs[0] / "rewards.png" if len(runs) == 1 and runs[0].is_dir() else Path("rewards_compare.png"))
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories (or metrics.csv files)")
    ap.add_argument("--terms", nargs="*", help="reward terms to draw (default: all recorded)")
    ap.add_argument("--per_step", action="store_true", help="divide by the mean episode length")
    ap.add_argument("--out", help="output png (default: <run>/rewards.png for one run, ./rewards_compare.png for several)")
    ap.add_argument("--list", action="store_true", help="print the recorded terms and exit")
    args = ap.parse_args()
    if args.list:
        for r in args.runs:
            print(f"{r}: {', '.join(terms_of(load(r)))}")
    else:
        plot(args.runs, args.terms, args.out, args.per_step)
