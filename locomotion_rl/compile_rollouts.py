"""Compile the rollout results that rollout.py writes next to each run (<run>/<env_name>_rollout[_<tag>].csv).

    PYTHONPATH=src/locomotion_rl venv/bin/python -m locomotion_rl.compile_rollouts logs/DrawnGait logs/Sluurm ...
        [--env CustomRobotJoystickRoughTerrain] [--scenarios forward turn] [--columns alive vx cot power_w]
        [--tag custom_0.5_0.2_1] [--matrix cot] [--out table.csv]

Default: one row per (run, evaluation env, scenario) with the chosen columns, so one checkpoint rolled out on several scenes
(rollout.py --env_name) can be compared side by side. --matrix COLUMN instead prints a runs x environments grid of that column
for the first scenario given (the cross-evaluation "trained on A, evaluated on B" table). Column names are those of the csv
header; --list prints them. Rollouts without --tag are read unless --tag is given.
"""
import argparse
import csv
from pathlib import Path

DEFAULT_COLUMNS = ["alive", "vx", "vx_rms_err", "cot", "power_w", "saturation_pct", "cadence_hz", "clearance_mm", "stride_mm"]


def load(runs, env=None, tag=""):
    """All csv rows of the given run dirs, with the evaluation env and run folder name attached."""
    rows = []
    for run in map(Path, runs):
        suffix = f"_{tag}" if tag else ""
        for path in sorted(run.glob(f"*_rollout{suffix}.csv")):
            with path.open() as f:
                for row in csv.DictReader(f):
                    if env is None or row["env_name"] == env:
                        rows.append(row)
    return rows


def fmt(value, width=9):
    if value in ("", None):
        return f"{'-':>{width}}"
    try:
        v = float(value)
    except ValueError:
        return f"{value:>{width}}"
    return f"{v:{width}.0f}" if v.is_integer() else f"{v:{width}.3f}"


def table(rows, columns):
    head = f"{'run':<28}{'eval env':<34}{'scenario':<12}" + "".join(f"{c[:11]:>12}" for c in columns)
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r['run'][-27:]:<28}{r['env_name']:<34}{r['scenario']:<12}" + "".join(fmt(r.get(c), 12) for c in columns))
    return "\n".join(lines)


def matrix(rows, column, scenario):
    rows = [r for r in rows if r["scenario"] == scenario]
    envs = sorted({r["env_name"] for r in rows})
    runs = sorted({r["run"] for r in rows})
    cell = {(r["run"], r["env_name"]): r.get(column) for r in rows}
    head = f"{column} @ {scenario}".ljust(30) + "".join(f"{e.replace('CustomRobotJoystick', ''):>22}" for e in envs)
    return "\n".join([head, "-" * len(head)] + [f"{run[-29:]:<30}" + "".join(fmt(cell.get((run, e)), 22) for e in envs) for run in runs])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--env", help="only rollouts evaluated on this env_name")
    ap.add_argument("--scenarios", nargs="*", help="only these scenarios (default: all)")
    ap.add_argument("--columns", nargs="*", default=DEFAULT_COLUMNS)
    ap.add_argument("--tag", default="", help="read the rollouts written with this --tag (default: the untagged ones)")
    ap.add_argument("--matrix", metavar="COLUMN", help="print a runs x envs grid of this column (first scenario given)")
    ap.add_argument("--out", help="also write the selected rows and columns to this csv")
    ap.add_argument("--list", action="store_true", help="print the available columns and exit")
    args = ap.parse_args()

    rows = load(args.runs, args.env, args.tag)
    if not rows:
        raise SystemExit("no <env_name>_rollout csv files found in the given runs (run rollout.py first)")
    if args.list:
        print(", ".join(rows[0]))
        return
    if args.scenarios:
        rows = [r for r in rows if r["scenario"] in args.scenarios]
    if args.matrix:
        print(matrix(rows, args.matrix, (args.scenarios or ["forward"])[0]))
    else:
        print(table(rows, args.columns))
    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["run", "env_name", "scenario"] + args.columns, extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
