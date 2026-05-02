from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def _read_rows(path: str, windows: set[str]) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("scope") != "regime":
                continue
            row_window = str(row.get("window", ""))
            if row_window not in windows:
                continue
            out.append(row)
    return out


def _collect(paths: List[str], windows: set[str]) -> Dict[Tuple[str, str, int], Dict[str, Dict]]:
    """
    Returns:
      keyed[(run_name, window, episode)][regime] = row
    """
    keyed: Dict[Tuple[str, str, int], Dict[str, Dict]] = defaultdict(dict)
    for path in paths:
        run_name = os.path.splitext(os.path.basename(path))[0]
        for row in _read_rows(path, windows=windows):
            ep = int(row.get("episode", -1))
            regime = str(row.get("key", ""))
            window = str(row.get("window", ""))
            if ep <= 0 or regime == "" or window == "":
                continue
            keyed[(run_name, window, ep)][regime] = row
    return keyed


def _flatten(
    keyed: Dict[Tuple[str, str, int], Dict[str, Dict]],
    hide_zero_rounds: bool,
) -> List[Dict]:
    rows = []
    for (run_name, window, ep), regimes in sorted(
        keyed.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])
    ):
        out = {"run": run_name, "window": window, "episode": ep}
        for regime in ("competitive", "mixed", "cooperative"):
            r = regimes.get(regime, {})
            n_rounds = int(r.get("n_rounds", 0))
            coop = float(r.get("coop_rate", 0.0))
            reward = float(r.get("avg_reward", 0.0))
            if hide_zero_rounds and n_rounds == 0:
                coop = float("nan")
                reward = float("nan")
            out[f"{regime}_coop"] = coop
            out[f"{regime}_reward"] = reward
            out[f"{regime}_rounds"] = n_rounds
        rows.append(out)
    return rows


def _write_csv(path: str, rows: List[Dict]):
    if len(rows) == 0:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fields = [
        "run",
        "window",
        "episode",
        "competitive_coop",
        "mixed_coop",
        "cooperative_coop",
        "competitive_reward",
        "mixed_reward",
        "cooperative_reward",
        "competitive_rounds",
        "mixed_rounds",
        "cooperative_rounds",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _latest_by_run(rows: List[Dict]) -> List[Dict]:
    latest = {}
    for row in rows:
        run = row["run"]
        window = row["window"]
        ep = int(row["episode"])
        key = (run, window)
        if key not in latest or ep > int(latest[key]["episode"]):
            latest[key] = row
    return [latest[k] for k in sorted(latest.keys())]


def _windows_from_arg(window: str) -> set[str]:
    w = str(window).strip().lower()
    if w == "both":
        return {"window", "cumulative"}
    if w in {"window", "cumulative"}:
        return {w}
    raise ValueError(f"unsupported window mode: {window}")


def _fmt(v: float) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "NA"
    return f"{float(v):.3f}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--metrics_glob",
        type=str,
        default="outputs/train/**/metrics/*.jsonl",
    )
    p.add_argument(
        "--out_csv",
        type=str,
        default="outputs/train/analysis/regime_metrics_summary.csv",
    )
    p.add_argument(
        "--window",
        type=str,
        choices=["window", "cumulative", "both"],
        default="both",
    )
    p.add_argument("--hide_zero_rounds", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    paths = sorted(glob.glob(args.metrics_glob, recursive=True))
    if len(paths) == 0:
        raise FileNotFoundError(f"no files matched: {args.metrics_glob}")

    windows = _windows_from_arg(args.window)
    keyed = _collect(paths, windows=windows)
    rows = _flatten(keyed, hide_zero_rounds=bool(args.hide_zero_rounds))
    _write_csv(args.out_csv, rows)

    print(
        f"[summary] files={len(paths)} rows={len(rows)} windows={sorted(windows)} "
        f"out_csv={args.out_csv}"
    )
    for row in _latest_by_run(rows):
        print(
            "[latest]",
            row["run"],
            row["window"],
            f"ep={row['episode']}",
            f"comp={_fmt(row['competitive_coop'])}",
            f"mixed={_fmt(row['mixed_coop'])}",
            f"coop={_fmt(row['cooperative_coop'])}",
        )


if __name__ == "__main__":
    main()
