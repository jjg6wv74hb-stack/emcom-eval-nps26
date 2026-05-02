from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


LINE_RE = re.compile(
    r"^\[episode\s+(\d+)\]\s+coop=([0-9.\-eE]+)\s+avg_reward=([0-9.\-eE]+)\s+loss=([0-9.\-eE]+)"
)
NAME_RE = re.compile(r"(cond[0-9]+)_seed([0-9]+)\.log$")


def _parse_log(path: str) -> Tuple[str, int, List[Dict]]:
    name = os.path.basename(path)
    m = NAME_RE.search(name)
    if m is None:
        return "unknown", -1, []
    condition = m.group(1)
    seed = int(m.group(2))

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            mline = LINE_RE.match(line.strip())
            if mline is None:
                continue
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "episode": int(mline.group(1)),
                    "coop_rate": float(mline.group(2)),
                    "avg_reward": float(mline.group(3)),
                    "loss_total": float(mline.group(4)),
                }
            )
    return condition, seed, rows


def _write_rows_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["condition", "seed", "episode", "coop_rate", "avg_reward", "loss_total"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate(rows: List[Dict], metric: str) -> Dict[str, List[Dict]]:
    # condition -> list of {"episode", "mean", "std", "n"}
    cond_ep_values = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cond_ep_values[row["condition"]][int(row["episode"])].append(float(row[metric]))

    out = {}
    for cond, ep_dict in cond_ep_values.items():
        series = []
        for ep in sorted(ep_dict.keys()):
            vals = np.asarray(ep_dict[ep], dtype=np.float32)
            series.append(
                {
                    "episode": int(ep),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": int(vals.size),
                }
            )
        out[cond] = series
    return out


def _plot_metric(agg: Dict[str, List[Dict]], metric: str, out_png: str):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(10, 6))
    palette = {
        "cond1": "#d62728",
        "cond2": "#1f77b4",
        "cond6": "#2ca02c",
    }

    for cond in sorted(agg.keys()):
        series = agg[cond]
        if len(series) == 0:
            continue
        x = np.asarray([r["episode"] for r in series], dtype=np.int32)
        y = np.asarray([r["mean"] for r in series], dtype=np.float32)
        s = np.asarray([r["std"] for r in series], dtype=np.float32)
        color = palette.get(cond, None)
        plt.plot(x, y, label=cond, color=color, linewidth=2.0)
        plt.fill_between(x, y - s, y + s, color=color, alpha=0.2)

    plt.xlabel("Episode")
    plt.ylabel(metric)
    plt.title(f"{metric} vs Episode (mean ± std across seeds)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logs_glob", type=str, default="outputs/train/grid/logs/cond*_seed*.log")
    p.add_argument("--out_dir", type=str, default="outputs/train/grid/analysis")
    return p.parse_args()


def main():
    args = parse_args()
    paths = sorted(glob.glob(args.logs_glob))
    if len(paths) == 0:
        raise FileNotFoundError(f"no logs matched: {args.logs_glob}")

    all_rows = []
    per_log_counts = {}
    for path in paths:
        condition, seed, rows = _parse_log(path)
        if condition == "unknown":
            continue
        all_rows.extend(rows)
        per_log_counts[os.path.basename(path)] = len(rows)

    if len(all_rows) == 0:
        raise RuntimeError("no episode metric lines were found in logs")

    os.makedirs(args.out_dir, exist_ok=True)
    rows_csv = os.path.join(args.out_dir, "learning_curves_rows.csv")
    _write_rows_csv(rows_csv, all_rows)

    agg_coop = _aggregate(all_rows, metric="coop_rate")
    agg_rew = _aggregate(all_rows, metric="avg_reward")
    _plot_metric(agg_coop, metric="coop_rate", out_png=os.path.join(args.out_dir, "coop_vs_episode.png"))
    _plot_metric(agg_rew, metric="avg_reward", out_png=os.path.join(args.out_dir, "reward_vs_episode.png"))

    print(f"[learning-curves] parsed_logs={len(per_log_counts)}")
    for name in sorted(per_log_counts.keys()):
        print(f"[learning-curves] {name}: points={per_log_counts[name]}")
    print(f"[learning-curves] rows_csv={rows_csv}")
    print(f"[learning-curves] coop_png={os.path.join(args.out_dir, 'coop_vs_episode.png')}")
    print(f"[learning-curves] reward_png={os.path.join(args.out_dir, 'reward_vs_episode.png')}")


if __name__ == "__main__":
    main()
