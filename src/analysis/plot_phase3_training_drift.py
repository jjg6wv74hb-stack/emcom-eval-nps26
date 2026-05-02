from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COMM_COLOR = "#1f77b4"
NOCOMM_COLOR = "#d62728"
COND_LABEL = {"cond1": "Comm", "cond2": "No-Comm"}
SCOPE_LABEL = {"overall": "Overall weighted average", "f_3.500": "Focal f=3.5"}
METRIC_LABEL = {
    "coop_rate": "Cooperation",
    "avg_reward": "Avg Reward",
    "avg_welfare": "Avg Welfare",
}


def _read_jsonl(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_csv_rows(path: str) -> List[Dict[str, object]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_sem(values: Sequence[float]) -> Tuple[float, float]:
    vals = [float(v) for v in values]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) / math.sqrt(len(arr)))


def _collect_metric_files(metrics_dirs: Sequence[str]) -> Dict[Tuple[str, int], List[str]]:
    files_by_key: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    for metrics_dir in metrics_dirs:
        if not os.path.isdir(metrics_dir):
            raise FileNotFoundError(f"metrics directory not found: {metrics_dir}")
        for fname in sorted(os.listdir(metrics_dir)):
            if not fname.endswith(".jsonl"):
                continue
            stem = fname[:-6]
            parts = stem.split("_")
            if len(parts) != 2:
                continue
            condition = parts[0]
            seed = int(parts[1].replace("seed", ""))
            files_by_key[(condition, seed)].append(os.path.join(metrics_dir, fname))
    return files_by_key


def _extract_rows_from_metrics_dirs(
    metrics_dirs: Sequence[str], focal_f_key: str
) -> List[Dict[str, object]]:
    rows_out: List[Dict[str, object]] = []
    files_by_key = _collect_metric_files(metrics_dirs)

    for (condition, seed), paths in sorted(files_by_key.items()):
        episode_map: Dict[int, Dict[str, Dict[str, float]]] = {}
        for path in sorted(paths):
            source_metrics_dir = os.path.abspath(os.path.dirname(path))
            for row in _read_jsonl(path):
                if row.get("scope") != "f_value" or row.get("window") != "window":
                    continue
                episode = int(row["episode"])
                episode_entry = episode_map.setdefault(episode, {})
                episode_entry[str(row["key"])] = {
                    "n_rounds": float(row["n_rounds"]),
                    "coop_rate": float(row["coop_rate"]),
                    "avg_reward": float(row["avg_reward"]),
                    "source_metrics_dir": source_metrics_dir,
                }

        for episode in sorted(episode_map):
            f_rows = episode_map[episode]
            total_rounds = sum(v["n_rounds"] for v in f_rows.values())
            if total_rounds <= 0:
                continue
            overall_coop = sum(v["coop_rate"] * v["n_rounds"] for v in f_rows.values()) / total_rounds
            overall_reward = sum(v["avg_reward"] * v["n_rounds"] for v in f_rows.values()) / total_rounds
            source_dirs = sorted({str(v["source_metrics_dir"]) for v in f_rows.values()})
            rows_out.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "episode": episode,
                    "scope": "overall",
                    "coop_rate": overall_coop,
                    "avg_reward": overall_reward,
                    "avg_welfare": overall_reward * 4.0,
                    "n_rounds": total_rounds,
                    "source_metrics_dir": "|".join(source_dirs),
                }
            )
            if focal_f_key in f_rows:
                focal = f_rows[focal_f_key]
                rows_out.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "episode": episode,
                        "scope": f"f_{focal_f_key}",
                        "coop_rate": focal["coop_rate"],
                        "avg_reward": focal["avg_reward"],
                        "avg_welfare": focal["avg_reward"] * 4.0,
                        "n_rounds": focal["n_rounds"],
                        "source_metrics_dir": str(focal["source_metrics_dir"]),
                    }
                )
    return rows_out


def _summarize_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), str(row["scope"]), int(row["episode"]))].append(dict(row))

    out: List[Dict[str, object]] = []
    for (condition, scope, episode), group in sorted(grouped.items()):
        coop_vals = [float(r["coop_rate"]) for r in group]
        reward_vals = [float(r["avg_reward"]) for r in group]
        welfare_vals = [float(r["avg_welfare"]) for r in group]
        mean_coop, sem_coop = _mean_sem(coop_vals)
        mean_reward, sem_reward = _mean_sem(reward_vals)
        mean_welfare, sem_welfare = _mean_sem(welfare_vals)
        out.append(
            {
                "condition": condition,
                "scope": scope,
                "episode": episode,
                "n_seeds": len(group),
                "mean_coop_rate": mean_coop,
                "sem_coop_rate": sem_coop,
                "mean_avg_reward": mean_reward,
                "sem_avg_reward": sem_reward,
                "mean_avg_welfare": mean_welfare,
                "sem_avg_welfare": sem_welfare,
            }
        )
    return out


def _coverage_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), str(row["scope"]), int(row["episode"]))].append(dict(row))
    out: List[Dict[str, object]] = []
    for (condition, scope, episode), group in sorted(grouped.items()):
        out.append(
            {
                "condition": condition,
                "scope": scope,
                "episode": episode,
                "n_seeds": len(group),
                "seed_list": ",".join(str(int(r["seed"])) for r in sorted(group, key=lambda x: int(x["seed"]))),
                "source_metrics_dirs": "|".join(sorted({str(r["source_metrics_dir"]) for r in group})),
            }
        )
    return out


def _series(summary_rows: Sequence[Dict[str, object]], condition: str, scope: str):
    rows = [
        r for r in summary_rows
        if str(r["condition"]) == condition and str(r["scope"]) == scope
    ]
    rows = sorted(rows, key=lambda r: int(r["episode"]))
    episodes = np.asarray([int(r["episode"]) for r in rows], dtype=float)
    return rows, episodes


def _plot(
    summary_rows: Sequence[Dict[str, object]],
    *,
    out_png: str,
    focal_f_key: str,
    figure_title: str,
    figure_note: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    metric_specs = [
        ("mean_coop_rate", "sem_coop_rate", "coop_rate"),
        ("mean_avg_reward", "sem_avg_reward", "avg_reward"),
        ("mean_avg_welfare", "sem_avg_welfare", "avg_welfare"),
    ]
    scopes = ["overall", f"f_{focal_f_key}"]
    colors = {"cond1": COMM_COLOR, "cond2": NOCOMM_COLOR}

    for row_idx, scope in enumerate(scopes):
        for col_idx, (mean_key, sem_key, metric_key) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            for condition in ["cond1", "cond2"]:
                rows, episodes = _series(summary_rows, condition=condition, scope=scope)
                if len(rows) == 0:
                    continue
                ys = np.asarray([float(r[mean_key]) for r in rows], dtype=float)
                sems = np.asarray([float(r[sem_key]) for r in rows], dtype=float)
                ax.plot(
                    episodes / 1000.0,
                    ys,
                    color=colors[condition],
                    linewidth=2.2,
                    label=COND_LABEL[condition],
                )
                ax.fill_between(
                    episodes / 1000.0,
                    ys - sems,
                    ys + sems,
                    color=colors[condition],
                    alpha=0.18,
                )
            if row_idx == 0:
                ax.set_title(METRIC_LABEL[metric_key], fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(SCOPE_LABEL[scope], fontsize=11)
            ax.grid(True, alpha=0.25)
            ax.set_xlim(0, 150)
            ax.axvline(50, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)
            ax.axvline(100, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)
            if row_idx == 1:
                ax.set_xlabel("Episode (x1k)", fontsize=10)
            if metric_key == "coop_rate":
                ax.set_ylim(0.0, 1.0)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.985), frameon=False)
    fig.suptitle(figure_title, fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, 0.02, figure_note, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.95))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _closest_row(
    summary_rows: Sequence[Dict[str, object]], condition: str, scope: str, episode: int
) -> Dict[str, object]:
    rows = [
        r for r in summary_rows
        if str(r["condition"]) == condition and str(r["scope"]) == scope
    ]
    if not rows:
        raise ValueError(f"no rows for {condition} {scope}")
    return min(rows, key=lambda r: abs(int(r["episode"]) - episode))


def _max_row(summary_rows: Sequence[Dict[str, object]], condition: str, scope: str, metric_key: str) -> Dict[str, object]:
    rows = [
        r for r in summary_rows
        if str(r["condition"]) == condition and str(r["scope"]) == scope
    ]
    if not rows:
        raise ValueError(f"no rows for {condition} {scope}")
    return max(rows, key=lambda r: float(r[metric_key]))


def _write_markdown(
    summary_rows: Sequence[Dict[str, object]],
    *,
    out_md: str,
    focal_f_key: str,
    training_family: str,
    source_repo: str,
    metrics_dirs: Sequence[str],
) -> None:
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    lines: List[str] = []
    lines.append("# Phase-3 Training Drift Summary")
    lines.append("")
    lines.append(f"- training_family: `{training_family}`")
    lines.append(f"- source_repo: `{source_repo}`")
    lines.append(f"- metrics_dirs: `{';'.join(os.path.abspath(v) for v in metrics_dirs)}`")
    lines.append("")
    lines.append(
        "This figure aggregates training-window JSONL metrics from the supplied straight-family "
        "training runs."
    )
    lines.append("")
    for scope in ["overall", f"f_{focal_f_key}"]:
        lines.append(f"## {SCOPE_LABEL[scope]}")
        lines.append("")
        for condition in ["cond1", "cond2"]:
            start = _closest_row(summary_rows, condition, scope, 1000)
            mid50 = _closest_row(summary_rows, condition, scope, 50000)
            mid100 = _closest_row(summary_rows, condition, scope, 100000)
            end = _closest_row(summary_rows, condition, scope, 150000)
            peak = _max_row(summary_rows, condition, scope, "mean_coop_rate")
            lines.append(
                f"- {COND_LABEL[condition]}: coop {float(start['mean_coop_rate']):.3f} near 1k, "
                f"{float(mid50['mean_coop_rate']):.3f} near 50k, "
                f"{float(mid100['mean_coop_rate']):.3f} near 100k, "
                f"{float(end['mean_coop_rate']):.3f} near 150k; "
                f"peak {float(peak['mean_coop_rate']):.3f} at {int(peak['episode'])//1000}k."
            )
            lines.append(
                f"  reward {float(start['mean_avg_reward']):.2f} -> "
                f"{float(mid50['mean_avg_reward']):.2f} -> {float(mid100['mean_avg_reward']):.2f} -> "
                f"{float(end['mean_avg_reward']):.2f}; welfare {float(start['mean_avg_welfare']):.2f} -> "
                f"{float(mid50['mean_avg_welfare']):.2f} -> {float(mid100['mean_avg_welfare']):.2f} -> "
                f"{float(end['mean_avg_welfare']):.2f}."
            )
        lines.append("")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_dirs", nargs="+", type=str, required=True)
    p.add_argument("--focal_f_key", type=str, default="3.500")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--stitched_csv", type=str, default="")
    p.add_argument("--training_family", type=str, default="phase3_vecstraight")
    p.add_argument("--source_repo", type=str, default="dsc-epgg-vectorized")
    p.add_argument(
        "--figure_title",
        type=str,
        default="Phase-3 training trajectories: communication vs no communication (vecstraight)",
    )
    p.add_argument(
        "--figure_note",
        type=str,
        default=(
            "Training-window JSONL metrics aggregated from supplied straight-family metrics dirs. "
            "Shaded bands show SEM across available seeds."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if str(args.stitched_csv or "").strip():
        stitched_rows = _read_csv_rows(args.stitched_csv)
    else:
        stitched_rows = _extract_rows_from_metrics_dirs(
            metrics_dirs=[os.path.abspath(v) for v in args.metrics_dirs],
            focal_f_key=str(args.focal_f_key),
        )
    summary_rows = _summarize_rows(stitched_rows)
    coverage = _coverage_rows(stitched_rows)
    os.makedirs(args.out_dir, exist_ok=True)
    _write_csv(os.path.join(args.out_dir, "phase3_training_metrics_stitched.csv"), stitched_rows)
    _write_csv(os.path.join(args.out_dir, "phase3_training_metrics_coverage.csv"), coverage)
    _write_csv(os.path.join(args.out_dir, "training_drift_summary.csv"), summary_rows)
    _plot(
        summary_rows=summary_rows,
        out_png=os.path.join(args.out_dir, "phase3_training_drift.png"),
        focal_f_key=str(args.focal_f_key),
        figure_title=str(args.figure_title),
        figure_note=str(args.figure_note),
    )
    _write_markdown(
        summary_rows=summary_rows,
        out_md=os.path.join(args.out_dir, "PHASE3_TRAINING_DRIFT.md"),
        focal_f_key=str(args.focal_f_key),
        training_family=str(args.training_family),
        source_repo=str(args.source_repo),
        metrics_dirs=[os.path.abspath(v) for v in args.metrics_dirs],
    )
    print(f"[ok] wrote {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
