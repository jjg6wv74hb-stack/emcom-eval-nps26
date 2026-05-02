from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _read_rows(paths: Iterable[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def _filter_rows(
    rows: Iterable[Dict[str, str]],
    condition: str,
    eval_policy: str,
    ablation: str,
    sender_remap: str,
    cross_play: str,
) -> List[Dict[str, str]]:
    out = []
    for row in rows:
        if row.get("condition") != condition:
            continue
        if row.get("eval_policy", "greedy") != eval_policy:
            continue
        if row.get("ablation", "none") != ablation:
            continue
        if row.get("sender_remap", "none") != sender_remap:
            continue
        if row.get("cross_play", "none") != cross_play:
            continue
        out.append(row)
    return out


def _group_mean(rows: Iterable[Dict[str, str]]) -> Dict[Tuple[int, str], float]:
    grouped: Dict[Tuple[int, str], List[float]] = defaultdict(list)
    for row in rows:
        if row.get("scope") != "regime":
            continue
        grouped[(int(row["checkpoint_episode"]), str(row["key"]))].append(float(row["coop_rate"]))
    return {k: (sum(v) / len(v)) for k, v in grouped.items() if v}


def _write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(path: Path, rows: List[Dict[str, object]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    lines.append("| Episode | Regime | New | Old | Delta |")
    lines.append("|---:|---|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['checkpoint_episode']} | {row['regime']} | "
            f"{100.0 * float(row['new_coop_rate']):.1f} pp | "
            f"{100.0 * float(row['old_coop_rate']):.1f} pp | "
            f"{100.0 * float(row['delta_coop_rate']):+.1f} pp |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--old_suite_csv", action="append", required=True)
    p.add_argument("--new_suite_csv", type=str, required=True)
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--out_md", type=str, required=True)
    p.add_argument("--condition", type=str, default="cond1")
    p.add_argument("--eval_policy", type=str, default="greedy")
    p.add_argument("--ablation", type=str, default="none")
    p.add_argument("--sender_remap", type=str, default="none")
    p.add_argument("--cross_play", type=str, default="none")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    old_rows = _filter_rows(
        _read_rows(args.old_suite_csv),
        condition=args.condition,
        eval_policy=args.eval_policy,
        ablation=args.ablation,
        sender_remap=args.sender_remap,
        cross_play=args.cross_play,
    )
    new_rows = _filter_rows(
        _read_rows([args.new_suite_csv]),
        condition=args.condition,
        eval_policy=args.eval_policy,
        ablation=args.ablation,
        sender_remap=args.sender_remap,
        cross_play=args.cross_play,
    )

    old_mean = _group_mean(old_rows)
    new_mean = _group_mean(new_rows)
    keys = sorted(set(old_mean) & set(new_mean))
    out_rows: List[Dict[str, object]] = []
    for episode, regime in keys:
        out_rows.append(
            {
                "checkpoint_episode": int(episode),
                "regime": str(regime),
                "new_coop_rate": float(new_mean[(episode, regime)]),
                "old_coop_rate": float(old_mean[(episode, regime)]),
                "delta_coop_rate": float(new_mean[(episode, regime)] - old_mean[(episode, regime)]),
            }
        )

    _write_rows(Path(args.out_csv), out_rows)
    _write_markdown(Path(args.out_md), out_rows, title="Vectorized Vs Old Checkpoint Suite Comparison")
    print(f"[compare] rows={len(out_rows)} out_csv={args.out_csv} out_md={args.out_md}")


if __name__ == "__main__":
    main()
