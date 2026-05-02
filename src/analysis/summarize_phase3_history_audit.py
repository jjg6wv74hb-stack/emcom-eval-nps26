from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


def _read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: str, rows: Sequence[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if len(rows) == 0:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / max(1, len(vals)))


def _target_f_values(raw_values: Sequence[str]) -> list[str]:
    return [f"{float(value):.3f}" for value in raw_values]


def _group_rows(
    rows: Sequence[Dict[str, str]],
    *,
    checkpoint_episode: int,
    target_f_values: Sequence[str],
    ablation: str,
) -> list[Dict]:
    grouped: dict[Tuple[str, str, str, int, str], list[Dict[str, str]]] = defaultdict(list)
    allowed_f = set(target_f_values)
    for row in rows:
        if str(row.get("eval_policy", "greedy")) != "greedy":
            continue
        if str(row.get("scope", "")) != "f_value":
            continue
        if int(float(row.get("checkpoint_episode", "0") or 0)) != int(checkpoint_episode):
            continue
        if str(row.get("cross_play", "none")) != "none":
            continue
        if str(row.get("sender_remap", "none")) != "none":
            continue
        if str(row.get("ablation", "none")) != str(ablation):
            continue
        key = str(row.get("key", ""))
        if key not in allowed_f:
            continue
        grouped[
            (
                str(row.get("condition", "")),
                str(row.get("history_intervention", "none")),
                str(row.get("ablation", "none")),
                int(checkpoint_episode),
                key,
            )
        ].append(row)

    out: list[Dict] = []
    for (condition, history_intervention, ablation_label, episode, f_value), cur in sorted(grouped.items()):
        avg_welfare_key = "avg_welfare" if "avg_welfare" in cur[0] else "avg_reward"
        out.append(
            {
                "condition": condition,
                "history_intervention": history_intervention,
                "ablation": ablation_label,
                "checkpoint_episode": int(episode),
                "f_value": f_value,
                "n_seeds": len(cur),
                "mean_coop_rate": _mean(float(row["coop_rate"]) for row in cur),
                "mean_avg_reward": _mean(float(row["avg_reward"]) for row in cur),
                "mean_avg_welfare": _mean(float(row[avg_welfare_key]) for row in cur),
            }
        )
    return out


def _write_markdown(
    path: str,
    *,
    summary_rows: Sequence[Dict],
    training_family: str,
    source_repo: str,
    source_train_root: str,
    source_eval_root: str,
    seed_count: int,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = ["# Phase-3 History Audit Summary", ""]
    lines.extend(
        [
            f"- training_family: `{training_family}`",
            f"- source_repo: `{source_repo}`",
            f"- source_train_root: `{source_train_root}`",
            f"- source_eval_root: `{source_eval_root}`",
            f"- seed_count: `{int(seed_count)}`",
            "",
            "| Condition | History | Ablation | Episode | f | Mean Coop | Mean Reward | Mean Welfare |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['condition']} | {row['history_intervention']} | {row['ablation']} | "
            f"{int(row['checkpoint_episode'])} | {float(row['f_value']):.1f} | "
            f"{float(row['mean_coop_rate']):.4f} | {float(row['mean_avg_reward']):.4f} | "
            f"{float(row['mean_avg_welfare']):.4f} |"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suite_main_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--checkpoint_episode", type=int, required=True)
    p.add_argument("--ablation", type=str, default="none")
    p.add_argument("--f_values", nargs="*", type=float, default=[3.5, 5.0])
    p.add_argument("--training_family", type=str, default="phase3_vecstraight")
    p.add_argument("--source_repo", type=str, default="dsc-epgg-vectorized")
    p.add_argument("--source_train_root", type=str, required=True)
    p.add_argument("--source_eval_root", type=str, required=True)
    p.add_argument("--seed_count", type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()
    rows = _read_rows(os.path.abspath(args.suite_main_csv))
    summary_rows = _group_rows(
        rows,
        checkpoint_episode=int(args.checkpoint_episode),
        target_f_values=_target_f_values(args.f_values),
        ablation=str(args.ablation),
    )
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    _write_rows(os.path.join(out_dir, "history_audit_summary.csv"), summary_rows)
    _write_markdown(
        os.path.join(out_dir, "history_audit_summary.md"),
        summary_rows=summary_rows,
        training_family=str(args.training_family),
        source_repo=str(args.source_repo),
        source_train_root=os.path.abspath(args.source_train_root),
        source_eval_root=os.path.abspath(args.source_eval_root),
        seed_count=int(args.seed_count),
    )
    print(f"[history-audit-summary] out_dir={out_dir}")


if __name__ == "__main__":
    main()
