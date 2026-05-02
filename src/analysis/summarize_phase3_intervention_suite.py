from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import random
from collections import defaultdict
from statistics import median, stdev
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


def _std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if len(vals) <= 1:
        return 0.0
    return float(stdev(vals))


def _sem(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if len(vals) <= 1:
        return 0.0
    return float(_std(vals) / math.sqrt(len(vals)))


def _target_f_values(raw_values: Sequence[str]) -> list[str]:
    out = []
    for value in raw_values:
        num = float(value)
        out.append(f"{num:.3f}")
    return out


def _as_int(value: str | None, default: int = 0) -> int:
    if value in ("", None):
        return int(default)
    return int(float(value))


def _as_float(value: str | None, default: float = 0.0) -> float:
    if value in ("", None):
        return float(default)
    return float(value)


def _pct_label(value: float) -> str:
    return f"{100.0 * float(value):+.1f} pp"


def _bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int,
    seed: int,
) -> Tuple[float, float]:
    vals = [float(v) for v in values]
    if len(vals) == 0:
        return 0.0, 0.0
    if len(vals) == 1:
        return vals[0], vals[0]
    rng = random.Random(int(seed))
    boot = []
    n = len(vals)
    for _ in range(max(1, int(n_boot))):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        boot.append(_mean(sample))
    boot.sort()
    lo_idx = int(0.025 * (len(boot) - 1))
    hi_idx = int(0.975 * (len(boot) - 1))
    return float(boot[lo_idx]), float(boot[hi_idx])


def _sign_flip_p_value(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return 1.0
    observed = abs(_mean(vals))
    if observed <= 0.0:
        return 1.0
    total = 0
    ge_count = 0
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        flipped = [sign * value for sign, value in zip(signs, vals)]
        total += 1
        if abs(_mean(flipped)) >= observed - 1e-15:
            ge_count += 1
    return float(ge_count / max(1, total))


def _collect_seed_rows(
    rows: Sequence[Dict[str, str]],
    *,
    checkpoint_episode: int,
    target_f_values: Sequence[str],
) -> list[Dict]:
    grouped: dict[Tuple[str, str, int, str, int], list[Dict[str, str]]] = defaultdict(list)
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
        key = str(row.get("key", ""))
        if key not in allowed_f:
            continue
        ablation = str(row.get("ablation", "none"))
        if str(row.get("condition", "")) == "cond2":
            ablation = "baseline_none"
        grouped[
            (
                str(row.get("condition", "")),
                ablation,
                int(checkpoint_episode),
                key,
                _as_int(row.get("train_seed"), -1),
            )
        ].append(row)

    out: list[Dict] = []
    for (condition, ablation, episode, f_value, train_seed), cur in sorted(grouped.items()):
        avg_welfare_key = "avg_welfare" if "avg_welfare" in cur[0] else "avg_reward"
        out.append(
            {
                "condition": condition,
                "ablation": ablation,
                "checkpoint_episode": int(episode),
                "f_value": f_value,
                "train_seed": int(train_seed),
                "coop_rate": _mean(float(row["coop_rate"]) for row in cur),
                "avg_reward": _mean(float(row["avg_reward"]) for row in cur),
                "avg_welfare": _mean(float(row[avg_welfare_key]) for row in cur),
            }
        )
    return out


def _summarize_seed_rows(seed_rows: Sequence[Dict]) -> list[Dict]:
    grouped: dict[Tuple[str, str, int, str], list[Dict]] = defaultdict(list)
    for row in seed_rows:
        grouped[
            (
                str(row["condition"]),
                str(row["ablation"]),
                int(row["checkpoint_episode"]),
                str(row["f_value"]),
            )
        ].append(row)

    out: list[Dict] = []
    for (condition, ablation, episode, f_value), cur in sorted(grouped.items()):
        coop_by_seed = [float(row["coop_rate"]) for row in cur]
        reward_by_seed = [float(row["avg_reward"]) for row in cur]
        welfare_by_seed = [float(row["avg_welfare"]) for row in cur]
        out.append(
            {
                "condition": condition,
                "ablation": ablation,
                "checkpoint_episode": int(episode),
                "f_value": f_value,
                "n_seeds": len(cur),
                "mean_coop_rate": _mean(coop_by_seed),
                "std_coop_rate": _std(coop_by_seed),
                "sem_coop_rate": _sem(coop_by_seed),
                "mean_avg_reward": _mean(reward_by_seed),
                "std_avg_reward": _std(reward_by_seed),
                "sem_avg_reward": _sem(reward_by_seed),
                "mean_avg_welfare": _mean(welfare_by_seed),
                "std_avg_welfare": _std(welfare_by_seed),
                "sem_avg_welfare": _sem(welfare_by_seed),
            }
        )
    return out


def _paired_stats(
    seed_rows: Sequence[Dict],
    *,
    reference_condition: str,
    reference_ablation: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[Dict]:
    values: dict[Tuple[str, str, int, str, int], float] = {}
    for row in seed_rows:
        values[
            (
                str(row["condition"]),
                str(row["ablation"]),
                int(row["checkpoint_episode"]),
                str(row["f_value"]),
                int(row["train_seed"]),
            )
        ] = float(row["coop_rate"])

    candidate_pairs = sorted(
        {
            (
                str(row["condition"]),
                str(row["ablation"]),
                int(row["checkpoint_episode"]),
                str(row["f_value"]),
            )
            for row in seed_rows
            if not (
                str(row["condition"]) == str(reference_condition)
                and str(row["ablation"]) == str(reference_ablation)
            )
        }
    )

    out = []
    for condition, ablation, checkpoint_episode, f_value in candidate_pairs:
        if str(condition) != str(reference_condition):
            continue
        paired = []
        seeds = sorted(
            {
                int(row["train_seed"])
                for row in seed_rows
                if int(row["checkpoint_episode"]) == int(checkpoint_episode)
                and str(row["f_value"]) == str(f_value)
                and str(row["condition"]) == str(reference_condition)
                and str(row["ablation"]) in (str(reference_ablation), str(ablation))
            }
        )
        for seed in seeds:
            ref_key = (
                str(reference_condition),
                str(reference_ablation),
                int(checkpoint_episode),
                str(f_value),
                int(seed),
            )
            cur_key = (str(condition), str(ablation), int(checkpoint_episode), str(f_value), int(seed))
            if ref_key not in values or cur_key not in values:
                continue
            natural_value = float(values[ref_key])
            intervention_value = float(values[cur_key])
            paired.append(
                {
                    "train_seed": int(seed),
                    "natural_value": natural_value,
                    "intervention_value": intervention_value,
                    "delta_natural_minus_intervention": float(natural_value - intervention_value),
                }
            )
        if len(paired) == 0:
            continue
        deltas = [float(row["delta_natural_minus_intervention"]) for row in paired]
        ci_lo, ci_hi = _bootstrap_ci(
            deltas,
            n_boot=int(bootstrap_samples),
            seed=int(bootstrap_seed) + int(checkpoint_episode) + len(ablation) + int(1000 * float(f_value)),
        )
        out.append(
            {
                "reference_condition": str(reference_condition),
                "reference_ablation": str(reference_ablation),
                "condition": str(condition),
                "ablation": str(ablation),
                "checkpoint_episode": int(checkpoint_episode),
                "f_value": str(f_value),
                "n_pairs": len(paired),
                "natural_mean_coop_rate": _mean(float(row["natural_value"]) for row in paired),
                "intervention_mean_coop_rate": _mean(float(row["intervention_value"]) for row in paired),
                "mean_delta_natural_minus_intervention": _mean(deltas),
                "median_delta_natural_minus_intervention": float(median(deltas)),
                "sem_delta_natural_minus_intervention": _sem(deltas),
                "bootstrap_ci_low": float(ci_lo),
                "bootstrap_ci_high": float(ci_hi),
                "n_positive": int(sum(delta > 0.0 for delta in deltas)),
                "n_negative": int(sum(delta < 0.0 for delta in deltas)),
                "n_zero": int(sum(abs(delta) <= 1e-12 for delta in deltas)),
                "sign_flip_p_value": _sign_flip_p_value(deltas),
            }
        )
    return out


def _write_markdown(
    path: str,
    *,
    seed_rows: Sequence[Dict],
    summary_rows: Sequence[Dict],
    paired_rows: Sequence[Dict],
    training_family: str,
    source_repo: str,
    source_train_root: str,
    source_eval_root: str,
    seed_count: int,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = ["# Phase-3 Intervention Suite Summary", ""]
    lines.extend(
        [
            f"- training_family: `{training_family}`",
            f"- source_repo: `{source_repo}`",
            f"- source_train_root: `{source_train_root}`",
            f"- source_eval_root: `{source_eval_root}`",
            f"- seed_count: `{int(seed_count)}`",
            "",
            "| Condition | Mode | Episode | f | Mean Coop | SEM Coop | Mean Reward | SEM Reward | Mean Welfare | SEM Welfare |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['condition']} | {row['ablation']} | {int(row['checkpoint_episode'])} | "
            f"{float(row['f_value']):.1f} | {float(row['mean_coop_rate']):.4f} | "
            f"{float(row['sem_coop_rate']):.4f} | {float(row['mean_avg_reward']):.4f} | "
            f"{float(row['sem_avg_reward']):.4f} | {float(row['mean_avg_welfare']):.4f} | "
            f"{float(row['sem_avg_welfare']):.4f} |"
        )
    lines.append("")
    if seed_rows:
        lines.append("## Seed-Level Natural vs Intervention Contrasts")
        lines.append("")
        lines.append("Positive deltas below mean the intervention lowered cooperation relative to the same seed under natural learned messages.")
        lines.append("")
        lines.append("| Mode | Episode | f | Natural - Intervention | Median | 95% bootstrap CI | Sign-flip p | Positive Seeds |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in paired_rows:
            lines.append(
                f"| {row['ablation']} | {int(row['checkpoint_episode'])} | {float(row['f_value']):.1f} | "
                f"{_pct_label(float(row['mean_delta_natural_minus_intervention']))} | "
                f"{_pct_label(float(row['median_delta_natural_minus_intervention']))} | "
                f"[{_pct_label(float(row['bootstrap_ci_low']))}, {_pct_label(float(row['bootstrap_ci_high']))}] | "
                f"{float(row['sign_flip_p_value']):.4f} | "
                f"{int(row['n_positive'])}/{int(row['n_pairs'])} |"
            )
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suite_main_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--checkpoint_episode", type=int, required=True)
    p.add_argument("--f_values", nargs="*", type=float, default=[3.5, 5.0])
    p.add_argument("--training_family", type=str, default="phase3_vecstraight")
    p.add_argument("--source_repo", type=str, default="dsc-epgg-vectorized")
    p.add_argument("--source_train_root", type=str, required=True)
    p.add_argument("--source_eval_root", type=str, required=True)
    p.add_argument("--seed_count", type=int, default=15)
    p.add_argument("--bootstrap_samples", type=int, default=20000)
    p.add_argument("--bootstrap_seed", type=int, default=7)
    return p.parse_args()


def main():
    args = parse_args()
    rows = _read_rows(os.path.abspath(args.suite_main_csv))
    seed_rows = _collect_seed_rows(
        rows,
        checkpoint_episode=int(args.checkpoint_episode),
        target_f_values=_target_f_values(args.f_values),
    )
    summary_rows = _summarize_seed_rows(seed_rows)
    paired_rows = _paired_stats(
        seed_rows,
        reference_condition="cond1",
        reference_ablation="none",
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    _write_rows(os.path.join(out_dir, "intervention_suite_raw.csv"), seed_rows)
    _write_rows(os.path.join(out_dir, "intervention_suite_summary.csv"), summary_rows)
    _write_rows(os.path.join(out_dir, "intervention_suite_paired_stats.csv"), paired_rows)
    _write_markdown(
        os.path.join(out_dir, "intervention_suite_summary.md"),
        seed_rows=seed_rows,
        summary_rows=summary_rows,
        paired_rows=paired_rows,
        training_family=str(args.training_family),
        source_repo=str(args.source_repo),
        source_train_root=os.path.abspath(args.source_train_root),
        source_eval_root=os.path.abspath(args.source_eval_root),
        seed_count=int(args.seed_count),
    )
    print(f"[intervention-suite-summary] out_dir={out_dir}")


if __name__ == "__main__":
    main()
