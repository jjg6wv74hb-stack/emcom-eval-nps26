from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import random
from statistics import stdev
from typing import Dict, Iterable, List, Sequence, Tuple

from src.analysis.checkpoint_artifacts import atomic_write_rows, atomic_write_text


def _read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _as_int(value: str | None, default: int = 0) -> int:
    if value in ("", None):
        return int(default)
    return int(float(value))


def _as_float(value: str | None, default: float = 0.0) -> float:
    if value in ("", None):
        return float(default)
    return float(value)


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


def _collect_rows(
    suite_csv: str,
    *,
    mode: str,
    source_kind: str,
    condition: str,
    f_values: Sequence[str],
) -> List[Dict]:
    rows = []
    allowed_f = {str(v) for v in f_values}
    for row in _read_rows(os.path.abspath(suite_csv)):
        if str(row.get("condition", "")) != str(condition):
            continue
        if str(row.get("scope", "")) != "f_value":
            continue
        if str(row.get("eval_policy", "greedy")) != "greedy":
            continue
        if str(row.get("ablation", "none")) != "none":
            continue
        if str(row.get("cross_play", "none")) != "none":
            continue
        if str(row.get("sender_remap", "none")) != "none":
            continue
        f_value = str(row.get("key", ""))
        if f_value not in allowed_f:
            continue
        rows.append(
            {
                "mode": str(mode),
                "source_kind": str(source_kind),
                "suite_csv": os.path.abspath(suite_csv),
                "condition": str(condition),
                "train_seed": _as_int(row.get("train_seed"), -1),
                "checkpoint_episode": _as_int(row.get("checkpoint_episode"), 0),
                "f_value": f_value,
                "coop_rate": _as_float(row.get("coop_rate")),
                "avg_reward": _as_float(row.get("avg_reward")),
                "avg_welfare": _as_float(row.get("avg_welfare")),
            }
        )
    return rows


def _summarize(rows: Sequence[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, str, str, int, str], List[Dict]] = {}
    for row in rows:
        key = (
            str(row["mode"]),
            str(row["source_kind"]),
            str(row["condition"]),
            int(row["checkpoint_episode"]),
            str(row["f_value"]),
        )
        grouped.setdefault(key, []).append(row)

    out = []
    for (mode, source_kind, condition, checkpoint_episode, f_value), cur in sorted(grouped.items()):
        coop_vals = [float(row["coop_rate"]) for row in cur]
        reward_vals = [float(row["avg_reward"]) for row in cur]
        welfare_vals = [float(row["avg_welfare"]) for row in cur]
        out.append(
            {
                "mode": mode,
                "source_kind": source_kind,
                "condition": condition,
                "checkpoint_episode": int(checkpoint_episode),
                "f_value": f_value,
                "n_seeds": len(cur),
                "mean_coop_rate": _mean(coop_vals),
                "sem_coop_rate": _sem(coop_vals),
                "mean_avg_reward": _mean(reward_vals),
                "sem_avg_reward": _sem(reward_vals),
                "mean_avg_welfare": _mean(welfare_vals),
                "sem_avg_welfare": _sem(welfare_vals),
            }
        )
    return out


def _paired_stats(
    rows: Sequence[Dict],
    *,
    reference_mode: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> List[Dict]:
    continuation_modes = {
        str(row["mode"])
        for row in rows
        if str(row.get("source_kind", "")) == "continuation_branch"
    }
    values: Dict[Tuple[str, int, str, int], float] = {}
    for row in rows:
        values[
            (
                str(row["mode"]),
                int(row["checkpoint_episode"]),
                str(row["f_value"]),
                int(row["train_seed"]),
            )
        ] = float(row["coop_rate"])

    mode_names = sorted(continuation_modes)
    episodes = sorted({int(row["checkpoint_episode"]) for row in rows})
    f_values = sorted({str(row["f_value"]) for row in rows})

    out = []
    for mode in mode_names:
        if mode == str(reference_mode):
            continue
        for checkpoint_episode in episodes:
            for f_value in f_values:
                paired = []
                seeds = sorted(
                    {
                        seed
                        for (candidate_mode, candidate_episode, candidate_f, seed) in values.keys()
                        if candidate_episode == checkpoint_episode
                        and candidate_f == f_value
                        and candidate_mode in (mode, str(reference_mode))
                    }
                )
                for seed in seeds:
                    ref_key = (str(reference_mode), checkpoint_episode, f_value, seed)
                    cur_key = (mode, checkpoint_episode, f_value, seed)
                    if ref_key not in values or cur_key not in values:
                        continue
                    paired.append(
                        {
                            "train_seed": int(seed),
                            "reference_value": float(values[ref_key]),
                            "mode_value": float(values[cur_key]),
                            "delta_mode_minus_reference": float(values[cur_key] - values[ref_key]),
                        }
                    )
                if len(paired) == 0:
                    continue
                deltas = [row["delta_mode_minus_reference"] for row in paired]
                ci_lo, ci_hi = _bootstrap_ci(
                    deltas,
                    n_boot=int(bootstrap_samples),
                    seed=int(bootstrap_seed) + int(checkpoint_episode) + len(mode) + int(1000 * float(f_value)),
                )
                out.append(
                    {
                        "reference_mode": str(reference_mode),
                        "mode": mode,
                        "checkpoint_episode": int(checkpoint_episode),
                        "f_value": f_value,
                        "n_pairs": len(paired),
                        "reference_mean_coop_rate": _mean(row["reference_value"] for row in paired),
                        "mode_mean_coop_rate": _mean(row["mode_value"] for row in paired),
                        "mean_delta_mode_minus_reference": _mean(deltas),
                        "sem_delta_mode_minus_reference": _sem(deltas),
                        "bootstrap_ci_low": float(ci_lo),
                        "bootstrap_ci_high": float(ci_hi),
                        "sign_flip_p_value": _sign_flip_p_value(deltas),
                    }
                )
    return out


def _write_markdown(
    path: str,
    *,
    summary_rows: Sequence[Dict],
    paired_rows: Sequence[Dict],
    reference_mode: str,
    baseline_mode: str,
) -> None:
    def _lookup(mode: str, checkpoint_episode: int, f_value: str) -> Dict | None:
        for row in summary_rows:
            if (
                str(row["mode"]) == str(mode)
                and int(row["checkpoint_episode"]) == int(checkpoint_episode)
                and str(row["f_value"]) == str(f_value)
            ):
                return row
        return None

    lines = ["# Same-Checkpoint Continuation Summary", ""]
    lines.append(f"- reference_mode: `{reference_mode}`")
    lines.append(f"- baseline_mode: `{baseline_mode}`")
    lines.append("")
    lines.append("## Available natural-evaluation branch summaries")
    lines.append("")
    lines.append("| Mode | Episode | f | Mean Coop | SEM | Seeds |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in summary_rows:
        lines.append(
            f"| {row['mode']} | {int(row['checkpoint_episode'])} | {float(row['f_value']):.1f} | "
            f"{float(row['mean_coop_rate']):.4f} | {float(row['sem_coop_rate']):.4f} | {int(row['n_seeds'])} |"
        )
    lines.append("")
    lines.append("## Paired continuation contrasts vs learned reference")
    lines.append("")
    lines.append("| Mode | Episode | f | Delta vs learned | 95% bootstrap CI | Sign-flip p | Pairs |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in paired_rows:
        lines.append(
            f"| {row['mode']} | {int(row['checkpoint_episode'])} | {float(row['f_value']):.1f} | "
            f"{_pct_label(float(row['mean_delta_mode_minus_reference']))} | "
            f"[{_pct_label(float(row['bootstrap_ci_low']))}, {_pct_label(float(row['bootstrap_ci_high']))}] | "
            f"{float(row['sign_flip_p_value']):.4f} | {int(row['n_pairs'])} |"
        )
    lines.append("")

    highlight_rows = [
        row
        for row in paired_rows
        if int(row["checkpoint_episode"]) == 150000 and str(row["f_value"]) in ("3.500", "5.000")
    ]
    if len(highlight_rows) > 0:
        lines.append("## Current interpretation")
        lines.append("")
        for row in highlight_rows:
            direction = "above" if float(row["mean_delta_mode_minus_reference"]) > 0 else "below"
            lines.append(
                "- "
                f"`{row['mode']}` at {int(row['checkpoint_episode'])} and "
                f"$f={float(row['f_value']):.1f}$ is {direction} the learned reference by "
                f"{_pct_label(float(row['mean_delta_mode_minus_reference']))} "
                f"(sign-flip p={float(row['sign_flip_p_value']):.4f})."
            )
        lines.append("")

    learned_150k_rows = [
        row
        for row in summary_rows
        if str(row["mode"]) == str(reference_mode) and int(row["checkpoint_episode"]) == 150000
    ]
    baseline_150k_rows = [
        row
        for row in summary_rows
        if str(row["mode"]) == str(baseline_mode) and int(row["checkpoint_episode"]) == 150000
    ]
    if len(learned_150k_rows) > 0 and len(baseline_150k_rows) > 0:
        lines.append("## Endpoint reference context")
        lines.append("")
        for f_value in ("3.500", "5.000"):
            learned = _lookup(str(reference_mode), 150000, f_value)
            baseline = _lookup(str(baseline_mode), 150000, f_value)
            if learned is None or baseline is None:
                continue
            lines.append(
                "- "
                f"At 150k and $f={float(f_value):.1f}$, the learned communication branch is "
                f"{_pct_label(float(learned['mean_coop_rate']) - float(baseline['mean_coop_rate']))} "
                "above the separately trained no-communication baseline."
            )
        lines.append("")

    atomic_write_text(path, "\n".join(lines) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_suite_csv", type=str, required=True)
    parser.add_argument(
        "--branch_suite",
        nargs=2,
        action="append",
        metavar=("MODE", "CSV"),
        default=[],
        help="Branch label and checkpoint_suite_main.csv path.",
    )
    parser.add_argument("--reference_mode", type=str, default="learned")
    parser.add_argument("--baseline_mode", type=str, default="no_comm_baseline")
    parser.add_argument("--comm_condition", type=str, default="cond1")
    parser.add_argument("--baseline_condition", type=str, default="cond2")
    parser.add_argument("--f_values", nargs="*", type=float, default=[3.5, 5.0])
    parser.add_argument("--bootstrap_samples", type=int, default=20000)
    parser.add_argument("--bootstrap_seed", type=int, default=7)
    parser.add_argument("--out_dir", type=str, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    f_values = [f"{float(v):.3f}" for v in args.f_values]

    raw_rows: List[Dict] = []
    raw_rows.extend(
        _collect_rows(
            args.reference_suite_csv,
            mode=str(args.reference_mode),
            source_kind="reference_comm",
            condition=str(args.comm_condition),
            f_values=f_values,
        )
    )
    raw_rows.extend(
        _collect_rows(
            args.reference_suite_csv,
            mode=str(args.baseline_mode),
            source_kind="reference_baseline",
            condition=str(args.baseline_condition),
            f_values=f_values,
        )
    )

    for mode, suite_csv in args.branch_suite:
        raw_rows.extend(
            _collect_rows(
                suite_csv,
                mode=str(mode),
                source_kind="continuation_branch",
                condition=str(args.comm_condition),
                f_values=f_values,
            )
        )

    summary_rows = _summarize(raw_rows)
    paired_rows = _paired_stats(
        raw_rows,
        reference_mode=str(args.reference_mode),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )

    atomic_write_rows(
        os.path.join(out_dir, "sameckpt_continuation_raw.csv"),
        raw_rows,
        fieldnames=[
            "mode",
            "source_kind",
            "suite_csv",
            "condition",
            "train_seed",
            "checkpoint_episode",
            "f_value",
            "coop_rate",
            "avg_reward",
            "avg_welfare",
        ],
    )
    atomic_write_rows(
        os.path.join(out_dir, "sameckpt_continuation_summary.csv"),
        summary_rows,
        fieldnames=[
            "mode",
            "source_kind",
            "condition",
            "checkpoint_episode",
            "f_value",
            "n_seeds",
            "mean_coop_rate",
            "sem_coop_rate",
            "mean_avg_reward",
            "sem_avg_reward",
            "mean_avg_welfare",
            "sem_avg_welfare",
        ],
    )
    atomic_write_rows(
        os.path.join(out_dir, "sameckpt_continuation_paired_stats.csv"),
        paired_rows,
        fieldnames=[
            "reference_mode",
            "mode",
            "checkpoint_episode",
            "f_value",
            "n_pairs",
            "reference_mean_coop_rate",
            "mode_mean_coop_rate",
            "mean_delta_mode_minus_reference",
            "sem_delta_mode_minus_reference",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "sign_flip_p_value",
        ],
    )
    _write_markdown(
        os.path.join(out_dir, "sameckpt_continuation_summary.md"),
        summary_rows=summary_rows,
        paired_rows=paired_rows,
        reference_mode=str(args.reference_mode),
        baseline_mode=str(args.baseline_mode),
    )
    print(f"[sameckpt-continuations] out_dir={out_dir}")


if __name__ == "__main__":
    main()
