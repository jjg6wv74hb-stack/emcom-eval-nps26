from __future__ import annotations

import argparse
import os
from statistics import median
from typing import Dict, List, Sequence

from src.analysis.summarize_phase3_intervention_suite import (
    _bootstrap_ci,
    _collect_seed_rows,
    _mean,
    _read_rows,
    _sem,
    _sign_flip_p_value,
    _summarize_seed_rows,
    _target_f_values,
    _write_rows,
)


_TEST_TO_ABLATION = {
    "natural": "none",
    "zeros": "zeros",
    "indep_random": "indep_random",
    "public_random": "public_random",
    "fixed0": "fixed0",
    "fixed1": "fixed1",
}


def _paired_seed_rows(
    seed_rows: Sequence[Dict],
    *,
    condition: str,
    target_ablation: str,
    checkpoint_episode: int,
    f_value: str,
) -> List[Dict]:
    values = {
        (
            str(row["condition"]),
            str(row["ablation"]),
            int(row["checkpoint_episode"]),
            str(row["f_value"]),
            int(row["train_seed"]),
        ): float(row["coop_rate"])
        for row in seed_rows
    }
    seeds = sorted(
        {
            int(row["train_seed"])
            for row in seed_rows
            if str(row["condition"]) == str(condition)
            and int(row["checkpoint_episode"]) == int(checkpoint_episode)
            and str(row["f_value"]) == str(f_value)
            and str(row["ablation"]) in ("none", str(target_ablation))
        }
    )
    out = []
    for seed in seeds:
        ref_key = (str(condition), "none", int(checkpoint_episode), str(f_value), int(seed))
        cur_key = (
            str(condition),
            str(target_ablation),
            int(checkpoint_episode),
            str(f_value),
            int(seed),
        )
        if ref_key not in values or cur_key not in values:
            continue
        natural_value = float(values[ref_key])
        intervention_value = float(values[cur_key])
        out.append(
            {
                "condition": str(condition),
                "ablation": str(target_ablation),
                "checkpoint_episode": int(checkpoint_episode),
                "f_value": str(f_value),
                "train_seed": int(seed),
                "natural_value": natural_value,
                "intervention_value": intervention_value,
                "delta_natural_minus_intervention": float(natural_value - intervention_value),
            }
        )
    return out


def _paired_summary_rows(
    paired_seed_rows: Sequence[Dict],
    *,
    target_ablation: str,
    checkpoint_episode: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> List[Dict]:
    grouped = {}
    for row in paired_seed_rows:
        grouped.setdefault(str(row["f_value"]), []).append(row)

    out = []
    for f_value, cur in sorted(grouped.items()):
        deltas = [float(row["delta_natural_minus_intervention"]) for row in cur]
        if len(deltas) == 0:
            continue
        ci_lo, ci_hi = _bootstrap_ci(
            deltas,
            n_boot=int(bootstrap_samples),
            seed=int(bootstrap_seed)
            + int(checkpoint_episode)
            + len(str(target_ablation))
            + int(1000 * float(f_value)),
        )
        out.append(
            {
                "reference_condition": "cond1",
                "reference_ablation": "none",
                "condition": "cond1",
                "ablation": str(target_ablation),
                "checkpoint_episode": int(checkpoint_episode),
                "f_value": str(f_value),
                "n_pairs": len(cur),
                "natural_mean_coop_rate": _mean(float(row["natural_value"]) for row in cur),
                "intervention_mean_coop_rate": _mean(
                    float(row["intervention_value"]) for row in cur
                ),
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


def _write_cell_reports(
    *,
    train_mode: str,
    suite_main_csv: str,
    out_base: str,
    run_kind: str,
    run_date: str,
    checkpoint_episode: int,
    f_values: Sequence[float],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> None:
    rows = _read_rows(os.path.abspath(suite_main_csv))
    seed_rows = _collect_seed_rows(
        rows,
        checkpoint_episode=int(checkpoint_episode),
        target_f_values=_target_f_values(f_values),
    )
    summary_rows = _summarize_seed_rows(seed_rows)

    for test_mode, ablation in _TEST_TO_ABLATION.items():
        cell_root = os.path.join(
            out_base,
            f"phase3_vecstraight_zeroaux_crossover_train_{train_mode}_test_{test_mode}_15seeds_{run_kind}_{run_date}",
            "report",
        )
        os.makedirs(cell_root, exist_ok=True)
        cell_seed_rows = [
            row
            for row in seed_rows
            if str(row["condition"]) == "cond1" and str(row["ablation"]) in ("none", ablation)
        ]
        cell_summary_rows = [
            row
            for row in summary_rows
            if str(row["condition"]) == "cond1" and str(row["ablation"]) == str(ablation)
        ]
        seed_diff_rows: List[Dict] = []
        for f_value in _target_f_values(f_values):
            seed_diff_rows.extend(
                _paired_seed_rows(
                    seed_rows,
                    condition="cond1",
                    target_ablation=str(ablation),
                    checkpoint_episode=int(checkpoint_episode),
                    f_value=str(f_value),
                )
            )
        paired_rows = _paired_summary_rows(
            seed_diff_rows,
            target_ablation=str(ablation),
            checkpoint_episode=int(checkpoint_episode),
            bootstrap_samples=int(bootstrap_samples),
            bootstrap_seed=int(bootstrap_seed),
        )
        _write_rows(os.path.join(cell_root, "intervention_suite_raw.csv"), cell_seed_rows)
        _write_rows(os.path.join(cell_root, "intervention_suite_summary.csv"), cell_summary_rows)
        _write_rows(os.path.join(cell_root, "intervention_suite_paired_stats.csv"), paired_rows)
        _write_rows(os.path.join(cell_root, "intervention_suite_seedwise_diffs.csv"), seed_diff_rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train_suite",
        nargs=2,
        action="append",
        metavar=("TRAIN_MODE", "CSV"),
        default=[],
        required=True,
    )
    p.add_argument("--out_base", type=str, required=True)
    p.add_argument("--run_kind", type=str, default="local")
    p.add_argument("--run_date", type=str, required=True)
    p.add_argument("--checkpoint_episode", type=int, default=150000)
    p.add_argument("--f_values", nargs="*", type=float, default=[3.5, 5.0])
    p.add_argument("--bootstrap_samples", type=int, default=20000)
    p.add_argument("--bootstrap_seed", type=int, default=7)
    return p.parse_args()


def main():
    args = parse_args()
    out_base = os.path.abspath(args.out_base)
    os.makedirs(out_base, exist_ok=True)
    for train_mode, suite_csv in list(args.train_suite or []):
        _write_cell_reports(
            train_mode=str(train_mode),
            suite_main_csv=str(suite_csv),
            out_base=out_base,
            run_kind=str(args.run_kind),
            run_date=str(args.run_date),
            checkpoint_episode=int(args.checkpoint_episode),
            f_values=[float(v) for v in args.f_values],
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_seed=int(args.bootstrap_seed),
        )
    print(f"[crossover-cell-reports] out_base={out_base}")


if __name__ == "__main__":
    main()
