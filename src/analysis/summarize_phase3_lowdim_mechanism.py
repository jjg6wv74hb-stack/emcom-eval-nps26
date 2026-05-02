from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


AGENT_IDS = [f"agent_{i}" for i in range(4)]
TRACE_USECOLS = [
    "condition",
    "train_seed",
    "eval_policy",
    "ablation",
    "history_intervention",
    "cross_play",
    "agent_id",
    "true_f",
    "f_hat",
    "action",
    "obs_last_coop_fraction",
    "obs_own_last_action",
    "obs_ewma_coop",
    "checkpoint_episode",
    "suite_kind",
    *[f"delivered_msg_{agent_id}" for agent_id in AGENT_IDS],
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trace_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--condition", type=str, default="cond1")
    p.add_argument("--checkpoint_episode", type=int, default=150000)
    p.add_argument("--ablation", type=str, default="none")
    p.add_argument("--history_intervention", type=str, default="none")
    p.add_argument("--eval_policy", type=str, default="greedy")
    p.add_argument("--suite_kind", type=str, default="comm")
    p.add_argument("--f_values", nargs="*", type=float, default=[3.5, 5.0])
    p.add_argument("--chunksize", type=int, default=250000)
    p.add_argument("--per_seed_sample", type=int, default=5000)
    p.add_argument("--min_pattern_obs", type=int, default=200)
    p.add_argument("--random_state", type=int, default=0)
    return p.parse_args()


def _sem(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def _format_pp(value: float) -> str:
    return f"{value * 100.0:+.1f}pp"


def _ensure_out_dir(path: str) -> Path:
    out_dir = Path(path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _load_filtered_trace(args: argparse.Namespace) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    f_values = {float(v) for v in args.f_values}
    delivered_cols = [f"delivered_msg_{agent_id}" for agent_id in AGENT_IDS]
    for chunk in pd.read_csv(args.trace_csv, usecols=TRACE_USECOLS, chunksize=args.chunksize):
        mask = (
            (chunk["condition"] == str(args.condition))
            & (chunk["eval_policy"] == str(args.eval_policy))
            & (chunk["ablation"] == str(args.ablation))
            & (chunk["history_intervention"] == str(args.history_intervention))
            & (chunk["cross_play"] == "none")
            & (pd.to_numeric(chunk["checkpoint_episode"], errors="coerce") == int(args.checkpoint_episode))
            & (chunk["suite_kind"] == str(args.suite_kind))
        )
        cur = chunk.loc[mask].copy()
        if cur.empty:
            continue

        cur["true_f"] = pd.to_numeric(cur["true_f"], errors="coerce")
        cur = cur[cur["true_f"].isin(f_values)].copy()
        if cur.empty:
            continue

        cur["train_seed"] = pd.to_numeric(cur["train_seed"], errors="coerce").astype(int)
        cur["f_hat"] = pd.to_numeric(cur["f_hat"], errors="coerce").astype(float)
        cur["action"] = pd.to_numeric(cur["action"], errors="coerce").astype(int)
        cur["obs_last_coop_fraction"] = pd.to_numeric(
            cur["obs_last_coop_fraction"], errors="coerce"
        ).astype(float)
        cur["obs_own_last_action"] = pd.to_numeric(
            cur["obs_own_last_action"], errors="coerce"
        ).astype(float)
        cur["obs_ewma_coop"] = pd.to_numeric(cur["obs_ewma_coop"], errors="coerce").astype(float)

        for agent_id in AGENT_IDS:
            source = f"delivered_msg_{agent_id}"
            dest = f"recv_from_{agent_id}"
            cur[dest] = pd.to_numeric(cur[source], errors="coerce").fillna(0).astype(int)
            cur.loc[cur["agent_id"] == agent_id, dest] = 0

        recv_cols = [f"recv_from_{agent_id}" for agent_id in AGENT_IDS]
        cur["count_ones"] = cur[recv_cols].sum(axis=1).astype(int)
        cur["any_token"] = (cur["count_ones"] > 0).astype(int)
        cur["pattern"] = cur.apply(
            lambda row: row["agent_id"]
            + "::"
            + "|".join(f"{agent_id}:{int(row[f'recv_from_{agent_id}'])}" for agent_id in AGENT_IDS if agent_id != row["agent_id"]),
            axis=1,
        )
        parts.append(
            cur[
                [
                    "train_seed",
                    "agent_id",
                    "true_f",
                    "f_hat",
                    "action",
                    "obs_last_coop_fraction",
                    "obs_own_last_action",
                    "obs_ewma_coop",
                    "count_ones",
                    "any_token",
                    "pattern",
                    *recv_cols,
                ]
            ].copy()
        )

    if not parts:
        raise ValueError("no rows matched the requested frozen-output slice")
    out = pd.concat(parts, ignore_index=True)
    out["true_f"] = out["true_f"].astype(float)
    return out


def _aggregate_seed_curves(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    seed_level = (
        df.groupby(["true_f", "train_seed", group_col], as_index=False)
        .agg(n_obs=("action", "size"), p_cooperate=("action", "mean"))
        .sort_values(["true_f", "train_seed", group_col])
    )
    summary = (
        seed_level.groupby(["true_f", group_col], as_index=False)
        .agg(
            n_seeds=("train_seed", "nunique"),
            total_obs=("n_obs", "sum"),
            mean_obs_per_seed=("n_obs", "mean"),
            mean_p_cooperate=("p_cooperate", "mean"),
            std_p_cooperate=("p_cooperate", "std"),
            sem_p_cooperate=("p_cooperate", _sem),
        )
        .sort_values(["true_f", group_col])
    )
    return summary


def _pattern_summaries(df: pd.DataFrame, min_pattern_obs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    pattern_df = (
        df.groupby(["true_f", "count_ones", "pattern"], as_index=False)
        .agg(n_obs=("action", "size"), p_cooperate=("action", "mean"))
        .sort_values(["true_f", "count_ones", "pattern"])
    )
    count_mean = (
        df.groupby(["true_f", "count_ones"], as_index=False)
        .agg(count_mean_p_cooperate=("action", "mean"), total_obs=("action", "size"))
    )
    merged = pattern_df.merge(count_mean, on=["true_f", "count_ones"], how="left")
    merged["abs_dev_pp"] = (merged["p_cooperate"] - merged["count_mean_p_cooperate"]).abs() * 100.0

    summary_rows: List[Dict] = []
    for (true_f, count_ones), cur in merged.groupby(["true_f", "count_ones"], sort=True):
        common = cur[cur["n_obs"] >= int(min_pattern_obs)].copy()
        if common.empty:
            common = cur.copy()
        common = common.sort_values(["p_cooperate", "pattern"])
        low = common.iloc[0]
        high = common.iloc[-1]
        weighted_abs_dev_pp = float(np.average(common["abs_dev_pp"], weights=common["n_obs"]))
        summary_rows.append(
            {
                "true_f": float(true_f),
                "count_ones": int(count_ones),
                "pattern_rows": int(len(cur)),
                "common_pattern_rows": int(len(common)),
                "weighted_abs_dev_pp": weighted_abs_dev_pp,
                "common_pattern_gap_pp": float((high["p_cooperate"] - low["p_cooperate"]) * 100.0),
                "low_pattern": str(low["pattern"]),
                "low_pattern_p_cooperate": float(low["p_cooperate"]),
                "low_pattern_n_obs": int(low["n_obs"]),
                "high_pattern": str(high["pattern"]),
                "high_pattern_p_cooperate": float(high["p_cooperate"]),
                "high_pattern_n_obs": int(high["n_obs"]),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["true_f", "count_ones"])
    top_examples = (
        pattern_df.sort_values(["true_f", "count_ones", "n_obs", "p_cooperate", "pattern"], ascending=[True, True, False, False, True])
        .groupby(["true_f", "count_ones"], as_index=False, group_keys=False)
        .head(5)
        .reset_index(drop=True)
    )
    return summary_df, top_examples


def _pattern_seed_summaries(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for (true_f, train_seed, count_ones), cur in df.groupby(
        ["true_f", "train_seed", "count_ones"], sort=True
    ):
        pattern_df = (
            cur.groupby("pattern", as_index=False)
            .agg(n_obs=("action", "size"), p_cooperate=("action", "mean"))
            .sort_values("pattern")
        )
        if len(pattern_df) < 2:
            continue
        count_mean = float(cur["action"].mean())
        pattern_df["abs_dev_pp"] = (pattern_df["p_cooperate"] - count_mean).abs() * 100.0
        low = pattern_df.nsmallest(1, "p_cooperate").iloc[0]
        high = pattern_df.nlargest(1, "p_cooperate").iloc[0]
        rows.append(
            {
                "true_f": float(true_f),
                "train_seed": int(train_seed),
                "count_ones": int(count_ones),
                "n_patterns": int(len(pattern_df)),
                "n_obs": int(len(cur)),
                "gap_pp": float((high["p_cooperate"] - low["p_cooperate"]) * 100.0),
                "weighted_abs_dev_pp": float(
                    np.average(pattern_df["abs_dev_pp"], weights=pattern_df["n_obs"])
                ),
            }
        )
    seed_df = pd.DataFrame(rows).sort_values(["true_f", "train_seed", "count_ones"]).reset_index(drop=True)
    seed_df["gap_gt_10pp"] = (seed_df["gap_pp"] > 10.0).astype(int)
    return seed_df


def _summarize_pattern_seed_df(seed_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        seed_df.groupby(["true_f", "count_ones"], as_index=False)
        .agg(
            n_seeds=("train_seed", "nunique"),
            mean_gap_pp=("gap_pp", "mean"),
            median_gap_pp=("gap_pp", "median"),
            sem_gap_pp=("gap_pp", _sem),
            mean_weighted_abs_dev_pp=("weighted_abs_dev_pp", "mean"),
            median_weighted_abs_dev_pp=("weighted_abs_dev_pp", "median"),
            share_gap_gt_10pp=("gap_gt_10pp", "mean"),
        )
        .sort_values(["true_f", "count_ones"])
        .reset_index(drop=True)
    )
    return summary


def _subsample_for_surrogate(df: pd.DataFrame, per_seed_sample: int, random_state: int) -> pd.DataFrame:
    if int(per_seed_sample) <= 0:
        return df.copy()
    parts = []
    for (_, train_seed), cur in df.groupby(["true_f", "train_seed"], sort=True):
        if len(cur) <= int(per_seed_sample):
            parts.append(cur)
            continue
        parts.append(cur.sample(n=int(per_seed_sample), random_state=int(random_state) + int(train_seed)))
    return pd.concat(parts, ignore_index=True)


def _build_model(model_name: str) -> tuple[Pipeline, List[str], List[str]]:
    base_numeric = ["f_hat", "obs_last_coop_fraction", "obs_own_last_action", "obs_ewma_coop"]
    categorical = ["agent_id"]
    if model_name == "history_only":
        numeric = list(base_numeric)
    elif model_name == "history_any":
        numeric = [*base_numeric, "any_token"]
    elif model_name == "history_count":
        numeric = [*base_numeric, "count_ones"]
    elif model_name == "history_sender_bits":
        numeric = [*base_numeric, *[f"recv_from_{agent_id}" for agent_id in AGENT_IDS]]
    elif model_name == "history_pattern":
        numeric = list(base_numeric)
        # Full received-pattern identity, including which receiver saw the pattern.
        categorical = ["pattern"]
    else:
        raise ValueError(f"unknown model_name={model_name!r}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    clf = LogisticRegression(max_iter=2000, solver="liblinear")
    model = Pipeline([("prep", preprocessor), ("clf", clf)])
    return model, numeric, categorical


def _evaluate_surrogates(df: pd.DataFrame) -> pd.DataFrame:
    model_names = [
        "history_only",
        "history_any",
        "history_count",
        "history_sender_bits",
        "history_pattern",
    ]
    rows: List[Dict] = []
    for true_f, cur in df.groupby("true_f", sort=True):
        cur = cur.reset_index(drop=True)
        groups = cur["train_seed"].to_numpy()
        y = cur["action"].to_numpy(dtype=int)
        n_groups = len(pd.unique(groups))
        if n_groups < 3:
            raise ValueError(f"need at least 3 train_seed groups for surrogate CV; got {n_groups}")
        splitter = GroupKFold(n_splits=min(5, n_groups))

        for model_name in model_names:
            fold_rows = []
            for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(cur, y, groups)):
                model, numeric_cols, categorical_cols = _build_model(model_name)
                x_train = cur.iloc[train_idx][numeric_cols + categorical_cols]
                x_test = cur.iloc[test_idx][numeric_cols + categorical_cols]
                y_train = y[train_idx]
                y_test = y[test_idx]

                model.fit(x_train, y_train)
                prob = model.predict_proba(x_test)[:, 1]
                pred = (prob >= 0.5).astype(int)
                fold_rows.append(
                    {
                        "true_f": float(true_f),
                        "model_name": model_name,
                        "fold": int(fold_idx),
                        "test_seeds": ",".join(str(v) for v in sorted(pd.unique(groups[test_idx]))),
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        "log_loss": float(log_loss(y_test, prob, labels=[0, 1])),
                        "accuracy": float(accuracy_score(y_test, pred)),
                        "brier": float(brier_score_loss(y_test, prob)),
                    }
                )
            fold_df = pd.DataFrame(fold_rows)
            rows.append(
                {
                    "true_f": float(true_f),
                    "model_name": model_name,
                    "n_folds": int(len(fold_df)),
                    "mean_log_loss": float(fold_df["log_loss"].mean()),
                    "std_log_loss": float(fold_df["log_loss"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
                    "mean_accuracy": float(fold_df["accuracy"].mean()),
                    "std_accuracy": float(fold_df["accuracy"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
                    "mean_brier": float(fold_df["brier"].mean()),
                    "std_brier": float(fold_df["brier"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
                    "fold_details": fold_df.to_dict(orient="records"),
                }
            )
    summary = pd.DataFrame(rows).sort_values(["true_f", "mean_log_loss", "model_name"]).reset_index(drop=True)
    # Flatten fold details into a separate CSV-friendly table.
    fold_rows_flat: List[Dict] = []
    for row in rows:
        for detail in row["fold_details"]:
            fold_rows_flat.append(detail)
    fold_df = pd.DataFrame(fold_rows_flat).sort_values(["true_f", "model_name", "fold"]).reset_index(drop=True)
    return summary, fold_df


def _surrogate_delta_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for true_f, cur in summary.groupby("true_f", sort=True):
        cur = cur.set_index("model_name")
        base_log_loss = float(cur.loc["history_only", "mean_log_loss"])
        count_log_loss = float(cur.loc["history_count", "mean_log_loss"])
        base_acc = float(cur.loc["history_only", "mean_accuracy"])
        count_acc = float(cur.loc["history_count", "mean_accuracy"])
        for model_name, row in cur.iterrows():
            rows.append(
                {
                    "true_f": float(true_f),
                    "model_name": str(model_name),
                    "mean_log_loss": float(row["mean_log_loss"]),
                    "mean_accuracy": float(row["mean_accuracy"]),
                    "delta_log_loss_vs_history_only": float(base_log_loss - float(row["mean_log_loss"])),
                    "delta_accuracy_vs_history_only": float(float(row["mean_accuracy"]) - base_acc),
                    "delta_log_loss_vs_history_count": float(count_log_loss - float(row["mean_log_loss"])),
                    "delta_accuracy_vs_history_count": float(float(row["mean_accuracy"]) - count_acc),
                }
            )
    return pd.DataFrame(rows).sort_values(["true_f", "model_name"]).reset_index(drop=True)


def _write_markdown(
    out_path: Path,
    args: argparse.Namespace,
    filtered_df: pd.DataFrame,
    count_df: pd.DataFrame,
    any_df: pd.DataFrame,
    pattern_summary_df: pd.DataFrame,
    pattern_seed_summary_df: pd.DataFrame,
    top_patterns_df: pd.DataFrame,
    surrogate_delta_df: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# Phase-3 Low-Dimensional Mechanism Summary")
    lines.append("")
    lines.append(f"- trace_csv: `{os.path.abspath(args.trace_csv)}`")
    lines.append(f"- condition: `{args.condition}`")
    lines.append(f"- checkpoint_episode: `{args.checkpoint_episode}`")
    lines.append(f"- ablation: `{args.ablation}`")
    lines.append(f"- history_intervention: `{args.history_intervention}`")
    lines.append(f"- suite_kind: `{args.suite_kind}`")
    lines.append(f"- focal_f_values: `{' '.join(str(v) for v in args.f_values)}`")
    lines.append(f"- rows_after_filter: `{len(filtered_df)}`")
    lines.append("")

    lines.append("## Count-of-Ones Response")
    lines.append("")
    lines.append("| true_f | count_ones | total_obs | mean_p_cooperate | sem_p_cooperate |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for _, row in count_df.iterrows():
        lines.append(
            f"| {row['true_f']:.1f} | {int(row['count_ones'])} | {int(row['total_obs'])} | "
            f"{row['mean_p_cooperate']:.4f} | {row['sem_p_cooperate']:.4f} |"
        )
    lines.append("")

    lines.append("## Any-Token Response")
    lines.append("")
    lines.append("| true_f | any_token | total_obs | mean_p_cooperate | sem_p_cooperate |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for _, row in any_df.iterrows():
        lines.append(
            f"| {row['true_f']:.1f} | {int(row['any_token'])} | {int(row['total_obs'])} | "
            f"{row['mean_p_cooperate']:.4f} | {row['sem_p_cooperate']:.4f} |"
        )
    lines.append("")

    lines.append("## Pattern Heterogeneity Within Count")
    lines.append("")
    lines.append("| true_f | count_ones | weighted_abs_dev_pp | common_pattern_gap_pp | low_pattern | high_pattern |")
    lines.append("| --- | ---: | ---: | ---: | --- | --- |")
    for _, row in pattern_summary_df.iterrows():
        lines.append(
            f"| {row['true_f']:.1f} | {int(row['count_ones'])} | {row['weighted_abs_dev_pp']:.2f} | "
            f"{row['common_pattern_gap_pp']:.2f} | `{row['low_pattern']}` | `{row['high_pattern']}` |"
        )
    lines.append("")

    lines.append("## Seed-Level Pattern Heterogeneity")
    lines.append("")
    lines.append(
        "| true_f | count_ones | n_seeds | mean_gap_pp | median_gap_pp | "
        "mean_weighted_abs_dev_pp | share_gap_gt_10pp |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, row in pattern_seed_summary_df.iterrows():
        lines.append(
            f"| {row['true_f']:.1f} | {int(row['count_ones'])} | {int(row['n_seeds'])} | "
            f"{row['mean_gap_pp']:.2f} | {row['median_gap_pp']:.2f} | "
            f"{row['mean_weighted_abs_dev_pp']:.2f} | {row['share_gap_gt_10pp']:.2f} |"
        )
    lines.append("")

    lines.append("## Surrogate Comparison")
    lines.append("")
    lines.append(
        "| true_f | model_name | mean_log_loss | improvement_vs_history_count | "
        "mean_accuracy | delta_acc_vs_history_count |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for _, row in surrogate_delta_df.iterrows():
        lines.append(
            f"| {row['true_f']:.1f} | `{row['model_name']}` | {row['mean_log_loss']:.4f} | "
            f"{row['delta_log_loss_vs_history_count']:.4f} | {row['mean_accuracy']:.4f} | "
            f"{row['delta_accuracy_vs_history_count']:.4f} |"
        )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    for true_f in sorted(pd.unique(count_df["true_f"])):
        cur_count = count_df[count_df["true_f"] == true_f].sort_values("count_ones")
        cur_any = any_df[any_df["true_f"] == true_f].sort_values("any_token")
        cur_pattern = pattern_summary_df[pattern_summary_df["true_f"] == true_f].sort_values("count_ones")
        cur_pattern_seed = pattern_seed_summary_df[
            pattern_seed_summary_df["true_f"] == true_f
        ].sort_values("count_ones")
        cur_sur = surrogate_delta_df[surrogate_delta_df["true_f"] == true_f].set_index("model_name")
        count_range = (
            float(cur_count["mean_p_cooperate"].min()),
            float(cur_count["mean_p_cooperate"].max()),
        )
        any_gap = float(
            cur_any.loc[cur_any["any_token"] == 1, "mean_p_cooperate"].iloc[0]
            - cur_any.loc[cur_any["any_token"] == 0, "mean_p_cooperate"].iloc[0]
        )
        common_gap_max = float(cur_pattern["common_pattern_gap_pp"].max())
        median_seed_gap_max = float(cur_pattern_seed["median_gap_pp"].max())
        seed_gap_share_max = float(cur_pattern_seed["share_gap_gt_10pp"].max())
        sender_bits_gain = float(cur_sur.loc["history_sender_bits", "delta_log_loss_vs_history_count"])
        pattern_gain = float(cur_sur.loc["history_pattern", "delta_log_loss_vs_history_count"])
        lines.append(
            f"- `f={true_f:.1f}`: count-of-ones responses span `{count_range[0]:.3f}` to `{count_range[1]:.3f}`, "
            f"`any_token` shifts cooperation by `{_format_pp(any_gap)}`, "
            f"the largest pooled within-count common-pattern gap is `{common_gap_max:.1f}pp`, "
            f"the largest median within-seed pattern gap is `{median_seed_gap_max:.1f}pp` "
            f"(with up to `{seed_gap_share_max:.2f}` of seeds above `10pp`), "
            f"the linear sender-bit surrogate changes log loss vs count by `{sender_bits_gain:.4f}`, "
            f"and the full pattern surrogate changes log loss vs count by `{pattern_gain:.4f}`."
        )
    lines.append("")

    lines.append("## Top Common Patterns")
    lines.append("")
    lines.append("| true_f | count_ones | n_obs | p_cooperate | pattern |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    for _, row in top_patterns_df.iterrows():
        lines.append(
            f"| {row['true_f']:.1f} | {int(row['count_ones'])} | {int(row['n_obs'])} | "
            f"{row['p_cooperate']:.4f} | `{row['pattern']}` |"
        )
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = _ensure_out_dir(args.out_dir)

    filtered_df = _load_filtered_trace(args)
    count_df = _aggregate_seed_curves(filtered_df, "count_ones")
    any_df = _aggregate_seed_curves(filtered_df, "any_token")
    pattern_summary_df, top_patterns_df = _pattern_summaries(
        filtered_df, min_pattern_obs=int(args.min_pattern_obs)
    )
    pattern_seed_df = _pattern_seed_summaries(filtered_df)
    pattern_seed_summary_df = _summarize_pattern_seed_df(pattern_seed_df)

    surrogate_input = _subsample_for_surrogate(
        filtered_df, per_seed_sample=int(args.per_seed_sample), random_state=int(args.random_state)
    )
    surrogate_summary_df, surrogate_folds_df = _evaluate_surrogates(surrogate_input)
    surrogate_delta_df = _surrogate_delta_table(surrogate_summary_df)

    filtered_df.to_csv(out_dir / "filtered_trace_slice.csv", index=False)
    count_df.to_csv(out_dir / "count_response_summary.csv", index=False)
    any_df.to_csv(out_dir / "any_token_response_summary.csv", index=False)
    pattern_summary_df.to_csv(out_dir / "pattern_identity_summary.csv", index=False)
    pattern_seed_df.to_csv(out_dir / "pattern_seed_details.csv", index=False)
    pattern_seed_summary_df.to_csv(out_dir / "pattern_seed_summary.csv", index=False)
    top_patterns_df.to_csv(out_dir / "pattern_top_examples.csv", index=False)
    surrogate_summary_df.drop(columns=["fold_details"]).to_csv(
        out_dir / "surrogate_model_summary.csv", index=False
    )
    surrogate_delta_df.to_csv(out_dir / "surrogate_model_deltas.csv", index=False)
    surrogate_folds_df.to_csv(out_dir / "surrogate_model_folds.csv", index=False)
    _write_markdown(
        out_dir / "lowdim_mechanism_summary.md",
        args=args,
        filtered_df=filtered_df,
        count_df=count_df,
        any_df=any_df,
        pattern_summary_df=pattern_summary_df,
        pattern_seed_summary_df=pattern_seed_summary_df,
        top_patterns_df=top_patterns_df,
        surrogate_delta_df=surrogate_delta_df,
    )
    print(f"[lowdim-mechanism] out_dir={out_dir}")


if __name__ == "__main__":
    main()
