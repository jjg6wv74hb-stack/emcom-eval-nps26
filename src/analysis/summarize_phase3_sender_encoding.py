from __future__ import annotations

import argparse
import csv
import math
import os
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REQUIRED_TRACE_COLS = [
    "checkpoint",
    "condition",
    "train_seed",
    "eval_seed",
    "eval_policy",
    "ablation",
    "history_intervention",
    "sender_remap",
    "cross_play",
    "agent_id",
    "true_f",
    "f_hat",
    "action",
    "obs_last_coop_fraction",
    "obs_own_last_action",
    "obs_ewma_coop",
    "own_sent_msg",
]
OPTIONAL_TRACE_COLS = ["intended_action", "checkpoint_episode", "suite_kind"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trace_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--condition", type=str, default="cond1")
    p.add_argument("--checkpoint_episode", type=int, default=150000)
    p.add_argument("--ablation", type=str, default="none")
    p.add_argument("--history_intervention", type=str, default="none")
    p.add_argument("--sender_remap", type=str, default="none")
    p.add_argument("--cross_play", type=str, default="none")
    p.add_argument("--eval_policy", type=str, default="greedy")
    p.add_argument("--suite_kind", type=str, default="comm")
    p.add_argument("--true_f_values", nargs="*", type=float, default=[])
    p.add_argument("--fhat_split", type=float, default=3.5)
    p.add_argument("--ewma_low", type=float, default=1.0 / 3.0)
    p.add_argument("--ewma_high", type=float, default=2.0 / 3.0)
    p.add_argument("--min_cell_obs", type=int, default=50)
    p.add_argument("--dominance_threshold_pp", type=float, default=10.0)
    p.add_argument("--chunksize", type=int, default=250000)
    return p.parse_args()


def _sem(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def _read_header(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader)


def _extract_episode_from_path(path: str) -> Optional[int]:
    m = re.search(r"_ep([0-9]+)", str(path))
    if not m:
        return None
    return int(m.group(1))


def _fhat_side_label(f_hat: float, split: float) -> str:
    return f"fhat<{split:g}" if float(f_hat) < float(split) else f"fhat>={split:g}"


def _ewma_bin_label(value: float, low: float, high: float) -> str:
    x = float(value)
    if x < float(low):
        return "ewma_low"
    if x < float(high):
        return "ewma_mid"
    return "ewma_high"


def _history_context_label(own_last_action: float, ewma_value: float, low: float, high: float) -> str:
    own_label = "prev_defect" if int(round(float(own_last_action))) == 0 else "prev_coop"
    return f"{own_label}__{_ewma_bin_label(ewma_value, low=low, high=high)}"


def _ensure_out_dir(path: str) -> Path:
    out_dir = Path(path).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _load_filtered_trace(args: argparse.Namespace) -> Tuple[pd.DataFrame, str]:
    header = _read_header(args.trace_csv)
    missing = [col for col in REQUIRED_TRACE_COLS if col not in header]
    if missing:
        raise ValueError(f"trace_csv is missing required columns: {missing}")

    usecols = [col for col in REQUIRED_TRACE_COLS + OPTIONAL_TRACE_COLS if col in header]
    parts: List[pd.DataFrame] = []
    true_f_filter = {float(v) for v in args.true_f_values}
    trace_episode = _extract_episode_from_path(args.trace_csv)

    for chunk in pd.read_csv(args.trace_csv, usecols=usecols, chunksize=args.chunksize):
        mask = (
            (chunk["condition"] == str(args.condition))
            & (chunk["eval_policy"] == str(args.eval_policy))
            & (chunk["ablation"] == str(args.ablation))
            & (chunk["history_intervention"] == str(args.history_intervention))
            & (chunk["sender_remap"] == str(args.sender_remap))
            & (chunk["cross_play"] == str(args.cross_play))
        )
        if "suite_kind" in chunk.columns:
            mask &= chunk["suite_kind"] == str(args.suite_kind)
        cur = chunk.loc[mask].copy()
        if cur.empty:
            continue

        cur["true_f"] = pd.to_numeric(cur["true_f"], errors="coerce")
        if true_f_filter:
            cur = cur[cur["true_f"].isin(true_f_filter)].copy()
            if cur.empty:
                continue

        if "checkpoint_episode" in cur.columns:
            cur["checkpoint_episode"] = pd.to_numeric(cur["checkpoint_episode"], errors="coerce")
            cur = cur[cur["checkpoint_episode"] == int(args.checkpoint_episode)].copy()
            if cur.empty:
                continue

        parts.append(cur)

    if not parts:
        raise ValueError("no rows matched the requested sender-encoding trace slice")

    df = pd.concat(parts, ignore_index=True)

    if "checkpoint_episode" not in df.columns:
        if trace_episode is not None:
            df["checkpoint_episode"] = int(trace_episode)
        else:
            df["checkpoint_episode"] = np.nan
        if pd.notna(df["checkpoint_episode"]).all():
            df = df[df["checkpoint_episode"] == int(args.checkpoint_episode)].copy()
    if df.empty:
        raise ValueError("no rows remained after checkpoint_episode filtering")

    if "suite_kind" not in df.columns:
        df["suite_kind"] = ""

    for col in [
        "train_seed",
        "eval_seed",
        "true_f",
        "f_hat",
        "action",
        "obs_last_coop_fraction",
        "obs_own_last_action",
        "obs_ewma_coop",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "intended_action" in df.columns:
        df["intended_action"] = pd.to_numeric(df["intended_action"], errors="coerce")

    action_column = "action"
    if "intended_action" in df.columns and df["intended_action"].notna().any():
        action_column = "intended_action"

    df["own_sent_msg"] = pd.to_numeric(df["own_sent_msg"], errors="coerce")
    df = df[df["own_sent_msg"].notna()].copy()
    df["message_one"] = (df["own_sent_msg"].astype(int) == 1).astype(int)
    df["action_used"] = pd.to_numeric(df[action_column], errors="coerce")

    df = df.dropna(
        subset=[
            "train_seed",
            "eval_seed",
            "true_f",
            "f_hat",
            "action_used",
            "obs_last_coop_fraction",
            "obs_own_last_action",
            "obs_ewma_coop",
        ]
    ).copy()
    if df.empty:
        raise ValueError("no rows remained after numeric coercion for sender-encoding analysis")

    df["train_seed"] = df["train_seed"].astype(int)
    df["eval_seed"] = df["eval_seed"].astype(int)
    df["action_used"] = df["action_used"].astype(int)
    df["fhat_side"] = df["f_hat"].map(lambda x: _fhat_side_label(x, split=args.fhat_split))
    df["history_context"] = df.apply(
        lambda row: _history_context_label(
            own_last_action=float(row["obs_own_last_action"]),
            ewma_value=float(row["obs_ewma_coop"]),
            low=float(args.ewma_low),
            high=float(args.ewma_high),
        ),
        axis=1,
    )
    return df.reset_index(drop=True), action_column


def _aggregate_conditionals(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    out = (
        df.groupby(list(group_cols), as_index=False)
        .agg(n_obs=("message_one", "size"), p_message_1=("message_one", "mean"))
        .sort_values(list(group_cols))
        .reset_index(drop=True)
    )
    return out


def _weighted_mean(values: List[Tuple[float, int]]) -> Tuple[float, float, int]:
    valid = [(float(v), int(w)) for v, w in values if w > 0 and v == v]
    if not valid:
        return float("nan"), float("nan"), 0
    total_w = float(sum(w for _v, w in valid))
    mean_val = float(sum(v * w for v, w in valid) / total_w)
    mean_abs = float(sum(abs(v) * w for v, w in valid) / total_w)
    return mean_val, mean_abs, int(len(valid))


def _conditional_delta(
    cur: pd.DataFrame,
    value_col: str,
    pos_label,
    neg_label,
    min_cell_obs: int,
    action_value: Optional[int] = None,
    fhat_side: Optional[str] = None,
) -> Tuple[float, int]:
    base = cur
    if action_value is not None:
        base = base[base["action_used"] == int(action_value)]
    if fhat_side is not None:
        base = base[base["fhat_side"] == str(fhat_side)]
    pos = base[base[value_col] == pos_label]
    neg = base[base[value_col] == neg_label]
    if len(pos) != 1 or len(neg) != 1:
        return float("nan"), 0
    if int(pos["n_obs"].iloc[0]) < int(min_cell_obs) or int(neg["n_obs"].iloc[0]) < int(min_cell_obs):
        return float("nan"), 0
    delta = float(pos["p_message_1"].iloc[0] - neg["p_message_1"].iloc[0])
    weight = int(pos["n_obs"].iloc[0] + neg["n_obs"].iloc[0])
    return delta, weight


def _summarize_sender_seed_effects(
    joint_df: pd.DataFrame,
    history_df: pd.DataFrame,
    min_cell_obs: int,
    dominance_threshold_pp: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    joint_group_cols = ["train_seed", "agent_id"]
    joint_lookup = {
        key: cur.copy()
        for key, cur in joint_df.groupby(joint_group_cols, sort=True)
    }
    history_lookup = {
        key: cur.copy()
        for key, cur in history_df.groupby(joint_group_cols, sort=True)
    }
    fhat_sides = sorted(joint_df["fhat_side"].dropna().unique().tolist())
    if len(fhat_sides) != 2:
        raise ValueError(
            "sender-encoding analysis requires two coarse f_hat sides after filtering; "
            f"got {fhat_sides}"
        )
    low_side, high_side = fhat_sides
    rows: List[Dict[str, object]] = []
    for key in sorted(joint_lookup):
        train_seed, sender_id = key
        cur_joint = joint_lookup[key]
        cur_hist = history_lookup.get(key, pd.DataFrame())

        uncond_fhat_effects: List[Tuple[float, int]] = []
        uncond_action_effects: List[Tuple[float, int]] = []
        row: Dict[str, object] = {
            "train_seed": int(train_seed),
            "sender_id": str(sender_id),
        }

        for action_value in (0, 1):
            delta, weight = _conditional_delta(
                cur_joint,
                value_col="fhat_side",
                pos_label=high_side,
                neg_label=low_side,
                min_cell_obs=min_cell_obs,
                action_value=action_value,
            )
            row[f"delta_fhat_high_minus_low_action{action_value}_pp"] = (
                float(delta * 100.0) if delta == delta else float("nan")
            )
            row[f"delta_fhat_high_minus_low_action{action_value}_n"] = int(weight)
            if weight > 0 and delta == delta:
                uncond_fhat_effects.append((delta * 100.0, weight))

        for side in (low_side, high_side):
            delta, weight = _conditional_delta(
                cur_joint,
                value_col="action_used",
                pos_label=1,
                neg_label=0,
                min_cell_obs=min_cell_obs,
                fhat_side=side,
            )
            side_key = "low" if side == low_side else "high"
            row[f"delta_action1_minus0_fhat_{side_key}_pp"] = (
                float(delta * 100.0) if delta == delta else float("nan")
            )
            row[f"delta_action1_minus0_fhat_{side_key}_n"] = int(weight)
            if weight > 0 and delta == delta:
                uncond_action_effects.append((delta * 100.0, weight))

        mean_delta, mean_abs, n_terms = _weighted_mean(uncond_fhat_effects)
        row["fhat_effect_mean_pp"] = mean_delta
        row["fhat_effect_abs_mean_pp"] = mean_abs
        row["fhat_effect_terms"] = int(n_terms)

        mean_delta, mean_abs, n_terms = _weighted_mean(uncond_action_effects)
        row["action_effect_mean_pp"] = mean_delta
        row["action_effect_abs_mean_pp"] = mean_abs
        row["action_effect_terms"] = int(n_terms)

        hist_fhat_effects: List[Tuple[float, int]] = []
        hist_action_effects: List[Tuple[float, int]] = []
        if not cur_hist.empty:
            for action_value in (0, 1):
                for _ctx, ctx_rows in cur_hist[cur_hist["action_used"] == int(action_value)].groupby(
                    "history_context", sort=True
                ):
                    delta, weight = _conditional_delta(
                        ctx_rows,
                        value_col="fhat_side",
                        pos_label=high_side,
                        neg_label=low_side,
                        min_cell_obs=min_cell_obs,
                    )
                    if weight > 0 and delta == delta:
                        hist_fhat_effects.append((delta * 100.0, weight))
            for side in (low_side, high_side):
                for _ctx, ctx_rows in cur_hist[cur_hist["fhat_side"] == str(side)].groupby(
                    "history_context", sort=True
                ):
                    delta, weight = _conditional_delta(
                        ctx_rows,
                        value_col="action_used",
                        pos_label=1,
                        neg_label=0,
                        min_cell_obs=min_cell_obs,
                    )
                    if weight > 0 and delta == delta:
                        hist_action_effects.append((delta * 100.0, weight))

        mean_delta, mean_abs, n_terms = _weighted_mean(hist_fhat_effects)
        row["history_adjusted_fhat_effect_mean_pp"] = mean_delta
        row["history_adjusted_fhat_effect_abs_mean_pp"] = mean_abs
        row["history_adjusted_fhat_effect_terms"] = int(n_terms)

        mean_delta, mean_abs, n_terms = _weighted_mean(hist_action_effects)
        row["history_adjusted_action_effect_mean_pp"] = mean_delta
        row["history_adjusted_action_effect_abs_mean_pp"] = mean_abs
        row["history_adjusted_action_effect_terms"] = int(n_terms)

        regime_abs = (
            float(row["history_adjusted_fhat_effect_abs_mean_pp"])
            if row["history_adjusted_fhat_effect_abs_mean_pp"]
            == row["history_adjusted_fhat_effect_abs_mean_pp"]
            else float(row["fhat_effect_abs_mean_pp"])
        )
        action_abs = (
            float(row["history_adjusted_action_effect_abs_mean_pp"])
            if row["history_adjusted_action_effect_abs_mean_pp"]
            == row["history_adjusted_action_effect_abs_mean_pp"]
            else float(row["action_effect_abs_mean_pp"])
        )
        row["dominance_regime_abs_pp"] = regime_abs
        row["dominance_action_abs_pp"] = action_abs
        row["dominance_basis"] = (
            "history_adjusted"
            if row["history_adjusted_fhat_effect_abs_mean_pp"]
            == row["history_adjusted_fhat_effect_abs_mean_pp"]
            or row["history_adjusted_action_effect_abs_mean_pp"]
            == row["history_adjusted_action_effect_abs_mean_pp"]
            else "unconditional"
        )
        if regime_abs >= float(dominance_threshold_pp) and action_abs >= float(dominance_threshold_pp):
            label = "mixed"
        elif regime_abs >= float(dominance_threshold_pp):
            label = "regime_leaning"
        elif action_abs >= float(dominance_threshold_pp):
            label = "action_leaning"
        else:
            label = "weak_or_unclear"
        row["dominance_label"] = label
        row["dominance_regime_minus_action_pp"] = float(regime_abs - action_abs)
        rows.append(row)

    sender_effect_df = pd.DataFrame(rows).sort_values(["train_seed", "sender_id"]).reset_index(drop=True)
    dominance_summary = (
        sender_effect_df.groupby("dominance_label", as_index=False)
        .agg(
            n_sender_seed_pairs=("sender_id", "size"),
            mean_regime_abs_pp=("dominance_regime_abs_pp", "mean"),
            mean_action_abs_pp=("dominance_action_abs_pp", "mean"),
            median_regime_abs_pp=("dominance_regime_abs_pp", "median"),
            median_action_abs_pp=("dominance_action_abs_pp", "median"),
        )
        .sort_values("dominance_label")
        .reset_index(drop=True)
    )
    return sender_effect_df, dominance_summary


def _build_model(model_name: str) -> Tuple[Pipeline, List[str], List[str]]:
    history_numeric = ["obs_last_coop_fraction", "obs_own_last_action", "obs_ewma_coop"]
    categorical = ["agent_id"]
    if model_name == "history_only":
        numeric = list(history_numeric)
    elif model_name == "history_fhat":
        numeric = [*history_numeric, "f_hat"]
    elif model_name == "history_action":
        numeric = [*history_numeric, "action_used"]
    elif model_name == "history_fhat_action":
        numeric = [*history_numeric, "f_hat", "action_used"]
    else:
        raise ValueError(f"unknown model_name={model_name!r}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    clf = LogisticRegression(max_iter=2000, solver="liblinear")
    return Pipeline([("prep", preprocessor), ("clf", clf)]), numeric, categorical


def _evaluate_models(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df["train_seed"].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame()
    if df["message_one"].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame()

    model_names = [
        "history_only",
        "history_fhat",
        "history_action",
        "history_fhat_action",
    ]
    groups = df["train_seed"].to_numpy()
    y = df["message_one"].to_numpy(dtype=int)
    splitter = GroupKFold(n_splits=min(5, int(pd.unique(groups).size)))

    fold_rows: List[Dict[str, object]] = []
    for model_name in model_names:
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(df, y, groups)):
            y_train = y[train_idx]
            y_test = y[test_idx]
            if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
                continue
            model, numeric_cols, categorical_cols = _build_model(model_name)
            x_train = df.iloc[train_idx][numeric_cols + categorical_cols]
            x_test = df.iloc[test_idx][numeric_cols + categorical_cols]
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="divide by zero encountered in matmul",
                    category=RuntimeWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="overflow encountered in matmul",
                    category=RuntimeWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="invalid value encountered in matmul",
                    category=RuntimeWarning,
                )
                model.fit(x_train, y_train)
                prob = model.predict_proba(x_test)[:, 1]
            pred = (prob >= 0.5).astype(int)
            fold_rows.append(
                {
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

    if not fold_rows:
        return pd.DataFrame(), pd.DataFrame()

    fold_df = pd.DataFrame(fold_rows).sort_values(["model_name", "fold"]).reset_index(drop=True)
    summary_df = (
        fold_df.groupby("model_name", as_index=False)
        .agg(
            n_folds=("fold", "size"),
            mean_log_loss=("log_loss", "mean"),
            sem_log_loss=("log_loss", _sem),
            mean_accuracy=("accuracy", "mean"),
            sem_accuracy=("accuracy", _sem),
            mean_brier=("brier", "mean"),
            sem_brier=("brier", _sem),
        )
        .sort_values(["mean_log_loss", "model_name"])
        .reset_index(drop=True)
    )

    if "history_only" in set(summary_df["model_name"]):
        base_log_loss = float(
            summary_df.loc[summary_df["model_name"] == "history_only", "mean_log_loss"].iloc[0]
        )
        base_accuracy = float(
            summary_df.loc[summary_df["model_name"] == "history_only", "mean_accuracy"].iloc[0]
        )
        summary_df["delta_log_loss_vs_history_only"] = base_log_loss - summary_df["mean_log_loss"]
        summary_df["delta_accuracy_vs_history_only"] = summary_df["mean_accuracy"] - base_accuracy
    else:
        summary_df["delta_log_loss_vs_history_only"] = np.nan
        summary_df["delta_accuracy_vs_history_only"] = np.nan
    return summary_df, fold_df


def _top_examples(
    sender_effect_df: pd.DataFrame,
    label: str,
    value_col: str,
    n_rows: int = 5,
) -> pd.DataFrame:
    cur = sender_effect_df[sender_effect_df["dominance_label"] == label].copy()
    if cur.empty:
        return cur
    return (
        cur.sort_values([value_col, "train_seed", "sender_id"], ascending=[False, True, True])
        .head(n_rows)
        .reset_index(drop=True)
    )


def _write_markdown(
    out_path: Path,
    args: argparse.Namespace,
    filtered_df: pd.DataFrame,
    action_column_used: str,
    pooled_joint_df: pd.DataFrame,
    dominance_summary_df: pd.DataFrame,
    sender_effect_df: pd.DataFrame,
    model_summary_df: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# Phase-3 Sender Encoding Decomposition")
    lines.append("")
    lines.append(f"- trace_csv: `{os.path.abspath(args.trace_csv)}`")
    lines.append(f"- action_column_used: `{action_column_used}`")
    lines.append(f"- condition: `{args.condition}`")
    lines.append(f"- checkpoint_episode: `{args.checkpoint_episode}`")
    lines.append(f"- ablation: `{args.ablation}`")
    lines.append(f"- history_intervention: `{args.history_intervention}`")
    lines.append(f"- sender_remap: `{args.sender_remap}`")
    lines.append(f"- cross_play: `{args.cross_play}`")
    lines.append(f"- eval_policy: `{args.eval_policy}`")
    lines.append(f"- suite_kind: `{args.suite_kind}`")
    lines.append(f"- min_cell_obs: `{args.min_cell_obs}`")
    lines.append(f"- rows_after_filter: `{len(filtered_df)}`")
    lines.append(f"- sender_seed_pairs: `{sender_effect_df[['train_seed', 'sender_id']].drop_duplicates().shape[0]}`")
    lines.append("")

    lines.append("## P(message=1 | fhat-side, action)")
    lines.append("")
    lines.append("| sender_id | fhat_side | action_used | total_obs | mean_p_message_1 | sem_p_message_1 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for _, row in pooled_joint_df.iterrows():
        lines.append(
            f"| {row['agent_id']} | {row['fhat_side']} | {int(row['action_used'])} | "
            f"{int(row['total_obs'])} | {row['mean_p_message_1']:.4f} | {row['sem_p_message_1']:.4f} |"
        )
    lines.append("")

    lines.append("## Sender-Seed Dominance Labels")
    lines.append("")
    lines.append(
        "| dominance_label | n_sender_seed_pairs | median_regime_abs_pp | median_action_abs_pp | "
        "mean_regime_abs_pp | mean_action_abs_pp |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for _, row in dominance_summary_df.iterrows():
        lines.append(
            f"| {row['dominance_label']} | {int(row['n_sender_seed_pairs'])} | "
            f"{row['median_regime_abs_pp']:.2f} | {row['median_action_abs_pp']:.2f} | "
            f"{row['mean_regime_abs_pp']:.2f} | {row['mean_action_abs_pp']:.2f} |"
        )
    lines.append("")

    if not model_summary_df.empty:
        lines.append("## Predicting Messages From History, f_hat, And Action")
        lines.append("")
        lines.append(
            "| model_name | n_folds | mean_log_loss | improvement_vs_history_only | "
            "mean_accuracy | delta_acc_vs_history_only |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for _, row in model_summary_df.iterrows():
            lines.append(
                f"| `{row['model_name']}` | {int(row['n_folds'])} | {row['mean_log_loss']:.4f} | "
                f"{row['delta_log_loss_vs_history_only']:.4f} | {row['mean_accuracy']:.4f} | "
                f"{row['delta_accuracy_vs_history_only']:.4f} |"
            )
        lines.append("")
    else:
        lines.append("## Predicting Messages From History, f_hat, And Action")
        lines.append("")
        lines.append("Not enough cross-seed variation was available to run grouped model comparisons.")
        lines.append("")

    top_regime = _top_examples(
        sender_effect_df,
        label="regime_leaning",
        value_col="dominance_regime_abs_pp",
    )
    top_action = _top_examples(
        sender_effect_df,
        label="action_leaning",
        value_col="dominance_action_abs_pp",
    )
    top_mixed = _top_examples(
        sender_effect_df,
        label="mixed",
        value_col="dominance_regime_abs_pp",
    )

    def _emit_examples(title: str, df: pd.DataFrame) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if df.empty:
            lines.append("No sender-seed pairs fell into this category under the current threshold.")
            lines.append("")
            return
        lines.append(
            "| train_seed | sender_id | dominance_basis | regime_abs_pp | action_abs_pp | "
            "history_adj_regime_abs_pp | history_adj_action_abs_pp |"
        )
        lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: |")
        for _, row in df.iterrows():
            lines.append(
                f"| {int(row['train_seed'])} | {row['sender_id']} | {row['dominance_basis']} | "
                f"{row['dominance_regime_abs_pp']:.2f} | {row['dominance_action_abs_pp']:.2f} | "
                f"{row['history_adjusted_fhat_effect_abs_mean_pp']:.2f} | "
                f"{row['history_adjusted_action_effect_abs_mean_pp']:.2f} |"
            )
        lines.append("")

    _emit_examples("Top Regime-Leaning Examples", top_regime)
    _emit_examples("Top Action-Leaning Examples", top_action)
    _emit_examples("Top Mixed Examples", top_mixed)

    lines.append("## Interpretation")
    lines.append("")
    mixed_count = int(
        dominance_summary_df.loc[
            dominance_summary_df["dominance_label"] == "mixed",
            "n_sender_seed_pairs",
        ].sum()
    )
    regime_count = int(
        dominance_summary_df.loc[
            dominance_summary_df["dominance_label"] == "regime_leaning",
            "n_sender_seed_pairs",
        ].sum()
    )
    action_count = int(
        dominance_summary_df.loc[
            dominance_summary_df["dominance_label"] == "action_leaning",
            "n_sender_seed_pairs",
        ].sum()
    )
    weak_count = int(
        dominance_summary_df.loc[
            dominance_summary_df["dominance_label"] == "weak_or_unclear",
            "n_sender_seed_pairs",
        ].sum()
    )
    lines.append(
        f"- Across sender-seed pairs, the coarse conditional tables split into `mixed={mixed_count}`, "
        f"`regime_leaning={regime_count}`, `action_leaning={action_count}`, and "
        f"`weak_or_unclear={weak_count}` under a `{args.dominance_threshold_pp:.1f}pp` dominance threshold."
    )
    if not model_summary_df.empty:
        best = model_summary_df.sort_values(["mean_log_loss", "model_name"]).iloc[0]
        lines.append(
            f"- In grouped seed-held-out prediction, the best message model is `{best['model_name']}` "
            f"with mean log loss `{best['mean_log_loss']:.4f}`. Relative to `history_only`, "
            f"adding the strongest available combination improves log loss by "
            f"`{best['delta_log_loss_vs_history_only']:.4f}`."
        )
    lines.append(
        "- The sender-side tables should be read as encoding diagnostics: they show whether message polarity "
        "still varies with `f_hat` after fixing action, and whether it still varies with action after fixing "
        "`f_hat`, with a separate history-adjusted summary based on coarse recent-context strata."
    )
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = _ensure_out_dir(args.out_dir)

    filtered_df, action_column_used = _load_filtered_trace(args)
    joint_df = _aggregate_conditionals(
        filtered_df,
        group_cols=["train_seed", "agent_id", "fhat_side", "action_used"],
    )
    history_df = _aggregate_conditionals(
        filtered_df,
        group_cols=["train_seed", "agent_id", "history_context", "fhat_side", "action_used"],
    )
    pooled_joint_df = (
        joint_df.groupby(["agent_id", "fhat_side", "action_used"], as_index=False)
        .agg(
            n_sender_seed=("train_seed", "nunique"),
            total_obs=("n_obs", "sum"),
            mean_p_message_1=("p_message_1", "mean"),
            sem_p_message_1=("p_message_1", _sem),
        )
        .sort_values(["agent_id", "fhat_side", "action_used"])
        .reset_index(drop=True)
    )
    sender_effect_df, dominance_summary_df = _summarize_sender_seed_effects(
        joint_df=joint_df,
        history_df=history_df,
        min_cell_obs=int(args.min_cell_obs),
        dominance_threshold_pp=float(args.dominance_threshold_pp),
    )
    model_summary_df, model_folds_df = _evaluate_models(filtered_df)

    filtered_df.to_csv(out_dir / "filtered_trace_slice.csv", index=False)
    joint_df.to_csv(out_dir / "sender_joint_conditionals.csv", index=False)
    history_df.to_csv(out_dir / "sender_history_conditionals.csv", index=False)
    pooled_joint_df.to_csv(out_dir / "sender_joint_pooled_summary.csv", index=False)
    sender_effect_df.to_csv(out_dir / "sender_effect_summary.csv", index=False)
    dominance_summary_df.to_csv(out_dir / "sender_dominance_summary.csv", index=False)
    if not model_summary_df.empty:
        model_summary_df.to_csv(out_dir / "sender_encoding_model_summary.csv", index=False)
        model_folds_df.to_csv(out_dir / "sender_encoding_model_folds.csv", index=False)

    _write_markdown(
        out_path=out_dir / "sender_encoding_summary.md",
        args=args,
        filtered_df=filtered_df,
        action_column_used=action_column_used,
        pooled_joint_df=pooled_joint_df,
        dominance_summary_df=dominance_summary_df,
        sender_effect_df=sender_effect_df,
        model_summary_df=model_summary_df,
    )
    print(
        "[sender-encoding] "
        f"out_dir={out_dir} action_column_used={action_column_used} "
        f"rows={len(filtered_df)} sender_seed_pairs={sender_effect_df.shape[0]}"
    )


if __name__ == "__main__":
    main()
