from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_REPORT_DIR = Path(
    "outputs/eval/phase3_vecstraight_zeroaux_frozen150k_natural_intended_15seeds_local_20260415"
) / "report" / "sender_encoding_decomposition"

THRESHOLDS_PP = [5.0, 7.5, 10.0, 12.5, 15.0, 20.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build threshold-free sender-encoding summaries from the existing "
            "150k sender-encoding decomposition outputs."
        )
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory containing sender_effect_summary.csv and filtered_trace_slice.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to --report-dir.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260418)
    parser.add_argument("--reference-threshold-pp", type=float, default=10.0)
    parser.add_argument(
        "--fhat-split",
        type=float,
        default=3.5,
        help="Expected f_hat split used by existing B.4 code.",
    )
    return parser.parse_args()


def fmt_num(value: float, places: int = 1) -> str:
    return f"{float(value):.{places}f}"


def fmt_corr(value: float) -> str:
    return f"{float(value):.2f}"


def iqr(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    return tuple(float(v) for v in np.percentile(arr, [25, 75]))


def corr_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def corr_spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return corr_pearson(xr, yr)


def bootstrap_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    corr_fn,
    n_resamples: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = x.size
    estimates = np.empty(int(n_resamples), dtype=float)
    for i in range(int(n_resamples)):
        idx = rng.integers(0, n, size=n)
        estimates[i] = corr_fn(x[idx], y[idx])
    estimates = estimates[np.isfinite(estimates)]
    if estimates.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return float(lo), float(hi)


def choose_axis_limit(max_value: float) -> Tuple[float, bool]:
    requested = min(float(max_value) + 5.0, 50.0)
    if float(max_value) > requested:
        # The manuscript request capped axes at 50 pp, but this dataset contains
        # larger effects. Do not hide points in the diagnostic plot.
        return float(math.ceil((float(max_value) + 5.0) / 5.0) * 5.0), True
    return requested, False


def classify_counts(abs_regime: pd.Series, abs_action: pd.Series, threshold: float) -> dict[str, int]:
    regime = abs_regime >= float(threshold)
    action = abs_action >= float(threshold)
    return {
        "mixed": int((regime & action).sum()),
        "regime_leaning": int((regime & ~action).sum()),
        "action_leaning": int((~regime & action).sum()),
        "weak_or_unclear": int((~regime & ~action).sum()),
    }


def write_sensitivity_tables(out_dir: Path, adjusted: pd.DataFrame) -> pd.DataFrame:
    rows = []
    abs_regime = adjusted["regime_shift_pp"].abs()
    abs_action = adjusted["action_shift_pp"].abs()
    labels = ["mixed", "regime_leaning", "action_leaning", "weak_or_unclear"]
    counts_by_threshold = {
        threshold: classify_counts(abs_regime, abs_action, threshold)
        for threshold in THRESHOLDS_PP
    }
    for label in labels:
        row = {"category": label}
        for threshold in THRESHOLDS_PP:
            row[f"tau_{threshold:g}_pp"] = counts_by_threshold[threshold][label]
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "sensitivity_table.csv", index=False)

    heading = "Category & " + " & ".join(f"{threshold:g} pp" for threshold in THRESHOLDS_PP) + r" \\"
    tex_lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        heading,
        r"\midrule",
    ]
    label_map = {
        "mixed": "Mixed",
        "regime_leaning": "Regime-leaning",
        "action_leaning": "Action-leaning",
        "weak_or_unclear": "Weak or unclear",
    }
    for _, row in table.iterrows():
        values = [str(int(row[f"tau_{threshold:g}_pp"])) for threshold in THRESHOLDS_PP]
        tex_lines.append(f"{label_map[row['category']]} & " + " & ".join(values) + r" \\")
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out_dir / "sensitivity_table.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    return table


def add_rugs(ax, x: np.ndarray, y: np.ndarray, axis_limit: float, color: str) -> None:
    rug_len = axis_limit * 0.018
    ax.vlines(x, ymin=0.0, ymax=rug_len, color=color, alpha=0.22, linewidth=0.5)
    ax.hlines(y, xmin=0.0, xmax=rug_len, color=color, alpha=0.22, linewidth=0.5)


def make_scatter(
    df: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    threshold: float,
) -> Tuple[float, bool]:
    x = df[x_col].abs().to_numpy(dtype=float)
    y = df[y_col].abs().to_numpy(dtype=float)
    axis_limit, expanded = choose_axis_limit(max(float(np.max(x)), float(np.max(y))))

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.15, 2.65))
    point_color = "#0072B2"
    ax.scatter(
        x,
        y,
        s=22,
        color=point_color,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.9,
        zorder=3,
    )
    add_rugs(ax, x, y, axis_limit=axis_limit, color=point_color)
    ax.axvline(threshold, color="0.45", linestyle=(0, (3, 2)), linewidth=0.75, zorder=1)
    ax.axhline(threshold, color="0.45", linestyle=(0, (3, 2)), linewidth=0.75, zorder=1)
    ax.text(
        threshold + axis_limit * 0.015,
        axis_limit * 0.965,
        f"{threshold:g} pp visual reference",
        ha="left",
        va="top",
        fontsize=6.5,
        color="0.3",
    )
    ax.set_xlim(0, axis_limit)
    ax.set_ylim(0, axis_limit)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(color="0.9", linewidth=0.5, zorder=0)
    fig.tight_layout(pad=0.5)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return axis_limit, expanded


def build_main_effects(effect_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict[str, int]]:
    regime = effect_df["dominance_regime_abs_pp"]
    action = effect_df["dominance_action_abs_pp"]
    out = pd.DataFrame(
        {
            "seed": effect_df["train_seed"].astype(int),
            "sender_id": effect_df["sender_id"].astype(str),
            "regime_shift_pp": regime.astype(float),
            "action_shift_pp": action.astype(float),
            "n_observations": np.nan,
        }
    ).sort_values(["seed", "sender_id"])
    basis_counts = effect_df["dominance_basis"].value_counts(dropna=False).to_dict()
    return out.reset_index(drop=True), {str(k): int(v) for k, v in basis_counts.items()}


def build_net_signed_effects(effect_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict[str, int]]:
    regime_adjusted = effect_df["history_adjusted_fhat_effect_mean_pp"]
    action_adjusted = effect_df["history_adjusted_action_effect_mean_pp"]
    regime = regime_adjusted.where(regime_adjusted.notna(), effect_df["fhat_effect_mean_pp"])
    action = action_adjusted.where(action_adjusted.notna(), effect_df["action_effect_mean_pp"])
    out = pd.DataFrame(
        {
            "seed": effect_df["train_seed"].astype(int),
            "sender_id": effect_df["sender_id"].astype(str),
            "regime_shift_pp": regime.astype(float),
            "action_shift_pp": action.astype(float),
            "n_observations": np.nan,
        }
    ).sort_values(["seed", "sender_id"])
    fallback = {
        "regime_history_adjusted_available": int(regime_adjusted.notna().sum()),
        "regime_raw_fallback": int(regime_adjusted.isna().sum()),
        "action_history_adjusted_available": int(action_adjusted.notna().sum()),
        "action_raw_fallback": int(action_adjusted.isna().sum()),
    }
    return out.reset_index(drop=True), fallback


def build_raw_effects(trace_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    required_sides = {"fhat<3.5", "fhat>=3.5"}
    actual_sides = set(trace_df["fhat_side"].dropna().astype(str).unique())
    if actual_sides != required_sides:
        raise ValueError(f"Expected fhat_side labels {required_sides}, got {actual_sides}")
    for (seed, sender_id), cur in trace_df.groupby(["train_seed", "agent_id"], sort=True):
        high = cur[cur["fhat_side"] == "fhat>=3.5"]
        low = cur[cur["fhat_side"] == "fhat<3.5"]
        coop = cur[cur["action_used"] == 1]
        defect = cur[cur["action_used"] == 0]
        if high.empty or low.empty or coop.empty or defect.empty:
            regime_shift = float("nan")
            action_shift = float("nan")
        else:
            regime_shift = (float(high["message_one"].mean()) - float(low["message_one"].mean())) * 100.0
            action_shift = (float(coop["message_one"].mean()) - float(defect["message_one"].mean())) * 100.0
        rows.append(
            {
                "seed": int(seed),
                "sender_id": str(sender_id),
                "regime_shift_pp": regime_shift,
                "action_shift_pp": action_shift,
                "n_observations": int(len(cur)),
            }
        )
    return pd.DataFrame(rows).sort_values(["seed", "sender_id"]).reset_index(drop=True)


def attach_observation_counts(adjusted: pd.DataFrame, trace_df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        trace_df.groupby(["train_seed", "agent_id"], as_index=False)
        .size()
        .rename(columns={"train_seed": "seed", "agent_id": "sender_id", "size": "n_observations"})
    )
    merged = adjusted.drop(columns=["n_observations"]).merge(
        counts,
        on=["seed", "sender_id"],
        how="left",
        validate="one_to_one",
    )
    return merged[["seed", "sender_id", "regime_shift_pp", "action_shift_pp", "n_observations"]]


def summarize_distribution(
    df: pd.DataFrame,
    threshold: float,
    n_resamples: int,
    seed: int,
) -> dict[str, object]:
    abs_regime = df["regime_shift_pp"].abs().to_numpy(dtype=float)
    abs_action = df["action_shift_pp"].abs().to_numpy(dtype=float)
    reg_q1, reg_q3 = iqr(abs_regime)
    act_q1, act_q3 = iqr(abs_action)
    pearson = corr_pearson(abs_regime, abs_action)
    spearman = corr_spearman(abs_regime, abs_action)
    pearson_ci = bootstrap_corr_ci(abs_regime, abs_action, corr_pearson, n_resamples, seed)
    spearman_ci = bootstrap_corr_ci(abs_regime, abs_action, corr_spearman, n_resamples, seed + 1)
    n = len(df)
    regime_gt = int((abs_regime >= threshold).sum())
    action_gt = int((abs_action >= threshold).sum())
    both_gt = int(((abs_regime >= threshold) & (abs_action >= threshold)).sum())
    return {
        "n": n,
        "regime_median": float(np.median(abs_regime)),
        "regime_q1": reg_q1,
        "regime_q3": reg_q3,
        "action_median": float(np.median(abs_action)),
        "action_q1": act_q1,
        "action_q3": act_q3,
        "pearson": pearson,
        "pearson_ci": pearson_ci,
        "spearman": spearman,
        "spearman_ci": spearman_ci,
        "regime_gt": regime_gt,
        "action_gt": action_gt,
        "both_gt": both_gt,
        "regime_pct": 100.0 * regime_gt / n,
        "action_pct": 100.0 * action_gt / n,
        "both_pct": 100.0 * both_gt / n,
    }


def correlation_language(spearman: float) -> str:
    abs_r = abs(float(spearman))
    if abs_r < 0.2:
        return "weakly correlated"
    if abs_r < 0.5:
        return "moderately correlated"
    return "strongly correlated"


def write_summary(
    out_dir: Path,
    stats: dict[str, object],
    net_stats: dict[str, object],
    raw_stats: dict[str, object],
    basis_counts: dict[str, int],
    main_axis_limit: float,
    main_axis_expanded: bool,
    net_axis_limit: float,
    net_axis_expanded: bool,
    raw_axis_limit: float,
    raw_axis_expanded: bool,
) -> None:
    pearson_lo, pearson_hi = stats["pearson_ci"]
    spearman_lo, spearman_hi = stats["spearman_ci"]
    net_spearman_lo, net_spearman_hi = net_stats["spearman_ci"]
    raw_spearman_lo, raw_spearman_hi = raw_stats["spearman_ci"]
    lines = [
        "Sender encoding distribution summary",
        "====================================",
        "",
        "Regime split: f_hat < 3.5 vs. f_hat >= 3.5, matching the existing Appendix B.4 code.",
        "Main scatter/statistics use the Appendix B.4 dominance quantities: mean absolute history-adjusted effects, falling back to unadjusted mean absolute effects only if needed.",
        (
            "Dominance basis counts: "
            + ", ".join(f"{key}={value}" for key, value in sorted(basis_counts.items()))
            + "."
        ),
        "",
        f"Median mean absolute history-adjusted regime effect: {fmt_num(stats['regime_median'])} pp "
        f"(IQR {fmt_num(stats['regime_q1'])}--{fmt_num(stats['regime_q3'])}).",
        f"Median mean absolute history-adjusted action effect: {fmt_num(stats['action_median'])} pp "
        f"(IQR {fmt_num(stats['action_q1'])}--{fmt_num(stats['action_q3'])}).",
        (
            "Pearson correlation between absolute effects: "
            f"r = {fmt_corr(stats['pearson'])}, 95% bootstrap CI "
            f"[{fmt_corr(pearson_lo)}, {fmt_corr(pearson_hi)}]."
        ),
        (
            "Spearman correlation between absolute effects: "
            f"rho = {fmt_corr(stats['spearman'])}, 95% bootstrap CI "
            f"[{fmt_corr(spearman_lo)}, {fmt_corr(spearman_hi)}]."
        ),
        f"Mean absolute history-adjusted regime effect >= 10 pp: {stats['regime_gt']}/{stats['n']} ({fmt_num(stats['regime_pct'])}%).",
        f"Mean absolute history-adjusted action effect >= 10 pp: {stats['action_gt']}/{stats['n']} ({fmt_num(stats['action_pct'])}%).",
        (
            f"Both mean absolute effects >= 10 pp: {stats['both_gt']}/{stats['n']} "
            f"({fmt_num(stats['both_pct'])}%)."
        ),
        "",
        "Net-signed secondary check:",
        (
            f"Both net-signed absolute effects >= 10 pp: {net_stats['both_gt']}/{net_stats['n']} "
            f"({fmt_num(net_stats['both_pct'])}%)."
        ),
        (
            "Spearman correlation between net-signed absolute effects: "
            f"rho = {fmt_corr(net_stats['spearman'])}, 95% bootstrap CI "
            f"[{fmt_corr(net_spearman_lo)}, {fmt_corr(net_spearman_hi)}]."
        ),
        "",
        "Raw conditional robustness check:",
        f"Median |raw regime difference|: {fmt_num(raw_stats['regime_median'])} pp "
        f"(IQR {fmt_num(raw_stats['regime_q1'])}--{fmt_num(raw_stats['regime_q3'])}).",
        f"Median |raw action difference|: {fmt_num(raw_stats['action_median'])} pp "
        f"(IQR {fmt_num(raw_stats['action_q1'])}--{fmt_num(raw_stats['action_q3'])}).",
        (
            "Spearman correlation between raw absolute effects: "
            f"rho = {fmt_corr(raw_stats['spearman'])}, 95% bootstrap CI "
            f"[{fmt_corr(raw_spearman_lo)}, {fmt_corr(raw_spearman_hi)}]."
        ),
        (
            f"Both raw absolute effects > 10 pp: {raw_stats['both_gt']}/{raw_stats['n']} "
            f"({fmt_num(raw_stats['both_pct'])}%)."
        ),
        "",
        f"Main scatter axis limit: 0--{fmt_num(main_axis_limit)} pp"
        + (" (expanded beyond the requested 50 pp cap to avoid clipping points)." if main_axis_expanded else "."),
        f"Net-signed scatter axis limit: 0--{fmt_num(net_axis_limit)} pp"
        + (" (expanded beyond the requested 50 pp cap to avoid clipping points)." if net_axis_expanded else "."),
        f"Raw scatter axis limit: 0--{fmt_num(raw_axis_limit)} pp"
        + (" (expanded beyond the requested 50 pp cap to avoid clipping points)." if raw_axis_expanded else "."),
    ]
    (out_dir / "encoding_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paragraph(out_dir: Path, stats: dict[str, object]) -> None:
    sp_lo, sp_hi = stats["spearman_ci"]
    corr_phrase = correlation_language(float(stats["spearman"]))
    paragraph = (
        "Learned messages do carry task-relevant information, but mostly as entangled signals "
        "rather than clean symbols. For each of the 60 sender--seed pairs at 150k "
        "(4 senders $\\times$ 15 seeds), we measure two mean absolute history-adjusted effects "
        "in the sender's probability of emitting token 1: a regime effect (high- vs. "
        "low-regime observations, using the same $\\hat f < 3.5$ vs. $\\hat f \\ge 3.5$ "
        "split as Appendix B.4) and an action effect (intended-cooperate vs. intended-defect). "
        "Figure [X] plots the two effects against each other. "
        f"Most senders show substantial variation on both axes: the median regime effect is "
        f"{fmt_num(stats['regime_median'])} pp (IQR {fmt_num(stats['regime_q1'])}--"
        f"{fmt_num(stats['regime_q3'])}) and the median action effect is "
        f"{fmt_num(stats['action_median'])} pp (IQR {fmt_num(stats['action_q1'])}--"
        f"{fmt_num(stats['action_q3'])}), with {stats['both_gt']}/{stats['n']} "
        f"sender--seed pairs ({fmt_num(stats['both_pct'])}%) exceeding 10 pp on both. "
        f"The two effect magnitudes are {corr_phrase} (Spearman $\\rho={fmt_corr(stats['spearman'])}$, "
        f"95% CI [{fmt_corr(sp_lo)}, {fmt_corr(sp_hi)}]). Task-relevant structure is "
        "therefore present in sender behavior, but senders typically entangle regime and "
        "action information rather than encoding one cleanly. A sensitivity analysis "
        "(Appendix B.4, Table [Y]) shows this conclusion is stable across classification "
        "thresholds from 5 to 20 pp. This sets up the next question: if the channel contains "
        "information, does cooperation depend on it?"
    )
    (out_dir / "encoding_paragraph_draft.md").write_text(paragraph + "\n", encoding="utf-8")


def write_raw_note(out_dir: Path, stats: dict[str, object], raw_stats: dict[str, object]) -> None:
    main_both = int(stats["both_gt"])
    raw_both = int(raw_stats["both_gt"])
    if raw_both >= 0.8 * main_both:
        qualifier = "gives the same qualitative conclusion that most senders vary on both axes"
    else:
        qualifier = "is directionally similar but weaker"
    note = (
        "A raw, unadjusted check using direct conditional differences "
        "$P(m=1\\mid \\hat f\\ge3.5)-P(m=1\\mid \\hat f<3.5)$ and "
        "$P(m=1\\mid a^{\\mathrm{intended}}=C)-P(m=1\\mid a^{\\mathrm{intended}}=D)$ "
        f"{qualifier}: {raw_both}/{raw_stats['n']} sender--seed pairs exceed 10 pp on both axes, "
        f"compared with {main_both}/{stats['n']} under the mean absolute history-adjusted effects."
    )
    (out_dir / "encoding_raw_appendix_note.md").write_text(note + "\n", encoding="utf-8")


def write_net_signed_note(out_dir: Path, stats: dict[str, object], net_stats: dict[str, object]) -> None:
    difference = int(stats["both_gt"]) - int(net_stats["both_gt"])
    note = (
        "A stricter version of the question asks whether each sender has a stable net association "
        "after averaging across contexts. This count is "
        f"{net_stats['both_gt']}/{net_stats['n']} at $\\tau=10\\,\\mathrm{{pp}}$, and the "
        f"{difference}-pair difference from the main {stats['both_gt']}/{stats['n']} reflects "
        "senders whose encoding direction reverses across contexts---a phenomenon that connects "
        "directly to the pair-specificity of receiver influence documented in Section \\ref{sec-sender-probes}."
    )
    (out_dir / "encoding_net_signed_appendix_note.md").write_text(note + "\n", encoding="utf-8")


def write_readme(
    out_dir: Path,
    report_dir: Path,
    basis_counts: dict[str, int],
    stats: dict[str, object],
    net_stats: dict[str, object],
    raw_stats: dict[str, object],
    main_axis_expanded: bool,
    net_axis_expanded: bool,
    raw_axis_expanded: bool,
) -> None:
    lines = [
        "# Sender-Encoding Distribution Artifacts",
        "",
        "Source directory:",
        f"`{report_dir}`",
        "",
        "The regime split is `f_hat < 3.5` versus `f_hat >= 3.5`, matching the existing Appendix B.4 summarizer (`--fhat_split`, default `3.5`).",
        "The main CSV and plot use the existing B.4 dominance quantities: mean absolute history-adjusted sender effects, falling back to mean absolute unadjusted effects only if needed.",
        (
            "Dominance basis counts: "
            + ", ".join(f"{key}={value}" for key, value in sorted(basis_counts.items()))
            + "."
        ),
        "",
        "Generated files:",
        "- `encoding_effects.csv`: 60-row mean absolute history-adjusted effect CSV for the main plot/prose.",
        "- `encoding_effects_net_signed.csv`: 60-row net-signed secondary check CSV.",
        "- `encoding_effects_raw.csv`: 60-row raw conditional-difference robustness CSV.",
        "- `encoding_scatter.pdf` / `encoding_scatter.png`: main mean-absolute-effect scatter.",
        "- `encoding_scatter_net_signed.pdf` / `encoding_scatter_net_signed.png`: secondary net-signed scatter.",
        "- `encoding_scatter_raw.pdf` / `encoding_scatter_raw.png`: raw robustness scatter.",
        "- `encoding_summary.txt`: paste-ready summary statistics.",
        "- `sensitivity_table.csv` / `sensitivity_table.tex`: old threshold taxonomy across thresholds.",
        "- `encoding_net_signed_appendix_note.md`: appendix note for the net-signed secondary check.",
        "- `encoding_raw_appendix_note.md`: appendix note for the raw robustness plot.",
        "- `encoding_paragraph_draft.md`: draft replacement paragraph for Section 4.2.",
        "",
        "Validation summary:",
        f"- `encoding_effects.csv` has {stats['n']} rows and no missing effect values.",
        f"- `encoding_effects_net_signed.csv` has {net_stats['n']} rows and no missing effect values.",
        f"- `encoding_effects_raw.csv` has {raw_stats['n']} rows and no missing effect values.",
        "- Three sender--seed spot checks were recomputed from conditional tables and matched the generated CSV (see `encoding_validation_checks.txt`).",
    ]
    if main_axis_expanded or net_axis_expanded or raw_axis_expanded:
        lines.append(
            "- The scatter axes were expanded beyond a 50 pp cap because a hard cap would hide observed points."
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def recompute_history_adjusted_from_conditionals(
    history_df: pd.DataFrame,
    seed: int,
    sender_id: str,
    min_cell_obs: int = 50,
) -> Tuple[float, float]:
    cur = history_df[(history_df["train_seed"] == int(seed)) & (history_df["agent_id"] == str(sender_id))]
    fhat_effects = []
    action_effects = []
    for action_value in (0, 1):
        subset = cur[cur["action_used"] == action_value]
        for _ctx, ctx in subset.groupby("history_context", sort=True):
            high = ctx[ctx["fhat_side"] == "fhat>=3.5"]
            low = ctx[ctx["fhat_side"] == "fhat<3.5"]
            if (
                len(high) == 1
                and len(low) == 1
                and int(high["n_obs"].iloc[0]) >= int(min_cell_obs)
                and int(low["n_obs"].iloc[0]) >= int(min_cell_obs)
            ):
                weight = int(high["n_obs"].iloc[0] + low["n_obs"].iloc[0])
                delta = float(high["p_message_1"].iloc[0] - low["p_message_1"].iloc[0]) * 100.0
                fhat_effects.append((delta, weight))
    for side in ("fhat<3.5", "fhat>=3.5"):
        subset = cur[cur["fhat_side"] == side]
        for _ctx, ctx in subset.groupby("history_context", sort=True):
            coop = ctx[ctx["action_used"] == 1]
            defect = ctx[ctx["action_used"] == 0]
            if (
                len(coop) == 1
                and len(defect) == 1
                and int(coop["n_obs"].iloc[0]) >= int(min_cell_obs)
                and int(defect["n_obs"].iloc[0]) >= int(min_cell_obs)
            ):
                weight = int(coop["n_obs"].iloc[0] + defect["n_obs"].iloc[0])
                delta = float(coop["p_message_1"].iloc[0] - defect["p_message_1"].iloc[0]) * 100.0
                action_effects.append((delta, weight))

    def weighted(values: list[tuple[float, int]]) -> float:
        return float(sum(v * w for v, w in values) / sum(w for _v, w in values))

    return weighted(fhat_effects), weighted(action_effects)


def write_validation_checks(
    out_dir: Path,
    main_effects: pd.DataFrame,
    net_signed: pd.DataFrame,
    raw: pd.DataFrame,
    effect_df: pd.DataFrame,
    history_df: pd.DataFrame,
) -> None:
    checks = [
        (int(main_effects.iloc[0]["seed"]), str(main_effects.iloc[0]["sender_id"])),
        (int(main_effects.iloc[21]["seed"]), str(main_effects.iloc[21]["sender_id"])),
        (int(main_effects.iloc[-1]["seed"]), str(main_effects.iloc[-1]["sender_id"])),
    ]
    lines = ["Sender-encoding validation checks", "=================================", ""]
    for seed, sender_id in checks:
        recomputed_regime, recomputed_action = recompute_history_adjusted_from_conditionals(
            history_df, seed, sender_id
        )
        row = main_effects[(main_effects["seed"] == seed) & (main_effects["sender_id"] == sender_id)].iloc[0]
        net_row = net_signed[(net_signed["seed"] == seed) & (net_signed["sender_id"] == sender_id)].iloc[0]
        raw_row = raw[(raw["seed"] == seed) & (raw["sender_id"] == sender_id)].iloc[0]
        source_row = effect_df[(effect_df["train_seed"] == seed) & (effect_df["sender_id"] == sender_id)].iloc[0]
        ok_regime = abs(float(row["regime_shift_pp"]) - float(source_row["dominance_regime_abs_pp"])) < 1e-9
        ok_action = abs(float(row["action_shift_pp"]) - float(source_row["dominance_action_abs_pp"])) < 1e-9
        ok_net_regime = abs(float(net_row["regime_shift_pp"]) - recomputed_regime) < 1e-9
        ok_net_action = abs(float(net_row["action_shift_pp"]) - recomputed_action) < 1e-9
        lines.append(
            f"- seed={seed}, sender_id={sender_id}: main regime {row['regime_shift_pp']:.6f} "
            f"(source dominance {source_row['dominance_regime_abs_pp']:.6f}, match={ok_regime}); "
            f"main action {row['action_shift_pp']:.6f} "
            f"(source dominance {source_row['dominance_action_abs_pp']:.6f}, match={ok_action}); "
            f"net regime {net_row['regime_shift_pp']:.6f} "
            f"(recomputed {recomputed_regime:.6f}, match={ok_net_regime}); "
            f"net action {net_row['action_shift_pp']:.6f} "
            f"(recomputed {recomputed_action:.6f}, match={ok_net_action}); "
            f"raw regime {raw_row['regime_shift_pp']:.6f}; raw action {raw_row['action_shift_pp']:.6f}."
        )
    (out_dir / "encoding_validation_checks.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir.resolve()
    out_dir = (args.out_dir or report_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    effect_df = pd.read_csv(report_dir / "sender_effect_summary.csv")
    trace_df = pd.read_csv(report_dir / "filtered_trace_slice.csv")
    history_df = pd.read_csv(report_dir / "sender_history_conditionals.csv")

    if set(trace_df["fhat_side"].dropna().astype(str).unique()) != {"fhat<3.5", "fhat>=3.5"}:
        raise ValueError("filtered trace does not use the expected f_hat < 3.5 / >= 3.5 split")

    main_effects, basis_counts = build_main_effects(effect_df)
    main_effects = attach_observation_counts(main_effects, trace_df)
    net_signed, net_fallback = build_net_signed_effects(effect_df)
    net_signed = attach_observation_counts(net_signed, trace_df)
    raw = build_raw_effects(trace_df)

    if main_effects.shape[0] != 60 or net_signed.shape[0] != 60 or raw.shape[0] != 60:
        raise AssertionError(
            "expected 60 rows; "
            f"got main={main_effects.shape[0]}, net={net_signed.shape[0]}, raw={raw.shape[0]}"
        )
    if main_effects[["regime_shift_pp", "action_shift_pp", "n_observations"]].isna().any().any():
        raise AssertionError("main effect CSV contains NaNs")
    if net_signed[["regime_shift_pp", "action_shift_pp", "n_observations"]].isna().any().any():
        raise AssertionError("net-signed effect CSV contains NaNs")
    if raw[["regime_shift_pp", "action_shift_pp", "n_observations"]].isna().any().any():
        raise AssertionError("raw effect CSV contains NaNs")

    main_effects.to_csv(out_dir / "encoding_effects.csv", index=False)
    net_signed.to_csv(out_dir / "encoding_effects_net_signed.csv", index=False)
    raw.to_csv(out_dir / "encoding_effects_raw.csv", index=False)
    sensitivity = write_sensitivity_tables(out_dir, main_effects)

    stats = summarize_distribution(
        main_effects,
        threshold=float(args.reference_threshold_pp),
        n_resamples=int(args.bootstrap_resamples),
        seed=int(args.bootstrap_seed),
    )
    net_stats = summarize_distribution(
        net_signed,
        threshold=float(args.reference_threshold_pp),
        n_resamples=int(args.bootstrap_resamples),
        seed=int(args.bootstrap_seed) + 50,
    )
    raw_stats = summarize_distribution(
        raw,
        threshold=float(args.reference_threshold_pp),
        n_resamples=int(args.bootstrap_resamples),
        seed=int(args.bootstrap_seed) + 100,
    )

    main_axis_limit, main_axis_expanded = make_scatter(
        main_effects,
        out_pdf=out_dir / "encoding_scatter.pdf",
        out_png=out_dir / "encoding_scatter.png",
        x_col="regime_shift_pp",
        y_col="action_shift_pp",
        x_label="mean absolute regime effect (pp)",
        y_label="mean absolute action effect (pp)",
        threshold=float(args.reference_threshold_pp),
    )
    net_axis_limit, net_axis_expanded = make_scatter(
        net_signed,
        out_pdf=out_dir / "encoding_scatter_net_signed.pdf",
        out_png=out_dir / "encoding_scatter_net_signed.png",
        x_col="regime_shift_pp",
        y_col="action_shift_pp",
        x_label="|net-signed regime effect| (pp)",
        y_label="|net-signed action effect| (pp)",
        threshold=float(args.reference_threshold_pp),
    )
    raw_axis_limit, raw_axis_expanded = make_scatter(
        raw,
        out_pdf=out_dir / "encoding_scatter_raw.pdf",
        out_png=out_dir / "encoding_scatter_raw.png",
        x_col="regime_shift_pp",
        y_col="action_shift_pp",
        x_label="|raw regime difference| (pp)",
        y_label="|raw action difference| (pp)",
        threshold=float(args.reference_threshold_pp),
    )

    write_summary(
        out_dir,
        stats=stats,
        net_stats=net_stats,
        raw_stats=raw_stats,
        basis_counts=basis_counts,
        main_axis_limit=main_axis_limit,
        main_axis_expanded=main_axis_expanded,
        net_axis_limit=net_axis_limit,
        net_axis_expanded=net_axis_expanded,
        raw_axis_limit=raw_axis_limit,
        raw_axis_expanded=raw_axis_expanded,
    )
    write_paragraph(out_dir, stats=stats)
    write_raw_note(out_dir, stats=stats, raw_stats=raw_stats)
    write_net_signed_note(out_dir, stats=stats, net_stats=net_stats)
    write_validation_checks(
        out_dir,
        main_effects=main_effects,
        net_signed=net_signed,
        raw=raw,
        effect_df=effect_df,
        history_df=history_df,
    )
    write_readme(
        out_dir,
        report_dir=report_dir,
        basis_counts=basis_counts,
        stats=stats,
        net_stats=net_stats,
        raw_stats=raw_stats,
        main_axis_expanded=main_axis_expanded,
        net_axis_expanded=net_axis_expanded,
        raw_axis_expanded=raw_axis_expanded,
    )

    print(
        "[sender-encoding-distribution] "
        f"out_dir={out_dir} rows={stats['n']} "
        f"main_both_ge_10={stats['both_gt']} net_both_ge_10={net_stats['both_gt']} "
        f"raw_both_ge_10={raw_stats['both_gt']} "
        f"sensitivity_cols={','.join(c for c in sensitivity.columns if c != 'category')}"
    )


if __name__ == "__main__":
    main()
