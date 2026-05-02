from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


FOCAL_F_VALUES = ["3.500", "5.000"]


def _read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _as_int(value: str | None, default: int = 0) -> int:
    if value in ("", None):
        return int(default)
    return int(float(value))


def _as_float(value: str | None, default: float = 0.0) -> float:
    if value in ("", None):
        return float(default)
    return float(value)


def _sign_with_eps(value: float, eps: float = 1e-8) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _filter_transfer_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for row in rows:
        if row.get("scope") != "f_value":
            continue
        if row.get("key") not in FOCAL_F_VALUES:
            continue
        if row.get("condition") != "cond1":
            continue
        if row.get("eval_policy", "greedy") != "greedy":
            continue
        if row.get("ablation", "none") != "none":
            continue
        if row.get("history_intervention", "none") != "none":
            continue
        out.append(row)
    return out


def _filter_reference_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for row in rows:
        if row.get("scope") != "f_value":
            continue
        if row.get("key") not in FOCAL_F_VALUES:
            continue
        if row.get("condition") != "cond1":
            continue
        if row.get("eval_policy", "greedy") != "greedy":
            continue
        if row.get("history_intervention", "none") != "none":
            continue
        if row.get("cross_play", "none") != "none":
            continue
        if row.get("sender_remap", "none") != "none":
            continue
        out.append(row)
    return out


def _filter_sender_semantics_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for row in rows:
        if row.get("condition") != "cond1":
            continue
        if row.get("eval_policy", "greedy") != "greedy":
            continue
        if row.get("ablation", "none") != "none":
            continue
        if row.get("history_intervention", "none") != "none":
            continue
        if row.get("sender_remap", "none") != "none":
            continue
        if row.get("cross_play", "none") != "none":
            continue
        if row.get("summary") != "p_msg1_given_fhat":
            continue
        if row.get("fhat_bin") not in {"fhat<1.5", "fhat>=4.5"}:
            continue
        out.append(row)
    return out


def _build_sender_polarity(
    rows: Iterable[Dict[str, str]]
) -> tuple[int, Dict[Tuple[int, str], float], Dict[int, float], List[Dict[str, object]]]:
    filtered = _filter_sender_semantics_rows(rows)
    if not filtered:
        raise ValueError("No natural sender-semantics rows available for non-oracle alignment rules.")
    checkpoint_episode = max(_as_int(row.get("checkpoint_episode"), 0) for row in filtered)
    filtered = [
        row
        for row in filtered
        if _as_int(row.get("checkpoint_episode"), 0) == int(checkpoint_episode)
    ]
    lookup: Dict[Tuple[int, str, str], float] = {}
    for row in filtered:
        lookup[
            (
                _as_int(row.get("train_seed"), -1),
                str(row.get("sender_id", "")),
                str(row.get("fhat_bin", "")),
            )
        ] = _as_float(row.get("p_message_1"))

    seed_sender_delta: Dict[Tuple[int, str], float] = {}
    polarity_rows: List[Dict[str, object]] = []
    seeds = sorted({_as_int(row.get("train_seed"), -1) for row in filtered})
    sender_ids = sorted({str(row.get("sender_id", "")) for row in filtered})
    for seed in seeds:
        for sender_id in sender_ids:
            low_key = (seed, sender_id, "fhat<1.5")
            high_key = (seed, sender_id, "fhat>=4.5")
            if low_key not in lookup or high_key not in lookup:
                continue
            delta = float(lookup[high_key] - lookup[low_key])
            seed_sender_delta[(seed, sender_id)] = delta
            polarity_rows.append(
                {
                    "checkpoint_episode": int(checkpoint_episode),
                    "train_seed": int(seed),
                    "sender_id": sender_id,
                    "delta_high_minus_low_fhat": float(delta),
                    "delta_sign": int(_sign_with_eps(delta)),
                }
            )

    seed_mean_delta: Dict[int, float] = {}
    for seed in seeds:
        deltas = [
            float(delta)
            for (candidate_seed, _sender_id), delta in seed_sender_delta.items()
            if int(candidate_seed) == int(seed)
        ]
        if deltas:
            seed_mean_delta[int(seed)] = _mean(deltas)
    return int(checkpoint_episode), seed_sender_delta, seed_mean_delta, polarity_rows


def _predict_alignment_rules(
    *,
    receiver_seed: int,
    donor_seed: int,
    seed_sender_delta: Dict[Tuple[int, str], float],
    seed_mean_delta: Dict[int, float],
) -> Dict[str, Dict[str, object]]:
    sender_ids = sorted(
        {
            sender_id
            for (seed, sender_id) in seed_sender_delta.keys()
            if int(seed) in (int(receiver_seed), int(donor_seed))
        }
    )
    common_sender_ids = [
        sender_id
        for sender_id in sender_ids
        if (int(receiver_seed), sender_id) in seed_sender_delta
        and (int(donor_seed), sender_id) in seed_sender_delta
    ]
    if not common_sender_ids:
        raise ValueError(f"No shared sender slots for receiver={receiver_seed}, donor={donor_seed}")

    predictions: Dict[str, Dict[str, object]] = {}

    receiver_mean = float(seed_mean_delta.get(int(receiver_seed), 0.0))
    donor_mean = float(seed_mean_delta.get(int(donor_seed), 0.0))
    seed_mean_score = float(receiver_mean * donor_mean)
    predictions["seed_mean_sign"] = {
        "predicted_alignment_label": "identity__noflip"
        if seed_mean_score >= 0.0
        else "identity__flipall",
        "rule_score": seed_mean_score,
        "n_slots_used": len(common_sender_ids),
    }

    majority_terms = []
    weighted_terms = []
    for sender_id in common_sender_ids:
        receiver_delta = float(seed_sender_delta[(int(receiver_seed), sender_id)])
        donor_delta = float(seed_sender_delta[(int(donor_seed), sender_id)])
        majority_terms.append(_sign_with_eps(receiver_delta) * _sign_with_eps(donor_delta))
        weighted_terms.append(receiver_delta * donor_delta)

    majority_score = float(sum(majority_terms))
    weighted_score = float(sum(weighted_terms))
    predictions["majority_slot_sign"] = {
        "predicted_alignment_label": "identity__noflip"
        if majority_score >= 0.0
        else "identity__flipall",
        "rule_score": majority_score,
        "n_slots_used": len(common_sender_ids),
    }
    predictions["weighted_slot_sign"] = {
        "predicted_alignment_label": "identity__noflip"
        if weighted_score >= 0.0
        else "identity__flipall",
        "rule_score": weighted_score,
        "n_slots_used": len(common_sender_ids),
    }
    return predictions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--transfer_main_csv", type=str, required=True)
    p.add_argument("--reference_main_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--sender_semantics_csv", type=str, default="")
    return p.parse_args()


def _write_nonoracle_markdown(
    *,
    out_dir: Path,
    checkpoint_episode: int,
    polarity_rows: List[Dict[str, object]],
    pairwise_rows: List[Dict[str, object]],
    receiver_rows: List[Dict[str, object]],
    summary_rows: List[Dict[str, object]],
    usage_rows: List[Dict[str, object]],
) -> None:
    metric_lookup = {
        (str(row["rule_name"]), str(row["f_value"]), str(row["metric"])): float(row["value"])
        for row in summary_rows
    }
    n_lookup = {
        (str(row["rule_name"]), str(row["f_value"]), str(row["metric"])): int(row["n"])
        for row in summary_rows
    }
    lines = ["# Non-Oracle Cross-Seed Alignment Summary", ""]
    lines.append(
        f"- sender_polarity_checkpoint_episode: `{int(checkpoint_episode)}`"
    )
    lines.append(
        f"- sender_polarity_rows: `{len(polarity_rows)}`"
    )
    lines.append(
        "Rules below are pre-specified polarity-based predictions evaluated on the already computed identity and flipall pairwise transfer outcomes."
    )
    lines.append("")

    rule_names = sorted({str(row["rule_name"]) for row in summary_rows})
    for rule_name in rule_names:
        lines.append(f"## {rule_name}")
        for f_value in FOCAL_F_VALUES:
            key = (rule_name, f_value, "predicted_mean")
            if key not in metric_lookup:
                continue
            predicted = 100.0 * metric_lookup[key]
            d_nat = 100.0 * metric_lookup[(rule_name, f_value, "predicted_minus_natural_mean")]
            d_pub = 100.0 * metric_lookup[(rule_name, f_value, "predicted_minus_public_random_mean")]
            d_shuf = 100.0 * metric_lookup[(rule_name, f_value, "predicted_minus_sender_shuffle_mean")]
            d_ident = 100.0 * metric_lookup[(rule_name, f_value, "predicted_minus_identity_mean")]
            d_flip = 100.0 * metric_lookup[(rule_name, f_value, "predicted_minus_flipall_mean")]
            d_best = 100.0 * metric_lookup[(rule_name, f_value, "predicted_minus_best_aligned_mean")]
            recv_d_nat = 100.0 * metric_lookup[(rule_name, f_value, "receiver_predicted_minus_natural_mean")]
            recv_pos_nat = int(round(metric_lookup[(rule_name, f_value, "receiver_positive_predicted_minus_natural_count")]))
            recv_n = n_lookup[(rule_name, f_value, "receiver_predicted_minus_natural_mean")]
            lines.append(f"### f={float(f_value):.1f}")
            lines.append(f"- Predicted-rule mean: {predicted:.1f}%")
            lines.append(f"- Predicted minus natural: {d_nat:+.1f} pp")
            lines.append(f"- Predicted minus public random: {d_pub:+.1f} pp")
            lines.append(f"- Predicted minus sender shuffle: {d_shuf:+.1f} pp")
            lines.append(f"- Predicted minus foreign identity: {d_ident:+.1f} pp")
            lines.append(f"- Predicted minus foreign flipall: {d_flip:+.1f} pp")
            lines.append(f"- Predicted minus oracle best-aligned: {d_best:+.1f} pp")
            lines.append(
                f"- Receiver-level robustness: mean predicted minus natural {recv_d_nat:+.1f} pp; positive for {recv_pos_nat}/{recv_n} receivers"
            )
            lines.append("")

    if usage_rows:
        lines.append("## Alignment Usage")
        for row in usage_rows:
            lines.append(
                f"- {row['rule_name']} at f={float(row['f_value']):.1f}: {row['predicted_alignment_label']} used {int(row['count'])} times"
            )
        lines.append("")

    (out_dir / "nonoracle_alignment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    transfer_rows = _filter_transfer_rows(_read_rows(args.transfer_main_csv))
    reference_rows = _filter_reference_rows(_read_rows(args.reference_main_csv))

    transfer_pair_rows: List[Dict[str, object]] = []
    best_rows: List[Dict[str, object]] = []
    receiver_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    reference_map: Dict[Tuple[int, str, str], float] = {}
    for row in reference_rows:
        receiver_seed = int(row["train_seed"])
        key = str(row["key"])
        ablation = str(row["ablation"])
        reference_map[(receiver_seed, key, ablation)] = float(row["coop_rate"])

    by_pair_f: Dict[Tuple[int, int, str], List[Dict[str, str]]] = defaultdict(list)
    identity_rows: Dict[Tuple[int, int, str], Dict[str, str]] = {}
    flip_rows: Dict[Tuple[int, int, str], Dict[str, str]] = {}
    for row in transfer_rows:
        receiver_seed = int(row["receiver_seed"])
        donor_seed = int(row["donor_seed"])
        f_value = str(row["key"])
        key = (receiver_seed, donor_seed, f_value)
        by_pair_f[key].append(row)
        if row.get("alignment_label") == "identity__noflip":
            identity_rows[key] = row
        if row.get("alignment_label") == "identity__flipall":
            flip_rows[key] = row

    best_alignment_counter: Counter[str] = Counter()
    for pair_key, rows in sorted(by_pair_f.items()):
        receiver_seed, donor_seed, f_value = pair_key
        natural = reference_map.get((receiver_seed, f_value, "none"))
        sender_shuffle = reference_map.get((receiver_seed, f_value, "sender_shuffle"))
        public_random = reference_map.get((receiver_seed, f_value, "public_random"))
        indep_random = reference_map.get((receiver_seed, f_value, "indep_random"))
        for row in rows:
            coop_rate = float(row["coop_rate"])
            transfer_pair_rows.append(
                {
                    "receiver_seed": receiver_seed,
                    "donor_seed": donor_seed,
                    "f_value": f_value,
                    "alignment_label": str(row["alignment_label"]),
                    "sender_remap": str(row["sender_remap"]),
                    "cross_play": str(row["cross_play"]),
                    "coop_rate": coop_rate,
                    "delta_vs_natural": (
                        "" if natural is None else float(coop_rate - float(natural))
                    ),
                    "delta_vs_sender_shuffle": (
                        "" if sender_shuffle is None else float(coop_rate - float(sender_shuffle))
                    ),
                    "delta_vs_public_random": (
                        "" if public_random is None else float(coop_rate - float(public_random))
                    ),
                    "delta_vs_indep_random": (
                        "" if indep_random is None else float(coop_rate - float(indep_random))
                    ),
                }
            )

        best = max(rows, key=lambda row: float(row["coop_rate"]))
        best_alignment_counter[str(best["alignment_label"])] += 1
        best_rate = float(best["coop_rate"])
        identity_rate = (
            float(identity_rows[pair_key]["coop_rate"])
            if pair_key in identity_rows
            else float("nan")
        )
        flip_rate = (
            float(flip_rows[pair_key]["coop_rate"])
            if pair_key in flip_rows
            else float("nan")
        )
        best_rows.append(
            {
                "receiver_seed": receiver_seed,
                "donor_seed": donor_seed,
                "f_value": f_value,
                "best_alignment_label": str(best["alignment_label"]),
                "best_coop_rate": best_rate,
                "identity_coop_rate": identity_rate,
                "flipall_coop_rate": flip_rate,
                "natural_coop_rate": natural,
                "sender_shuffle_coop_rate": sender_shuffle,
                "public_random_coop_rate": public_random,
                "indep_random_coop_rate": indep_random,
                "best_minus_natural": "" if natural is None else float(best_rate - float(natural)),
                "best_minus_public_random": (
                    "" if public_random is None else float(best_rate - float(public_random))
                ),
                "best_minus_sender_shuffle": (
                    "" if sender_shuffle is None else float(best_rate - float(sender_shuffle))
                ),
            }
        )

    best_by_f: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in best_rows:
        best_by_f[str(row["f_value"])].append(row)

    by_receiver_f: Dict[Tuple[int, str], List[Dict[str, object]]] = defaultdict(list)
    for row in best_rows:
        by_receiver_f[(int(row["receiver_seed"]), str(row["f_value"]))].append(row)

    receiver_by_f: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for (receiver_seed, f_value), rows in sorted(by_receiver_f.items()):
        natural_vals = [
            float(row["natural_coop_rate"])
            for row in rows
            if row["natural_coop_rate"] is not None
        ]
        sender_shuffle_vals = [
            float(row["sender_shuffle_coop_rate"])
            for row in rows
            if row["sender_shuffle_coop_rate"] is not None
        ]
        public_random_vals = [
            float(row["public_random_coop_rate"])
            for row in rows
            if row["public_random_coop_rate"] is not None
        ]
        indep_random_vals = [
            float(row["indep_random_coop_rate"])
            for row in rows
            if row["indep_random_coop_rate"] is not None
        ]
        receiver_row = {
            "receiver_seed": receiver_seed,
            "f_value": f_value,
            "n_donors": len(rows),
            "best_coop_rate_mean": _mean(float(row["best_coop_rate"]) for row in rows),
            "identity_coop_rate_mean": _mean(
                float(row["identity_coop_rate"])
                for row in rows
                if row["identity_coop_rate"] == row["identity_coop_rate"]
            ),
            "flipall_coop_rate_mean": _mean(
                float(row["flipall_coop_rate"])
                for row in rows
                if row["flipall_coop_rate"] == row["flipall_coop_rate"]
            ),
            "natural_coop_rate": _mean(natural_vals),
            "sender_shuffle_coop_rate": _mean(sender_shuffle_vals),
            "public_random_coop_rate": _mean(public_random_vals),
            "indep_random_coop_rate": _mean(indep_random_vals),
            "best_minus_natural_mean": _mean(
                float(row["best_minus_natural"])
                for row in rows
                if row["best_minus_natural"] != ""
            ),
            "best_minus_public_random_mean": _mean(
                float(row["best_minus_public_random"])
                for row in rows
                if row["best_minus_public_random"] != ""
            ),
            "best_minus_sender_shuffle_mean": _mean(
                float(row["best_minus_sender_shuffle"])
                for row in rows
                if row["best_minus_sender_shuffle"] != ""
            ),
        }
        receiver_rows.append(receiver_row)
        receiver_by_f[f_value].append(receiver_row)

    for f_value in FOCAL_F_VALUES:
        cur = best_by_f.get(f_value, [])
        if not cur:
            continue
        receiver_cur = receiver_by_f.get(f_value, [])
        identity_vals = [float(row["identity_coop_rate"]) for row in cur if row["identity_coop_rate"] == row["identity_coop_rate"]]
        flip_vals = [float(row["flipall_coop_rate"]) for row in cur if row["flipall_coop_rate"] == row["flipall_coop_rate"]]
        best_vals = [float(row["best_coop_rate"]) for row in cur]
        natural_vals = [float(row["natural_coop_rate"]) for row in cur if row["natural_coop_rate"] is not None]
        sender_shuffle_vals = [float(row["sender_shuffle_coop_rate"]) for row in cur if row["sender_shuffle_coop_rate"] is not None]
        public_random_vals = [float(row["public_random_coop_rate"]) for row in cur if row["public_random_coop_rate"] is not None]
        indep_random_vals = [float(row["indep_random_coop_rate"]) for row in cur if row["indep_random_coop_rate"] is not None]
        summary_rows.extend(
            [
                {"f_value": f_value, "metric": "natural_mean", "value": _mean(natural_vals), "n": len(natural_vals), "sample_unit": "ordered_pairs"},
                {"f_value": f_value, "metric": "sender_shuffle_mean", "value": _mean(sender_shuffle_vals), "n": len(sender_shuffle_vals), "sample_unit": "ordered_pairs"},
                {"f_value": f_value, "metric": "public_random_mean", "value": _mean(public_random_vals), "n": len(public_random_vals), "sample_unit": "ordered_pairs"},
                {"f_value": f_value, "metric": "indep_random_mean", "value": _mean(indep_random_vals), "n": len(indep_random_vals), "sample_unit": "ordered_pairs"},
                {"f_value": f_value, "metric": "foreign_identity_mean", "value": _mean(identity_vals), "n": len(identity_vals), "sample_unit": "ordered_pairs"},
                {"f_value": f_value, "metric": "foreign_flipall_mean", "value": _mean(flip_vals), "n": len(flip_vals), "sample_unit": "ordered_pairs"},
                {"f_value": f_value, "metric": "foreign_best_aligned_mean", "value": _mean(best_vals), "n": len(best_vals), "sample_unit": "ordered_pairs"},
                {
                    "f_value": f_value,
                    "metric": "foreign_best_minus_natural_mean",
                    "value": _mean(
                        float(row["best_minus_natural"])
                        for row in cur
                        if row["best_minus_natural"] != ""
                    ),
                    "n": sum(row["best_minus_natural"] != "" for row in cur),
                    "sample_unit": "ordered_pairs",
                },
                {
                    "f_value": f_value,
                    "metric": "foreign_best_minus_public_random_mean",
                    "value": _mean(
                        float(row["best_minus_public_random"])
                        for row in cur
                        if row["best_minus_public_random"] != ""
                    ),
                    "n": sum(row["best_minus_public_random"] != "" for row in cur),
                    "sample_unit": "ordered_pairs",
                },
                {
                    "f_value": f_value,
                    "metric": "foreign_best_minus_sender_shuffle_mean",
                    "value": _mean(
                        float(row["best_minus_sender_shuffle"])
                        for row in cur
                        if row["best_minus_sender_shuffle"] != ""
                    ),
                    "n": sum(row["best_minus_sender_shuffle"] != "" for row in cur),
                    "sample_unit": "ordered_pairs",
                },
                {
                    "f_value": f_value,
                    "metric": "receiver_natural_mean",
                    "value": _mean(float(row["natural_coop_rate"]) for row in receiver_cur),
                    "n": len(receiver_cur),
                    "sample_unit": "receivers",
                },
                {
                    "f_value": f_value,
                    "metric": "receiver_foreign_identity_mean",
                    "value": _mean(float(row["identity_coop_rate_mean"]) for row in receiver_cur),
                    "n": len(receiver_cur),
                    "sample_unit": "receivers",
                },
                {
                    "f_value": f_value,
                    "metric": "receiver_foreign_best_aligned_mean",
                    "value": _mean(float(row["best_coop_rate_mean"]) for row in receiver_cur),
                    "n": len(receiver_cur),
                    "sample_unit": "receivers",
                },
                {
                    "f_value": f_value,
                    "metric": "receiver_best_minus_natural_mean",
                    "value": _mean(
                        float(row["best_minus_natural_mean"]) for row in receiver_cur
                    ),
                    "n": len(receiver_cur),
                    "sample_unit": "receivers",
                },
                {
                    "f_value": f_value,
                    "metric": "receiver_best_minus_public_random_mean",
                    "value": _mean(
                        float(row["best_minus_public_random_mean"])
                        for row in receiver_cur
                    ),
                    "n": len(receiver_cur),
                    "sample_unit": "receivers",
                },
                {
                    "f_value": f_value,
                    "metric": "receiver_best_minus_sender_shuffle_mean",
                    "value": _mean(
                        float(row["best_minus_sender_shuffle_mean"])
                        for row in receiver_cur
                    ),
                    "n": len(receiver_cur),
                    "sample_unit": "receivers",
                },
                {
                    "f_value": f_value,
                    "metric": "receiver_positive_best_minus_natural_count",
                    "value": float(
                        sum(float(row["best_minus_natural_mean"]) > 0.0 for row in receiver_cur)
                    ),
                    "n": len(receiver_cur),
                    "sample_unit": "receivers",
                },
                {
                    "f_value": f_value,
                    "metric": "receiver_positive_best_minus_public_random_count",
                    "value": float(
                        sum(
                            float(row["best_minus_public_random_mean"]) > 0.0
                            for row in receiver_cur
                        )
                    ),
                    "n": len(receiver_cur),
                    "sample_unit": "receivers",
                },
                {
                    "f_value": f_value,
                    "metric": "receiver_positive_best_minus_sender_shuffle_count",
                    "value": float(
                        sum(
                            float(row["best_minus_sender_shuffle_mean"]) > 0.0
                            for row in receiver_cur
                        )
                    ),
                    "n": len(receiver_cur),
                    "sample_unit": "receivers",
                },
            ]
        )

    usage_rows = [
        {"alignment_label": label, "count": count}
        for label, count in sorted(best_alignment_counter.items())
    ]

    _write_rows(out_dir / "pairwise_transfer_results.csv", transfer_pair_rows)
    _write_rows(out_dir / "best_alignment_results.csv", best_rows)
    _write_rows(out_dir / "receiver_level_summary.csv", receiver_rows)
    _write_rows(out_dir / "summary_by_f.csv", summary_rows)
    _write_rows(out_dir / "best_alignment_usage.csv", usage_rows)

    metric_lookup = {(row["f_value"], row["metric"]): float(row["value"]) for row in summary_rows}
    n_lookup = {(row["f_value"], row["metric"]): int(row["n"]) for row in summary_rows}
    lines = ["# Cross-Seed Transfer Summary", ""]
    lines.append(
        "Pairwise means below average across ordered receiver-donor pairs; receiver-level robustness is summarized separately across receiver seeds."
    )
    lines.append(
        "Alignment comparisons use matched eval_seed within each receiver-donor pair."
    )
    lines.append("")
    for f_value in FOCAL_F_VALUES:
        if (f_value, "foreign_best_aligned_mean") not in metric_lookup:
            continue
        nat = metric_lookup.get((f_value, "natural_mean"), float("nan")) * 100.0
        ident = metric_lookup.get((f_value, "foreign_identity_mean"), float("nan")) * 100.0
        best = metric_lookup.get((f_value, "foreign_best_aligned_mean"), float("nan")) * 100.0
        pub = metric_lookup.get((f_value, "public_random_mean"), float("nan")) * 100.0
        shuf = metric_lookup.get((f_value, "sender_shuffle_mean"), float("nan")) * 100.0
        d_nat = metric_lookup.get((f_value, "foreign_best_minus_natural_mean"), float("nan")) * 100.0
        d_pub = metric_lookup.get((f_value, "foreign_best_minus_public_random_mean"), float("nan")) * 100.0
        d_shuf = metric_lookup.get((f_value, "foreign_best_minus_sender_shuffle_mean"), float("nan")) * 100.0
        recv_d_nat = metric_lookup.get((f_value, "receiver_best_minus_natural_mean"), float("nan")) * 100.0
        recv_pos_nat = int(
            round(metric_lookup.get((f_value, "receiver_positive_best_minus_natural_count"), float("nan")))
        )
        recv_n = int(n_lookup.get((f_value, "receiver_best_minus_natural_mean"), 0))
        lines.append(f"## f={float(f_value):.1f}")
        lines.append(
            f"- Natural same-seed mean: {nat:.1f}%"
        )
        lines.append(
            f"- Foreign identity mean: {ident:.1f}%"
        )
        lines.append(
            f"- Foreign best-aligned mean: {best:.1f}%"
        )
        lines.append(
            f"- Public random mean: {pub:.1f}%"
        )
        lines.append(
            f"- Sender shuffle mean: {shuf:.1f}%"
        )
        lines.append(
            f"- Best aligned minus natural: {d_nat:+.1f} pp"
        )
        lines.append(
            f"- Best aligned minus public random: {d_pub:+.1f} pp"
        )
        lines.append(
            f"- Best aligned minus sender shuffle: {d_shuf:+.1f} pp"
        )
        lines.append(
            f"- Receiver-level robustness: mean best aligned minus natural {recv_d_nat:+.1f} pp; positive for {recv_pos_nat}/{recv_n} receivers"
        )
        lines.append("")

    if usage_rows:
        lines.append("## Best Alignment Usage")
        for row in usage_rows:
            lines.append(f"- {row['alignment_label']}: {int(row['count'])}")
        lines.append("")

    (out_dir / "cross_seed_transfer_summary.md").write_text("\n".join(lines), encoding="utf-8")

    if str(args.sender_semantics_csv).strip():
        (
            polarity_checkpoint_episode,
            seed_sender_delta,
            seed_mean_delta,
            polarity_rows,
        ) = _build_sender_polarity(_read_rows(args.sender_semantics_csv))

        pair_lookup = {
            (
                int(row["receiver_seed"]),
                int(row["donor_seed"]),
                str(row["f_value"]),
                str(row["alignment_label"]),
            ): row
            for row in transfer_pair_rows
        }
        best_lookup = {
            (
                int(row["receiver_seed"]),
                int(row["donor_seed"]),
                str(row["f_value"]),
            ): row
            for row in best_rows
        }

        predicted_pairwise_rows: List[Dict[str, object]] = []
        predicted_receiver_rows: List[Dict[str, object]] = []
        predicted_summary_rows: List[Dict[str, object]] = []
        predicted_usage_counter: Counter[Tuple[str, str, str]] = Counter()

        predicted_by_rule_receiver_f: Dict[Tuple[str, int, str], List[Dict[str, object]]] = defaultdict(list)
        predicted_by_rule_f: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)

        for pair_key in sorted(by_pair_f.keys()):
            receiver_seed, donor_seed, f_value = pair_key
            predictions = _predict_alignment_rules(
                receiver_seed=int(receiver_seed),
                donor_seed=int(donor_seed),
                seed_sender_delta=seed_sender_delta,
                seed_mean_delta=seed_mean_delta,
            )
            best_row = best_lookup[(int(receiver_seed), int(donor_seed), str(f_value))]
            for rule_name, prediction in sorted(predictions.items()):
                predicted_alignment_label = str(prediction["predicted_alignment_label"])
                predicted_usage_counter[(rule_name, str(f_value), predicted_alignment_label)] += 1
                chosen_row = pair_lookup[(int(receiver_seed), int(donor_seed), str(f_value), predicted_alignment_label)]
                coop_rate = float(chosen_row["coop_rate"])
                record = {
                    "rule_name": rule_name,
                    "receiver_seed": int(receiver_seed),
                    "donor_seed": int(donor_seed),
                    "f_value": str(f_value),
                    "predicted_alignment_label": predicted_alignment_label,
                    "rule_score": float(prediction["rule_score"]),
                    "n_slots_used": int(prediction["n_slots_used"]),
                    "coop_rate": coop_rate,
                    "natural_coop_rate": float(best_row["natural_coop_rate"]),
                    "sender_shuffle_coop_rate": float(best_row["sender_shuffle_coop_rate"]),
                    "public_random_coop_rate": float(best_row["public_random_coop_rate"]),
                    "indep_random_coop_rate": float(best_row["indep_random_coop_rate"]),
                    "identity_coop_rate": float(best_row["identity_coop_rate"]),
                    "flipall_coop_rate": float(best_row["flipall_coop_rate"]),
                    "best_aligned_coop_rate": float(best_row["best_coop_rate"]),
                    "delta_vs_natural": float(coop_rate - float(best_row["natural_coop_rate"])),
                    "delta_vs_sender_shuffle": float(coop_rate - float(best_row["sender_shuffle_coop_rate"])),
                    "delta_vs_public_random": float(coop_rate - float(best_row["public_random_coop_rate"])),
                    "delta_vs_indep_random": float(coop_rate - float(best_row["indep_random_coop_rate"])),
                    "delta_vs_identity": float(coop_rate - float(best_row["identity_coop_rate"])),
                    "delta_vs_flipall": float(coop_rate - float(best_row["flipall_coop_rate"])),
                    "delta_vs_best_aligned": float(coop_rate - float(best_row["best_coop_rate"])),
                }
                predicted_pairwise_rows.append(record)
                predicted_by_rule_receiver_f[(rule_name, int(receiver_seed), str(f_value))].append(record)
                predicted_by_rule_f[(rule_name, str(f_value))].append(record)

        predicted_receiver_by_rule_f: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
        for (rule_name, receiver_seed, f_value), rows in sorted(predicted_by_rule_receiver_f.items()):
            receiver_row = {
                "rule_name": rule_name,
                "receiver_seed": int(receiver_seed),
                "f_value": str(f_value),
                "n_donors": len(rows),
                "predicted_coop_rate_mean": _mean(float(row["coop_rate"]) for row in rows),
                "natural_coop_rate": _mean(float(row["natural_coop_rate"]) for row in rows),
                "sender_shuffle_coop_rate": _mean(float(row["sender_shuffle_coop_rate"]) for row in rows),
                "public_random_coop_rate": _mean(float(row["public_random_coop_rate"]) for row in rows),
                "identity_coop_rate_mean": _mean(float(row["identity_coop_rate"]) for row in rows),
                "flipall_coop_rate_mean": _mean(float(row["flipall_coop_rate"]) for row in rows),
                "best_aligned_coop_rate_mean": _mean(float(row["best_aligned_coop_rate"]) for row in rows),
                "predicted_minus_natural_mean": _mean(float(row["delta_vs_natural"]) for row in rows),
                "predicted_minus_sender_shuffle_mean": _mean(float(row["delta_vs_sender_shuffle"]) for row in rows),
                "predicted_minus_public_random_mean": _mean(float(row["delta_vs_public_random"]) for row in rows),
                "predicted_minus_identity_mean": _mean(float(row["delta_vs_identity"]) for row in rows),
                "predicted_minus_flipall_mean": _mean(float(row["delta_vs_flipall"]) for row in rows),
                "predicted_minus_best_aligned_mean": _mean(float(row["delta_vs_best_aligned"]) for row in rows),
            }
            predicted_receiver_rows.append(receiver_row)
            predicted_receiver_by_rule_f[(rule_name, str(f_value))].append(receiver_row)

        for rule_name, f_value in sorted(predicted_by_rule_f.keys()):
            cur = predicted_by_rule_f[(rule_name, f_value)]
            receiver_cur = predicted_receiver_by_rule_f[(rule_name, f_value)]
            predicted_summary_rows.extend(
                [
                    {"rule_name": rule_name, "f_value": f_value, "metric": "predicted_mean", "value": _mean(float(row["coop_rate"]) for row in cur), "n": len(cur), "sample_unit": "ordered_pairs"},
                    {"rule_name": rule_name, "f_value": f_value, "metric": "predicted_minus_natural_mean", "value": _mean(float(row["delta_vs_natural"]) for row in cur), "n": len(cur), "sample_unit": "ordered_pairs"},
                    {"rule_name": rule_name, "f_value": f_value, "metric": "predicted_minus_sender_shuffle_mean", "value": _mean(float(row["delta_vs_sender_shuffle"]) for row in cur), "n": len(cur), "sample_unit": "ordered_pairs"},
                    {"rule_name": rule_name, "f_value": f_value, "metric": "predicted_minus_public_random_mean", "value": _mean(float(row["delta_vs_public_random"]) for row in cur), "n": len(cur), "sample_unit": "ordered_pairs"},
                    {"rule_name": rule_name, "f_value": f_value, "metric": "predicted_minus_identity_mean", "value": _mean(float(row["delta_vs_identity"]) for row in cur), "n": len(cur), "sample_unit": "ordered_pairs"},
                    {"rule_name": rule_name, "f_value": f_value, "metric": "predicted_minus_flipall_mean", "value": _mean(float(row["delta_vs_flipall"]) for row in cur), "n": len(cur), "sample_unit": "ordered_pairs"},
                    {"rule_name": rule_name, "f_value": f_value, "metric": "predicted_minus_best_aligned_mean", "value": _mean(float(row["delta_vs_best_aligned"]) for row in cur), "n": len(cur), "sample_unit": "ordered_pairs"},
                    {"rule_name": rule_name, "f_value": f_value, "metric": "receiver_predicted_mean", "value": _mean(float(row["predicted_coop_rate_mean"]) for row in receiver_cur), "n": len(receiver_cur), "sample_unit": "receivers"},
                    {"rule_name": rule_name, "f_value": f_value, "metric": "receiver_predicted_minus_natural_mean", "value": _mean(float(row["predicted_minus_natural_mean"]) for row in receiver_cur), "n": len(receiver_cur), "sample_unit": "receivers"},
                    {"rule_name": rule_name, "f_value": f_value, "metric": "receiver_positive_predicted_minus_natural_count", "value": float(sum(float(row["predicted_minus_natural_mean"]) > 0.0 for row in receiver_cur)), "n": len(receiver_cur), "sample_unit": "receivers"},
                ]
            )

        predicted_usage_rows = [
            {
                "rule_name": rule_name,
                "f_value": f_value,
                "predicted_alignment_label": predicted_alignment_label,
                "count": count,
            }
            for (rule_name, f_value, predicted_alignment_label), count in sorted(predicted_usage_counter.items())
        ]

        _write_rows(out_dir / "nonoracle_alignment_sender_polarity.csv", polarity_rows)
        _write_rows(out_dir / "nonoracle_alignment_pairwise_results.csv", predicted_pairwise_rows)
        _write_rows(out_dir / "nonoracle_alignment_receiver_summary.csv", predicted_receiver_rows)
        _write_rows(out_dir / "nonoracle_alignment_summary_by_f.csv", predicted_summary_rows)
        _write_rows(out_dir / "nonoracle_alignment_usage.csv", predicted_usage_rows)
        _write_nonoracle_markdown(
            out_dir=out_dir,
            checkpoint_episode=int(polarity_checkpoint_episode),
            polarity_rows=polarity_rows,
            pairwise_rows=predicted_pairwise_rows,
            receiver_rows=predicted_receiver_rows,
            summary_rows=predicted_summary_rows,
            usage_rows=predicted_usage_rows,
        )

    print(f"[xseed-summary] out_dir={out_dir}")


if __name__ == "__main__":
    main()
