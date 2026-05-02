from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_F_VALUES = ["3.500", "5.000"]
DEFAULT_CHECKPOINTS = [25000, 50000, 100000, 150000]


def _normalize_f_value(value: str | float) -> str:
    return f"{float(value):.3f}"


def mean_and_sem(values: Sequence[float]) -> Tuple[float, float]:
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("expected at least one value")
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, 0.0
    sem = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) / math.sqrt(len(vals))
    return mean, sem


def load_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_exact_f_gap_rows(
    suite_rows: Iterable[Dict[str, str]],
    *,
    comm_condition: str = "cond1",
    baseline_condition: str = "cond2",
    f_values: Sequence[str] = DEFAULT_F_VALUES,
    checkpoints: Sequence[int] = DEFAULT_CHECKPOINTS,
) -> List[Dict[str, object]]:
    target_f_values = [_normalize_f_value(value) for value in f_values]
    target_checkpoints = [int(value) for value in checkpoints]

    seedwise: dict[tuple[str, int, str], dict[int, float]] = defaultdict(dict)
    for row in suite_rows:
        if str(row.get("scope", "")).strip() != "f_value":
            continue
        condition = str(row.get("condition", "")).strip()
        if condition not in {str(comm_condition), str(baseline_condition)}:
            continue
        checkpoint_episode = int(row["checkpoint_episode"])
        if checkpoint_episode not in target_checkpoints:
            continue
        f_value = _normalize_f_value(row["key"])
        if f_value not in target_f_values:
            continue
        train_seed = int(row["train_seed"])
        coop_rate = float(row["coop_rate"])
        key = (condition, checkpoint_episode, f_value)
        if train_seed in seedwise[key]:
            raise ValueError(
                f"duplicate suite row for condition={condition} checkpoint={checkpoint_episode} "
                f"f_value={f_value} seed={train_seed}"
            )
        seedwise[key][train_seed] = coop_rate

    out_rows: List[Dict[str, object]] = []
    for f_value in target_f_values:
        for checkpoint_episode in target_checkpoints:
            comm_values = seedwise.get((str(comm_condition), checkpoint_episode, f_value), {})
            baseline_values = seedwise.get((str(baseline_condition), checkpoint_episode, f_value), {})
            if not comm_values:
                raise ValueError(
                    f"missing comm rows for condition={comm_condition} checkpoint={checkpoint_episode} "
                    f"f_value={f_value}"
                )
            if not baseline_values:
                raise ValueError(
                    f"missing baseline rows for condition={baseline_condition} checkpoint={checkpoint_episode} "
                    f"f_value={f_value}"
                )
            comm_mean, comm_sem = mean_and_sem(list(comm_values.values()))
            baseline_mean, baseline_sem = mean_and_sem(list(baseline_values.values()))
            out_rows.append(
                {
                    "f_value": f_value,
                    "checkpoint_episode": str(checkpoint_episode),
                    "new_cond1_mean": comm_mean,
                    "new_cond1_sem": comm_sem,
                    "new_cond2_mean": baseline_mean,
                    "new_cond2_sem": baseline_sem,
                    "new_gap": comm_mean - baseline_mean,
                }
            )
    return out_rows


def write_csv(path: str | Path, rows: Sequence[Dict[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "f_value",
        "checkpoint_episode",
        "new_cond1_mean",
        "new_cond1_sem",
        "new_cond2_mean",
        "new_cond2_sem",
        "new_gap",
    ]
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_main_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--provenance_json", default="")
    parser.add_argument("--comm_condition", default="cond1")
    parser.add_argument("--baseline_condition", default="cond2")
    parser.add_argument("--comm_source_label", default="")
    parser.add_argument("--baseline_source_label", default="")
    parser.add_argument("--f_values", nargs="*", default=DEFAULT_F_VALUES)
    parser.add_argument("--checkpoints", nargs="*", type=int, default=DEFAULT_CHECKPOINTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_rows = load_csv_rows(args.suite_main_csv)
    rows = build_exact_f_gap_rows(
        suite_rows,
        comm_condition=str(args.comm_condition),
        baseline_condition=str(args.baseline_condition),
        f_values=[str(value) for value in args.f_values],
        checkpoints=[int(value) for value in args.checkpoints],
    )
    write_csv(args.out_csv, rows)

    if str(args.provenance_json).strip():
        payload = {
            "suite_main_csv": str(Path(args.suite_main_csv).resolve()),
            "out_csv": str(Path(args.out_csv).resolve()),
            "comm_condition": str(args.comm_condition),
            "baseline_condition": str(args.baseline_condition),
            "comm_source_label": str(args.comm_source_label),
            "baseline_source_label": str(args.baseline_source_label),
            "f_values": [_normalize_f_value(value) for value in args.f_values],
            "checkpoints": [int(value) for value in args.checkpoints],
            "row_count": len(rows),
        }
        provenance_path = Path(args.provenance_json)
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[base-gap] rows={len(rows)} out_csv={Path(args.out_csv).resolve()}")


if __name__ == "__main__":
    main()
