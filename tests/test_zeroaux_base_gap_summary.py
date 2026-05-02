from __future__ import annotations

import csv
from pathlib import Path

from src.analysis.summarize_phase3_base_gap_from_suite import build_exact_f_gap_rows, write_csv


def _suite_row(condition: str, seed: int, checkpoint_episode: int, f_value: str, coop_rate: float) -> dict[str, str]:
    return {
        "condition": condition,
        "train_seed": str(seed),
        "scope": "f_value",
        "key": f_value,
        "coop_rate": str(coop_rate),
        "checkpoint_episode": str(checkpoint_episode),
    }


def test_build_exact_f_gap_rows_computes_means_and_sems(tmp_path: Path) -> None:
    rows = [
        _suite_row("cond1", 101, 25000, "3.5", 0.6),
        _suite_row("cond1", 202, 25000, "3.5", 0.8),
        _suite_row("cond2", 101, 25000, "3.5", 0.3),
        _suite_row("cond2", 202, 25000, "3.5", 0.5),
        _suite_row("cond1", 101, 25000, "5.0", 0.9),
        _suite_row("cond1", 202, 25000, "5.0", 0.7),
        _suite_row("cond2", 101, 25000, "5.0", 0.4),
        _suite_row("cond2", 202, 25000, "5.0", 0.2),
    ]

    out_rows = build_exact_f_gap_rows(rows, f_values=["3.500", "5.000"], checkpoints=[25000])
    assert len(out_rows) == 2

    row35 = next(row for row in out_rows if row["f_value"] == "3.500")
    assert row35["checkpoint_episode"] == "25000"
    assert abs(float(row35["new_cond1_mean"]) - 0.7) < 1e-12
    assert abs(float(row35["new_cond2_mean"]) - 0.4) < 1e-12
    assert abs(float(row35["new_gap"]) - 0.3) < 1e-12

    out_csv = tmp_path / "exact_f_gap_table.csv"
    write_csv(out_csv, out_rows)
    with out_csv.open("r", encoding="utf-8") as f:
        written_rows = list(csv.DictReader(f))
    assert written_rows[0]["f_value"] == "3.500"
    assert "new_cond1_sem" in written_rows[0]


def test_build_exact_f_gap_rows_rejects_duplicate_seed_rows() -> None:
    rows = [
        _suite_row("cond1", 101, 25000, "3.5", 0.6),
        _suite_row("cond1", 101, 25000, "3.5", 0.7),
        _suite_row("cond2", 101, 25000, "3.5", 0.3),
    ]

    try:
        build_exact_f_gap_rows(rows, f_values=["3.500"], checkpoints=[25000])
    except ValueError as exc:
        assert "duplicate suite row" in str(exc)
    else:
        raise AssertionError("expected duplicate seed rows to raise")
