from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


DEFAULT_TRAIN_MODES = ("learned", "uniform", "public_random", "fixed0")
DEFAULT_TEST_MODES = ("natural", "zeros", "indep_random", "public_random", "fixed0", "fixed1")


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: str | float | int | None, default: float = 0.0) -> float:
    if value in ("", None):
        return float(default)
    return float(value)


def _cell_report_dir(
    root: Path,
    *,
    train_mode: str,
    test_mode: str,
    run_kind: str,
    run_date: str,
) -> Path:
    return (
        root
        / f"phase3_vecstraight_zeroaux_crossover_train_{train_mode}_test_{test_mode}_15seeds_{run_kind}_{run_date}"
        / "report"
    )


def _load_cell_rows(
    root: Path,
    *,
    train_mode: str,
    test_mode: str,
    run_kind: str,
    run_date: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    report_dir = _cell_report_dir(
        root,
        train_mode=train_mode,
        test_mode=test_mode,
        run_kind=run_kind,
        run_date=run_date,
    )
    summary_path = report_dir / "intervention_suite_summary.csv"
    paired_path = report_dir / "intervention_suite_paired_stats.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not paired_path.exists():
        raise FileNotFoundError(paired_path)
    return _read_rows(summary_path), _read_rows(paired_path)


def _index_by_f(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row["f_value"]): row for row in rows}


def _message_tokens(row: dict[str, str], senders: Sequence[str]) -> list[int] | None:
    tokens: list[int] = []
    for sender in senders:
        tok0 = row.get(f"obs_msg_{sender}_tok0", "")
        tok1 = row.get(f"obs_msg_{sender}_tok1", "")
        if tok0 == "" or tok1 == "":
            return None
        pair = (float(tok0), float(tok1))
        if pair == (1.0, 0.0):
            tokens.append(0)
        elif pair == (0.0, 1.0):
            tokens.append(1)
        else:
            return None
    return tokens


def _validate_message_sample(path: Path, *, test_mode: str) -> dict[str, object]:
    if not path.exists():
        return {
            "message_sample_path": str(path),
            "message_sample_rows": 0,
            "obs_msg_cols": 0,
            "message_validation_ok": 0,
            "message_validation_detail": "missing",
        }
    rows = _read_rows(path)
    if not rows:
        return {
            "message_sample_path": str(path),
            "message_sample_rows": 0,
            "obs_msg_cols": 0,
            "message_validation_ok": 0,
            "message_validation_detail": "empty",
        }
    senders = ("agent_0", "agent_1", "agent_2", "agent_3")
    obs_cols = [key for key in rows[0].keys() if key.startswith("obs_msg_")]
    tokens_by_row = [_message_tokens(row, senders) for row in rows]
    valid_token_rows = [tokens for tokens in tokens_by_row if tokens is not None]
    all_values = {
        float(row[col])
        for row in rows
        for col in obs_cols
        if str(row.get(col, "")) != ""
    }
    nonshared_rows = sum(1 for tokens in valid_token_rows if len(set(tokens)) > 1)
    shared_rows = sum(1 for tokens in valid_token_rows if len(set(tokens)) == 1)

    ok = True
    detail = "not_checked"
    if test_mode == "zeros":
        ok = all_values == {0.0}
        detail = "zeros_all_zero" if ok else f"zeros_values={sorted(all_values)}"
    elif test_mode == "fixed0":
        ok = len(valid_token_rows) == len(rows) and all(
            all(token == 0 for token in tokens) for tokens in valid_token_rows
        )
        detail = "all_fixed0" if ok else "not_all_fixed0"
    elif test_mode == "fixed1":
        ok = len(valid_token_rows) == len(rows) and all(
            all(token == 1 for token in tokens) for tokens in valid_token_rows
        )
        detail = "all_fixed1" if ok else "not_all_fixed1"
    elif test_mode == "public_random":
        ok = len(valid_token_rows) == len(rows) and shared_rows == len(rows)
        detail = "shared_public_bit" if ok else f"nonshared_rows={nonshared_rows}"
    elif test_mode == "indep_random":
        ok = len(valid_token_rows) == len(rows) and nonshared_rows > 0
        detail = f"nonshared_rows={nonshared_rows}"
    elif test_mode == "natural":
        detail = "natural_no_fixed_expectation"

    return {
        "message_sample_path": str(path),
        "message_sample_rows": len(rows),
        "obs_msg_cols": len(obs_cols),
        "valid_onehot_rows": len(valid_token_rows),
        "shared_rows": shared_rows,
        "nonshared_rows": nonshared_rows,
        "message_validation_ok": int(bool(ok)),
        "message_validation_detail": detail,
    }


def build_crossover_summary(
    *,
    crossover_root: Path,
    out_dir: Path,
    train_modes: Sequence[str] = DEFAULT_TRAIN_MODES,
    test_modes: Sequence[str] = DEFAULT_TEST_MODES,
    run_kind: str = "iwr",
    run_date: str = "20260417",
) -> dict[str, Path]:
    summary_rows: list[dict[str, object]] = []
    message_rows: list[dict[str, object]] = []

    for train_mode in train_modes:
        for test_mode in test_modes:
            summary, paired = _load_cell_rows(
                crossover_root,
                train_mode=train_mode,
                test_mode=test_mode,
                run_kind=run_kind,
                run_date=run_date,
            )
            paired_by_f = _index_by_f(paired)
            for row in summary:
                f_value = str(row["f_value"])
                paired_row = paired_by_f.get(f_value, {})
                natural_minus_intervention = _as_float(
                    paired_row.get("mean_delta_natural_minus_intervention"),
                    0.0,
                )
                summary_rows.append(
                    {
                        "train_mode": train_mode,
                        "test_mode": test_mode,
                        "f_value": f_value,
                        "n_seeds": int(float(row["n_seeds"])),
                        "mean_coop_rate": _as_float(row["mean_coop_rate"]),
                        "sem_coop_rate": _as_float(row["sem_coop_rate"]),
                        "mean_coop_pct": 100.0 * _as_float(row["mean_coop_rate"]),
                        "sem_coop_pp": 100.0 * _as_float(row["sem_coop_rate"]),
                        "natural_mean_coop_rate": _as_float(
                            paired_row.get("natural_mean_coop_rate"),
                            _as_float(row["mean_coop_rate"]),
                        ),
                        "delta_intervention_minus_natural": -natural_minus_intervention,
                        "delta_intervention_minus_natural_pp": -100.0
                        * natural_minus_intervention,
                        "sign_flip_p_value": _as_float(
                            paired_row.get("sign_flip_p_value"),
                            1.0,
                        ),
                        "n_intervention_gt_natural": int(
                            float(paired_row.get("n_negative", 0) or 0)
                        ),
                        "n_intervention_lt_natural": int(
                            float(paired_row.get("n_positive", 0) or 0)
                        ),
                        "n_intervention_eq_natural": int(
                            float(paired_row.get("n_zero", 0) or 0)
                        ),
                    }
                )

            report_dir = _cell_report_dir(
                crossover_root,
                train_mode=train_mode,
                test_mode=test_mode,
                run_kind=run_kind,
                run_date=run_date,
            )
            message_row = {
                "train_mode": train_mode,
                "test_mode": test_mode,
            }
            message_row.update(
                _validate_message_sample(
                    report_dir / "message_stream_sample.csv",
                    test_mode=test_mode,
                )
            )
            message_rows.append(message_row)

    priority_pairs = [
        ("uniform", "zeros"),
        ("uniform", "indep_random"),
        ("uniform", "fixed0"),
        ("learned", "zeros"),
    ]
    priority_rows = [
        row
        for row in summary_rows
        if (str(row["train_mode"]), str(row["test_mode"])) in priority_pairs
    ]
    priority_order = {pair: idx for idx, pair in enumerate(priority_pairs)}
    priority_rows.sort(
        key=lambda row: (
            priority_order[(str(row["train_mode"]), str(row["test_mode"]))],
            float(str(row["f_value"])),
        )
    )

    matrix_rows = sorted(
        summary_rows,
        key=lambda row: (
            float(str(row["f_value"])),
            list(train_modes).index(str(row["train_mode"])),
            list(test_modes).index(str(row["test_mode"])),
        ),
    )
    message_rows.sort(
        key=lambda row: (
            list(train_modes).index(str(row["train_mode"])),
            list(test_modes).index(str(row["test_mode"])),
        )
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_csv = out_dir / "crossover_matrix_summary.csv"
    priority_csv = out_dir / "crossover_priority_contrasts.csv"
    message_csv = out_dir / "crossover_message_stream_validation.csv"
    _write_rows(matrix_csv, matrix_rows)
    _write_rows(priority_csv, priority_rows)
    _write_rows(message_csv, message_rows)

    for f_value in sorted({str(row["f_value"]) for row in matrix_rows}, key=float):
        wide_rows: list[dict[str, object]] = []
        for train_mode in train_modes:
            wide_row: dict[str, object] = {"train_mode": train_mode, "f_value": f_value}
            for test_mode in test_modes:
                match = next(
                    row
                    for row in matrix_rows
                    if row["train_mode"] == train_mode
                    and row["test_mode"] == test_mode
                    and row["f_value"] == f_value
                )
                wide_row[f"{test_mode}_mean_coop_pct"] = match["mean_coop_pct"]
                wide_row[f"{test_mode}_sem_pp"] = match["sem_coop_pp"]
                wide_row[f"{test_mode}_delta_vs_natural_pp"] = match[
                    "delta_intervention_minus_natural_pp"
                ]
            wide_rows.append(wide_row)
        suffix = str(f_value).replace(".", "p")
        _write_rows(out_dir / f"crossover_matrix_wide_f{suffix}.csv", wide_rows)

    md_path = out_dir / "crossover_summary.md"
    md_path.write_text(_render_markdown(matrix_rows, priority_rows, message_rows), encoding="utf-8")
    return {
        "matrix_csv": matrix_csv,
        "priority_csv": priority_csv,
        "message_csv": message_csv,
        "markdown": md_path,
    }


def _fmt_pct(value: object) -> str:
    return f"{float(value):.1f}"


def _render_markdown(
    matrix_rows: Sequence[dict[str, object]],
    priority_rows: Sequence[dict[str, object]],
    message_rows: Sequence[dict[str, object]],
) -> str:
    lines = [
        "# Phase-3 Zero-Aux Crossover Summary",
        "",
        "Cooperation values are percentages; delta is intervention minus the train-family natural self-eval.",
        "",
        "## Priority Contrasts",
        "",
        "| train | test | f | coop % | SEM pp | delta pp | sign-flip p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in priority_rows:
        lines.append(
            "| {train} | {test} | {f} | {coop} | {sem} | {delta} | {p:.4f} |".format(
                train=row["train_mode"],
                test=row["test_mode"],
                f=row["f_value"],
                coop=_fmt_pct(row["mean_coop_pct"]),
                sem=_fmt_pct(row["sem_coop_pp"]),
                delta=_fmt_pct(row["delta_intervention_minus_natural_pp"]),
                p=float(row["sign_flip_p_value"]),
            )
        )

    lines.extend(
        [
            "",
            "## Matrix",
            "",
            "| f | train | test | coop % | SEM pp | delta pp |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in matrix_rows:
        lines.append(
            "| {f} | {train} | {test} | {coop} | {sem} | {delta} |".format(
                f=row["f_value"],
                train=row["train_mode"],
                test=row["test_mode"],
                coop=_fmt_pct(row["mean_coop_pct"]),
                sem=_fmt_pct(row["sem_coop_pp"]),
                delta=_fmt_pct(row["delta_intervention_minus_natural_pp"]),
            )
        )

    failed = [
        row
        for row in message_rows
        if int(row.get("message_validation_ok", 0)) != 1
        and row.get("test_mode") != "natural"
    ]
    lines.extend(["", "## Message Stream Validation", ""])
    if failed:
        lines.append(f"{len(failed)} non-natural message-stream checks failed.")
        for row in failed:
            lines.append(
                f"- {row['train_mode']} / {row['test_mode']}: {row['message_validation_detail']}"
            )
    else:
        lines.append("All non-natural message-stream checks passed.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--crossover_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--run_kind", default="iwr")
    p.add_argument("--run_date", default="20260417")
    p.add_argument("--train_modes", nargs="*", default=list(DEFAULT_TRAIN_MODES))
    p.add_argument("--test_modes", nargs="*", default=list(DEFAULT_TEST_MODES))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_crossover_summary(
        crossover_root=Path(args.crossover_root).resolve(),
        out_dir=Path(args.out_dir).resolve(),
        train_modes=[str(v) for v in args.train_modes],
        test_modes=[str(v) for v in args.test_modes],
        run_kind=str(args.run_kind),
        run_date=str(args.run_date),
    )
    for key, path in outputs.items():
        print(f"{key}={path}")


if __name__ == "__main__":
    main()
