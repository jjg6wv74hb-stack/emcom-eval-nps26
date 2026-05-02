from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.analysis.checkpoint_artifacts import csv_has_data_rows
from src.analysis.checkpoint_suite_manifest import validate_checkpoint_suite_manifest


def validate_checkpoint_suite_outputs(
    manifest_path: str | Path,
    *,
    suite_dir: str | Path | None = None,
    expected_seeds: Sequence[int] | None = None,
    expected_episodes: Sequence[int] | None = None,
    expected_interventions: Sequence[str] | None = None,
) -> None:
    manifest = Path(manifest_path)
    suite_root = manifest.parent if suite_dir is None else Path(suite_dir)

    validate_checkpoint_suite_manifest(
        manifest,
        expected_seeds=expected_seeds,
        expected_episodes=expected_episodes,
        expected_interventions=expected_interventions,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected list manifest: {manifest}")

    missing: list[str] = []
    invalid: list[str] = []

    for task in payload:
        if not isinstance(task, dict):
            continue
        required_keys = [
            "out_csv",
            "out_condition_csv",
        ]
        if str(task.get("suite_kind", "")).strip().lower() != "baseline":
            required_keys.append("out_comm_csv")
        required_keys.extend(
            [
                "out_trace_csv",
                "out_sender_csv",
                "out_receiver_csv",
                "out_posterior_csv",
                "out_sender_causal_csv",
            ]
        )
        for key in required_keys:
            path = str(task.get(key, "") or "").strip()
            if path == "":
                continue
            p = Path(path)
            if not p.is_absolute():
                p = suite_root / p
            if not p.exists():
                missing.append(str(p))
                continue
            if not csv_has_data_rows(p):
                invalid.append(str(p))

    for filename in (
        "checkpoint_suite_main.csv",
        "checkpoint_suite_comm.csv",
        "checkpoint_suite_condition.csv",
    ):
        p = suite_root / filename
        if not p.exists():
            missing.append(str(p))
        elif not csv_has_data_rows(p):
            invalid.append(str(p))

    if missing or invalid:
        parts: list[str] = []
        if missing:
            parts.append(f"missing={missing}")
        if invalid:
            parts.append(f"invalid={invalid}")
        raise ValueError("; ".join(parts))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate checkpoint-suite raw and aggregate outputs."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--suite_dir", default=None)
    parser.add_argument("--expected-seeds", nargs="*", type=int, default=None)
    parser.add_argument("--expected-episodes", nargs="*", type=int, default=None)
    parser.add_argument("--expected-interventions", nargs="*", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        validate_checkpoint_suite_outputs(
            args.manifest,
            suite_dir=args.suite_dir,
            expected_seeds=args.expected_seeds,
            expected_episodes=args.expected_episodes,
            expected_interventions=args.expected_interventions,
        )
    except ValueError as exc:
        print(f"checkpoint suite output validation failed: {exc}")
        return 1
    print("checkpoint suite outputs valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
