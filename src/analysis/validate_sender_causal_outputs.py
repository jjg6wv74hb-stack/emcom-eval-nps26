from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.analysis.checkpoint_artifacts import csv_has_data_rows


def validate_sender_causal_outputs(
    manifest_path: str | Path,
    *,
    suite_dir: str | Path | None = None,
    expected_seeds: Sequence[int] | None = None,
    expected_episodes: Sequence[int] | None = None,
) -> None:
    manifest = Path(manifest_path)
    suite_root = manifest.parent if suite_dir is None else Path(suite_dir)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected list manifest: {manifest}")

    seed_filter = None if expected_seeds is None else {int(seed) for seed in expected_seeds}
    episode_filter = None if expected_episodes is None else {int(ep) for ep in expected_episodes}

    missing: list[str] = []
    invalid: list[str] = []
    seen_pairs: set[tuple[int, int]] = set()

    for task in payload:
        if not isinstance(task, dict):
            continue
        seed = int(task.get("seed", -1))
        episode = int(task.get("episode", -1))
        if seed_filter is not None and seed not in seed_filter:
            raise ValueError(f"unexpected seed={seed} in sender-causal manifest")
        if episode_filter is not None and episode not in episode_filter:
            raise ValueError(f"unexpected episode={episode} in sender-causal manifest")
        seen_pairs.add((seed, episode))
        for key in (
            "out_csv",
            "out_condition_csv",
            "out_comm_csv",
            "out_sender_causal_csv",
        ):
            path = str(task.get(key, "") or "").strip()
            if path == "":
                raise ValueError(f"missing manifest key={key} for seed={seed} episode={episode}")
            p = Path(path)
            if not p.is_absolute():
                p = suite_root / p
            if not p.exists():
                missing.append(str(p))
                continue
            if not csv_has_data_rows(p):
                invalid.append(str(p))

    if seed_filter is not None and episode_filter is not None:
        expected_pairs = {(int(seed), int(ep)) for seed in seed_filter for ep in episode_filter}
        missing_pairs = sorted(expected_pairs - seen_pairs)
        if missing_pairs:
            raise ValueError(f"missing manifest pairs={missing_pairs}")

    for filename in (
        "sender_causal_matrix.csv",
        "sender_causal_checkpoint_main.csv",
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
        description="Validate sender-causal raw and aggregate outputs."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--suite_dir", default=None)
    parser.add_argument("--expected-seeds", nargs="*", type=int, default=None)
    parser.add_argument("--expected-episodes", nargs="*", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        validate_sender_causal_outputs(
            args.manifest,
            suite_dir=args.suite_dir,
            expected_seeds=args.expected_seeds,
            expected_episodes=args.expected_episodes,
        )
    except ValueError as exc:
        print(f"sender-causal output validation failed: {exc}")
        return 1
    print("sender-causal outputs valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
