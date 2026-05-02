from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


_SEED_RE = re.compile(r"seed(\d+)")


@dataclass(frozen=True)
class CheckpointSuiteManifestSummary:
    task_count: int
    seeds: tuple[int, ...]
    episodes: tuple[int, ...]
    interventions: tuple[str, ...]


def _extract_seed(task: dict) -> int | None:
    for field in ("name", "checkpoint"):
        value = str(task.get(field, ""))
        match = _SEED_RE.search(value)
        if match:
            return int(match.group(1))
    return None


def summarize_checkpoint_suite_manifest(
    manifest_path: str | Path,
) -> CheckpointSuiteManifestSummary:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected a list manifest, got {type(payload).__name__}: {path}")

    seeds: set[int] = set()
    episodes: set[int] = set()
    interventions: set[str] = set()
    for idx, task in enumerate(payload):
        if not isinstance(task, dict):
            raise ValueError(f"manifest task {idx} is not a dict: {path}")
        seed = _extract_seed(task)
        if seed is not None:
            seeds.add(seed)
        episode = task.get("episode")
        if episode is not None:
            episodes.add(int(episode))
        intervention = task.get("intervention")
        if intervention is not None:
            interventions.add(str(intervention))

    return CheckpointSuiteManifestSummary(
        task_count=len(payload),
        seeds=tuple(sorted(seeds)),
        episodes=tuple(sorted(episodes)),
        interventions=tuple(sorted(interventions)),
    )


def _set_diff_message(
    label: str,
    actual: Sequence[object],
    expected: Iterable[object],
) -> str | None:
    actual_set = set(actual)
    expected_set = set(expected)
    if actual_set == expected_set:
        return None
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    return f"{label} mismatch: missing={missing} extra={extra}"


def validate_checkpoint_suite_manifest(
    manifest_path: str | Path,
    *,
    expected_seeds: Sequence[int] | None = None,
    expected_episodes: Sequence[int] | None = None,
    expected_interventions: Sequence[str] | None = None,
) -> CheckpointSuiteManifestSummary:
    summary = summarize_checkpoint_suite_manifest(manifest_path)
    errors: list[str] = []
    if expected_seeds is not None:
        msg = _set_diff_message("seed set", summary.seeds, expected_seeds)
        if msg:
            errors.append(msg)
    if expected_episodes is not None:
        msg = _set_diff_message("episode set", summary.episodes, expected_episodes)
        if msg:
            errors.append(msg)
    if expected_interventions is not None:
        msg = _set_diff_message("intervention set", summary.interventions, expected_interventions)
        if msg:
            errors.append(msg)
    if errors:
        raise ValueError("; ".join(errors))
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize and validate a checkpoint-suite manifest."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-seeds", nargs="*", type=int, default=None)
    parser.add_argument("--expected-episodes", nargs="*", type=int, default=None)
    parser.add_argument("--expected-interventions", nargs="*", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        summary = validate_checkpoint_suite_manifest(
            args.manifest,
            expected_seeds=args.expected_seeds,
            expected_episodes=args.expected_episodes,
            expected_interventions=args.expected_interventions,
        )
    except ValueError as exc:
        print(f"manifest validation failed: {exc}", file=sys.stderr)
        return 1

    print(f"tasks={summary.task_count}")
    print(f"seeds={','.join(str(seed) for seed in summary.seeds)}")
    print(f"episodes={','.join(str(episode) for episode in summary.episodes)}")
    print(f"interventions={','.join(summary.interventions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
