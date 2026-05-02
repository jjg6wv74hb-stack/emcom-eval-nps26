from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


_CHECKPOINT_RE = re.compile(
    r"(?P<condition>cond[0-9]+)_seed(?P<seed>[0-9]+)(?:_ep(?P<local>[0-9]+))?\.pt$"
)


@dataclass(frozen=True)
class CheckpointArtifact:
    path: str
    condition: str
    seed: int
    local_episode: int | None
    episode_offset: int
    absolute_episode: int | None
    is_final: bool


def _load_checkpoint_config(path: str) -> dict:
    sidecar_path = Path(path).with_suffix(".run.json")
    if sidecar_path.exists():
        try:
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            config = payload.get("config", {})
            if isinstance(config, dict) and len(config) > 0:
                return config

    try:
        import torch
    except Exception:
        return {}
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    config = payload.get("config", {})
    return config if isinstance(config, dict) else {}


def _parse_checkpoint_filename(path: str) -> tuple[str | None, int | None, int | None]:
    m = _CHECKPOINT_RE.search(os.path.basename(path))
    if not m:
        return None, None, None
    condition = m.group("condition")
    seed = int(m.group("seed"))
    local = m.group("local")
    return condition, seed, (int(local) if local is not None else None)


def _artifact_from_path(path: str) -> CheckpointArtifact | None:
    parsed_condition, parsed_seed, local_from_name = _parse_checkpoint_filename(path)
    if parsed_condition is None or parsed_seed is None:
        return None
    config = _load_checkpoint_config(path)
    episode_offset = int(config.get("episode_offset", 0) or 0)
    if local_from_name is not None:
        # train_ppo names intermediate checkpoints with the absolute episode
        # number (for example *_ep100000.pt during a 50000 -> 150000 continuation).
        absolute_episode = int(local_from_name)
        local_episode = int(absolute_episode - episode_offset)
        if local_episode < 0:
            local_episode = 0
        is_final = False
    else:
        final_local = int(config.get("n_episodes", 0) or 0)
        local_episode = final_local if final_local > 0 else None
        absolute_episode = None if local_episode is None else episode_offset + int(local_episode)
        is_final = True
    return CheckpointArtifact(
        path=str(os.path.abspath(path)),
        condition=str(parsed_condition),
        seed=int(parsed_seed),
        local_episode=local_episode,
        episode_offset=episode_offset,
        absolute_episode=absolute_episode,
        is_final=bool(is_final),
    )


def _read_manifest_lines(manifest_path: str | Path) -> list[str]:
    path = Path(manifest_path)
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            entry = str(line).strip()
            if entry == "" or entry.startswith("#"):
                continue
            lines.append(str(Path(entry).expanduser().resolve()))
    return lines


def scan_checkpoints(
    checkpoint_dir: str,
    *,
    condition: str | None = None,
    seeds: Sequence[int] | None = None,
) -> list[CheckpointArtifact]:
    root = os.path.abspath(checkpoint_dir)
    seed_filter = None if seeds is None else {int(seed) for seed in seeds}
    out: list[CheckpointArtifact] = []
    for path in sorted(glob.glob(os.path.join(root, "*.pt"))):
        artifact = _artifact_from_path(path)
        if artifact is None:
            continue
        if condition is not None and artifact.condition != str(condition):
            continue
        if seed_filter is not None and artifact.seed not in seed_filter:
            continue
        out.append(artifact)
    return out


def scan_checkpoint_manifest(
    manifest_path: str | Path,
    *,
    condition: str | None = None,
    seeds: Sequence[int] | None = None,
) -> list[CheckpointArtifact]:
    seed_filter = None if seeds is None else {int(seed) for seed in seeds}
    out: list[CheckpointArtifact] = []
    for path in _read_manifest_lines(manifest_path):
        artifact = _artifact_from_path(path)
        if artifact is None:
            continue
        if condition is not None and artifact.condition != str(condition):
            continue
        if seed_filter is not None and artifact.seed not in seed_filter:
            continue
        out.append(artifact)
    return out


def _resolve_matches(matches: Sequence[CheckpointArtifact], *, label: str) -> str:
    if len(matches) == 0:
        raise FileNotFoundError(label)
    ranked = sorted(
        matches,
        key=lambda item: (
            0 if item.is_final else 1,
            0 if item.local_episode is None else -int(item.local_episode),
            item.path,
        ),
    )
    return ranked[0].path


def resolve_checkpoint_path(
    checkpoint_dir: str,
    condition: str,
    seed: int,
    episode: int,
) -> str:
    matches = [
        item
        for item in scan_checkpoints(checkpoint_dir, condition=condition, seeds=[seed])
        if item.absolute_episode == int(episode)
    ]
    return _resolve_matches(
        matches,
        label=f"checkpoint missing for {condition} seed={seed} episode={episode}",
    )


def resolve_manifest_checkpoint_path(
    manifest_path: str | Path,
    condition: str,
    seed: int,
    episode: int,
) -> str:
    matches = [
        item
        for item in scan_checkpoint_manifest(manifest_path, condition=condition, seeds=[seed])
        if item.absolute_episode == int(episode)
    ]
    return _resolve_matches(
        matches,
        label=f"manifest checkpoint missing for {condition} seed={seed} episode={episode}",
    )


def infer_absolute_milestones(
    checkpoint_dir: str,
    *,
    condition: str | None = None,
    seeds: Sequence[int] | None = None,
) -> list[int]:
    episodes = {
        int(item.absolute_episode)
        for item in scan_checkpoints(checkpoint_dir, condition=condition, seeds=seeds)
        if item.absolute_episode is not None
    }
    return sorted(episodes)


def infer_manifest_absolute_milestones(
    manifest_path: str | Path,
    *,
    condition: str | None = None,
    seeds: Sequence[int] | None = None,
) -> list[int]:
    episodes = {
        int(item.absolute_episode)
        for item in scan_checkpoint_manifest(manifest_path, condition=condition, seeds=seeds)
        if item.absolute_episode is not None
    }
    return sorted(episodes)


def csv_has_data_rows(path: str | Path) -> bool:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return False
    if p.stat().st_size <= 0:
        return False
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return False
        if len(header) == 0:
            return False
        for row in reader:
            if len(row) == 0:
                continue
            if any(str(cell).strip() != "" for cell in row):
                return True
    return False


def atomic_write_rows(path: str | Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    target = Path(path)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{target.name}.tmp.",
        suffix=".csv",
        dir=str(parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(fieldnames))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{target.name}.tmp.",
        suffix=".json",
        dir=str(parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{target.name}.tmp.",
        suffix=".txt",
        dir=str(parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect checkpoint artifacts.")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--checkpoint_manifest", default=None)
    parser.add_argument("--condition", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument(
        "--print",
        dest="print_what",
        choices=["milestones", "json"],
        default="milestones",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if bool(args.checkpoint_dir) == bool(args.checkpoint_manifest):
        raise ValueError("provide exactly one of --checkpoint_dir or --checkpoint_manifest")
    if args.checkpoint_dir:
        scan_fn = scan_checkpoints
        milestone_fn = infer_absolute_milestones
        source = args.checkpoint_dir
    else:
        scan_fn = scan_checkpoint_manifest
        milestone_fn = infer_manifest_absolute_milestones
        source = args.checkpoint_manifest

    if args.print_what == "milestones":
        milestones = milestone_fn(
            source,
            condition=args.condition,
            seeds=args.seeds,
        )
        print(" ".join(str(ep) for ep in milestones))
        return 0

    payload = [
        {
            "path": item.path,
            "condition": item.condition,
            "seed": item.seed,
            "local_episode": item.local_episode,
            "episode_offset": item.episode_offset,
            "absolute_episode": item.absolute_episode,
            "is_final": item.is_final,
        }
        for item in scan_fn(
            source,
            condition=args.condition,
            seeds=args.seeds,
        )
    ]
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
