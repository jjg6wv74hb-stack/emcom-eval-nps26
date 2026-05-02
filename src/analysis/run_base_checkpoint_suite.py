from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


_ROOT = Path(__file__).resolve().parents[2]


def _read_text_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _collect_checkpoints(
    checkpoint_paths: Sequence[str],
    checkpoint_globs: Sequence[str],
    checkpoint_manifest: str,
) -> List[str]:
    out: List[str] = []
    seen = set()

    def _add(path: str):
        norm = os.path.abspath(path)
        if norm in seen:
            return
        seen.add(norm)
        out.append(norm)

    for path in checkpoint_paths:
        if str(path).strip():
            _add(str(path).strip())

    import glob

    for pattern in checkpoint_globs:
        for path in sorted(glob.glob(str(pattern))):
            _add(path)

    if str(checkpoint_manifest).strip():
        for path in _read_text_lines(str(checkpoint_manifest)):
            _add(path)

    return out


def _infer_checkpoint_episode(path: str) -> int:
    m = re.search(r"_ep([0-9]+)\.pt$", os.path.basename(path))
    if m:
        return int(m.group(1))

    sidecar = Path(path).with_suffix(".run.json")
    if sidecar.exists():
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        cfg = data.get("config", {})
        if "episode_offset" in cfg and "n_episodes" in cfg:
            return int(cfg.get("episode_offset", 0)) + int(cfg.get("n_episodes", 0))

    import torch

    payload = torch.load(path, map_location="cpu")
    cfg = payload.get("config", {})
    if "episode_offset" in cfg and "n_episodes" in cfg:
        return int(cfg.get("episode_offset", 0)) + int(cfg.get("n_episodes", 0))
    raise ValueError(f"could not infer checkpoint episode for {path}")


def _task_name(path: str, episode: int) -> str:
    stem = Path(path).stem
    if stem.endswith(f"_ep{int(episode)}"):
        return stem
    return f"{stem}_ep{int(episode)}"


def _run_task(task: Dict[str, object], raw_dir: Path, log_dir: Path, skip_existing: bool) -> Dict[str, object]:
    out_csv = raw_dir / f"{task['name']}.csv"
    out_condition_csv = raw_dir / f"{task['name']}_condition.csv"
    log_path = log_dir / f"{task['name']}.log"

    if skip_existing and out_csv.exists() and out_condition_csv.exists():
        return {**task, "out_csv": str(out_csv), "out_condition_csv": str(out_condition_csv), "skipped": True}

    cmd = [
        sys.executable,
        "-m",
        "src.analysis.evaluate_regime_conditional",
        "--checkpoints_glob",
        str(task["checkpoint"]),
        "--n_eval_episodes",
        str(int(task["n_eval_episodes"])),
        "--eval_seed",
        str(int(task["eval_seed"])),
        "--out_csv",
        str(out_csv),
        "--out_condition_csv",
        str(out_condition_csv),
    ]
    if bool(task.get("greedy", True)):
        cmd.append("--greedy")

    env = os.environ.copy()
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("MPLBACKEND", "Agg")
    with log_path.open("w", encoding="utf-8") as log_f:
        subprocess.run(
            cmd,
            cwd=str(_ROOT),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return {**task, "out_csv": str(out_csv), "out_condition_csv": str(out_condition_csv), "skipped": False}


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
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


def _suite_kind(row: Dict[str, str]) -> str:
    return "comm" if str(row.get("comm_enabled", "0")) == "1" else "baseline"


def _aggregate_results(results: Sequence[Dict[str, object]], out_dir: Path) -> None:
    main_rows: List[Dict[str, object]] = []
    condition_rows: List[Dict[str, object]] = []

    for result in results:
        extra = {
            "checkpoint_episode": int(result["episode"]),
            "suite_kind": str(result["suite_kind"]),
            "ablation": "none",
            "sender_remap": "none",
            "cross_play": "none",
        }
        for row in _read_csv_rows(str(result["out_csv"])):
            suite_kind = _suite_kind(row)
            main_rows.append({**row, **extra, "suite_kind": suite_kind})
        for row in _read_csv_rows(str(result["out_condition_csv"])):
            condition_rows.append(
                {
                    **row,
                    "checkpoint_episode": int(result["episode"]),
                    "suite_kind": str(result["suite_kind"]),
                    "ablation": "none",
                    "sender_remap": "none",
                    "cross_play": "none",
                }
            )

    _write_rows(out_dir / "checkpoint_suite_main.csv", main_rows)
    _write_rows(out_dir / "checkpoint_suite_condition.csv", condition_rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", action="append", default=[], help="Explicit checkpoint path. Can be repeated.")
    p.add_argument("--checkpoint_glob", action="append", default=[], help="Glob for checkpoint paths. Can be repeated.")
    p.add_argument(
        "--checkpoint_manifest",
        type=str,
        default="",
        help="Text file with one checkpoint path per line.",
    )
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--n_eval_episodes", type=int, default=300)
    p.add_argument("--eval_seed", type=int, default=9001)
    p.add_argument("--sample_policy", action="store_true")
    p.add_argument("--max_workers", type=int, default=4)
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = _collect_checkpoints(
        checkpoint_paths=[str(v) for v in args.checkpoint],
        checkpoint_globs=[str(v) for v in args.checkpoint_glob],
        checkpoint_manifest=str(args.checkpoint_manifest or ""),
    )
    if not checkpoints:
        raise FileNotFoundError("no checkpoints provided")

    out_dir = Path(args.out_dir).resolve()
    raw_dir = out_dir / "raw"
    log_dir = out_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    tasks: List[Dict[str, object]] = []
    for idx, checkpoint in enumerate(sorted(checkpoints, key=lambda p: (_infer_checkpoint_episode(p), p))):
        episode = _infer_checkpoint_episode(checkpoint)
        tasks.append(
            {
                "name": _task_name(checkpoint, episode),
                "checkpoint": checkpoint,
                "episode": episode,
                "n_eval_episodes": int(args.n_eval_episodes),
                "eval_seed": int(args.eval_seed) + idx,
                "greedy": not bool(args.sample_policy),
                "suite_kind": "comm",
            }
        )

    manifest_path = out_dir / "checkpoint_suite_manifest.json"
    manifest_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")

    results: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
        future_map = {
            ex.submit(_run_task, task, raw_dir, log_dir, bool(args.skip_existing)): task for task in tasks
        }
        for fut in as_completed(future_map):
            results.append(fut.result())

    results.sort(key=lambda row: (int(row["episode"]), str(row["checkpoint"])))
    _aggregate_results(results, out_dir)
    print(f"[suite] checkpoints={len(tasks)} out_dir={out_dir}")


if __name__ == "__main__":
    main()
