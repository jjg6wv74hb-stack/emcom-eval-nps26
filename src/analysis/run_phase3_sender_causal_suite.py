from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from src.analysis.checkpoint_artifacts import (
    atomic_write_json,
    atomic_write_rows,
    csv_has_data_rows,
    resolve_checkpoint_path,
    resolve_manifest_checkpoint_path,
)


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _checkpoint_path(
    checkpoint_dir: str,
    checkpoint_manifest: str,
    condition: str,
    seed: int,
    episode: int,
) -> str:
    manifest = str(checkpoint_manifest or "").strip()
    if manifest != "":
        return resolve_manifest_checkpoint_path(manifest, condition=condition, seed=seed, episode=episode)
    return resolve_checkpoint_path(checkpoint_dir, condition=condition, seed=seed, episode=episode)


def _task_name(condition: str, seed: int, episode: int) -> str:
    return f"{condition}_seed{int(seed)}_ep{int(episode)}"


def _read_csv_rows(path: str) -> List[Dict]:
    if path == "" or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: str, rows: List[Dict]) -> None:
    if len(rows) == 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    atomic_write_rows(path, rows, fieldnames)


def _run_task(task: Dict, raw_dir: str, log_dir: str, skip_existing: bool):
    out_csv = os.path.join(raw_dir, f"{task['name']}.csv")
    out_condition_csv = os.path.join(raw_dir, f"{task['name']}_condition.csv")
    out_comm_csv = os.path.join(raw_dir, f"{task['name']}_comm.csv")
    out_sender_causal_csv = os.path.join(raw_dir, f"{task['name']}_sender_causal.csv")
    log_path = os.path.join(log_dir, f"{task['name']}.log")
    expected = [out_csv, out_condition_csv, out_comm_csv, out_sender_causal_csv]
    if skip_existing and all(csv_has_data_rows(path) for path in expected):
        return {
            **task,
            "out_csv": out_csv,
            "out_condition_csv": out_condition_csv,
            "out_comm_csv": out_comm_csv,
            "out_sender_causal_csv": out_sender_causal_csv,
            "skipped": True,
        }

    cmd = [
        sys.executable,
        "-m",
        "src.analysis.evaluate_regime_conditional",
        "--checkpoints_glob",
        task["checkpoint"],
        "--n_eval_episodes",
        str(int(task["n_eval_episodes"])),
        "--eval_seed",
        str(int(task["eval_seed"])),
        "--greedy",
        "--msg_intervention",
        "none",
        "--out_csv",
        out_csv,
        "--out_comm_csv",
        out_comm_csv,
        "--out_condition_csv",
        out_condition_csv,
        "--out_sender_causal_csv",
        out_sender_causal_csv,
    ]
    eval_sigmas = task.get("eval_sigmas")
    if eval_sigmas is not None and len(eval_sigmas) > 0:
        cmd.extend(["--eval_sigmas", *[str(float(v)) for v in eval_sigmas]])

    env = os.environ.copy()
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("MPLBACKEND", "Agg")
    with open(log_path, "w", encoding="utf-8") as log_f:
        subprocess.run(
            cmd,
            cwd=_ROOT,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return {
        **task,
        "out_csv": out_csv,
        "out_condition_csv": out_condition_csv,
        "out_comm_csv": out_comm_csv,
        "out_sender_causal_csv": out_sender_causal_csv,
        "skipped": False,
    }


def _aggregate_results(results: List[Dict], out_dir: str) -> None:
    causal_rows: List[Dict] = []
    main_rows: List[Dict] = []
    for result in results:
        extra = {"checkpoint_episode": int(result["episode"])}
        for row in _read_csv_rows(result["out_sender_causal_csv"]):
            causal_rows.append({**row, **extra})
        for row in _read_csv_rows(result["out_csv"]):
            main_rows.append({**row, **extra})
    _write_rows(os.path.join(out_dir, "sender_causal_matrix.csv"), causal_rows)
    _write_rows(os.path.join(out_dir, "sender_causal_checkpoint_main.csv"), main_rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=str, default="")
    p.add_argument("--checkpoint_manifest", type=str, default="")
    p.add_argument("--out_dir", type=str, default="outputs/eval/phase3/sender_causal")
    p.add_argument("--condition", type=str, default="cond1")
    p.add_argument("--seeds", nargs="*", type=int, default=[101, 202])
    p.add_argument("--milestones", nargs="*", type=int, default=[150000])
    p.add_argument("--n_eval_episodes", type=int, default=300)
    p.add_argument("--eval_seed", type=int, default=9001)
    p.add_argument("--max_workers", type=int, default=4)
    p.add_argument("--eval_sigmas", nargs="*", type=float, default=None)
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    raw_dir = os.path.join(out_dir, "raw")
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    tasks = []
    for seed in args.seeds:
        for episode in args.milestones:
            tasks.append(
                {
                    "name": _task_name(args.condition, seed, episode),
                    "condition": str(args.condition),
                    "seed": int(seed),
                    "checkpoint": _checkpoint_path(
                        checkpoint_dir=args.checkpoint_dir,
                        checkpoint_manifest=args.checkpoint_manifest,
                        condition=args.condition,
                        seed=int(seed),
                        episode=int(episode),
                    ),
                    "episode": int(episode),
                    "n_eval_episodes": int(args.n_eval_episodes),
                    "eval_seed": int(args.eval_seed),
                    "eval_sigmas": (
                        [float(v) for v in args.eval_sigmas]
                        if args.eval_sigmas is not None and len(args.eval_sigmas) > 0
                        else None
                    ),
                }
            )

    results = []
    total_tasks = len(tasks)
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
        future_to_task = {
            ex.submit(_run_task, task, raw_dir, log_dir, bool(args.skip_existing)): task
            for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            result = future.result()
            results.append(result)
            print(
                f"[sender-causal] done {task['name']} skipped={bool(result.get('skipped', False))}"
            )
            print(f"[sender-causal] progress current={len(results)} total={total_tasks}")

    results = sorted(results, key=lambda item: item["name"])
    _aggregate_results(results=results, out_dir=out_dir)
    atomic_write_json(os.path.join(out_dir, "sender_causal_manifest.json"), results)
    print(f"[sender-causal] tasks={len(results)} out_dir={out_dir}")


if __name__ == "__main__":
    main()
