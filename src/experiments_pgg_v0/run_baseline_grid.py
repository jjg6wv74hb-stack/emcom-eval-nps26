from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class GridJob:
    condition: str
    seed: int
    save_path: str
    log_path: str
    cmd: List[str]


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _condition_config(condition: str, n_agents: int) -> Dict:
    if condition == "cond6":
        return {
            "sigmas": [0.0] * n_agents,
            "epsilon_tremble": 0.0,
            "comm_enabled": False,
            "vocab_size": 2,
        }
    if condition == "cond2":
        return {
            "sigmas": [0.5] * n_agents,
            "epsilon_tremble": 0.05,
            "comm_enabled": False,
            "vocab_size": 2,
        }
    if condition == "cond1":
        return {
            "sigmas": [0.5] * n_agents,
            "epsilon_tremble": 0.05,
            "comm_enabled": True,
            "vocab_size": 2,
        }
    raise ValueError(f"unknown condition: {condition}")


def _build_job(args, condition: str, seed: int) -> GridJob:
    cfg = _condition_config(condition, n_agents=args.n_agents)
    save_path = os.path.join(args.save_dir, f"{condition}_seed{seed}.pt")
    log_path = os.path.join(args.log_dir, f"{condition}_seed{seed}.log")

    cmd = [
        sys.executable,
        "-u",
        "src/experiments_pgg_v0/train_ppo.py",
        "--n_episodes",
        str(args.n_episodes),
        "--T",
        str(args.T),
        "--n_agents",
        str(args.n_agents),
        "--sigmas",
        *[str(x) for x in cfg["sigmas"]],
        "--epsilon_tremble",
        str(cfg["epsilon_tremble"]),
        "--seed",
        str(seed),
        "--save_path",
        save_path,
        "--log_interval",
        str(args.log_interval),
        "--lr_schedule",
        args.lr_schedule,
        "--min_lr",
        str(args.min_lr),
    ]
    if cfg["comm_enabled"]:
        cmd.extend(["--comm_enabled", "--vocab_size", str(cfg["vocab_size"])])

    return GridJob(
        condition=condition,
        seed=int(seed),
        save_path=save_path,
        log_path=log_path,
        cmd=cmd,
    )


def _extract_last_metrics(log_path: str) -> Dict:
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            text = f.read()
        marker = '{\n  "episodes":'
        idx = text.rfind(marker)
        if idx < 0:
            return {}
        payload = json.loads(text[idx:])
        return payload.get("last", {})
    except Exception:
        return {}


def _run_job(job: GridJob, cwd: str) -> Dict:
    start = time.time()
    os.makedirs(os.path.dirname(job.log_path), exist_ok=True)
    os.makedirs(os.path.dirname(job.save_path), exist_ok=True)

    with open(job.log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# condition={job.condition} seed={job.seed}\n")
        logf.write("# cmd: " + " ".join(job.cmd) + "\n\n")
        logf.flush()
        proc = subprocess.run(
            job.cmd,
            cwd=cwd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )

    duration_sec = time.time() - start
    last = _extract_last_metrics(job.log_path)
    return {
        "condition": job.condition,
        "seed": job.seed,
        "returncode": int(proc.returncode),
        "duration_sec": float(duration_sec),
        "save_path": job.save_path,
        "log_path": job.log_path,
        "last_metrics": last,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", nargs="*", default=["cond6", "cond2", "cond1"])
    p.add_argument("--seeds", nargs="*", type=int, default=[101, 202, 303, 404, 505])
    p.add_argument("--n_episodes", type=int, default=2000)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--n_agents", type=int, default=4)
    p.add_argument("--max_workers", type=int, default=3)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--lr_schedule", type=str, choices=["none", "linear"], default="linear")
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--save_dir", type=str, default="outputs/train/grid")
    p.add_argument("--log_dir", type=str, default="outputs/train/grid/logs")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    root = _repo_root()
    os.makedirs(os.path.join(root, args.save_dir), exist_ok=True)
    os.makedirs(os.path.join(root, args.log_dir), exist_ok=True)

    jobs: List[GridJob] = []
    for condition in args.conditions:
        for seed in args.seeds:
            job = _build_job(args, condition, seed)
            if args.skip_existing and os.path.exists(os.path.join(root, job.save_path)):
                print(f"[skip-existing] {job.condition} seed={job.seed} -> {job.save_path}")
                continue
            jobs.append(job)

    print(
        f"[grid] jobs={len(jobs)} conditions={args.conditions} "
        f"seeds={args.seeds} episodes={args.n_episodes} workers={args.max_workers}"
    )
    if args.dry_run:
        for job in jobs:
            print(" ".join(job.cmd))
        return
    if len(jobs) == 0:
        print("[grid] nothing to run.")
        return

    start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        fut_to_job = {pool.submit(_run_job, job, root): job for job in jobs}
        for fut in as_completed(fut_to_job):
            job = fut_to_job[fut]
            try:
                out = fut.result()
            except Exception as exc:
                out = {
                    "condition": job.condition,
                    "seed": job.seed,
                    "returncode": -1,
                    "duration_sec": None,
                    "save_path": job.save_path,
                    "log_path": job.log_path,
                    "error": str(exc),
                }
            results.append(out)
            rc = out.get("returncode", -1)
            dur = out.get("duration_sec")
            status = "ok" if rc == 0 else "fail"
            if dur is None:
                print(f"[done] {status} {job.condition} seed={job.seed} rc={rc}")
            else:
                print(
                    f"[done] {status} {job.condition} seed={job.seed} rc={rc} "
                    f"duration={dur:.1f}s"
                )

    elapsed = time.time() - start
    ok = sum(1 for r in results if int(r.get("returncode", -1)) == 0)
    fail = len(results) - ok

    summary = {
        "n_jobs": len(results),
        "ok": ok,
        "fail": fail,
        "elapsed_sec": float(elapsed),
        "results": sorted(results, key=lambda x: (x["condition"], x["seed"])),
    }
    summary_path = os.path.join(root, args.save_dir, "grid_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[grid] complete ok={ok} fail={fail} elapsed={elapsed:.1f}s "
        f"summary={summary_path}"
    )


if __name__ == "__main__":
    main()
