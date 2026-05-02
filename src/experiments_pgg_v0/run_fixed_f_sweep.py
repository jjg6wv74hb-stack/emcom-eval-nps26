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
class FixedFJob:
    f_value: float
    seed: int
    save_path: str
    log_path: str
    metrics_jsonl_path: str
    cmd: List[str]


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _f_slug(f_value: float) -> str:
    return str(float(f_value)).replace("-", "m").replace(".", "p")


def _build_job(args, f_value: float, seed: int) -> FixedFJob:
    slug = _f_slug(f_value)
    save_path = os.path.join(args.save_dir, f"fixedf_{slug}_seed{seed}.pt")
    log_path = os.path.join(args.log_dir, f"fixedf_{slug}_seed{seed}.log")
    metrics_jsonl_path = os.path.join(args.metrics_dir, f"fixedf_{slug}_seed{seed}.jsonl")
    condition_name = f"fixedf_{slug}"

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
        "--F",
        str(float(f_value)),
        "--rho",
        str(args.rho),
        "--sigmas",
        *[str(v) for v in args.sigmas],
        "--epsilon_tremble",
        str(args.epsilon_tremble),
        "--gamma",
        str(args.gamma),
        "--lam",
        str(args.lam),
        "--seed",
        str(seed),
        "--save_path",
        save_path,
        "--condition_name",
        condition_name,
        "--log_interval",
        str(args.log_interval),
        "--regime_log_interval",
        str(args.regime_log_interval),
        "--metrics_jsonl_path",
        metrics_jsonl_path,
        "--lr_schedule",
        args.lr_schedule,
        "--min_lr",
        str(args.min_lr),
        "--reward_scale",
        str(args.reward_scale),
        "--early_stop_patience",
        str(args.early_stop_patience),
        "--early_stop_min_delta",
        str(args.early_stop_min_delta),
    ]
    if args.comm_enabled:
        cmd.extend(["--comm_enabled", "--vocab_size", str(args.vocab_size)])

    return FixedFJob(
        f_value=float(f_value),
        seed=int(seed),
        save_path=save_path,
        log_path=log_path,
        metrics_jsonl_path=metrics_jsonl_path,
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


def _run_job(job: FixedFJob, cwd: str) -> Dict:
    start = time.time()
    os.makedirs(os.path.dirname(job.log_path), exist_ok=True)
    os.makedirs(os.path.dirname(job.save_path), exist_ok=True)
    os.makedirs(os.path.dirname(job.metrics_jsonl_path), exist_ok=True)

    with open(job.log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# fixed_f={job.f_value} seed={job.seed}\n")
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
        "f_value": float(job.f_value),
        "seed": int(job.seed),
        "returncode": int(proc.returncode),
        "duration_sec": float(duration_sec),
        "save_path": job.save_path,
        "log_path": job.log_path,
        "metrics_jsonl_path": job.metrics_jsonl_path,
        "last_metrics": last,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--f_values", nargs="*", type=float, default=[0.5, 1.5, 2.5, 3.5, 5.0])
    p.add_argument("--seeds", nargs="*", type=int, default=[101])
    p.add_argument("--n_episodes", type=int, default=50000)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--n_agents", type=int, default=4)
    p.add_argument("--rho", type=float, default=0.0)
    p.add_argument("--sigmas", nargs="*", type=float, default=[0.0, 0.0, 0.0, 0.0])
    p.add_argument("--epsilon_tremble", type=float, default=0.0)
    p.add_argument("--comm_enabled", action="store_true")
    p.add_argument("--vocab_size", type=int, default=2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--reward_scale", type=float, default=20.0)
    p.add_argument("--early_stop_patience", type=int, default=0)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-3)
    p.add_argument("--log_interval", type=int, default=500)
    p.add_argument("--regime_log_interval", type=int, default=500)
    p.add_argument("--lr_schedule", type=str, choices=["none", "linear"], default="linear")
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--max_workers", type=int, default=2)
    p.add_argument("--save_dir", type=str, default="outputs/train/fixed_f_grid")
    p.add_argument("--log_dir", type=str, default="outputs/train/fixed_f_grid/logs")
    p.add_argument("--metrics_dir", type=str, default="outputs/train/fixed_f_grid/metrics")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    root = _repo_root()
    if len(args.sigmas) != int(args.n_agents):
        raise ValueError("len(sigmas) must equal n_agents")

    os.makedirs(os.path.join(root, args.save_dir), exist_ok=True)
    os.makedirs(os.path.join(root, args.log_dir), exist_ok=True)
    os.makedirs(os.path.join(root, args.metrics_dir), exist_ok=True)

    jobs: List[FixedFJob] = []
    for f_value in args.f_values:
        for seed in args.seeds:
            job = _build_job(args=args, f_value=float(f_value), seed=int(seed))
            if args.skip_existing and os.path.exists(os.path.join(root, job.save_path)):
                print(f"[skip-existing] f={job.f_value} seed={job.seed} -> {job.save_path}")
                continue
            jobs.append(job)

    print(
        f"[fixed-f] jobs={len(jobs)} f_values={args.f_values} seeds={args.seeds} "
        f"episodes={args.n_episodes} workers={args.max_workers}"
    )
    if args.dry_run:
        for job in jobs:
            print(" ".join(job.cmd))
        return
    if len(jobs) == 0:
        print("[fixed-f] nothing to run.")
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
                    "f_value": float(job.f_value),
                    "seed": int(job.seed),
                    "returncode": -1,
                    "duration_sec": None,
                    "save_path": job.save_path,
                    "log_path": job.log_path,
                    "metrics_jsonl_path": job.metrics_jsonl_path,
                    "error": str(exc),
                }
            results.append(out)
            rc = int(out.get("returncode", -1))
            dur = out.get("duration_sec")
            status = "ok" if rc == 0 else "fail"
            if dur is None:
                print(f"[done] {status} f={job.f_value} seed={job.seed} rc={rc}")
            else:
                print(
                    f"[done] {status} f={job.f_value} seed={job.seed} rc={rc} "
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
        "results": sorted(results, key=lambda x: (x["f_value"], x["seed"])),
    }
    summary_path = os.path.join(root, args.save_dir, "fixed_f_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[fixed-f] complete ok={ok} fail={fail} elapsed={elapsed:.1f}s "
        f"summary={summary_path}"
    )


if __name__ == "__main__":
    main()
