from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BASELINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(_ROOT), "dsc-epgg"))
_DEFAULT_FIXED_F_DIR = os.path.join(_BASELINE_ROOT, "outputs", "train", "fixed_f_grid")


@dataclass
class Job:
    condition: str
    seed: int
    save_path: str
    metrics_path: str
    log_path: str
    init_ckpt: str
    cmd: List[str]


def _fixedf_ckpt(fixed_f_dir: str, seed: int) -> str:
    path = os.path.join(fixed_f_dir, f"fixedf_5p0_seed{int(seed)}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing fixed-f warm start checkpoint: {path}")
    return path


def _condition_cfg(condition: str, vocab_size: int) -> Dict:
    if str(condition) == "cond1":
        return {
            "comm_enabled": True,
            "vocab_size": int(vocab_size),
        }
    if str(condition) == "cond2":
        return {
            "comm_enabled": False,
            "vocab_size": int(vocab_size),
        }
    raise ValueError(f"unknown condition: {condition}")


def _build_job(args, condition: str, seed: int) -> Job:
    cond_cfg = _condition_cfg(condition, vocab_size=args.vocab_size)
    save_path = os.path.join(args.out_dir, f"{condition}_seed{int(seed)}.pt")
    metrics_path = os.path.join(args.out_dir, "metrics", f"{condition}_seed{int(seed)}.jsonl")
    log_path = os.path.join(args.out_dir, "logs", f"{condition}_seed{int(seed)}.log")
    init_ckpt = _fixedf_ckpt(args.fixed_f_dir, seed)

    cmd = [
        sys.executable,
        "-m",
        "src.experiments_pgg_v0.train_ppo",
        "--n_episodes",
        str(int(args.n_episodes)),
        "--num_envs",
        str(int(args.num_envs)),
        "--count_env_episodes",
        "--T",
        str(int(args.T)),
        "--n_agents",
        str(int(args.n_agents)),
        "--F",
        *[str(float(v)) for v in args.F],
        "--sigmas",
        *[str(float(v)) for v in args.sigmas],
        "--rho",
        str(float(args.rho)),
        "--epsilon_tremble",
        str(float(args.epsilon_tremble)),
        "--gamma",
        str(float(args.gamma)),
        "--reward_scale",
        str(float(args.reward_scale)),
        "--lr",
        str(float(args.lr)),
        "--min_lr",
        str(float(args.min_lr)),
        "--lr_schedule",
        str(args.lr_schedule),
        "--entropy_coeff",
        str(float(args.entropy_coeff)),
        "--entropy_schedule",
        str(args.entropy_schedule),
        "--entropy_coeff_final",
        str(float(args.entropy_coeff_final)),
        "--msg_entropy_coeff",
        str(float(args.msg_entropy_coeff)),
        "--msg_entropy_coeff_final",
        str(float(args.msg_entropy_coeff_final)),
        "--lam",
        str(float(args.lam)),
        "--clip_ratio",
        str(float(args.clip_ratio)),
        "--hidden_size",
        str(int(args.hidden_size)),
        "--log_interval",
        str(int(args.log_interval)),
        "--regime_log_interval",
        str(int(args.regime_log_interval)),
        "--checkpoint_interval",
        str(int(args.checkpoint_interval)),
        "--seed",
        str(int(seed)),
        "--init_ckpt",
        init_ckpt,
        "--save_path",
        save_path,
        "--metrics_jsonl_path",
        metrics_path,
        "--condition_name",
        str(condition),
    ]
    if cond_cfg["comm_enabled"]:
        cmd.extend(["--comm_enabled", "--vocab_size", str(int(cond_cfg["vocab_size"]))])

    return Job(
        condition=str(condition),
        seed=int(seed),
        save_path=save_path,
        metrics_path=metrics_path,
        log_path=log_path,
        init_ckpt=init_ckpt,
        cmd=cmd,
    )


def _run_job(job: Job, skip_existing: bool) -> Dict:
    if skip_existing and os.path.exists(job.save_path):
        return {"job": job.__dict__, "skipped": True, "returncode": 0}

    os.makedirs(os.path.dirname(job.save_path), exist_ok=True)
    os.makedirs(os.path.dirname(job.metrics_path), exist_ok=True)
    os.makedirs(os.path.dirname(job.log_path), exist_ok=True)

    env = os.environ.copy()
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    with open(job.log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            job.cmd,
            cwd=_ROOT,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, job.cmd)
    return {"job": job.__dict__, "skipped": False, "returncode": int(proc.returncode)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out_dir",
        type=str,
        default="outputs/train/phase3_vectorized_uninterrupted_150k_pilot3",
    )
    p.add_argument("--fixed_f_dir", type=str, default=_DEFAULT_FIXED_F_DIR)
    p.add_argument("--conditions", nargs="*", type=str, default=["cond1", "cond2"])
    p.add_argument("--seeds", nargs="*", type=int, default=[101, 202, 303])
    p.add_argument("--n_agents", type=int, default=4)
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--n_episodes", type=int, default=150000)
    p.add_argument("--F", nargs="*", type=float, default=[0.5, 1.5, 2.5, 3.5, 5.0])
    p.add_argument("--sigmas", nargs="*", type=float, default=[0.5, 0.5, 0.5, 0.5])
    p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--epsilon_tremble", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--reward_scale", type=float, default=20.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--lr_schedule", type=str, choices=["none", "linear", "cosine"], default="cosine")
    p.add_argument("--entropy_coeff", type=float, default=0.01)
    p.add_argument(
        "--entropy_schedule",
        type=str,
        choices=["none", "linear", "cosine"],
        default="linear",
    )
    p.add_argument("--entropy_coeff_final", type=float, default=0.001)
    p.add_argument("--msg_entropy_coeff", type=float, default=0.01)
    p.add_argument("--msg_entropy_coeff_final", type=float, default=0.0)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip_ratio", type=float, default=0.2)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--vocab_size", type=int, default=2)
    p.add_argument("--log_interval", type=int, default=1000)
    p.add_argument("--regime_log_interval", type=int, default=1000)
    p.add_argument("--checkpoint_interval", type=int, default=50000)
    p.add_argument("--max_workers", type=int, default=1)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    jobs = [
        _build_job(args, condition=str(condition), seed=int(seed))
        for condition in args.conditions
        for seed in args.seeds
    ]
    os.makedirs(os.path.abspath(args.out_dir), exist_ok=True)

    if args.dry_run:
        for job in jobs:
            print(" ".join(job.cmd))
        return

    results = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
        future_to_job = {
            ex.submit(_run_job, job, bool(args.skip_existing)): job for job in jobs
        }
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            result = future.result()
            results.append(result)
            print(
                "[phase3-uninterrupted] done "
                f"{job.condition} seed={job.seed} skipped={bool(result['skipped'])}"
            )

    results = sorted(results, key=lambda item: (item["job"]["condition"], item["job"]["seed"]))
    manifest_path = os.path.join(os.path.abspath(args.out_dir), "phase3_uninterrupted_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[phase3-uninterrupted] jobs={len(results)} manifest={manifest_path}")


if __name__ == "__main__":
    main()
