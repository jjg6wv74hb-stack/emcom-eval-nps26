from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List

from src.analysis.checkpoint_artifacts import (
    resolve_checkpoint_path,
    resolve_manifest_checkpoint_path,
)


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


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


def _train_ckpt(
    *,
    checkpoint_dir: str,
    checkpoint_manifest: str,
    condition: str,
    seed: int,
    episode: int,
) -> str:
    manifest = str(checkpoint_manifest or "").strip()
    if manifest != "":
        return resolve_manifest_checkpoint_path(
            manifest, condition=condition, seed=seed, episode=episode
        )
    if str(checkpoint_dir or "").strip() == "":
        raise FileNotFoundError(
            f"missing init checkpoint source for {condition} seed={seed} episode={episode}"
        )
    return resolve_checkpoint_path(
        checkpoint_dir, condition=condition, seed=seed, episode=episode
    )


def _job_name(condition: str, seed: int) -> str:
    return f"{condition}_seed{int(seed)}"


def _build_job(args, condition: str, seed: int) -> Job:
    name = _job_name(condition=condition, seed=seed)
    save_path = os.path.join(args.out_dir, f"{name}.pt")
    metrics_path = os.path.join(args.out_dir, "metrics", f"{name}.jsonl")
    log_path = os.path.join(args.out_dir, "logs", f"{name}.log")
    if str(args.init_checkpoint_dir or "").strip() != "" or str(args.init_checkpoint_manifest or "").strip() != "":
        init_ckpt = _train_ckpt(
            checkpoint_dir=str(args.init_checkpoint_dir),
            checkpoint_manifest=str(args.init_checkpoint_manifest),
            condition=str(condition),
            seed=int(seed),
            episode=int(args.init_episode),
        )
    else:
        init_ckpt = _fixedf_ckpt(args.fixed_f_dir, seed)

    cmd = [
        sys.executable,
        "-m",
        "src.experiments_pgg_v0.train_ppo",
        "--n_episodes",
        str(int(args.n_episodes)),
        "--num_envs",
        str(int(args.num_envs)),
        "--env_backend",
        str(args.env_backend),
        "--env_start_method",
        str(args.env_start_method),
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
        "--episode_offset",
        str(int(args.episode_offset)),
        "--schedule_total_episodes",
        str(int(args.schedule_total_episodes)),
    ]
    if bool(args.count_env_episodes):
        cmd.append("--count_env_episodes")
    if condition == "cond1":
        intervention = str(args.msg_training_intervention).strip().lower()
        sign_lambda = (
            0.0 if (args.sign_lambda is None and intervention != "none") else args.sign_lambda
        )
        list_lambda = (
            0.0 if (args.list_lambda is None and intervention != "none") else args.list_lambda
        )
        if sign_lambda is None:
            sign_lambda = 0.1
        if list_lambda is None:
            list_lambda = 0.1
        cmd.extend(
            [
                "--comm_enabled",
                "--vocab_size",
                str(int(args.vocab_size)),
                "--sign_lambda",
                str(float(sign_lambda)),
                "--list_lambda",
                str(float(list_lambda)),
            ]
        )
        if bool(args.disable_comm_fallback):
            cmd.append("--disable_comm_fallback")
        if intervention != "none":
            cmd.extend(
                [
                    "--msg_training_intervention",
                    str(args.msg_training_intervention),
                    "--msg_training_history_len",
                    str(int(args.msg_training_history_len)),
                ]
            )
    return Job(
        condition=condition,
        seed=int(seed),
        save_path=save_path,
        metrics_path=metrics_path,
        log_path=log_path,
        init_ckpt=init_ckpt,
        cmd=cmd,
    )


def _run_job(job: Job, skip_existing: bool) -> Dict:
    lock_path = job.save_path + ".lock"
    if skip_existing and os.path.exists(job.save_path):
        return {"job": job.__dict__, "skipped": True, "returncode": 0}
    os.makedirs(os.path.dirname(job.save_path), exist_ok=True)
    os.makedirs(os.path.dirname(job.metrics_path), exist_ok=True)
    os.makedirs(os.path.dirname(job.log_path), exist_ok=True)
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return {
            "job": job.__dict__,
            "skipped": True,
            "returncode": 0,
            "reason": "locked",
        }
    with os.fdopen(lock_fd, "w", encoding="utf-8") as f:
        f.write(f"pid={os.getpid()}\n")

    env = os.environ.copy()
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("MPLBACKEND", "Agg")
    try:
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
    finally:
        if os.path.exists(lock_path):
            os.remove(lock_path)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="outputs/train/phase3_trimmed")
    p.add_argument("--fixed_f_dir", type=str, default="outputs/train/fixed_f_grid")
    p.add_argument("--init_checkpoint_dir", type=str, default="")
    p.add_argument("--init_checkpoint_manifest", type=str, default="")
    p.add_argument("--init_episode", type=int, default=0)
    p.add_argument("--conditions", nargs="*", type=str, default=["cond1"])
    p.add_argument("--seeds", nargs="*", type=int, default=[101, 202, 303, 404, 505])
    p.add_argument("--n_agents", type=int, default=4)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--n_episodes", type=int, default=200000)
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--env_backend", type=str, choices=["serial", "subproc"], default="subproc")
    p.add_argument(
        "--env_start_method",
        type=str,
        choices=["spawn", "forkserver", "fork"],
        default="spawn",
    )
    p.add_argument("--count_env_episodes", dest="count_env_episodes", action="store_true")
    p.add_argument("--no_count_env_episodes", dest="count_env_episodes", action="store_false")
    p.set_defaults(count_env_episodes=True)
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
    p.add_argument("--entropy_schedule", type=str, choices=["none", "linear", "cosine"], default="linear")
    p.add_argument("--entropy_coeff_final", type=float, default=0.001)
    p.add_argument("--msg_entropy_coeff", type=float, default=0.01)
    p.add_argument("--msg_entropy_coeff_final", type=float, default=0.0)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip_ratio", type=float, default=0.2)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--vocab_size", type=int, default=2)
    p.add_argument("--sign_lambda", type=float, default=None)
    p.add_argument("--list_lambda", type=float, default=None)
    p.add_argument("--log_interval", type=int, default=1000)
    p.add_argument("--regime_log_interval", type=int, default=400)
    p.add_argument("--checkpoint_interval", type=int, default=25000)
    p.add_argument("--episode_offset", type=int, default=0)
    p.add_argument("--schedule_total_episodes", type=int, default=0)
    p.add_argument("--disable_comm_fallback", dest="disable_comm_fallback", action="store_true")
    p.add_argument("--enable_comm_fallback", dest="disable_comm_fallback", action="store_false")
    p.set_defaults(disable_comm_fallback=True)
    p.add_argument(
        "--msg_training_intervention",
        type=str,
        choices=["none", "uniform", "public_random", "fixed0", "fixed1", "sender_shuffle"],
        default="none",
    )
    p.add_argument("--msg_training_history_len", type=int, default=4096)
    p.add_argument("--max_workers", type=int, default=2)
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args(argv)


def main():
    args = parse_args()
    jobs = [
        _build_job(args, condition=str(condition), seed=int(seed))
        for condition in args.conditions
        for seed in args.seeds
    ]
    results = []
    total_jobs = len(jobs)
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
        future_to_job = {
            ex.submit(_run_job, job, bool(args.skip_existing)): job for job in jobs
        }
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            result = future.result()
            results.append(result)
            print(
                "[phase3-seeds] done "
                f"{job.condition} seed={job.seed} skipped={bool(result['skipped'])}"
            )
            print(f"[phase3-seeds] progress current={len(results)} total={total_jobs}")

    results = sorted(results, key=lambda item: (item["job"]["condition"], item["job"]["seed"]))
    manifest_path = os.path.join(os.path.abspath(args.out_dir), "phase3_seed_expansion_manifest.json")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[phase3-seeds] jobs={len(results)} manifest={manifest_path}")


if __name__ == "__main__":
    main()
