from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _run(cmd: List[str], cwd: str):
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--n_agents", type=int, default=4)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--stage1_episodes", type=int, default=50000)
    p.add_argument("--stage2_episodes", type=int, default=200000)
    p.add_argument("--stage1_gamma", type=float, default=0.0)
    p.add_argument("--stage2_gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--epsilon_tremble", type=float, default=0.05)
    p.add_argument("--sigmas", nargs="*", type=float, default=[0.5, 0.5, 0.5, 0.5])
    p.add_argument("--F_stage1", nargs="*", type=float, default=[5.0])
    p.add_argument("--F_stage2", nargs="*", type=float, default=[0.5, 1.5, 2.5, 3.5, 5.0])
    p.add_argument("--comm_enabled", action="store_true")
    p.add_argument("--vocab_size", type=int, default=2)
    p.add_argument("--log_interval", type=int, default=500)
    p.add_argument("--out_dir", type=str, default="outputs/train/curriculum")
    return p.parse_args()


def main():
    args = parse_args()
    root = _repo_root()
    os.makedirs(os.path.join(root, args.out_dir), exist_ok=True)

    stage1_ckpt = os.path.join(args.out_dir, f"stage1_seed{args.seed}.pt")
    stage2_ckpt = os.path.join(args.out_dir, f"stage2_seed{args.seed}.pt")

    cmd1 = [
        sys.executable,
        "src/experiments_pgg_v0/train_ppo.py",
        "--n_episodes",
        str(args.stage1_episodes),
        "--T",
        str(args.T),
        "--n_agents",
        str(args.n_agents),
        "--F",
        *[str(v) for v in args.F_stage1],
        "--rho",
        "0.0",
        "--sigmas",
        *["0.0"] * int(args.n_agents),
        "--epsilon_tremble",
        "0.0",
        "--gamma",
        str(args.stage1_gamma),
        "--lam",
        str(args.lam),
        "--seed",
        str(args.seed),
        "--save_path",
        stage1_ckpt,
        "--log_interval",
        str(args.log_interval),
    ]
    if args.comm_enabled:
        cmd1.extend(["--comm_enabled", "--vocab_size", str(args.vocab_size)])

    cmd2 = [
        sys.executable,
        "src/experiments_pgg_v0/train_ppo.py",
        "--n_episodes",
        str(args.stage2_episodes),
        "--T",
        str(args.T),
        "--n_agents",
        str(args.n_agents),
        "--F",
        *[str(v) for v in args.F_stage2],
        "--rho",
        str(args.rho),
        "--sigmas",
        *[str(v) for v in args.sigmas],
        "--epsilon_tremble",
        str(args.epsilon_tremble),
        "--gamma",
        str(args.stage2_gamma),
        "--lam",
        str(args.lam),
        "--seed",
        str(args.seed),
        "--init_ckpt",
        stage1_ckpt,
        "--save_path",
        stage2_ckpt,
        "--log_interval",
        str(args.log_interval),
    ]
    if args.comm_enabled:
        cmd2.extend(["--comm_enabled", "--vocab_size", str(args.vocab_size)])

    _run(cmd1, cwd=root)
    _run(cmd2, cwd=root)

    print("[curriculum] done")
    print("[curriculum] stage1_ckpt=", stage1_ckpt)
    print("[curriculum] stage2_ckpt=", stage2_ckpt)


if __name__ == "__main__":
    main()
