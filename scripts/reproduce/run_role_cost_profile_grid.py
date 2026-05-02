#!/usr/bin/env python3
"""Run small role-allocation training grids across cost profiles."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from src.experiments_role_allocation.common import write_json
from src.experiments_role_allocation.train_ppo_vec import (
    MESSAGE_SOURCE_CHOICES,
    VectorizedTrainConfig,
    train_vec,
)


COST_PROFILES = {
    # Small named cost spreads used in the role-allocation probe.
    # Keeping the numbers here makes the run manifest readable after the fact.
    "equal": (1.0, 1.0, 1.0, 1.0),
    "narrow": (0.75, 0.9, 1.1, 1.25),
    "moderate": (0.5, 0.9, 1.3, 1.7),
    "wide": (0.25, 0.75, 1.25, 1.75),
}


def _expand_conditions(conditions: Iterable[str], fixed_mode: str) -> list[str]:
    # The CLI accepts "fixed" as a shortcut, but the trainer wants fixed0/fixed1.
    expanded: list[str] = []
    for condition in conditions:
        if condition == "fixed":
            if fixed_mode == "both":
                expanded.extend(["fixed0", "fixed1"])
            else:
                expanded.append(fixed_mode)
        else:
            expanded.append(condition)
    return expanded


def _parse_seeds(raw: str) -> list[int]:
    # Comma-separated seeds keep the documented shell commands short.
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_args() -> argparse.Namespace:
    # Defaults are deliberately small enough for a local smoke run.
    # Longer paper-scale commands are listed in docs/COMMANDS.md.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/role_cost_grid"))
    parser.add_argument("--profiles", nargs="+", default=["equal", "narrow", "moderate", "wide"])
    parser.add_argument("--conditions", nargs="+", default=["no_comm", "learned"])
    parser.add_argument("--fixed-mode", choices=("fixed0", "fixed1", "both"), default="both")
    parser.add_argument("--seeds", default="101")
    parser.add_argument("--total-episodes", type=int, default=1024)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--env-mode", choices=("current", "crossed", "informant_executor"), default="informant_executor")
    parser.add_argument("--checkpoint-interval-episodes", type=int, default=0)
    parser.add_argument("--eval-slot-permutation", action="store_true")
    parser.add_argument("--eval-message-shuffle", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    conditions = _expand_conditions(args.conditions, args.fixed_mode)

    # Fail before launching jobs if a shorthand expands to something unsupported.
    unknown = sorted(set(conditions) - set(MESSAGE_SOURCE_CHOICES))
    if unknown:
        raise SystemExit(f"Unknown conditions: {', '.join(unknown)}")

    results = []
    for profile_name in args.profiles:
        # Treat profile names as part of the public run contract.
        if profile_name not in COST_PROFILES:
            raise SystemExit(f"Unknown cost profile: {profile_name}")
        cost_levels = COST_PROFILES[profile_name]
        for condition in conditions:
            for seed in _parse_seeds(args.seeds):
                # Group runs by profile so failed or partial grids are easy to inspect.
                run_dir = args.out_dir / profile_name
                cfg = VectorizedTrainConfig(
                    condition=condition,
                    seed=seed,
                    total_episodes=args.total_episodes,
                    num_envs=args.num_envs,
                    # The vectorized trainer currently expects one full episode per rollout.
                    rollout_len=args.horizon,
                    horizon=args.horizon,
                    eval_episodes=args.eval_episodes,
                    out_dir=str(run_dir),
                    env_mode=args.env_mode,
                    cost_levels=tuple(float(x) for x in cost_levels),
                    checkpoint_interval_episodes=args.checkpoint_interval_episodes,
                    eval_slot_permutation=bool(args.eval_slot_permutation),
                    eval_message_shuffle=bool(args.eval_message_shuffle),
                )
                result = train_vec(cfg)
                results.append(
                    {
                        # Save enough metadata to reconstruct what each run actually used.
                        "profile": profile_name,
                        "condition": condition,
                        "seed": seed,
                        "config": asdict(cfg),
                        "result": result,
                    }
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "grid_manifest.json", {"runs": results})


if __name__ == "__main__":
    main()
