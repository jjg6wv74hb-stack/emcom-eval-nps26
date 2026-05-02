from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from src.algos.PPO import PPOTrainer
from src.experiments_role_allocation.common import (
    AGENT_IDS,
    append_jsonl,
    parse_float_list,
    seed_everything,
    write_json,
)
from src.experiments_role_allocation.train_ppo import (
    MESSAGE_SOURCE_CHOICES,
    TrainConfig,
    _comm_enabled,
    _env_config,
    _make_agents,
    _make_wrapper,
    _checkpoint_payload,
    _scheduled_hyperparams,
    _set_agent_learning_rate,
    rotated_policy_map,
    _validate_condition,
    _write_trace,
    evaluate,
)
from src.experiments_role_allocation.vectorized_rollout import VectorizedRoleRunner


@dataclass
class VectorizedTrainConfig(TrainConfig):
    total_episodes: int = 1024
    num_envs: int = 32
    rollout_len: int = 128


def train_vec(cfg: VectorizedTrainConfig) -> Dict[str, Any]:
    cfg.condition = _validate_condition(cfg.condition)
    if int(cfg.rollout_len) != int(cfg.horizon):
        raise ValueError(
            "train_ppo_vec currently requires rollout_len == horizon. "
            "Zero bootstrap values are only valid when every vectorized rollout "
            "ends at an episode boundary."
        )
    seed_everything(int(cfg.seed))

    out_dir = Path(cfg.out_dir) / f"{cfg.condition}_seed{cfg.seed}_vec{cfg.num_envs}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    template_wrapper = _make_wrapper(cfg, eval_mode=False)
    agents = _make_agents(cfg, obs_dim=template_wrapper.obs_dim)
    trainer = PPOTrainer(
        agents,
        clip_ratio=float(cfg.clip_ratio),
        value_coeff=float(cfg.value_coeff),
        entropy_coeff=float(cfg.entropy_coeff),
        msg_entropy_coeff=float(cfg.msg_entropy_coeff),
        max_grad_norm=float(cfg.max_grad_norm),
        ppo_epochs=int(cfg.ppo_epochs),
        mini_batch_size=int(cfg.mini_batch_size),
    )
    runner = VectorizedRoleRunner(
        cfg,
        num_envs=int(cfg.num_envs),
        rng=np.random.default_rng(int(cfg.seed) + 1701),
    )

    start_time = time.time()
    completed_episodes = 0
    update_idx = 0
    train_trace_rows = []
    checkpoint_interval = max(0, int(cfg.checkpoint_interval_episodes))
    next_checkpoint_target = checkpoint_interval if checkpoint_interval > 0 else None
    periodic_checkpoints = []

    while completed_episodes < int(cfg.total_episodes):
        result = runner.collect(
            agents,
            rollout_len=int(cfg.rollout_len),
            trace_episodes=int(cfg.train_trace_episodes),
        )
        flat_buffer = result.buffer.flatten()
        # rollout_len == horizon guarantees every env is terminal here, so zero
        # bootstrap values are correct. If this guard is relaxed, compute
        # value(next_obs) for non-terminal envs instead.
        last_values = np.zeros((int(cfg.num_envs), len(AGENT_IDS)), dtype=np.float32)
        advantages, returns = result.buffer.compute_gae(
            last_values=last_values,
            gamma=float(cfg.gamma),
            lam=float(cfg.gae_lambda),
        )
        schedule_values = _scheduled_hyperparams(
            cfg,
            completed_episodes=completed_episodes + int(result.completed_episodes),
            total_episodes=int(cfg.total_episodes),
        )
        trainer.entropy_coeff = float(schedule_values["entropy_coeff"])
        trainer.msg_entropy_coeff = float(schedule_values["msg_entropy_coeff"])
        _set_agent_learning_rate(agents, float(schedule_values["lr"]))
        update_start = time.time()
        ppo_metrics = trainer.update(flat_buffer, advantages, returns)
        update_sec = time.time() - update_start

        completed_episodes += int(result.completed_episodes)
        update_idx += 1
        train_trace_rows.extend(result.trace_rows)

        row = {
            "update": update_idx,
            "completed_episodes": completed_episodes,
            "target_episodes": int(cfg.total_episodes),
            "elapsed_sec": time.time() - start_time,
            "num_envs": int(cfg.num_envs),
            "rollout_len": int(cfg.rollout_len),
            "rollout_sec": float(result.rollout_sec),
            "update_sec": float(update_sec),
            "scheduled_entropy_coeff": float(schedule_values["entropy_coeff"]),
            "scheduled_msg_entropy_coeff": float(schedule_values["msg_entropy_coeff"]),
            "scheduled_lr": float(schedule_values["lr"]),
            "env_steps_per_sec": (
                float(cfg.num_envs) * float(cfg.rollout_len) / max(float(result.rollout_sec), 1e-12)
            ),
            "agent_steps_per_sec": (
                float(cfg.num_envs)
                * float(cfg.rollout_len)
                * float(len(AGENT_IDS))
                / max(float(result.rollout_sec), 1e-12)
            ),
        }
        row.update(result.metrics)
        row.update({f"ppo_{k}": v for k, v in ppo_metrics.items()})
        append_jsonl(metrics_path, row)

        while (
            next_checkpoint_target is not None
            and next_checkpoint_target <= completed_episodes
            and next_checkpoint_target <= int(cfg.total_episodes)
        ):
            checkpoint_path = out_dir / f"checkpoint_ep{next_checkpoint_target:06d}.pt"
            torch.save(
                _checkpoint_payload(
                    cfg,
                    agents,
                    completed_episodes=completed_episodes,
                    update=update_idx,
                ),
                checkpoint_path,
            )
            periodic_checkpoints.append(
                {
                    "target_episodes": int(next_checkpoint_target),
                    "completed_episodes": int(completed_episodes),
                    "update": int(update_idx),
                    "checkpoint": str(checkpoint_path),
                }
            )
            next_checkpoint_target += checkpoint_interval

    if train_trace_rows:
        _write_trace(out_dir / "train_trace.csv", train_trace_rows)

    eval_trace_path = out_dir / "eval_trace.csv"
    eval_metrics = evaluate(cfg, agents, out_trace_path=eval_trace_path)
    slot_permutation_map = rotated_policy_map()
    slot_permutation_eval_metrics = None
    slot_permutation_trace_path = None
    if bool(cfg.eval_slot_permutation):
        slot_permutation_trace_path = out_dir / "eval_slot_permuted_trace.csv"
        slot_permutation_eval_metrics = evaluate(
            cfg,
            agents,
            out_trace_path=slot_permutation_trace_path,
            policy_map=slot_permutation_map,
            trace_phase="eval_slot_permuted",
        )
    message_shuffle_eval_metrics = None
    message_shuffle_trace_path = None
    if bool(cfg.eval_message_shuffle):
        message_shuffle_trace_path = out_dir / "eval_message_shuffled_trace.csv"
        message_shuffle_eval_metrics = evaluate(
            cfg,
            agents,
            out_trace_path=message_shuffle_trace_path,
            trace_phase="eval_message_shuffled",
            message_shuffle=True,
        )

    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(
        _checkpoint_payload(
            cfg,
            agents,
            eval_metrics=eval_metrics,
            slot_permutation_policy_map=slot_permutation_map,
            slot_permutation_eval_metrics=slot_permutation_eval_metrics,
            message_shuffle_eval_metrics=message_shuffle_eval_metrics,
            completed_episodes=completed_episodes,
            update=update_idx,
        ),
        checkpoint_path,
    )
    write_json(
        out_dir / "manifest.json",
        {
            "config": asdict(cfg),
            "env_config": _env_config(cfg),
            "metrics_jsonl": str(metrics_path),
            "eval_trace_csv": str(eval_trace_path),
            "slot_permutation_eval_trace_csv": (
                None if slot_permutation_trace_path is None else str(slot_permutation_trace_path)
            ),
            "message_shuffle_eval_trace_csv": (
                None if message_shuffle_trace_path is None else str(message_shuffle_trace_path)
            ),
            "checkpoint": str(checkpoint_path),
            "periodic_checkpoints": periodic_checkpoints,
            "eval_metrics": eval_metrics,
            "slot_permutation_policy_map": slot_permutation_map,
            "slot_permutation_eval_metrics": slot_permutation_eval_metrics,
            "message_shuffle_eval_metrics": message_shuffle_eval_metrics,
        },
    )
    return {
        "out_dir": str(out_dir),
        "metrics_jsonl": str(metrics_path),
        "eval_trace_csv": str(eval_trace_path),
        "slot_permutation_eval_trace_csv": (
            None if slot_permutation_trace_path is None else str(slot_permutation_trace_path)
        ),
        "message_shuffle_eval_trace_csv": (
            None if message_shuffle_trace_path is None else str(message_shuffle_trace_path)
        ),
        "checkpoint": str(checkpoint_path),
        "eval_metrics": eval_metrics,
        "slot_permutation_eval_metrics": slot_permutation_eval_metrics,
        "message_shuffle_eval_metrics": message_shuffle_eval_metrics,
    }


def parse_args() -> VectorizedTrainConfig:
    parser = argparse.ArgumentParser(description="Vectorized role-allocation PPO trainer.")
    parser.add_argument("--condition", choices=MESSAGE_SOURCE_CHOICES, default="no_comm")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--total-episodes", type=int, default=1024)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-len", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default="outputs/train/role_allocation/vectorized")
    parser.add_argument(
        "--env-mode",
        choices=("current", "crossed", "informant_executor"),
        default="current",
    )
    parser.add_argument("--uncertainty", type=float, default=0.5)
    parser.add_argument("--cost-mode", choices=("constant", "balanced", "iid"), default="balanced")
    parser.add_argument("--cost-levels", type=str, default="0.25,0.75,1.25,1.75")
    parser.add_argument("--benefit", type=float, default=4.0)
    parser.add_argument("--miss-loss", type=float, default=2.0)
    parser.add_argument("--redundant-penalty", type=float, default=0.0)
    parser.add_argument("--eligibility-prob", type=float, default=1.0)
    parser.add_argument(
        "--invalid-volunteer-penalty",
        type=float,
        default=None,
        help="Override invalid volunteer cost. Omit to use max(cost_levels).",
    )
    parser.add_argument("--informant-sigma", type=float, default=0.3)
    parser.add_argument("--capable-sigma", type=float, default=0.1)
    parser.add_argument("--neither-sigma", type=float, default=1.0)
    parser.add_argument("--neither-eligibility-prob", type=float, default=0.3)
    parser.add_argument("--prohibitive-cost", type=float, default=None)
    parser.add_argument("--target-volunteers", type=int, default=1)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--epsilon-tremble", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coeff", type=float, default=0.5)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument("--msg-entropy-coeff", type=float, default=0.01)
    parser.add_argument("--entropy-coeff-end", type=float, default=None)
    parser.add_argument("--msg-entropy-coeff-end", type=float, default=None)
    parser.add_argument("--entropy-decay-end-fraction", type=float, default=1.0)
    parser.add_argument("--msg-entropy-decay-end-fraction", type=float, default=1.0)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--mini-batch-size", type=int, default=512)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--vocab-size", type=int, default=2)
    parser.add_argument("--msg-dropout", type=float, default=0.1)
    parser.add_argument("--ewma-decay", type=float, default=0.9)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--lr-end", type=float, default=None)
    parser.add_argument("--lr-decay-end-fraction", type=float, default=1.0)
    parser.add_argument("--train-trace-episodes", type=int, default=0)
    parser.add_argument("--stochastic-eval", action="store_true")
    parser.add_argument("--eval-slot-permutation", action="store_true")
    parser.add_argument("--eval-message-shuffle", action="store_true")
    parser.add_argument(
        "--checkpoint-interval-episodes",
        type=int,
        default=0,
        help="Save checkpoint_epXXXXXX.pt whenever training crosses this episode interval.",
    )
    args = parser.parse_args()

    cost_levels = parse_float_list(args.cost_levels)
    if cost_levels is None:
        cost_levels = [0.25, 0.75, 1.25, 1.75]
    rollout_len = int(args.rollout_len)
    horizon = rollout_len if args.horizon is None else int(args.horizon)
    return VectorizedTrainConfig(
        condition=args.condition,
        seed=args.seed,
        episodes=args.total_episodes,
        horizon=horizon,
        update_episodes=args.num_envs,
        eval_episodes=args.eval_episodes,
        out_dir=args.out_dir,
        env_mode=args.env_mode,
        uncertainty=args.uncertainty,
        cost_mode=args.cost_mode,
        cost_levels=tuple(float(x) for x in cost_levels),
        benefit=args.benefit,
        miss_loss=args.miss_loss,
        redundant_penalty=args.redundant_penalty,
        eligibility_prob=args.eligibility_prob,
        invalid_volunteer_penalty=args.invalid_volunteer_penalty,
        informant_sigma=args.informant_sigma,
        capable_sigma=args.capable_sigma,
        neither_sigma=args.neither_sigma,
        neither_eligibility_prob=args.neither_eligibility_prob,
        prohibitive_cost=args.prohibitive_cost,
        target_volunteers=args.target_volunteers,
        rho=args.rho,
        epsilon_tremble=args.epsilon_tremble,
        lr=args.lr,
        hidden_size=args.hidden_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        value_coeff=args.value_coeff,
        entropy_coeff=args.entropy_coeff,
        msg_entropy_coeff=args.msg_entropy_coeff,
        entropy_coeff_end=args.entropy_coeff_end,
        msg_entropy_coeff_end=args.msg_entropy_coeff_end,
        entropy_decay_end_fraction=args.entropy_decay_end_fraction,
        msg_entropy_decay_end_fraction=args.msg_entropy_decay_end_fraction,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        max_grad_norm=args.max_grad_norm,
        vocab_size=args.vocab_size,
        msg_dropout=args.msg_dropout,
        ewma_decay=args.ewma_decay,
        reward_scale=args.reward_scale,
        lr_end=args.lr_end,
        lr_decay_end_fraction=args.lr_decay_end_fraction,
        train_trace_episodes=args.train_trace_episodes,
        eval_greedy=not bool(args.stochastic_eval),
        eval_slot_permutation=bool(args.eval_slot_permutation),
        eval_message_shuffle=bool(args.eval_message_shuffle),
        checkpoint_interval_episodes=int(args.checkpoint_interval_episodes),
        total_episodes=args.total_episodes,
        num_envs=args.num_envs,
        rollout_len=rollout_len,
    )


def main() -> None:
    result = train_vec(parse_args())
    print(f"wrote {result['out_dir']}")
    print(f"eval_metrics {result['eval_metrics']}")


if __name__ == "__main__":
    main()
