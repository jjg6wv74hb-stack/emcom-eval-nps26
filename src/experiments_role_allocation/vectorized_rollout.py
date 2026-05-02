from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from src.algos.PPO import PPOAgentV2
from src.algos.trajectory_buffer import VectorizedTrajectoryBuffer
from src.environments.role_allocation import role_allocation_parallel_v0
from src.experiments_role_allocation.common import AGENT_IDS, RoleMetrics
from src.experiments_role_allocation.train_ppo import (
    TrainConfig,
    _comm_enabled,
    _env_config,
    _learned_messages,
    _make_wrapper,
    _sample_exogenous_messages,
    _trace_rows_for_step,
)


@dataclass
class VectorizedRolloutResult:
    buffer: VectorizedTrajectoryBuffer
    metrics: Dict[str, float]
    trace_rows: List[Dict[str, Any]]
    rollout_sec: float
    completed_episodes: int


class VectorizedRoleRunner:
    """Synchronous vectorized rollout over multiple independent env replicas."""

    def __init__(
        self,
        cfg: TrainConfig,
        *,
        num_envs: int,
        rng: np.random.Generator,
    ) -> None:
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.rng = rng
        self.envs = [
            role_allocation_parallel_v0.parallel_env(_env_config(cfg))
            for _ in range(self.num_envs)
        ]
        self.wrappers = [_make_wrapper(cfg, eval_mode=False) for _ in range(self.num_envs)]
        self.raw_obs_batch = []
        self.episode_ids = [0 for _ in range(self.num_envs)]
        self.completed_episodes = 0
        for env, wrapper in zip(self.envs, self.wrappers):
            self.raw_obs_batch.append(env.reset())
            wrapper.reset(AGENT_IDS)

    @property
    def obs_dim(self) -> int:
        return self.wrappers[0].obs_dim

    def _make_buffer(self, rollout_len: int) -> VectorizedTrajectoryBuffer:
        return VectorizedTrajectoryBuffer(
            agent_ids=AGENT_IDS,
            T=int(rollout_len),
            obs_dim=self.obs_dim,
            n_envs=self.num_envs,
            comm_enabled=_comm_enabled(self.cfg.condition),
            vocab_size=int(self.cfg.vocab_size),
            sender_ids=AGENT_IDS if _comm_enabled(self.cfg.condition) else [],
        )

    def collect(
        self,
        agents: Mapping[str, PPOAgentV2],
        *,
        rollout_len: int,
        trace_episodes: int = 0,
    ) -> VectorizedRolloutResult:
        start = time.time()
        buffer = self._make_buffer(rollout_len)
        metrics = RoleMetrics()
        trace_rows: List[Dict[str, Any]] = []
        completed_before = self.completed_episodes

        for t in range(int(rollout_len)):
            obs_batch: List[Dict[str, np.ndarray]] = []
            action_batch: List[Dict[str, int]] = []
            reward_batch: List[Dict[str, float]] = []
            scaled_reward_batch: List[Dict[str, float]] = []
            value_batch: List[Dict[str, float]] = []
            log_prob_batch: List[Dict[str, float]] = []
            done_batch: List[bool] = []
            executed_batch: List[Dict[str, int]] = []
            flips_batch: List[Dict[str, bool]] = []
            true_need_batch: List[float] = []
            f_hat_batch: List[Mapping[str, Any]] = []
            messages_batch: List[Optional[Dict[str, int]]] = []
            message_actions_batch: List[Optional[Dict[str, int]]] = []
            message_log_probs_batch: List[Optional[Dict[str, float]]] = []

            delivered_batch, source_batch, msg_log_prob_batch = self._message_batch(agents)
            obs_by_agent_batch, actions_batch, log_probs_batch, values_batch = (
                self._action_batch(agents, delivered_batch)
            )

            for env_idx, env in enumerate(self.envs):
                raw_obs = self.raw_obs_batch[env_idx]
                rewards: Dict[str, float]
                next_obs, rewards, done, infos = env.step(actions_batch[env_idx])
                scaled_rewards = {
                    agent_id: float(rewards[agent_id]) / float(self.cfg.reward_scale)
                    for agent_id in AGENT_IDS
                }

                metrics.update(infos, rewards)
                if self.episode_ids[env_idx] < int(trace_episodes):
                    trace_rows.extend(
                        _trace_rows_for_step(
                            phase="train",
                            cfg=self.cfg,
                            episode=self.episode_ids[env_idx],
                            t=t,
                            raw_obs=raw_obs,
                            obs_by_agent=obs_by_agent_batch[env_idx],
                            rewards=rewards,
                            infos=infos,
                            source_messages=source_batch[env_idx],
                            delivered_messages=delivered_batch[env_idx],
                            wrapper=self.wrappers[env_idx],
                        )
                    )

                obs_batch.append(obs_by_agent_batch[env_idx])
                action_batch.append(actions_batch[env_idx])
                reward_batch.append(rewards)
                scaled_reward_batch.append(scaled_rewards)
                value_batch.append(values_batch[env_idx])
                log_prob_batch.append(log_probs_batch[env_idx])
                done_batch.append(bool(done))
                executed_batch.append(infos["executed_actions"])
                flips_batch.append(infos["flips"])
                true_need_batch.append(float(infos["true_need"]))
                f_hat_batch.append(raw_obs)
                messages_batch.append(delivered_batch[env_idx])
                if _learned_messages(self.cfg.condition):
                    message_actions_batch.append(source_batch[env_idx])
                    message_log_probs_batch.append(msg_log_prob_batch[env_idx])
                else:
                    message_actions_batch.append(None)
                    message_log_probs_batch.append(None)

                self.wrappers[env_idx].update(infos["executed_actions"])
                if done:
                    self.completed_episodes += 1
                    self.episode_ids[env_idx] += 1
                    next_obs = env.reset()
                    self.wrappers[env_idx].reset(AGENT_IDS)
                self.raw_obs_batch[env_idx] = next_obs

            buffer.store_step(
                obs_batch=obs_batch,
                actions_batch=action_batch,
                rewards_batch=scaled_reward_batch,
                values_batch=value_batch,
                log_probs_batch=log_prob_batch,
                done_batch=done_batch,
                executed_actions_batch=executed_batch,
                flips_batch=flips_batch,
                true_f_batch=true_need_batch,
                f_hats_batch=f_hat_batch,
                raw_rewards_batch=reward_batch,
                messages_batch=messages_batch,
                message_actions_batch=message_actions_batch,
                message_log_probs_batch=message_log_probs_batch,
            )

        return VectorizedRolloutResult(
            buffer=buffer,
            metrics=metrics.summary(),
            trace_rows=trace_rows,
            rollout_sec=time.time() - start,
            completed_episodes=self.completed_episodes - completed_before,
        )

    def _message_batch(
        self,
        agents: Mapping[str, PPOAgentV2],
    ) -> tuple[
        List[Optional[Dict[str, int]]],
        List[Optional[Dict[str, int]]],
        List[Optional[Dict[str, float]]],
    ]:
        if not _comm_enabled(self.cfg.condition):
            none_batch: List[Optional[Dict[str, int]]] = [None for _ in range(self.num_envs)]
            none_log_probs: List[Optional[Dict[str, float]]] = [
                None for _ in range(self.num_envs)
            ]
            return none_batch, none_batch[:], none_log_probs

        source_batch: List[Optional[Dict[str, int]]] = [
            {} for _ in range(self.num_envs)
        ]
        log_prob_batch: List[Optional[Dict[str, float]]] = [
            {} for _ in range(self.num_envs)
        ]

        if _learned_messages(self.cfg.condition):
            for agent_id in AGENT_IDS:
                obs_stack = []
                for env_idx in range(self.num_envs):
                    sender_obs = self.wrappers[env_idx].build_obs(
                        agent_id,
                        self.raw_obs_batch[env_idx][agent_id],
                        messages=None,
                    )
                    obs_stack.append(sender_obs)
                msgs, log_probs, _entropy, _probs = agents[agent_id].sample_message_batch(
                    np.asarray(obs_stack, dtype=np.float32)
                )
                if msgs is None or log_probs is None:
                    raise RuntimeError("learned sender failed to sample batched messages")
                for env_idx in range(self.num_envs):
                    source_batch[env_idx][agent_id] = int(msgs[env_idx])
                    log_prob_batch[env_idx][agent_id] = float(log_probs[env_idx])
        else:
            for env_idx in range(self.num_envs):
                source_batch[env_idx] = _sample_exogenous_messages(
                    self.cfg.condition,
                    rng=self.rng,
                    vocab_size=int(self.cfg.vocab_size),
                )
                log_prob_batch[env_idx] = {agent_id: 0.0 for agent_id in AGENT_IDS}

        delivered_batch: List[Optional[Dict[str, int]]] = []
        for env_idx, source_messages in enumerate(source_batch):
            assert source_messages is not None
            for sender_id, msg in source_messages.items():
                self.wrappers[env_idx].update_msg_marginals(sender_id, int(msg))
            delivered_batch.append(self.wrappers[env_idx].apply_msg_dropout(source_messages))
        return delivered_batch, source_batch, log_prob_batch

    def _action_batch(
        self,
        agents: Mapping[str, PPOAgentV2],
        delivered_batch: List[Optional[Dict[str, int]]],
    ) -> tuple[
        List[Dict[str, np.ndarray]],
        List[Dict[str, int]],
        List[Dict[str, float]],
        List[Dict[str, float]],
    ]:
        obs_batch: List[Dict[str, np.ndarray]] = [{} for _ in range(self.num_envs)]
        actions_batch: List[Dict[str, int]] = [{} for _ in range(self.num_envs)]
        log_probs_batch: List[Dict[str, float]] = [{} for _ in range(self.num_envs)]
        values_batch: List[Dict[str, float]] = [{} for _ in range(self.num_envs)]

        for agent_id in AGENT_IDS:
            obs_stack = []
            for env_idx in range(self.num_envs):
                obs_vec = self.wrappers[env_idx].build_obs(
                    agent_id,
                    self.raw_obs_batch[env_idx][agent_id],
                    messages=delivered_batch[env_idx],
                )
                obs_batch[env_idx][agent_id] = obs_vec
                obs_stack.append(obs_vec)

            actions, log_probs, values, _entropy, _probs = agents[agent_id].sample_action_batch(
                np.asarray(obs_stack, dtype=np.float32)
            )
            for env_idx in range(self.num_envs):
                actions_batch[env_idx][agent_id] = int(actions[env_idx])
                log_probs_batch[env_idx][agent_id] = float(log_probs[env_idx])
                values_batch[env_idx][agent_id] = float(values[env_idx])

        return obs_batch, actions_batch, log_probs_batch, values_batch


def collect_env_only_steps(
    cfg: TrainConfig,
    *,
    num_envs: int,
    rollout_len: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    envs = [
        role_allocation_parallel_v0.parallel_env(_env_config(cfg))
        for _ in range(int(num_envs))
    ]
    raw_obs_batch = [env.reset() for env in envs]
    del raw_obs_batch
    metrics = RoleMetrics()
    start = time.time()
    completed = 0
    for _t in range(int(rollout_len)):
        for env in envs:
            actions = {
                agent_id: int(rng.integers(0, 2))
                for agent_id in AGENT_IDS
            }
            _obs, rewards, done, infos = env.step(actions)
            metrics.update(infos, rewards)
            if done:
                completed += 1
                env.reset()
    elapsed = time.time() - start
    out = metrics.summary()
    out.update(
        {
            "elapsed_sec": elapsed,
            "completed_episodes": float(completed),
            "env_steps_per_sec": float(num_envs) * float(rollout_len) / max(elapsed, 1e-12),
            "agent_steps_per_sec": (
                float(num_envs) * float(rollout_len) * float(len(AGENT_IDS)) / max(elapsed, 1e-12)
            ),
        }
    )
    return out
