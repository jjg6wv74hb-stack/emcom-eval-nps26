from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np


def _to_numpy_1d(value) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


class TrajectoryBuffer:
    """
    Fixed-length trajectory storage for PPO/GAE in multi-agent sessions.
    """

    def __init__(
        self,
        agent_ids: Iterable[str],
        T: int,
        obs_dim: int,
        value_obs_dim: Optional[int] = None,
        comm_enabled: bool = False,
        vocab_size: int = 0,
        sender_ids: Optional[Iterable[str]] = None,
    ) -> None:
        self.agent_ids = list(agent_ids)
        self.agent_to_idx = {agent_id: i for i, agent_id in enumerate(self.agent_ids)}
        self.n_agents = len(self.agent_ids)
        self.T = int(T)
        self.obs_dim = int(obs_dim)
        self.value_obs_dim = int(value_obs_dim) if value_obs_dim is not None else int(obs_dim)

        self.observations = np.zeros((self.T, self.n_agents, self.obs_dim), dtype=np.float32)
        self.value_observations = np.zeros(
            (self.T, self.n_agents, self.value_obs_dim), dtype=np.float32
        )
        self.actions = np.zeros((self.T, self.n_agents), dtype=np.int32)
        self.rewards = np.zeros((self.T, self.n_agents), dtype=np.float32)
        self.values = np.zeros((self.T, self.n_agents), dtype=np.float32)
        self.log_probs = np.zeros((self.T, self.n_agents), dtype=np.float32)
        self.dones = np.zeros((self.T,), dtype=bool)

        # Set-C / diagnostics.
        self.executed_actions = np.zeros((self.T, self.n_agents), dtype=np.int32)
        self.flips = np.zeros((self.T, self.n_agents), dtype=bool)
        self.true_f = np.zeros((self.T,), dtype=np.float32)
        self.f_hats = np.zeros((self.T, self.n_agents), dtype=np.float32)
        self.agent_rewards = np.zeros((self.T, self.n_agents), dtype=np.float32)

        # Optional communication traces.
        self.comm_enabled = bool(comm_enabled)
        self.sender_ids = list(sender_ids) if sender_ids is not None else []
        self.sender_to_idx = {sender_id: i for i, sender_id in enumerate(self.sender_ids)}
        if self.comm_enabled and len(self.sender_ids) > 0:
            n_senders = len(self.sender_ids)
            self.messages = np.zeros((self.T, n_senders), dtype=np.int32)
            self.message_actions = np.full((self.T, self.n_agents), -1, dtype=np.int32)
            self.message_log_probs = np.zeros((self.T, self.n_agents), dtype=np.float32)
        else:
            self.messages = None
            self.message_actions = None
            self.message_log_probs = None

        # Listening auxiliary signal from rollout-time distribution shift.
        self.listening_bonus = np.zeros((self.T, self.n_agents), dtype=np.float32)

        self.t = 0

    def _agent_index(self, agent_id: str) -> int:
        return self.agent_to_idx[agent_id]

    def store(
        self,
        obs: Dict[str, np.ndarray],
        actions: Dict[str, int],
        rewards: Dict[str, float],
        values: Dict[str, float],
        log_probs: Dict[str, float],
        done: bool,
        executed_actions: Dict[str, int],
        flips: Dict[str, bool],
        true_f: float,
        f_hats: Dict[str, np.ndarray],
        raw_rewards: Optional[Dict[str, float]] = None,
        value_obs: Optional[Dict[str, np.ndarray]] = None,
        messages: Optional[Dict[str, int]] = None,
        message_actions: Optional[Dict[str, int]] = None,
        message_log_probs: Optional[Dict[str, float]] = None,
        listening_bonus: Optional[Dict[str, float]] = None,
    ) -> None:
        if self.t >= self.T:
            raise IndexError("trajectory buffer overflow")

        for agent_id in self.agent_ids:
            i = self._agent_index(agent_id)
            self.observations[self.t, i] = np.asarray(obs[agent_id], dtype=np.float32)
            if value_obs is not None:
                self.value_observations[self.t, i] = np.asarray(
                    value_obs[agent_id], dtype=np.float32
                )
            else:
                self.value_observations[self.t, i] = np.asarray(obs[agent_id], dtype=np.float32)
            self.actions[self.t, i] = int(actions[agent_id])
            self.rewards[self.t, i] = float(rewards[agent_id])
            if raw_rewards is not None:
                self.agent_rewards[self.t, i] = float(raw_rewards[agent_id])
            else:
                self.agent_rewards[self.t, i] = float(rewards[agent_id])
            self.values[self.t, i] = float(values[agent_id])
            self.log_probs[self.t, i] = float(log_probs[agent_id])
            self.executed_actions[self.t, i] = int(executed_actions[agent_id])
            self.flips[self.t, i] = bool(flips[agent_id])

            raw = _to_numpy_1d(f_hats[agent_id])
            self.f_hats[self.t, i] = float(raw[0]) if raw.size > 0 else 0.0

            if listening_bonus is not None:
                self.listening_bonus[self.t, i] = float(listening_bonus.get(agent_id, 0.0))

        self.true_f[self.t] = float(true_f)
        self.dones[self.t] = bool(done)

        if self.messages is not None and messages is not None:
            for sender_id, msg in messages.items():
                s_idx = self.sender_to_idx.get(sender_id)
                if s_idx is not None:
                    self.messages[self.t, s_idx] = int(msg)

        if self.message_actions is not None and message_actions is not None:
            for agent_id, action in message_actions.items():
                i = self._agent_index(agent_id)
                self.message_actions[self.t, i] = int(action)

        if self.message_log_probs is not None and message_log_probs is not None:
            for agent_id, lp in message_log_probs.items():
                i = self._agent_index(agent_id)
                self.message_log_probs[self.t, i] = float(lp)

        self.t += 1

    def compute_gae(
        self,
        last_values: np.ndarray,
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        if self.t == 0:
            return np.zeros((0, self.n_agents), dtype=np.float32), np.zeros(
                (0, self.n_agents), dtype=np.float32
            )

        last_values = np.asarray(last_values, dtype=np.float32).reshape(self.n_agents)
        rewards = self.rewards[: self.t]
        values = self.values[: self.t]
        dones = self.dones[: self.t]

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = np.zeros((self.n_agents,), dtype=np.float32)

        for step in reversed(range(self.t)):
            if step == self.t - 1:
                next_values = last_values
                next_non_terminal = 1.0 - float(dones[step])
            else:
                next_values = values[step + 1]
                next_non_terminal = 1.0 - float(dones[step])

            delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[step] = last_gae

        returns = advantages + values
        return advantages.astype(np.float32), returns.astype(np.float32)

    def reset(self) -> None:
        self.t = 0


class VectorizedTrajectoryBuffer:
    """
    Fixed-length rollout storage for synchronous collection from multiple env replicas.
    Flattening preserves the TrajectoryBuffer interface expected by the PPO trainer.
    """

    def __init__(
        self,
        agent_ids: Iterable[str],
        T: int,
        obs_dim: int,
        n_envs: int,
        value_obs_dim: Optional[int] = None,
        comm_enabled: bool = False,
        vocab_size: int = 0,
        sender_ids: Optional[Iterable[str]] = None,
    ) -> None:
        self.agent_ids = list(agent_ids)
        self.agent_to_idx = {agent_id: i for i, agent_id in enumerate(self.agent_ids)}
        self.n_agents = len(self.agent_ids)
        self.T = int(T)
        self.n_envs = int(n_envs)
        self.obs_dim = int(obs_dim)
        self.value_obs_dim = int(value_obs_dim) if value_obs_dim is not None else int(obs_dim)
        self.comm_enabled = bool(comm_enabled)
        self.vocab_size = int(vocab_size)
        self.sender_ids = list(sender_ids) if sender_ids is not None else []
        self.sender_to_idx = {sender_id: i for i, sender_id in enumerate(self.sender_ids)}

        self.observations = np.zeros(
            (self.T, self.n_envs, self.n_agents, self.obs_dim), dtype=np.float32
        )
        self.value_observations = np.zeros(
            (self.T, self.n_envs, self.n_agents, self.value_obs_dim), dtype=np.float32
        )
        self.actions = np.zeros((self.T, self.n_envs, self.n_agents), dtype=np.int32)
        self.rewards = np.zeros((self.T, self.n_envs, self.n_agents), dtype=np.float32)
        self.values = np.zeros((self.T, self.n_envs, self.n_agents), dtype=np.float32)
        self.log_probs = np.zeros((self.T, self.n_envs, self.n_agents), dtype=np.float32)
        self.dones = np.zeros((self.T, self.n_envs), dtype=bool)

        self.executed_actions = np.zeros((self.T, self.n_envs, self.n_agents), dtype=np.int32)
        self.flips = np.zeros((self.T, self.n_envs, self.n_agents), dtype=bool)
        self.true_f = np.zeros((self.T, self.n_envs), dtype=np.float32)
        self.f_hats = np.zeros((self.T, self.n_envs, self.n_agents), dtype=np.float32)
        self.agent_rewards = np.zeros((self.T, self.n_envs, self.n_agents), dtype=np.float32)

        if self.comm_enabled and len(self.sender_ids) > 0:
            n_senders = len(self.sender_ids)
            self.messages = np.zeros((self.T, self.n_envs, n_senders), dtype=np.int32)
            self.message_actions = np.full(
                (self.T, self.n_envs, self.n_agents), -1, dtype=np.int32
            )
            self.message_log_probs = np.zeros(
                (self.T, self.n_envs, self.n_agents), dtype=np.float32
            )
        else:
            self.messages = None
            self.message_actions = None
            self.message_log_probs = None

        self.listening_bonus = np.zeros((self.T, self.n_envs, self.n_agents), dtype=np.float32)
        self.t = 0

    def store_step(
        self,
        obs_batch,
        actions_batch,
        rewards_batch,
        values_batch,
        log_probs_batch,
        done_batch,
        executed_actions_batch,
        flips_batch,
        true_f_batch,
        f_hats_batch,
        raw_rewards_batch=None,
        value_obs_batch=None,
        messages_batch=None,
        message_actions_batch=None,
        message_log_probs_batch=None,
        listening_bonus_batch=None,
    ) -> None:
        if self.t >= self.T:
            raise IndexError("trajectory buffer overflow")
        if len(obs_batch) != self.n_envs:
            raise ValueError(f"expected {self.n_envs} env observations, got {len(obs_batch)}")

        for env_idx in range(self.n_envs):
            obs = obs_batch[env_idx]
            actions = actions_batch[env_idx]
            rewards = rewards_batch[env_idx]
            values = values_batch[env_idx]
            log_probs = log_probs_batch[env_idx]
            executed_actions = executed_actions_batch[env_idx]
            flips = flips_batch[env_idx]
            f_hats = f_hats_batch[env_idx]
            raw_rewards = raw_rewards_batch[env_idx] if raw_rewards_batch is not None else None
            value_obs = value_obs_batch[env_idx] if value_obs_batch is not None else None
            messages = messages_batch[env_idx] if messages_batch is not None else None
            message_actions = (
                message_actions_batch[env_idx] if message_actions_batch is not None else None
            )
            message_log_probs = (
                message_log_probs_batch[env_idx]
                if message_log_probs_batch is not None
                else None
            )
            listening_bonus = (
                listening_bonus_batch[env_idx] if listening_bonus_batch is not None else None
            )

            for agent_id in self.agent_ids:
                agent_idx = self.agent_to_idx[agent_id]
                self.observations[self.t, env_idx, agent_idx] = np.asarray(
                    obs[agent_id], dtype=np.float32
                )
                if value_obs is not None:
                    self.value_observations[self.t, env_idx, agent_idx] = np.asarray(
                        value_obs[agent_id], dtype=np.float32
                    )
                else:
                    self.value_observations[self.t, env_idx, agent_idx] = np.asarray(
                        obs[agent_id], dtype=np.float32
                    )
                self.actions[self.t, env_idx, agent_idx] = int(actions[agent_id])
                self.rewards[self.t, env_idx, agent_idx] = float(rewards[agent_id])
                if raw_rewards is not None:
                    self.agent_rewards[self.t, env_idx, agent_idx] = float(raw_rewards[agent_id])
                else:
                    self.agent_rewards[self.t, env_idx, agent_idx] = float(rewards[agent_id])
                self.values[self.t, env_idx, agent_idx] = float(values[agent_id])
                self.log_probs[self.t, env_idx, agent_idx] = float(log_probs[agent_id])
                self.executed_actions[self.t, env_idx, agent_idx] = int(
                    executed_actions[agent_id]
                )
                self.flips[self.t, env_idx, agent_idx] = bool(flips[agent_id])

                raw = _to_numpy_1d(f_hats[agent_id])
                self.f_hats[self.t, env_idx, agent_idx] = float(raw[0]) if raw.size > 0 else 0.0

                if listening_bonus is not None:
                    self.listening_bonus[self.t, env_idx, agent_idx] = float(
                        listening_bonus.get(agent_id, 0.0)
                    )

            self.true_f[self.t, env_idx] = float(true_f_batch[env_idx])
            self.dones[self.t, env_idx] = bool(done_batch[env_idx])

            if self.messages is not None and messages is not None:
                for sender_id, msg in messages.items():
                    sender_idx = self.sender_to_idx.get(sender_id)
                    if sender_idx is not None:
                        self.messages[self.t, env_idx, sender_idx] = int(msg)

            if self.message_actions is not None and message_actions is not None:
                for agent_id, action in message_actions.items():
                    agent_idx = self.agent_to_idx[agent_id]
                    self.message_actions[self.t, env_idx, agent_idx] = int(action)

            if self.message_log_probs is not None and message_log_probs is not None:
                for agent_id, lp in message_log_probs.items():
                    agent_idx = self.agent_to_idx[agent_id]
                    self.message_log_probs[self.t, env_idx, agent_idx] = float(lp)

        self.t += 1

    def compute_gae(
        self,
        last_values: np.ndarray,
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        if self.t == 0:
            empty = np.zeros((0, self.n_agents), dtype=np.float32)
            return empty, empty

        last_values = np.asarray(last_values, dtype=np.float32).reshape(self.n_envs, self.n_agents)
        rewards = self.rewards[: self.t]
        values = self.values[: self.t]
        dones = self.dones[: self.t]

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = np.zeros((self.n_envs, self.n_agents), dtype=np.float32)

        for step in reversed(range(self.t)):
            if step == self.t - 1:
                next_values = last_values
                next_non_terminal = 1.0 - dones[step].astype(np.float32)
            else:
                next_values = values[step + 1]
                next_non_terminal = 1.0 - dones[step].astype(np.float32)

            next_non_terminal = next_non_terminal[:, None]
            delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[step] = last_gae

        returns = advantages + values
        flat_adv = advantages.reshape(self.t * self.n_envs, self.n_agents)
        flat_ret = returns.reshape(self.t * self.n_envs, self.n_agents)
        return flat_adv.astype(np.float32), flat_ret.astype(np.float32)

    def flatten(self) -> TrajectoryBuffer:
        flat_t = self.t * self.n_envs
        flat = TrajectoryBuffer(
            agent_ids=self.agent_ids,
            T=max(1, flat_t),
            obs_dim=self.obs_dim,
            value_obs_dim=self.value_obs_dim,
            comm_enabled=self.comm_enabled,
            vocab_size=self.vocab_size,
            sender_ids=self.sender_ids,
        )
        if flat_t == 0:
            flat.t = 0
            return flat

        flat.observations[:flat_t] = self.observations[: self.t].reshape(
            flat_t, self.n_agents, self.obs_dim
        )
        flat.value_observations[:flat_t] = self.value_observations[: self.t].reshape(
            flat_t, self.n_agents, self.value_obs_dim
        )
        flat.actions[:flat_t] = self.actions[: self.t].reshape(flat_t, self.n_agents)
        flat.rewards[:flat_t] = self.rewards[: self.t].reshape(flat_t, self.n_agents)
        flat.values[:flat_t] = self.values[: self.t].reshape(flat_t, self.n_agents)
        flat.log_probs[:flat_t] = self.log_probs[: self.t].reshape(flat_t, self.n_agents)
        flat.dones[:flat_t] = self.dones[: self.t].reshape(flat_t)
        flat.executed_actions[:flat_t] = self.executed_actions[: self.t].reshape(
            flat_t, self.n_agents
        )
        flat.flips[:flat_t] = self.flips[: self.t].reshape(flat_t, self.n_agents)
        flat.true_f[:flat_t] = self.true_f[: self.t].reshape(flat_t)
        flat.f_hats[:flat_t] = self.f_hats[: self.t].reshape(flat_t, self.n_agents)
        flat.agent_rewards[:flat_t] = self.agent_rewards[: self.t].reshape(flat_t, self.n_agents)
        flat.listening_bonus[:flat_t] = self.listening_bonus[: self.t].reshape(
            flat_t, self.n_agents
        )

        if flat.messages is not None and self.messages is not None:
            flat.messages[:flat_t] = self.messages[: self.t].reshape(
                flat_t, len(self.sender_ids)
            )
        if flat.message_actions is not None and self.message_actions is not None:
            flat.message_actions[:flat_t] = self.message_actions[: self.t].reshape(
                flat_t, self.n_agents
            )
        if flat.message_log_probs is not None and self.message_log_probs is not None:
            flat.message_log_probs[:flat_t] = self.message_log_probs[: self.t].reshape(
                flat_t, self.n_agents
            )

        flat.t = flat_t
        return flat

    def to_single_env_buffer(self, env_idx: int) -> TrajectoryBuffer:
        if env_idx < 0 or env_idx >= self.n_envs:
            raise IndexError(f"env_idx {env_idx} out of range for n_envs={self.n_envs}")
        buf = TrajectoryBuffer(
            agent_ids=self.agent_ids,
            T=max(1, self.t),
            obs_dim=self.obs_dim,
            value_obs_dim=self.value_obs_dim,
            comm_enabled=self.comm_enabled,
            vocab_size=self.vocab_size,
            sender_ids=self.sender_ids,
        )
        if self.t == 0:
            buf.t = 0
            return buf

        buf.observations[: self.t] = self.observations[: self.t, env_idx]
        buf.value_observations[: self.t] = self.value_observations[: self.t, env_idx]
        buf.actions[: self.t] = self.actions[: self.t, env_idx]
        buf.rewards[: self.t] = self.rewards[: self.t, env_idx]
        buf.values[: self.t] = self.values[: self.t, env_idx]
        buf.log_probs[: self.t] = self.log_probs[: self.t, env_idx]
        buf.dones[: self.t] = self.dones[: self.t, env_idx]
        buf.executed_actions[: self.t] = self.executed_actions[: self.t, env_idx]
        buf.flips[: self.t] = self.flips[: self.t, env_idx]
        buf.true_f[: self.t] = self.true_f[: self.t, env_idx]
        buf.f_hats[: self.t] = self.f_hats[: self.t, env_idx]
        buf.agent_rewards[: self.t] = self.agent_rewards[: self.t, env_idx]
        buf.listening_bonus[: self.t] = self.listening_bonus[: self.t, env_idx]

        if buf.messages is not None and self.messages is not None:
            buf.messages[: self.t] = self.messages[: self.t, env_idx]
        if buf.message_actions is not None and self.message_actions is not None:
            buf.message_actions[: self.t] = self.message_actions[: self.t, env_idx]
        if buf.message_log_probs is not None and self.message_log_probs is not None:
            buf.message_log_probs[: self.t] = self.message_log_probs[: self.t, env_idx]

        buf.t = self.t
        return buf

    def reset(self) -> None:
        self.t = 0
