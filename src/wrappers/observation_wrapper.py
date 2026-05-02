from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import torch


_HISTORY_MODE_CHOICES = ("full", "reduced")


def _canonical_history_mode(history_mode: str) -> str:
    mode = str(history_mode or "full").strip().lower() or "full"
    if mode not in _HISTORY_MODE_CHOICES:
        raise ValueError(
            "unknown history_mode="
            f"{history_mode!r}; expected one of: {','.join(_HISTORY_MODE_CHOICES)}"
        )
    return mode


class ObservationWrapper:
    """
    Trainer-side wrapper that builds Set-A observations from raw env tensors,
    lagged social features, and optional messages.
    """

    def __init__(
        self,
        n_agents: int,
        ewma_decay: float = 0.9,
        comm_enabled: bool = False,
        n_senders: int = 0,
        sender_ids: Optional[Iterable[str]] = None,
        vocab_size: int = 2,
        msg_dropout: float = 0.1,
        default_endowment: float = 4.0,
        default_local_cost: Optional[float] = None,
        msg_marginal_alpha: float = 0.01,
        history_mode: str = "full",
        include_eligibility: bool = False,
    ) -> None:
        self.n_agents = int(n_agents)
        self.ewma_decay = float(ewma_decay)
        self.comm_enabled = bool(comm_enabled)
        self.n_senders = int(n_senders)
        self.vocab_size = int(vocab_size)
        self.msg_dropout = float(msg_dropout)
        self.default_local_cost = (
            float(default_local_cost)
            if default_local_cost is not None
            else float(default_endowment)
        )
        self.default_endowment = self.default_local_cost
        self.msg_marginal_alpha = float(msg_marginal_alpha)
        self.history_mode = _canonical_history_mode(history_mode)
        self.include_eligibility = bool(include_eligibility)

        if sender_ids is not None:
            self.sender_ids = list(sender_ids)
        else:
            self.sender_ids = [f"agent_{i}" for i in range(self.n_senders)]

        self.last_actions: Dict[str, int] = {}
        self.last_coop_fraction: float = 0.0
        self.ewma_coop: float = 0.0
        self.msg_marginals: Dict[str, np.ndarray] = {}

    def reset(self, agent_ids: Optional[Iterable[str]] = None) -> None:
        if agent_ids is None:
            ids = [f"agent_{i}" for i in range(self.n_agents)]
        else:
            ids = list(agent_ids)

        self.last_actions = {agent_id: 0 for agent_id in ids}
        self.last_coop_fraction = 0.0
        self.ewma_coop = 0.0
        self.msg_marginals = {
            sender_id: np.ones(self.vocab_size, dtype=np.float32) / float(self.vocab_size)
            for sender_id in self.sender_ids
        }

    def _to_numpy_1d(self, raw_obs) -> np.ndarray:
        if isinstance(raw_obs, torch.Tensor):
            arr = raw_obs.detach().cpu().numpy()
        else:
            arr = np.asarray(raw_obs)
        return np.array(arr, dtype=np.float32).reshape(-1)

    def _canonicalize_raw_obs(self, raw_obs) -> Dict[str, float]:
        if isinstance(raw_obs, dict):
            need_estimate = float(raw_obs.get("need_hat", raw_obs.get("f_hat", 0.0)))
            local_cost = float(
                raw_obs.get(
                    "local_cost",
                    raw_obs.get(
                        "volunteer_cost",
                        raw_obs.get("cost", raw_obs.get("endowment", self.default_local_cost)),
                    ),
                )
            )
            eligibility = float(
                raw_obs.get("eligible", raw_obs.get("eligibility", raw_obs.get("available", 1.0)))
            )
            return {
                "need_hat": need_estimate,
                "local_cost": local_cost,
                "eligibility": eligibility,
            }

        arr = self._to_numpy_1d(raw_obs)
        if arr.size >= 2:
            eligibility = float(arr[2]) if arr.size > 2 else 1.0
            return {
                "need_hat": float(arr[0]),
                "local_cost": float(arr[1]),
                "eligibility": eligibility,
            }
        if arr.size == 1:
            return {
                "need_hat": float(arr[0]),
                "local_cost": self.default_local_cost,
                "eligibility": 1.0,
            }
        return {"need_hat": 0.0, "local_cost": self.default_local_cost, "eligibility": 1.0}

    def update(self, executed_actions: Dict[str, int]) -> None:
        if len(executed_actions) == 0:
            self.last_coop_fraction = 0.0
            return
        k = float(sum(int(v) for v in executed_actions.values()))
        denom = float(len(executed_actions))
        self.last_coop_fraction = k / denom
        self.ewma_coop = (
            self.ewma_decay * self.ewma_coop
            + (1.0 - self.ewma_decay) * self.last_coop_fraction
        )
        self.last_actions = {agent_id: int(v) for agent_id, v in executed_actions.items()}

    def update_msg_marginals(self, sender_id: str, message: int) -> None:
        if sender_id not in self.msg_marginals:
            self.msg_marginals[sender_id] = (
                np.ones(self.vocab_size, dtype=np.float32) / float(self.vocab_size)
            )
        onehot = np.zeros(self.vocab_size, dtype=np.float32)
        onehot[int(message)] = 1.0
        self.msg_marginals[sender_id] = (
            (1.0 - self.msg_marginal_alpha) * self.msg_marginals[sender_id]
            + self.msg_marginal_alpha * onehot
        )

    def apply_msg_dropout(self, messages: Dict[str, int]) -> Dict[str, int]:
        dropped = {}
        for sender_id, msg in messages.items():
            msg_int = int(msg)
            if np.random.random() < self.msg_dropout:
                probs = self.msg_marginals.get(sender_id)
                if probs is None:
                    probs = np.ones(self.vocab_size, dtype=np.float32) / float(self.vocab_size)
                dropped[sender_id] = int(np.random.choice(self.vocab_size, p=probs))
            else:
                dropped[sender_id] = msg_int
        return dropped

    def build_obs(
        self,
        agent_id: str,
        raw_env_obs,
        messages: Optional[Dict[str, int]] = None,
        f_hat_override: Optional[float] = None,
    ) -> np.ndarray:
        raw = self._canonicalize_raw_obs(raw_env_obs)
        if f_hat_override is not None:
            raw["need_hat"] = float(f_hat_override)

        obs = [
            raw["need_hat"],
            raw["local_cost"],
        ]
        if self.include_eligibility:
            obs.append(raw["eligibility"])
        obs.extend(
            [
                float(self.last_coop_fraction),
                float(self.last_actions.get(agent_id, 0)),
                float(self.ewma_coop),
            ]
        )

        if self.comm_enabled:
            messages = messages or {}
            for sender_id in self.sender_ids:
                onehot = np.zeros(self.vocab_size, dtype=np.float32)
                msg_val = int(messages.get(sender_id, 0))
                onehot[msg_val] = 1.0
                obs.extend(onehot.tolist())

        obs_arr = np.asarray(obs, dtype=np.float32)
        if self.history_mode == "reduced":
            obs_arr[self.temporal_feature_slice] = 0.0
        return obs_arr

    @property
    def obs_dim(self) -> int:
        base = self.base_feature_count
        if self.comm_enabled:
            base += len(self.sender_ids) * self.vocab_size
        return base

    @property
    def base_feature_count(self) -> int:
        return 6 if self.include_eligibility else 5

    @property
    def message_start_idx(self) -> int:
        # Base Set-A fields before message one-hot blocks.
        return self.base_feature_count

    @property
    def last_coop_idx(self) -> int:
        return 3 if self.include_eligibility else 2

    @property
    def own_last_action_idx(self) -> int:
        return 4 if self.include_eligibility else 3

    @property
    def ewma_coop_idx(self) -> int:
        return 5 if self.include_eligibility else 4

    @property
    def eligibility_idx(self) -> Optional[int]:
        return 2 if self.include_eligibility else None

    @property
    def temporal_feature_slice(self) -> slice:
        return slice(self.last_coop_idx, self.ewma_coop_idx + 1)
