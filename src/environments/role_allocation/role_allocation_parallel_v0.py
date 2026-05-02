from __future__ import annotations

import functools
import random
from typing import Any, Mapping

import numpy as np
import torch
try:
    from gym.spaces import Box, Discrete
except ImportError:  # pragma: no cover - compatibility path
    from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _config_items(config: Any):
    if isinstance(config, Mapping):
        return config.items()
    if hasattr(config, "items"):
        return config.items()
    return vars(config).items()


def env(config):
    wrapped = raw_env(config)
    if not bool(getattr(config, "fraction", False)):
        wrapped = wrappers.AssertOutOfBoundsWrapper(wrapped)
    wrapped = wrappers.OrderEnforcingWrapper(wrapped)
    return wrapped


def raw_env(config):
    return parallel_to_aec(parallel_env(config))


class parallel_env(ParallelEnv):
    """Hidden-need volunteer dilemma with role allocation.

    Action 1 means volunteer; action 0 means hold back. The latent need state is
    observed only through noisy private signals. True need, intended actions, executed
    actions, and tremble flips are logged in infos only.
    """

    metadata = {"render.modes": ["human"], "name": "role_allocation_parallel_v0"}

    def __init__(self, config):
        for key, val in _config_items(config):
            setattr(self, key, val)

        self.n_agents = int(getattr(self, "n_agents", 4))
        self.num_game_iterations = int(getattr(self, "num_game_iterations", 100))
        self.env_mode = str(getattr(self, "env_mode", "current")).strip().lower()
        if self.env_mode not in {"current", "crossed", "informant_executor"}:
            raise ValueError("env_mode must be 'current', 'crossed', or 'informant_executor'")
        self.need_levels = [float(x) for x in getattr(self, "need_levels", [0.0, 1.0])]
        if len(self.need_levels) == 0:
            self.need_levels = [0.0, 1.0]
        self.uncertainties = getattr(self, "uncertainties", [0.5] * self.n_agents)
        if self.uncertainties is not None:
            if len(self.uncertainties) != self.n_agents:
                raise ValueError("uncertainties must have one value per agent")
            self.uncertainties_dict = {
                f"agent_{idx}": float(sigma) for idx, sigma in enumerate(self.uncertainties)
            }
        else:
            self.uncertainties_dict = {f"agent_{idx}": 0.0 for idx in range(self.n_agents)}

        self.rho = float(getattr(self, "rho", 0.05))
        self.epsilon_tremble = float(getattr(self, "epsilon_tremble", 0.0))
        self.volunteer_cost = float(getattr(self, "volunteer_cost", 1.0))
        raw_cost_levels = getattr(self, "cost_levels", None)
        default_cost_mode = "balanced" if raw_cost_levels is not None else "constant"
        self.cost_mode = str(getattr(self, "cost_mode", default_cost_mode)).strip().lower()
        if raw_cost_levels is None:
            self.cost_levels = [self.volunteer_cost] * self.n_agents
        else:
            self.cost_levels = [float(x) for x in raw_cost_levels]
            if len(self.cost_levels) == 1:
                self.cost_levels = self.cost_levels * self.n_agents
            if len(self.cost_levels) != self.n_agents:
                raise ValueError("cost_levels must have one value per agent")
        if self.cost_mode not in {"constant", "balanced", "iid"}:
            raise ValueError("cost_mode must be 'constant', 'balanced', or 'iid'")
        self.benefit = float(getattr(self, "benefit", 4.0))
        self.miss_loss = float(getattr(self, "miss_loss", 2.0))
        self.redundant_penalty = float(getattr(self, "redundant_penalty", 0.0))
        self.invalid_volunteer_penalty = float(
            getattr(self, "invalid_volunteer_penalty", 0.5)
        )
        self.eligibility_prob = float(getattr(self, "eligibility_prob", 1.0))
        if not 0.0 <= self.eligibility_prob <= 1.0:
            raise ValueError("eligibility_prob must be in [0, 1]")
        self.informant_sigma = float(getattr(self, "informant_sigma", 0.3))
        self.capable_sigma = float(getattr(self, "capable_sigma", 0.1))
        self.neither_sigma = float(getattr(self, "neither_sigma", 1.0))
        self.neither_eligibility_prob = float(
            getattr(self, "neither_eligibility_prob", 0.3)
        )
        if not 0.0 <= self.neither_eligibility_prob <= 1.0:
            raise ValueError("neither_eligibility_prob must be in [0, 1]")
        raw_prohibitive_cost = getattr(self, "prohibitive_cost", None)
        self.prohibitive_cost = (
            2.0 * self.benefit
            if raw_prohibitive_cost is None
            else float(raw_prohibitive_cost)
        )
        self.target_volunteers = int(getattr(self, "target_volunteers", 1))
        self.fraction = bool(getattr(self, "fraction", False))

        self.possible_agents = [f"agent_{idx}" for idx in range(self.n_agents)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = {
            agent: idx for idx, agent in enumerate(self.possible_agents)
        }
        self.n_actions = 2
        self.obs_dim = 3 if self.env_mode in {"crossed", "informant_executor"} else 2
        self.current_need = torch.tensor([0.0], dtype=torch.float32, device=device)
        self.current_costs = {
            agent: self.volunteer_cost for agent in self.possible_agents
        }
        self.current_eligibility = {agent: 1 for agent in self.possible_agents}
        self.current_roles = {agent: "standard" for agent in self.possible_agents}
        self._last_intended_actions: dict[str, int] = {}
        self._last_executed_actions: dict[str, int] = {}
        self._last_flips: dict[str, bool] = {}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if self.fraction:
            return Box(low=np.array([0.0], dtype=np.float32), high=np.array([1.0], dtype=np.float32))
        return Discrete(self.n_actions)

    def close(self):
        pass

    def _set_current_need(self, value: float) -> None:
        self.current_need = torch.tensor([float(value)], dtype=torch.float32, device=device)

    def _sample_initial_need(self) -> None:
        self._set_current_need(random.choice(self.need_levels))

    def _update_need(self) -> None:
        if len(self.need_levels) <= 1:
            return
        if np.random.random() < self.rho:
            current = float(self.current_need.item())
            other_values = [value for value in self.need_levels if value != current]
            if other_values:
                self._set_current_need(random.choice(other_values))

    def _need_prior_mean(self) -> float:
        return float(sum(self.need_levels) / max(len(self.need_levels), 1))

    def _sample_current_roles(self) -> None:
        if self.env_mode != "informant_executor":
            self.current_roles = {agent: "standard" for agent in self.possible_agents}
            return

        roles = ["informed", "capable", "capable", "neither"]
        if self.n_agents != len(roles):
            raise ValueError("informant_executor currently requires exactly 4 agents")
        shuffled_roles = roles[:]
        random.shuffle(shuffled_roles)
        self.current_roles = {
            agent: shuffled_roles[idx] for idx, agent in enumerate(self.possible_agents)
        }

    def _sample_current_costs(self) -> None:
        if self.env_mode == "informant_executor":
            capable_costs = np.random.choice(
                np.asarray(self.cost_levels, dtype=np.float32),
                size=sum(1 for role in self.current_roles.values() if role == "capable"),
                replace=True,
            )
            capable_idx = 0
            costs: dict[str, float] = {}
            for agent in self.possible_agents:
                role = self.current_roles.get(agent, "standard")
                if role == "capable":
                    costs[agent] = float(capable_costs[capable_idx])
                    capable_idx += 1
                else:
                    costs[agent] = float(self.prohibitive_cost)
            self.current_costs = costs
            return

        if self.cost_mode == "constant":
            self.current_costs = {
                agent: self.volunteer_cost for agent in self.possible_agents
            }
            return

        if self.cost_mode == "iid":
            shuffled = np.random.choice(
                np.asarray(self.cost_levels, dtype=np.float32),
                size=self.n_agents,
                replace=True,
            )
        else:
            shuffled = np.random.permutation(
                np.asarray(self.cost_levels, dtype=np.float32)
            )
        self.current_costs = {
            agent: float(shuffled[idx]) for idx, agent in enumerate(self.possible_agents)
        }

    def _sample_current_eligibility(self) -> None:
        if self.env_mode == "informant_executor":
            self.current_eligibility = {
                agent: (
                    1
                    if self.current_roles.get(agent) == "capable"
                    else (
                        int(np.random.random() < self.neither_eligibility_prob)
                        if self.current_roles.get(agent) == "neither"
                        else 0
                    )
                )
                for agent in self.possible_agents
            }
            return

        if self.env_mode != "crossed":
            self.current_eligibility = {agent: 1 for agent in self.possible_agents}
            return

        draws = {
            agent: int(np.random.random() < self.eligibility_prob)
            for agent in self.possible_agents
        }
        need_present = float(self.current_need.item()) > 0.0
        if need_present and sum(draws.values()) == 0:
            draws[random.choice(self.possible_agents)] = 1
        self.current_eligibility = draws

    def _sample_need_hat(self, agent: str, true_need: float) -> float:
        if self.env_mode != "informant_executor":
            sigma = float(self.uncertainties_dict[agent])
            return float(np.random.normal(true_need, sigma))

        role = self.current_roles.get(agent, "standard")
        if role == "informed":
            return float(np.random.normal(true_need, self.informant_sigma))
        if role == "capable":
            return float(np.random.normal(self._need_prior_mean(), self.capable_sigma))
        if role == "neither":
            return float(np.random.normal(true_need, self.neither_sigma))
        sigma = float(self.uncertainties_dict[agent])
        return float(np.random.normal(true_need, sigma))

    def _coerce_binary_action(self, action_value) -> int:
        if isinstance(action_value, torch.Tensor):
            value = float(action_value.detach().cpu().view(-1)[0].item())
        else:
            arr = np.asarray(action_value)
            value = float(arr.reshape(-1)[0]) if arr.size else 0.0
        if self.fraction:
            return 1 if float(np.clip(value, 0.0, 1.0)) >= 0.5 else 0
        return int(round(value))

    def observe(self):
        observations = {}
        true_need = float(self.current_need.item())
        for agent in self.agents:
            need_hat = self._sample_need_hat(agent, true_need)
            observations[agent] = torch.tensor(
                (
                    [float(need_hat), float(self.current_costs[agent])]
                    if self.obs_dim == 2
                    else [
                        float(need_hat),
                        float(self.current_costs[agent]),
                        float(self.current_eligibility[agent]),
                    ]
                ),
                dtype=torch.float32,
                device=device,
            )
        return observations

    def reset(self, need_in: float | None = None):
        self.agents = self.possible_agents[:]
        self.dones = {agent: False for agent in self.agents}
        self.num_moves = 0
        if need_in is None:
            self._sample_initial_need()
        else:
            self._set_current_need(need_in)
        self._sample_current_roles()
        self._sample_current_costs()
        self._sample_current_eligibility()
        self._last_intended_actions = {}
        self._last_executed_actions = {}
        self._last_flips = {}
        return self.observe()

    def _rewards_for(
        self,
        true_need: float,
        executed_actions: dict[str, int],
        volunteer_costs: dict[str, float],
        eligibility: dict[str, int],
    ) -> dict[str, float]:
        n_volunteers = int(sum(executed_actions.values()))
        n_effective_volunteers = int(
            sum(
                int(executed_actions[agent]) * int(eligibility.get(agent, 1))
                for agent in executed_actions
            )
        )
        rewards: dict[str, float] = {}
        need_present = true_need > 0.0
        produced = need_present and n_effective_volunteers >= self.target_volunteers
        redundant = max(0, n_volunteers - self.target_volunteers)

        for agent, volunteered in executed_actions.items():
            reward = 0.0
            if need_present:
                if produced:
                    reward += self.benefit * true_need
                else:
                    reward -= self.miss_loss * true_need
            if volunteered:
                if int(eligibility.get(agent, 1)) == 1:
                    reward -= float(volunteer_costs[agent])
                else:
                    if self.env_mode == "informant_executor":
                        reward -= max(
                            float(self.invalid_volunteer_penalty),
                            float(volunteer_costs[agent]),
                        )
                    else:
                        reward -= self.invalid_volunteer_penalty
                if produced and redundant > 0:
                    reward -= self.redundant_penalty * redundant
            rewards[agent] = float(reward)
        return rewards

    def step(self, actions):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        intended_actions: dict[str, int] = {}
        executed_actions: dict[str, int] = {}
        flips: dict[str, bool] = {}
        for agent in self.agents:
            intended = self._coerce_binary_action(actions[agent])
            flip = bool(np.random.random() < self.epsilon_tremble)
            intended_actions[agent] = intended
            executed_actions[agent] = 1 - intended if flip else intended
            flips[agent] = flip

        step_need = float(self.current_need.item())
        step_costs = {
            agent: float(self.current_costs[agent]) for agent in self.agents
        }
        step_eligibility = {
            agent: int(self.current_eligibility.get(agent, 1)) for agent in self.agents
        }
        step_roles = {
            agent: str(self.current_roles.get(agent, "standard")) for agent in self.agents
        }
        rewards = self._rewards_for(
            step_need, executed_actions, step_costs, step_eligibility
        )
        n_volunteers = int(sum(executed_actions.values()))
        n_effective_volunteers = int(
            sum(
                executed_actions[agent] * step_eligibility.get(agent, 1)
                for agent in executed_actions
            )
        )
        n_invalid_volunteers = int(n_volunteers - n_effective_volunteers)
        need_present = step_need > 0.0
        feasible_need = bool(
            need_present and sum(step_eligibility.values()) >= self.target_volunteers
        )
        produced = bool(
            need_present and n_effective_volunteers >= self.target_volunteers
        )

        self.num_moves += 1
        env_done = self.num_moves >= self.num_game_iterations
        if not env_done:
            self._update_need()
            self._sample_current_roles()
            self._sample_current_costs()
            self._sample_current_eligibility()

        observations = self.observe()
        infos = {
            agent: {
                "intended_action": intended_actions[agent],
                "executed_action": executed_actions[agent],
                "flipped": flips[agent],
                "true_need": step_need,
                "volunteer_cost": step_costs[agent],
                "eligible": step_eligibility[agent],
                "need_present": need_present,
                "feasible_need": feasible_need,
                "n_volunteers": n_volunteers,
                "n_effective_volunteers": n_effective_volunteers,
                "n_invalid_volunteers": n_invalid_volunteers,
                "target_volunteers": self.target_volunteers,
                "role": step_roles[agent],
                "produced": produced,
                "redundant_volunteers": max(0, n_volunteers - self.target_volunteers),
                "redundant_effective_volunteers": max(
                    0, n_effective_volunteers - self.target_volunteers
                ),
            }
            for agent in self.agents
        }
        infos["intended_actions"] = intended_actions
        infos["executed_actions"] = executed_actions
        infos["flips"] = flips
        infos["true_need"] = step_need
        infos["volunteer_costs"] = step_costs
        infos["eligibility"] = step_eligibility
        infos["roles"] = step_roles
        infos["n_volunteers"] = n_volunteers
        infos["n_effective_volunteers"] = n_effective_volunteers
        infos["n_invalid_volunteers"] = n_invalid_volunteers
        infos["target_volunteers"] = self.target_volunteers
        infos["feasible_need"] = feasible_need
        infos["produced"] = produced

        self._last_intended_actions = intended_actions
        self._last_executed_actions = executed_actions
        self._last_flips = flips

        if env_done:
            self.agents = []
        return observations, rewards, env_done, infos
