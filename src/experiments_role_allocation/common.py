from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch


AGENT_IDS = [f"agent_{idx}" for idx in range(4)]


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def default_env_config() -> Dict[str, Any]:
    return {
        "n_agents": 4,
        "num_game_iterations": 100,
        "env_mode": "current",
        "need_levels": [0.0, 1.0],
        "uncertainties": [0.5, 0.5, 0.5, 0.5],
        "rho": 0.05,
        "epsilon_tremble": 0.05,
        "volunteer_cost": 1.0,
        "cost_mode": "balanced",
        "cost_levels": [0.25, 0.75, 1.25, 1.75],
        "benefit": 4.0,
        "miss_loss": 2.0,
        "redundant_penalty": 0.0,
        "eligibility_prob": 1.0,
        "invalid_volunteer_penalty": 0.5,
        "informant_sigma": 0.3,
        "capable_sigma": 0.1,
        "neither_sigma": 1.0,
        "neither_eligibility_prob": 0.3,
        "prohibitive_cost": None,
        "target_volunteers": 1,
        "fraction": False,
    }


def resolve_invalid_volunteer_penalty(
    cost_levels: Iterable[float],
    invalid_volunteer_penalty: Optional[float] = None,
) -> float:
    costs = [float(x) for x in cost_levels]
    if invalid_volunteer_penalty is not None:
        return float(invalid_volunteer_penalty)
    if not costs:
        return float(default_env_config()["invalid_volunteer_penalty"])
    return float(max(costs))


def config_for_run(
    *,
    horizon: int,
    uncertainty: float,
    cost_levels: Iterable[float],
    miss_loss: float,
    redundant_penalty: float,
    target_volunteers: int = 1,
    benefit: float = 4.0,
    rho: float = 0.05,
    epsilon_tremble: float = 0.05,
    env_mode: str = "current",
    cost_mode: str = "balanced",
    eligibility_prob: float = 1.0,
    invalid_volunteer_penalty: Optional[float] = None,
    informant_sigma: float = 0.3,
    capable_sigma: float = 0.1,
    neither_sigma: float = 1.0,
    neither_eligibility_prob: float = 0.3,
    prohibitive_cost: Optional[float] = None,
) -> Dict[str, Any]:
    cost_values = [float(x) for x in cost_levels]
    resolved_invalid_penalty = resolve_invalid_volunteer_penalty(
        cost_values,
        invalid_volunteer_penalty,
    )
    cfg = default_env_config()
    cfg.update(
        {
            "num_game_iterations": int(horizon),
            "env_mode": str(env_mode),
            "uncertainties": [float(uncertainty)] * int(cfg["n_agents"]),
            "cost_mode": str(cost_mode),
            "cost_levels": cost_values,
            "miss_loss": float(miss_loss),
            "redundant_penalty": float(redundant_penalty),
            "eligibility_prob": float(eligibility_prob),
            "invalid_volunteer_penalty": resolved_invalid_penalty,
            "informant_sigma": float(informant_sigma),
            "capable_sigma": float(capable_sigma),
            "neither_sigma": float(neither_sigma),
            "neither_eligibility_prob": float(neither_eligibility_prob),
            "prohibitive_cost": (
                None if prohibitive_cost is None else float(prohibitive_cost)
            ),
            "target_volunteers": int(target_volunteers),
            "benefit": float(benefit),
            "rho": float(rho),
            "epsilon_tremble": float(epsilon_tremble),
        }
    )
    return cfg


def to_numpy_1d(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def need_hat(raw_obs_agent: Any) -> float:
    arr = to_numpy_1d(raw_obs_agent)
    return float(arr[0]) if arr.size else 0.0


def local_cost(raw_obs_agent: Any) -> float:
    arr = to_numpy_1d(raw_obs_agent)
    return float(arr[1]) if arr.size > 1 else 0.0


def eligibility(raw_obs_agent: Any) -> float:
    arr = to_numpy_1d(raw_obs_agent)
    return float(arr[2]) if arr.size > 2 else 1.0


def lowest_cost_agent(costs: Mapping[str, float]) -> str:
    return min(costs.keys(), key=lambda agent_id: (float(costs[agent_id]), agent_id))


def lowest_eligible_cost_agent(
    costs: Mapping[str, float],
    eligibility_by_agent: Mapping[str, int],
) -> Optional[str]:
    eligible_costs = {
        agent_id: float(cost)
        for agent_id, cost in costs.items()
        if int(eligibility_by_agent.get(agent_id, 1)) == 1
    }
    if not eligible_costs:
        return None
    return lowest_cost_agent(eligible_costs)


class RoleMetrics:
    def __init__(self) -> None:
        self.total_rounds = 0
        self.need_rounds = 0
        self.feasible_need_rounds = 0
        self.no_need_rounds = 0
        self.any_volunteer_need = 0
        self.any_effective_volunteer_need = 0
        self.exactly_one_need = 0
        self.exactly_one_effective_need = 0
        self.missed_need = 0
        self.missed_service_need = 0
        self.missed_feasible_need = 0
        self.redundant_need = 0
        self.redundant_effective_need = 0
        self.any_volunteer_absent = 0
        self.lowest_cost_volunteers = 0
        self.lowest_cost_volunteers_need = 0
        self.lowest_eligible_cost_volunteers_need = 0
        self.invalid_volunteer_rounds = 0
        self.invalid_volunteers = 0
        self.welfare_sum = 0.0
        self.volunteer_cost_need_success_sum = 0.0
        self.volunteer_cost_need_success_count = 0

    def update(self, infos: Mapping[str, Any], rewards: Mapping[str, float]) -> None:
        executed = {k: int(v) for k, v in infos["executed_actions"].items()}
        costs = {k: float(v) for k, v in infos["volunteer_costs"].items()}
        eligibility_by_agent = {
            k: int(v) for k, v in infos.get("eligibility", {}).items()
        }
        if not eligibility_by_agent:
            eligibility_by_agent = {agent_id: 1 for agent_id in executed}
        n_volunteers = int(infos["n_volunteers"])
        target_volunteers = int(infos.get("target_volunteers", 1))
        n_effective_volunteers = int(
            infos.get(
                "n_effective_volunteers",
                sum(executed[agent_id] * eligibility_by_agent.get(agent_id, 1) for agent_id in executed),
            )
        )
        n_invalid_volunteers = int(
            infos.get("n_invalid_volunteers", max(0, n_volunteers - n_effective_volunteers))
        )
        need_present = bool(infos.get("need_present", float(infos["true_need"]) > 0.0))
        feasible_need = bool(infos.get("feasible_need", need_present))
        produced = bool(infos["produced"])
        low_agent = lowest_cost_agent(costs)
        low_eligible_agent = lowest_eligible_cost_agent(costs, eligibility_by_agent)

        self.total_rounds += 1
        self.welfare_sum += float(sum(float(v) for v in rewards.values()))
        self.lowest_cost_volunteers += int(executed.get(low_agent, 0) == 1)
        self.invalid_volunteer_rounds += int(n_invalid_volunteers > 0)
        self.invalid_volunteers += int(n_invalid_volunteers)

        if need_present:
            self.need_rounds += 1
            self.feasible_need_rounds += int(feasible_need)
            self.any_volunteer_need += int(n_volunteers > 0)
            self.any_effective_volunteer_need += int(n_effective_volunteers > 0)
            self.exactly_one_need += int(n_volunteers == 1)
            self.exactly_one_effective_need += int(
                n_effective_volunteers == target_volunteers
            )
            self.missed_need += int(n_volunteers == 0)
            self.missed_service_need += int(not produced)
            if feasible_need:
                self.missed_feasible_need += int(not produced)
            self.redundant_need += int(n_volunteers > 1)
            self.redundant_effective_need += int(
                n_effective_volunteers > target_volunteers
            )
            self.lowest_cost_volunteers_need += int(executed.get(low_agent, 0) == 1)
            if low_eligible_agent is not None:
                self.lowest_eligible_cost_volunteers_need += int(
                    executed.get(low_eligible_agent, 0) == 1
                )
            if produced and n_volunteers > 0:
                for agent_id, action in executed.items():
                    if int(action) == 1 and int(eligibility_by_agent.get(agent_id, 1)) == 1:
                        self.volunteer_cost_need_success_sum += float(costs[agent_id])
                        self.volunteer_cost_need_success_count += 1
        else:
            self.no_need_rounds += 1
            self.any_volunteer_absent += int(n_volunteers > 0)

    @staticmethod
    def _rate(num: float, den: float) -> float:
        return float(num) / float(den) if den else float("nan")

    def summary(self) -> Dict[str, float]:
        return {
            "rounds": float(self.total_rounds),
            "need_rounds": float(self.need_rounds),
            "feasible_need_rounds": float(self.feasible_need_rounds),
            "no_need_rounds": float(self.no_need_rounds),
            "p_any_volunteer_given_need": self._rate(
                self.any_volunteer_need, self.need_rounds
            ),
            "p_any_effective_volunteer_given_need": self._rate(
                self.any_effective_volunteer_need, self.need_rounds
            ),
            "p_exactly_one_given_need": self._rate(self.exactly_one_need, self.need_rounds),
            "p_exactly_one_effective_given_need": self._rate(
                self.exactly_one_effective_need, self.need_rounds
            ),
            "missed_need_rate": self._rate(self.missed_need, self.need_rounds),
            "missed_service_need_rate": self._rate(
                self.missed_service_need, self.need_rounds
            ),
            "missed_feasible_need_rate": self._rate(
                self.missed_feasible_need, self.feasible_need_rounds
            ),
            "redundant_volunteer_rate": self._rate(self.redundant_need, self.need_rounds),
            "redundant_effective_volunteer_rate": self._rate(
                self.redundant_effective_need, self.need_rounds
            ),
            "false_volunteer_rate": self._rate(
                self.any_volunteer_absent, self.no_need_rounds
            ),
            "invalid_volunteer_round_rate": self._rate(
                self.invalid_volunteer_rounds, self.total_rounds
            ),
            "invalid_volunteers_per_round": self._rate(
                self.invalid_volunteers, self.total_rounds
            ),
            "lowest_cost_volunteer_rate": self._rate(
                self.lowest_cost_volunteers, self.total_rounds
            ),
            "lowest_cost_volunteer_rate_given_need": self._rate(
                self.lowest_cost_volunteers_need, self.need_rounds
            ),
            "lowest_eligible_cost_volunteer_rate_given_need": self._rate(
                self.lowest_eligible_cost_volunteers_need, self.feasible_need_rounds
            ),
            "mean_welfare": self._rate(self.welfare_sum, self.total_rounds),
            "mean_volunteer_cost_when_need_produced": self._rate(
                self.volunteer_cost_need_success_sum,
                self.volunteer_cost_need_success_count,
            ),
        }


def write_csv(path: Path, rows: List[Mapping[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_ready(row), sort_keys=True) + "\n")


def json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def cost_profile(name: str) -> List[float]:
    profiles = {
        "equal": [1.0, 1.0, 1.0, 1.0],
        "narrow": [0.75, 0.9, 1.1, 1.25],
        "mild": [0.5, 0.8, 1.2, 1.5],
        "moderate": [0.5, 0.9, 1.3, 1.7],
        "current": [0.25, 0.75, 1.25, 1.75],
        "strong": [0.1, 0.6, 1.4, 1.9],
    }
    if name not in profiles:
        raise ValueError(f"unknown cost profile {name!r}; expected one of {sorted(profiles)}")
    return profiles[name]


def parse_float_list(text: Optional[str]) -> Optional[List[float]]:
    if text is None or str(text).strip() == "":
        return None
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]
