from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

from src.algos.PPO import PPOAgentV2, PPOTrainer, device
from src.algos.trajectory_buffer import TrajectoryBuffer
from src.environments.role_allocation import role_allocation_parallel_v0
from src.experiments_role_allocation.common import (
    AGENT_IDS,
    RoleMetrics,
    append_jsonl,
    config_for_run,
    eligibility,
    local_cost,
    lowest_eligible_cost_agent,
    lowest_cost_agent,
    need_hat,
    parse_float_list,
    seed_everything,
    write_csv,
    write_json,
)
from src.wrappers import ObservationWrapper


MESSAGE_SOURCE_CHOICES = ("no_comm", "learned", "uniform", "public_random", "fixed0", "fixed1")


@dataclass
class TrainConfig:
    condition: str = "no_comm"
    seed: int = 101
    episodes: int = 100
    horizon: int = 100
    update_episodes: int = 4
    eval_episodes: int = 10
    out_dir: str = "outputs/train/role_allocation/smoke"
    env_mode: str = "current"
    uncertainty: float = 0.5
    cost_mode: str = "balanced"
    cost_levels: Tuple[float, ...] = (0.25, 0.75, 1.25, 1.75)
    benefit: float = 4.0
    miss_loss: float = 2.0
    redundant_penalty: float = 0.0
    eligibility_prob: float = 1.0
    invalid_volunteer_penalty: Optional[float] = None
    informant_sigma: float = 0.3
    capable_sigma: float = 0.1
    neither_sigma: float = 1.0
    neither_eligibility_prob: float = 0.3
    prohibitive_cost: Optional[float] = None
    target_volunteers: int = 1
    rho: float = 0.05
    epsilon_tremble: float = 0.05
    lr: float = 3e-4
    hidden_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    msg_entropy_coeff: float = 0.01
    entropy_coeff_end: Optional[float] = None
    msg_entropy_coeff_end: Optional[float] = None
    entropy_decay_end_fraction: float = 1.0
    msg_entropy_decay_end_fraction: float = 1.0
    ppo_epochs: int = 4
    mini_batch_size: int = 512
    max_grad_norm: float = 0.5
    vocab_size: int = 2
    msg_dropout: float = 0.1
    ewma_decay: float = 0.9
    reward_scale: float = 1.0
    lr_end: Optional[float] = None
    lr_decay_end_fraction: float = 1.0
    train_trace_episodes: int = 0
    eval_greedy: bool = True
    eval_slot_permutation: bool = False
    eval_message_shuffle: bool = False
    checkpoint_interval_episodes: int = 0


def _train_config_payload(cfg: TrainConfig) -> Dict[str, Any]:
    return {field.name: getattr(cfg, field.name) for field in fields(TrainConfig)}


def _validate_condition(condition: str) -> str:
    mode = str(condition).strip().lower()
    if mode not in MESSAGE_SOURCE_CHOICES:
        raise ValueError(
            f"unknown condition={condition!r}; expected one of {','.join(MESSAGE_SOURCE_CHOICES)}"
        )
    return mode


def _linear_schedule_value(
    start: float,
    end: Optional[float],
    *,
    progress: float,
    decay_end_fraction: float,
) -> float:
    if end is None:
        return float(start)
    frac = max(float(decay_end_fraction), 1e-8)
    phase = min(max(float(progress), 0.0) / frac, 1.0)
    return float(start) + phase * (float(end) - float(start))


def _set_agent_learning_rate(agents: Mapping[str, PPOAgentV2], lr: float) -> None:
    for agent in agents.values():
        for group in agent.optimizer.param_groups:
            group["lr"] = float(lr)


def _scheduled_hyperparams(
    cfg: TrainConfig,
    *,
    completed_episodes: int,
    total_episodes: int,
) -> Dict[str, float]:
    progress = 0.0 if int(total_episodes) <= 0 else float(completed_episodes) / float(total_episodes)
    return {
        "entropy_coeff": _linear_schedule_value(
            float(cfg.entropy_coeff),
            cfg.entropy_coeff_end,
            progress=progress,
            decay_end_fraction=float(cfg.entropy_decay_end_fraction),
        ),
        "msg_entropy_coeff": _linear_schedule_value(
            float(cfg.msg_entropy_coeff),
            cfg.msg_entropy_coeff_end,
            progress=progress,
            decay_end_fraction=float(cfg.msg_entropy_decay_end_fraction),
        ),
        "lr": _linear_schedule_value(
            float(cfg.lr),
            cfg.lr_end,
            progress=progress,
            decay_end_fraction=float(cfg.lr_decay_end_fraction),
        ),
    }


def _env_config(cfg: TrainConfig) -> Dict[str, Any]:
    return config_for_run(
        horizon=int(cfg.horizon),
        uncertainty=float(cfg.uncertainty),
        cost_levels=list(cfg.cost_levels),
        miss_loss=float(cfg.miss_loss),
        redundant_penalty=float(cfg.redundant_penalty),
        target_volunteers=int(cfg.target_volunteers),
        benefit=float(cfg.benefit),
        rho=float(cfg.rho),
        epsilon_tremble=float(cfg.epsilon_tremble),
        env_mode=str(cfg.env_mode),
        cost_mode=str(cfg.cost_mode),
        eligibility_prob=float(cfg.eligibility_prob),
        invalid_volunteer_penalty=cfg.invalid_volunteer_penalty,
        informant_sigma=float(cfg.informant_sigma),
        capable_sigma=float(cfg.capable_sigma),
        neither_sigma=float(cfg.neither_sigma),
        neither_eligibility_prob=float(cfg.neither_eligibility_prob),
        prohibitive_cost=cfg.prohibitive_cost,
    )


def _comm_enabled(condition: str) -> bool:
    return _validate_condition(condition) != "no_comm"


def _learned_messages(condition: str) -> bool:
    return _validate_condition(condition) == "learned"


def rotated_policy_map(offset: int = 1) -> Dict[str, str]:
    """Map each acting slot to a different trained policy for slot-sensitivity eval."""
    n_agents = len(AGENT_IDS)
    return {
        agent_id: AGENT_IDS[(idx + int(offset)) % n_agents]
        for idx, agent_id in enumerate(AGENT_IDS)
    }


def _policy_id_for(
    acting_agent_id: str,
    policy_map: Optional[Mapping[str, str]],
) -> str:
    if policy_map is None:
        return acting_agent_id
    policy_id = str(policy_map.get(acting_agent_id, acting_agent_id))
    if policy_id not in AGENT_IDS:
        raise ValueError(f"policy_map for {acting_agent_id!r} points to unknown policy {policy_id!r}")
    return policy_id


def _make_wrapper(cfg: TrainConfig, *, eval_mode: bool = False) -> ObservationWrapper:
    comm_enabled = _comm_enabled(cfg.condition)
    return ObservationWrapper(
        n_agents=4,
        ewma_decay=float(cfg.ewma_decay),
        comm_enabled=comm_enabled,
        n_senders=4 if comm_enabled else 0,
        sender_ids=AGENT_IDS if comm_enabled else [],
        vocab_size=int(cfg.vocab_size),
        msg_dropout=0.0 if eval_mode else float(cfg.msg_dropout),
        include_eligibility=(
            str(cfg.env_mode).strip().lower() in {"crossed", "informant_executor"}
        ),
    )


def _make_agents(cfg: TrainConfig, obs_dim: int) -> Dict[str, PPOAgentV2]:
    can_send = _learned_messages(cfg.condition)
    return {
        agent_id: PPOAgentV2(
            obs_dim=obs_dim,
            action_size=2,
            can_send=can_send,
            vocab_size=int(cfg.vocab_size),
            hidden_size=int(cfg.hidden_size),
            lr=float(cfg.lr),
        )
        for agent_id in AGENT_IDS
    }


def _sample_exogenous_messages(
    condition: str,
    rng: np.random.Generator,
    vocab_size: int,
) -> Dict[str, int]:
    mode = _validate_condition(condition)
    if mode == "uniform":
        return {agent_id: int(rng.integers(0, int(vocab_size))) for agent_id in AGENT_IDS}
    if mode == "public_random":
        shared = int(rng.integers(0, int(vocab_size)))
        return {agent_id: shared for agent_id in AGENT_IDS}
    if mode == "fixed0":
        return {agent_id: 0 for agent_id in AGENT_IDS}
    if mode == "fixed1":
        if int(vocab_size) < 2:
            raise ValueError("fixed1 requires vocab_size >= 2")
        return {agent_id: 1 for agent_id in AGENT_IDS}
    raise ValueError(f"condition={condition!r} has no exogenous message source")


def _greedy_action(agent: PPOAgentV2, obs: np.ndarray) -> int:
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    with torch.no_grad():
        probs = agent.action_distribution(obs_t)
    return int(torch.argmax(probs, dim=-1).item())


def _greedy_message(agent: PPOAgentV2, obs: np.ndarray) -> int:
    if not agent.can_send or agent.message_actor is None:
        return 0
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = agent.message_actor(obs_t)
    return int(torch.argmax(logits, dim=-1).item())


def _message_step(
    *,
    cfg: TrainConfig,
    agents: Dict[str, PPOAgentV2],
    wrapper: ObservationWrapper,
    raw_obs: Mapping[str, Any],
    rng: np.random.Generator,
    greedy: bool,
    policy_map: Optional[Mapping[str, str]] = None,
) -> Tuple[Optional[Dict[str, int]], Optional[Dict[str, int]], Optional[Dict[str, float]]]:
    if not _comm_enabled(cfg.condition):
        return None, None, None

    source_messages: Dict[str, int] = {}
    message_log_probs: Dict[str, float] = {}

    if _learned_messages(cfg.condition):
        for agent_id in AGENT_IDS:
            sender_obs = wrapper.build_obs(agent_id, raw_obs[agent_id], messages=None)
            policy_id = _policy_id_for(agent_id, policy_map)
            if greedy:
                msg = _greedy_message(agents[policy_id], sender_obs)
                log_prob = 0.0
            else:
                msg, log_prob, _entropy, _probs = agents[policy_id].sample_message(sender_obs)
                if msg is None or log_prob is None:
                    raise RuntimeError("learned sender failed to sample a message")
            source_messages[agent_id] = int(msg)
            message_log_probs[agent_id] = float(log_prob)
    else:
        source_messages = _sample_exogenous_messages(
            cfg.condition, rng=rng, vocab_size=int(cfg.vocab_size)
        )
        message_log_probs = {agent_id: 0.0 for agent_id in AGENT_IDS}

    for sender_id, msg in source_messages.items():
        wrapper.update_msg_marginals(sender_id, int(msg))

    delivered = source_messages if greedy else wrapper.apply_msg_dropout(source_messages)
    return delivered, source_messages, message_log_probs


def _action_step(
    *,
    cfg: TrainConfig,
    agents: Dict[str, PPOAgentV2],
    wrapper: ObservationWrapper,
    raw_obs: Mapping[str, Any],
    delivered_messages: Optional[Dict[str, int]],
    greedy: bool,
    policy_map: Optional[Mapping[str, str]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int], Dict[str, float], Dict[str, float]]:
    obs_by_agent: Dict[str, np.ndarray] = {}
    actions: Dict[str, int] = {}
    log_probs: Dict[str, float] = {}
    values: Dict[str, float] = {}
    for agent_id in AGENT_IDS:
        obs_vec = wrapper.build_obs(agent_id, raw_obs[agent_id], messages=delivered_messages)
        obs_by_agent[agent_id] = obs_vec
        policy_id = _policy_id_for(agent_id, policy_map)
        if greedy:
            actions[agent_id] = _greedy_action(agents[policy_id], obs_vec)
            log_probs[agent_id] = 0.0
            with torch.no_grad():
                value = agents[policy_id].value(
                    torch.tensor(obs_vec, dtype=torch.float32, device=device)
                )
            values[agent_id] = float(value.item())
        else:
            action, log_prob, value, _entropy, _probs = agents[policy_id].sample_action(obs_vec)
            actions[agent_id] = int(action)
            log_probs[agent_id] = float(log_prob)
            values[agent_id] = float(value)
    return obs_by_agent, actions, log_probs, values


def _collect_eval_message_pool(
    cfg: TrainConfig,
    agents: Dict[str, PPOAgentV2],
    *,
    policy_map: Optional[Mapping[str, str]] = None,
) -> List[Dict[str, int]]:
    """Collect same-policy eval messages for cross-episode shuffle diagnostics."""
    if not _comm_enabled(cfg.condition):
        return []

    seed_everything(int(cfg.seed) + 12345)
    env = role_allocation_parallel_v0.parallel_env(_env_config(cfg))
    wrapper = _make_wrapper(cfg, eval_mode=True)
    rng = np.random.default_rng(int(cfg.seed) + 9001)
    pool: List[Dict[str, int]] = []

    for _episode in range(int(cfg.eval_episodes)):
        raw_obs = env.reset()
        wrapper.reset(AGENT_IDS)
        done = False
        while not done:
            delivered, source, _msg_log_probs = _message_step(
                cfg=cfg,
                agents=agents,
                wrapper=wrapper,
                raw_obs=raw_obs,
                rng=rng,
                greedy=bool(cfg.eval_greedy),
                policy_map=policy_map,
            )
            if source is not None:
                pool.append(dict(source))
            _obs_by_agent, actions, _log_probs, _values = _action_step(
                cfg=cfg,
                agents=agents,
                wrapper=wrapper,
                raw_obs=raw_obs,
                delivered_messages=delivered,
                greedy=bool(cfg.eval_greedy),
                policy_map=policy_map,
            )
            next_obs, _rewards, done, infos = env.step(actions)
            wrapper.update(infos["executed_actions"])
            raw_obs = next_obs
    return pool


TRACE_FIELDNAMES = [
    "phase",
    "condition",
    "seed",
    "episode",
    "t",
    "agent_id",
    "true_need",
    "need_present",
    "need_hat",
    "local_cost",
    "local_eligible",
    "local_role",
    "cost_agent_0",
    "cost_agent_1",
    "cost_agent_2",
    "cost_agent_3",
    "eligible_agent_0",
    "eligible_agent_1",
    "eligible_agent_2",
    "eligible_agent_3",
    "role_agent_0",
    "role_agent_1",
    "role_agent_2",
    "role_agent_3",
    "lowest_cost_agent",
    "is_lowest_cost_agent",
    "lowest_cost_agent_volunteered",
    "lowest_eligible_cost_agent",
    "is_lowest_eligible_cost_agent",
    "lowest_eligible_cost_agent_volunteered",
    "intended_action",
    "executed_action",
    "flipped",
    "reward",
    "round_welfare",
    "n_volunteers",
    "n_effective_volunteers",
    "n_invalid_volunteers",
    "feasible_need",
    "produced",
    "redundant_volunteers",
    "redundant_effective_volunteers",
    "obs_last_volunteer_fraction",
    "obs_own_last_action",
    "obs_ewma_volunteer",
    "own_sent_msg",
    "delivered_msg_agent_0",
    "delivered_msg_agent_1",
    "delivered_msg_agent_2",
    "delivered_msg_agent_3",
]


def _trace_rows_for_step(
    *,
    phase: str,
    cfg: TrainConfig,
    episode: int,
    t: int,
    raw_obs: Mapping[str, Any],
    obs_by_agent: Mapping[str, np.ndarray],
    rewards: Mapping[str, float],
    infos: Mapping[str, Any],
    source_messages: Optional[Dict[str, int]],
    delivered_messages: Optional[Dict[str, int]],
    wrapper: ObservationWrapper,
) -> List[Dict[str, Any]]:
    costs = {agent_id: float(infos["volunteer_costs"][agent_id]) for agent_id in AGENT_IDS}
    eligibility_by_agent = {
        agent_id: int(infos.get("eligibility", {}).get(agent_id, 1))
        for agent_id in AGENT_IDS
    }
    roles_by_agent = {
        agent_id: str(infos.get("roles", {}).get(agent_id, "standard"))
        for agent_id in AGENT_IDS
    }
    low_agent = lowest_cost_agent(costs)
    low_eligible_agent = lowest_eligible_cost_agent(costs, eligibility_by_agent)
    executed = {agent_id: int(infos["executed_actions"][agent_id]) for agent_id in AGENT_IDS}
    rows = []
    for agent_id in AGENT_IDS:
        obs_vec = obs_by_agent[agent_id]
        row = {
            "phase": phase,
            "condition": cfg.condition,
            "seed": int(cfg.seed),
            "episode": int(episode),
            "t": int(t),
            "agent_id": agent_id,
            "true_need": float(infos["true_need"]),
            "need_present": int(float(infos["true_need"]) > 0.0),
            "need_hat": need_hat(raw_obs[agent_id]),
            "local_cost": local_cost(raw_obs[agent_id]),
            "local_eligible": eligibility(raw_obs[agent_id]),
            "local_role": roles_by_agent[agent_id],
            "cost_agent_0": costs["agent_0"],
            "cost_agent_1": costs["agent_1"],
            "cost_agent_2": costs["agent_2"],
            "cost_agent_3": costs["agent_3"],
            "eligible_agent_0": eligibility_by_agent["agent_0"],
            "eligible_agent_1": eligibility_by_agent["agent_1"],
            "eligible_agent_2": eligibility_by_agent["agent_2"],
            "eligible_agent_3": eligibility_by_agent["agent_3"],
            "role_agent_0": roles_by_agent["agent_0"],
            "role_agent_1": roles_by_agent["agent_1"],
            "role_agent_2": roles_by_agent["agent_2"],
            "role_agent_3": roles_by_agent["agent_3"],
            "lowest_cost_agent": low_agent,
            "is_lowest_cost_agent": int(agent_id == low_agent),
            "lowest_cost_agent_volunteered": int(executed.get(low_agent, 0) == 1),
            "lowest_eligible_cost_agent": "" if low_eligible_agent is None else low_eligible_agent,
            "is_lowest_eligible_cost_agent": int(agent_id == low_eligible_agent),
            "lowest_eligible_cost_agent_volunteered": (
                ""
                if low_eligible_agent is None
                else int(executed.get(low_eligible_agent, 0) == 1)
            ),
            "intended_action": int(infos["intended_actions"][agent_id]),
            "executed_action": executed[agent_id],
            "flipped": int(bool(infos["flips"][agent_id])),
            "reward": float(rewards[agent_id]),
            "round_welfare": float(sum(float(v) for v in rewards.values())),
            "n_volunteers": int(infos["n_volunteers"]),
            "n_effective_volunteers": int(infos.get("n_effective_volunteers", infos["n_volunteers"])),
            "n_invalid_volunteers": int(infos.get("n_invalid_volunteers", 0)),
            "feasible_need": int(bool(infos.get("feasible_need", infos["true_need"] > 0.0))),
            "produced": int(bool(infos["produced"])),
            "redundant_volunteers": int(infos[agent_id]["redundant_volunteers"]),
            "redundant_effective_volunteers": int(
                infos[agent_id].get("redundant_effective_volunteers", 0)
            ),
            "obs_last_volunteer_fraction": float(obs_vec[wrapper.last_coop_idx]),
            "obs_own_last_action": float(obs_vec[wrapper.own_last_action_idx]),
            "obs_ewma_volunteer": float(obs_vec[wrapper.ewma_coop_idx]),
            "own_sent_msg": (
                ""
                if source_messages is None or agent_id not in source_messages
                else int(source_messages[agent_id])
            ),
        }
        for sender_id in AGENT_IDS:
            row[f"delivered_msg_{sender_id}"] = (
                ""
                if delivered_messages is None or sender_id not in delivered_messages
                else int(delivered_messages[sender_id])
            )
        rows.append(row)
    return rows


def _write_trace(path: Path, rows: List[Mapping[str, Any]]) -> None:
    write_csv(path, rows, TRACE_FIELDNAMES)


def _state_dict_for_save(agents: Mapping[str, PPOAgentV2]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for agent_id, agent in agents.items():
        out[agent_id] = {
            "action_actor": agent.action_actor.state_dict(),
            "value_net": agent.value_net.state_dict(),
            "message_actor": (
                None if agent.message_actor is None else agent.message_actor.state_dict()
            ),
        }
    return out


def _checkpoint_payload(
    cfg: TrainConfig,
    agents: Mapping[str, PPOAgentV2],
    *,
    eval_metrics: Optional[Dict[str, float]] = None,
    slot_permutation_policy_map: Optional[Mapping[str, str]] = None,
    slot_permutation_eval_metrics: Optional[Dict[str, float]] = None,
    message_shuffle_eval_metrics: Optional[Dict[str, float]] = None,
    completed_episodes: Optional[int] = None,
    update: Optional[int] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "config": asdict(cfg),
        "env_config": _env_config(cfg),
        "agents": _state_dict_for_save(agents),
        "eval_metrics": eval_metrics,
        "slot_permutation_policy_map": slot_permutation_policy_map,
        "slot_permutation_eval_metrics": slot_permutation_eval_metrics,
        "message_shuffle_eval_metrics": message_shuffle_eval_metrics,
    }
    if completed_episodes is not None:
        payload["completed_episodes"] = int(completed_episodes)
    if update is not None:
        payload["update"] = int(update)
    return payload


def evaluate(
    cfg: TrainConfig,
    agents: Dict[str, PPOAgentV2],
    *,
    out_trace_path: Optional[Path] = None,
    policy_map: Optional[Mapping[str, str]] = None,
    trace_phase: str = "eval",
    message_shuffle: bool = False,
) -> Dict[str, float]:
    eval_cfg = TrainConfig(**_train_config_payload(cfg))
    eval_cfg.epsilon_tremble = 0.0 if bool(cfg.eval_greedy) else float(cfg.epsilon_tremble)
    message_pool: List[Dict[str, int]] = []
    if bool(message_shuffle):
        message_pool = _collect_eval_message_pool(
            eval_cfg,
            agents,
            policy_map=policy_map,
        )
    # Re-seed numpy/python/torch global RNGs so repeated evaluate() calls (e.g. the
    # default pass and the slot-permuted pass) see identical environment
    # realizations. Without this, the first pass consumes RNG state that the
    # second pass then sees shifted, confounding the slot-permutation diagnostic
    # with ordinary eval-episode noise.
    seed_everything(int(cfg.seed) + 12345)
    env = role_allocation_parallel_v0.parallel_env(_env_config(eval_cfg))
    wrapper = _make_wrapper(eval_cfg, eval_mode=True)
    rng = np.random.default_rng(int(cfg.seed) + 9001)
    metrics = RoleMetrics()
    trace_rows: List[Dict[str, Any]] = []
    message_step_idx = 0
    shuffle_offset = int(cfg.horizon) if len(message_pool) > int(cfg.horizon) else 1

    for episode in range(int(cfg.eval_episodes)):
        raw_obs = env.reset()
        wrapper.reset(AGENT_IDS)
        done = False
        t = 0
        while not done:
            delivered, source, _msg_log_probs = _message_step(
                cfg=eval_cfg,
                agents=agents,
                wrapper=wrapper,
                raw_obs=raw_obs,
                rng=rng,
                greedy=bool(cfg.eval_greedy),
                policy_map=policy_map,
            )
            delivered_for_action = delivered
            if message_pool:
                delivered_for_action = dict(
                    message_pool[(message_step_idx + shuffle_offset) % len(message_pool)]
                )
            obs_by_agent, actions, _log_probs, _values = _action_step(
                cfg=eval_cfg,
                agents=agents,
                wrapper=wrapper,
                raw_obs=raw_obs,
                delivered_messages=delivered_for_action,
                greedy=bool(cfg.eval_greedy),
                policy_map=policy_map,
            )
            next_obs, rewards, done, infos = env.step(actions)
            metrics.update(infos, rewards)
            if out_trace_path is not None:
                trace_rows.extend(
                    _trace_rows_for_step(
                        phase=trace_phase,
                        cfg=eval_cfg,
                        episode=episode,
                        t=t,
                        raw_obs=raw_obs,
                        obs_by_agent=obs_by_agent,
                        rewards=rewards,
                        infos=infos,
                        source_messages=source,
                        delivered_messages=delivered_for_action,
                        wrapper=wrapper,
                    )
                )
            wrapper.update(infos["executed_actions"])
            raw_obs = next_obs
            t += 1
            message_step_idx += 1

    if out_trace_path is not None:
        _write_trace(out_trace_path, trace_rows)
    return metrics.summary()


def train(cfg: TrainConfig) -> Dict[str, Any]:
    cfg.condition = _validate_condition(cfg.condition)
    seed_everything(int(cfg.seed))

    out_dir = Path(cfg.out_dir) / f"{cfg.condition}_seed{cfg.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    env = role_allocation_parallel_v0.parallel_env(_env_config(cfg))
    wrapper = _make_wrapper(cfg, eval_mode=False)
    wrapper.reset(AGENT_IDS)
    agents = _make_agents(cfg, obs_dim=wrapper.obs_dim)
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

    train_trace_rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(int(cfg.seed) + 17)
    start_time = time.time()
    episode = 0
    update_idx = 0
    checkpoint_interval = max(0, int(cfg.checkpoint_interval_episodes))
    next_checkpoint_target = checkpoint_interval if checkpoint_interval > 0 else None
    periodic_checkpoints: List[Dict[str, Any]] = []

    while episode < int(cfg.episodes):
        batch_episodes = min(int(cfg.update_episodes), int(cfg.episodes) - episode)
        buffer = TrajectoryBuffer(
            agent_ids=AGENT_IDS,
            T=max(1, batch_episodes * int(cfg.horizon)),
            obs_dim=wrapper.obs_dim,
            comm_enabled=_comm_enabled(cfg.condition),
            vocab_size=int(cfg.vocab_size),
            sender_ids=AGENT_IDS if _comm_enabled(cfg.condition) else [],
        )
        rollout_metrics = RoleMetrics()
        batch_start_episode = episode

        for _ in range(batch_episodes):
            raw_obs = env.reset()
            wrapper.reset(AGENT_IDS)
            done = False
            t = 0
            while not done:
                delivered, source, msg_log_probs = _message_step(
                    cfg=cfg,
                    agents=agents,
                    wrapper=wrapper,
                    raw_obs=raw_obs,
                    rng=rng,
                    greedy=False,
                )
                obs_by_agent, actions, log_probs, values = _action_step(
                    cfg=cfg,
                    agents=agents,
                    wrapper=wrapper,
                    raw_obs=raw_obs,
                    delivered_messages=delivered,
                    greedy=False,
                )
                next_obs, raw_rewards, done, infos = env.step(actions)
                scaled_rewards = {
                    agent_id: float(raw_rewards[agent_id]) / float(cfg.reward_scale)
                    for agent_id in AGENT_IDS
                }
                buffer.store(
                    obs=obs_by_agent,
                    actions=actions,
                    rewards=scaled_rewards,
                    values=values,
                    log_probs=log_probs,
                    done=done,
                    executed_actions=infos["executed_actions"],
                    flips=infos["flips"],
                    true_f=float(infos["true_need"]),
                    f_hats=raw_obs,
                    raw_rewards=raw_rewards,
                    messages=delivered,
                    message_actions=source if _learned_messages(cfg.condition) else None,
                    message_log_probs=msg_log_probs if _learned_messages(cfg.condition) else None,
                )
                rollout_metrics.update(infos, raw_rewards)
                if episode < int(cfg.train_trace_episodes):
                    train_trace_rows.extend(
                        _trace_rows_for_step(
                            phase="train",
                            cfg=cfg,
                            episode=episode,
                            t=t,
                            raw_obs=raw_obs,
                            obs_by_agent=obs_by_agent,
                            rewards=raw_rewards,
                            infos=infos,
                            source_messages=source,
                            delivered_messages=delivered,
                            wrapper=wrapper,
                        )
                    )
                wrapper.update(infos["executed_actions"])
                raw_obs = next_obs
                t += 1
            episode += 1

        last_values = np.zeros((len(AGENT_IDS),), dtype=np.float32)
        advantages, returns = buffer.compute_gae(
            last_values=last_values,
            gamma=float(cfg.gamma),
            lam=float(cfg.gae_lambda),
        )
        schedule_values = _scheduled_hyperparams(
            cfg,
            completed_episodes=episode,
            total_episodes=int(cfg.episodes),
        )
        trainer.entropy_coeff = float(schedule_values["entropy_coeff"])
        trainer.msg_entropy_coeff = float(schedule_values["msg_entropy_coeff"])
        _set_agent_learning_rate(agents, float(schedule_values["lr"]))
        ppo_metrics = trainer.update(buffer, advantages, returns)
        update_idx += 1

        row = {
            "update": update_idx,
            "episode_start": batch_start_episode,
            "episode_end": episode,
            "elapsed_sec": time.time() - start_time,
            "scheduled_entropy_coeff": float(schedule_values["entropy_coeff"]),
            "scheduled_msg_entropy_coeff": float(schedule_values["msg_entropy_coeff"]),
            "scheduled_lr": float(schedule_values["lr"]),
        }
        row.update(rollout_metrics.summary())
        row.update({f"ppo_{k}": v for k, v in ppo_metrics.items()})
        append_jsonl(metrics_path, row)

        while (
            next_checkpoint_target is not None
            and next_checkpoint_target <= episode
            and next_checkpoint_target <= int(cfg.episodes)
        ):
            checkpoint_path = out_dir / f"checkpoint_ep{next_checkpoint_target:06d}.pt"
            torch.save(
                _checkpoint_payload(
                    cfg,
                    agents,
                    completed_episodes=episode,
                    update=update_idx,
                ),
                checkpoint_path,
            )
            periodic_checkpoints.append(
                {
                    "target_episodes": int(next_checkpoint_target),
                    "completed_episodes": int(episode),
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
    slot_permutation_eval_metrics: Optional[Dict[str, float]] = None
    slot_permutation_trace_path: Optional[Path] = None
    if bool(cfg.eval_slot_permutation):
        slot_permutation_trace_path = out_dir / "eval_slot_permuted_trace.csv"
        slot_permutation_eval_metrics = evaluate(
            cfg,
            agents,
            out_trace_path=slot_permutation_trace_path,
            policy_map=slot_permutation_map,
            trace_phase="eval_slot_permuted",
        )
    message_shuffle_eval_metrics: Optional[Dict[str, float]] = None
    message_shuffle_trace_path: Optional[Path] = None
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
            completed_episodes=episode,
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


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Minimal role-allocation PPO smoke trainer.")
    parser.add_argument("--condition", choices=MESSAGE_SOURCE_CHOICES, default="no_comm")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--update-episodes", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--out-dir", type=str, default="outputs/train/role_allocation/smoke")
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
    return TrainConfig(
        condition=args.condition,
        seed=args.seed,
        episodes=args.episodes,
        horizon=args.horizon,
        update_episodes=args.update_episodes,
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
    )


def main() -> None:
    result = train(parse_args())
    print(f"wrote {result['out_dir']}")
    print(f"eval_metrics {result['eval_metrics']}")


if __name__ == "__main__":
    main()
