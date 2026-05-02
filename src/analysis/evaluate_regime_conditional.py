from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import os
import random
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Allow direct execution: `python src/analysis/evaluate_regime_conditional.py ...`
if __package__ is None or __package__ == "":
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from src.algos.PPO import PPOAgentV2
from src.analysis.checkpoint_artifacts import atomic_write_rows
from src.environments import pgg_parallel_v0
from src.wrappers import ObservationWrapper

_MSG_SOURCE_MODE_CHOICES = ("learned", "uniform", "public_random", "fixed0", "fixed1")


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _as_float(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def _condition_seed_from_path(path: str) -> Tuple[str, int]:
    name = os.path.basename(path)
    # Allow extra control-mode suffixes such as:
    #   cond1_seed101_fixed0_ep25000.pt
    #   cond1_seed101_public_random.pt
    m = re.search(r"(cond[0-9]+)_seed([0-9]+)", name)
    if not m:
        return "unknown", -1
    return m.group(1), int(m.group(2))


def _regime_label(f_val: float, n_agents: int) -> str:
    if f_val <= 1.0:
        return "competitive"
    if f_val <= float(n_agents):
        return "mixed"
    return "cooperative"


def _mutual_information_from_counts(counts: np.ndarray) -> float:
    arr = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(arr))
    if total <= 0.0:
        return 0.0
    p_xy = arr / total
    p_x = np.sum(p_xy, axis=1, keepdims=True)
    p_y = np.sum(p_xy, axis=0, keepdims=True)
    mask = p_xy > 0.0
    denom = np.maximum(p_x * p_y, 1e-12)
    return float(np.sum(p_xy[mask] * np.log2(p_xy[mask] / denom[mask])))


def _entropy_from_counts_1d(counts: np.ndarray) -> float:
    arr = np.asarray(counts, dtype=np.float64).reshape(-1)
    total = float(np.sum(arr))
    if total <= 0.0:
        return 0.0
    p = arr / total
    mask = p > 0.0
    return float(-np.sum(p[mask] * np.log2(p[mask])))


def _mi_null_independence_stats(
    counts: np.ndarray,
    n_perms: int,
    alpha: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    arr = np.asarray(counts, dtype=np.float64)
    observed_mi = _mutual_information_from_counts(arr)
    total = int(round(float(np.sum(arr))))
    n_perms = int(max(1, n_perms))
    alpha = float(alpha)
    if total <= 0:
        return {
            "mi_perm_p95": 0.0,
            "mi_p_value": 1.0,
            "mi_significant": False,
            "mi_observed": float(observed_mi),
        }

    p_x = np.sum(arr, axis=1)
    p_y = np.sum(arr, axis=0)
    sx = float(np.sum(p_x))
    sy = float(np.sum(p_y))
    if sx <= 0.0 or sy <= 0.0:
        return {
            "mi_perm_p95": 0.0,
            "mi_p_value": 1.0,
            "mi_significant": False,
            "mi_observed": float(observed_mi),
        }

    p_x = p_x / sx
    p_y = p_y / sy
    p_ind = np.outer(p_x, p_y).reshape(-1)
    p_ind = p_ind / np.maximum(np.sum(p_ind), 1e-12)

    null_mi = np.zeros((n_perms,), dtype=np.float64)
    for i in range(n_perms):
        sampled = rng.multinomial(total, p_ind).reshape(arr.shape)
        null_mi[i] = _mutual_information_from_counts(sampled)
    p95 = float(np.percentile(null_mi, 95.0))
    p_value = float((1.0 + np.sum(null_mi >= observed_mi)) / float(n_perms + 1))
    return {
        "mi_perm_p95": p95,
        "mi_p_value": p_value,
        "mi_significant": bool(p_value < alpha and observed_mi > 1e-8),
        "mi_observed": float(observed_mi),
    }


def _env_cfg_from_train_cfg(
    cfg: Dict,
    greedy: bool = False,
    eval_sigmas: Optional[Sequence[float]] = None,
) -> Dict:
    if eval_sigmas is not None and len(eval_sigmas) > 0:
        sigmas = [float(v) for v in eval_sigmas]
        if len(sigmas) == 1:
            sigmas = sigmas * int(cfg["n_agents"])
        elif len(sigmas) != int(cfg["n_agents"]):
            raise ValueError(
                "eval_sigmas must have length 1 or n_agents="
                f"{int(cfg['n_agents'])}, got {len(sigmas)}"
            )
    else:
        sigmas = [float(v) for v in cfg["sigmas"]]
    return dict(
        n_agents=int(cfg["n_agents"]),
        num_game_iterations=int(cfg["T"]),
        mult_fact=[float(v) for v in cfg["F"]],
        F=[float(v) for v in cfg["F"]],
        uncertainties=sigmas,
        fraction=False,
        rho=float(cfg.get("rho", 0.05)),
        # Greedy eval measures policy preference, so disable tremble noise.
        epsilon_tremble=0.0 if greedy else float(cfg.get("epsilon_tremble", 0.05)),
        endowment=float(cfg.get("endowment", 4.0)),
    )


def _canonical_msg_source_mode(mode: str) -> str:
    resolved = str(mode or "learned").strip().lower() or "learned"
    if resolved not in _MSG_SOURCE_MODE_CHOICES:
        raise ValueError(
            "unknown msg_source_mode="
            f"{mode!r}; expected one of: {','.join(_MSG_SOURCE_MODE_CHOICES)}"
        )
    return resolved


def _sample_exogenous_messages(
    source_mode: str,
    sender_ids: List[str],
    vocab_size: int,
) -> Dict[str, int]:
    mode = _canonical_msg_source_mode(source_mode)
    if mode == "learned":
        raise ValueError("_sample_exogenous_messages requires msg_source_mode != learned")
    if mode == "uniform":
        return {
            sender_id: int(np.random.randint(0, vocab_size))
            for sender_id in sender_ids
        }
    if mode == "public_random":
        shared = int(np.random.randint(0, vocab_size))
        return {sender_id: shared for sender_id in sender_ids}
    if mode == "fixed0":
        return {sender_id: 0 for sender_id in sender_ids}
    if mode == "fixed1":
        if int(vocab_size) < 2:
            raise ValueError("msg_source_mode=fixed1 requires vocab_size >= 2")
        return {sender_id: 1 for sender_id in sender_ids}
    raise ValueError(
        "unknown msg_source_mode="
        f"{source_mode!r}; expected one of: {','.join(_MSG_SOURCE_MODE_CHOICES)}"
    )


def _build_eval_objects(
    payload: Dict,
    greedy: bool = False,
    eval_sigmas: Optional[Sequence[float]] = None,
):
    cfg = payload["config"]
    n_agents = int(cfg["n_agents"])
    agent_ids = [f"agent_{i}" for i in range(n_agents)]
    comm_enabled = bool(cfg.get("comm_enabled", False))
    n_senders = int(cfg.get("n_senders", 0))
    if comm_enabled and n_senders == 0:
        n_senders = n_agents
    sender_ids = agent_ids[:n_senders]
    vocab_size = int(cfg.get("vocab_size", 2))

    wrapper = ObservationWrapper(
        n_agents=n_agents,
        comm_enabled=comm_enabled,
        n_senders=n_senders,
        sender_ids=sender_ids,
        vocab_size=vocab_size,
        # Keep greedy mode deterministic by disabling message dropout.
        msg_dropout=0.0 if greedy else float(cfg.get("msg_dropout", 0.1)),
        default_endowment=float(cfg.get("endowment", 4.0)),
        history_mode=str(cfg.get("history_mode", "full")),
    )
    value_time_feature = bool(cfg.get("value_time_feature", False))
    value_obs_dim = wrapper.obs_dim + (1 if value_time_feature else 0)
    msg_source_mode = _canonical_msg_source_mode(cfg.get("msg_source_mode", "learned"))

    agents = {}
    for agent_id in agent_ids:
        can_send = comm_enabled and (agent_id in sender_ids) and msg_source_mode == "learned"
        agent = PPOAgentV2(
            obs_dim=wrapper.obs_dim,
            action_size=2,
            can_send=can_send,
            vocab_size=vocab_size,
            hidden_size=int(cfg.get("hidden_size", 64)),
            value_obs_dim=value_obs_dim,
            lr=float(cfg.get("lr", 3e-4)),
        )
        state = payload["agents"][agent_id]
        agent.action_actor.load_state_dict(state["action_actor"])
        agent.value_net.load_state_dict(state["value_net"])
        if can_send and agent.message_actor is not None and state.get("message_actor") is not None:
            agent.message_actor.load_state_dict(state["message_actor"])
        agent.action_actor.eval()
        agent.value_net.eval()
        if agent.message_actor is not None:
            agent.message_actor.eval()
        agents[agent_id] = agent

    env = pgg_parallel_v0.parallel_env(
        _env_cfg_from_train_cfg(cfg, greedy=greedy, eval_sigmas=eval_sigmas)
    )
    return cfg, env, wrapper, agents, agent_ids, sender_ids, value_time_feature


def _propose_eval_messages(
    *,
    msg_cfg: Dict,
    msg_agents: Dict[str, PPOAgentV2],
    sender_ids: List[str],
    aug_obs: Dict[str, np.ndarray],
    greedy: bool,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    if len(sender_ids) == 0:
        return {}, {}

    msg_source_mode = _canonical_msg_source_mode(msg_cfg.get("msg_source_mode", "learned"))
    vocab_size = int(msg_cfg.get("vocab_size", 2))
    if msg_source_mode != "learned":
        return _sample_exogenous_messages(
            source_mode=msg_source_mode,
            sender_ids=sender_ids,
            vocab_size=vocab_size,
        ), {}

    proposed: Dict[str, int] = {}
    policy_entropy: Dict[str, float] = {}
    for sender_id in sender_ids:
        msg_agent = msg_agents[sender_id]
        if msg_agent.message_actor is None:
            raise RuntimeError(
                f"message_actor missing for learned msg_source_mode sender_id={sender_id}"
            )
        msg_device = msg_agent.action_actor.net[0].weight.device
        obs_t = torch.tensor(
            aug_obs[sender_id],
            dtype=torch.float32,
            device=msg_device,
        )
        logits = msg_agent.message_actor(obs_t)
        msg_probs = torch.softmax(logits, dim=-1)
        policy_entropy[sender_id] = float(
            -torch.sum(msg_probs * torch.log2(torch.clamp(msg_probs, min=1e-12)))
            .detach()
            .cpu()
            .item()
        )
        if greedy:
            msg = int(torch.argmax(logits).item())
        else:
            msg, _lp, _ent, _probs = msg_agent.sample_message(aug_obs[sender_id])
        proposed[sender_id] = int(msg)
    return proposed, policy_entropy


def _build_message_combos(vocab_size: int, n_senders: int):
    if n_senders <= 0:
        return [tuple()]
    return list(itertools.product(range(vocab_size), repeat=n_senders))


def _extract_f_hat(raw_obs_agent) -> float:
    if isinstance(raw_obs_agent, dict):
        return float(raw_obs_agent.get("f_hat", 0.0))
    if isinstance(raw_obs_agent, torch.Tensor):
        arr = raw_obs_agent.detach().cpu().numpy().reshape(-1)
        return float(arr[0]) if arr.size > 0 else 0.0
    arr = np.asarray(raw_obs_agent, dtype=np.float32).reshape(-1)
    return float(arr[0]) if arr.size > 0 else 0.0


_OBSERVABILITY_MODE_CHOICES = (
    "private",
    "public_noisy",
    "public_exact",
)


def _canonical_observability_mode(observability_mode: str) -> str:
    mode = str(observability_mode or "private").strip().lower() or "private"
    if mode not in _OBSERVABILITY_MODE_CHOICES:
        raise ValueError(
            "unknown observability_mode="
            f"{observability_mode!r}; expected one of: {','.join(_OBSERVABILITY_MODE_CHOICES)}"
        )
    return mode


def _resolve_public_signal_sigma(
    cfg: Dict,
    eval_sigmas: Optional[Sequence[float]],
    public_signal_sigma: Optional[float],
) -> float:
    if public_signal_sigma is not None:
        sigma = float(public_signal_sigma)
    else:
        source = (
            [float(v) for v in eval_sigmas]
            if eval_sigmas is not None and len(eval_sigmas) > 0
            else [float(v) for v in cfg["sigmas"]]
        )
        if len(source) == 0:
            raise ValueError("public_noisy observability requires at least one sigma source")
        if len(source) > 1 and not np.allclose(source, source[0]):
            raise ValueError(
                "public_noisy observability requires a scalar sigma; pass --public_signal_sigma "
                "or use uniform eval_sigmas"
            )
        sigma = float(source[0])
    if sigma < 0.0:
        raise ValueError("public_signal_sigma must be >= 0")
    return sigma


def _observed_f_hat_by_agent(
    raw_obs: Dict[str, object],
    agent_ids: List[str],
    true_f: float,
    observability_mode: str,
    public_signal_sigma: Optional[float],
) -> Dict[str, float]:
    mode = _canonical_observability_mode(observability_mode)
    if mode == "private":
        return {agent_id: _extract_f_hat(raw_obs[agent_id]) for agent_id in agent_ids}

    if mode == "public_exact":
        shared_f_hat = float(true_f)
    else:
        sigma = float(public_signal_sigma if public_signal_sigma is not None else 0.0)
        shared_f_hat = float(np.random.normal(loc=float(true_f), scale=sigma))
    return {agent_id: shared_f_hat for agent_id in agent_ids}


def _build_aug_obs(
    wrapper: ObservationWrapper,
    raw_obs: Dict[str, object],
    agent_ids: List[str],
    current_messages: Optional[Dict[str, int]],
    observed_f_hat_by_agent: Dict[str, float],
    history_intervention: str,
) -> Dict[str, np.ndarray]:
    return {
        agent_id: _apply_history_intervention(
            wrapper.build_obs(
                agent_id,
                raw_obs[agent_id],
                current_messages,
                f_hat_override=observed_f_hat_by_agent.get(agent_id),
            ),
            wrapper=wrapper,
            history_intervention=history_intervention,
        )
        for agent_id in agent_ids
    }


def _binary_action_policy_stats(agent: PPOAgentV2, obs) -> Dict[str, float]:
    obs_t = agent._obs_tensor(obs)
    logits_t = agent.action_actor(obs_t)
    probs_t = torch.softmax(logits_t, dim=-1)
    logits = logits_t.detach().cpu().numpy().reshape(-1)
    probs = probs_t.detach().cpu().numpy().reshape(-1)
    p_cooperate = float(probs[1]) if probs.size >= 2 else float(probs[0])
    if logits.size >= 2:
        logit_margin = float(logits[1] - logits[0])
    else:
        logit_margin = float(logits[0])
    greedy_action = int(np.argmax(logits))
    return {
        "p_cooperate": p_cooperate,
        "logit_margin": logit_margin,
        "greedy_action": greedy_action,
    }


def _apply_message_intervention(
    intervention: str,
    delivered: Dict[str, int],
    wrapper: ObservationWrapper,
    sender_ids: List[str],
    vocab_size: int,
    sender_shuffle_state: Optional[Dict[str, Dict[str, object]]] = None,
) -> Tuple[Dict[str, int], bool]:
    mode = str(intervention or "none").strip().lower()
    out = {sender_id: int(delivered[sender_id]) for sender_id in sender_ids}
    if mode == "none":
        return out, False
    if mode == "marginal":
        replaced = {}
        for sender_id in sender_ids:
            probs = wrapper.msg_marginals.get(sender_id)
            if probs is None:
                probs = np.ones((vocab_size,), dtype=np.float32) / float(vocab_size)
            replaced[sender_id] = int(np.random.choice(vocab_size, p=probs))
        return replaced, False
    if mode == "zeros":
        return out, True
    if mode == "fixed0":
        return {sender_id: 0 for sender_id in sender_ids}, False
    if mode == "fixed1":
        if vocab_size < 2:
            raise ValueError("msg_intervention=fixed1 requires vocab_size >= 2")
        return {sender_id: 1 for sender_id in sender_ids}, False
    if mode == "flip":
        if vocab_size != 2:
            raise ValueError("msg_intervention=flip requires vocab_size == 2")
        return {sender_id: 1 - int(out[sender_id]) for sender_id in sender_ids}, False
    if mode in ("uniform", "indep_random"):
        return {
            sender_id: int(np.random.randint(0, vocab_size))
            for sender_id in sender_ids
        }, False
    if mode == "public_random":
        shared = int(np.random.randint(0, vocab_size))
        return {sender_id: shared for sender_id in sender_ids}, False
    if mode == "public_marginal":
        shared_probs = np.zeros((vocab_size,), dtype=np.float64)
        for sender_id in sender_ids:
            probs = wrapper.msg_marginals.get(sender_id)
            if probs is None:
                probs = np.ones((vocab_size,), dtype=np.float32) / float(vocab_size)
            shared_probs += np.asarray(probs, dtype=np.float64)
        if len(sender_ids) > 0:
            shared_probs /= float(len(sender_ids))
        total = float(np.sum(shared_probs))
        if total <= 0.0:
            shared_probs[:] = 1.0 / float(vocab_size)
        else:
            shared_probs /= total
        shared = int(np.random.choice(vocab_size, p=shared_probs))
        return {sender_id: shared for sender_id in sender_ids}, False
    if mode == "sender_shuffle":
        if sender_shuffle_state is None:
            raise ValueError("msg_intervention=sender_shuffle requires sender shuffle state")
        replaced = {}
        for sender_id in sender_ids:
            state = sender_shuffle_state.get(sender_id)
            if state is None:
                raise ValueError(
                    f"msg_intervention=sender_shuffle missing sender_id={sender_id}"
                )
            tokens = list(state.get("tokens", []))
            if len(tokens) == 0:
                raise ValueError(
                    f"msg_intervention=sender_shuffle empty pool for sender_id={sender_id}"
                )
            cursor = int(state.get("cursor", 0))
            replaced[sender_id] = int(tokens[cursor % len(tokens)])
            state["cursor"] = cursor + 1
        return replaced, False
    if mode == "permute_slots":
        if len(sender_ids) <= 1:
            return out, False
        values = [int(out[sender_id]) for sender_id in sender_ids]
        perm = np.random.permutation(len(sender_ids))
        return {
            sender_ids[idx]: int(values[int(perm[idx])]) for idx in range(len(sender_ids))
        }, False
    raise ValueError(
        "unknown msg_intervention="
        f"{intervention!r}; expected one of: "
        "none,marginal,zeros,fixed0,fixed1,flip,uniform,indep_random,public_random,public_marginal,sender_shuffle,permute_slots"
    )


def _resolve_eval_objects(
    payload: Dict,
    greedy: bool,
    cross_play_checkpoint: str,
    eval_sigmas: Optional[Sequence[float]] = None,
):
    cfg, env, wrapper, agents, agent_ids, sender_ids, value_time_feature = _build_eval_objects(
        payload, greedy=greedy, eval_sigmas=eval_sigmas
    )
    comm_enabled = bool(cfg.get("comm_enabled", False))
    vocab_size = int(cfg.get("vocab_size", 2))
    msg_agents = agents
    msg_cfg = cfg
    cross_play_label = "none"
    cross_play_checkpoint = str(cross_play_checkpoint or "").strip()
    if cross_play_checkpoint != "":
        xp_payload = torch.load(cross_play_checkpoint, map_location="cpu")
        (
            xp_cfg,
            xp_env,
            xp_wrapper,
            xp_agents,
            xp_agent_ids,
            xp_sender_ids,
            _xp_value_time_feature,
        ) = _build_eval_objects(
            xp_payload,
            greedy=greedy,
            eval_sigmas=eval_sigmas,
        )
        if int(xp_cfg["n_agents"]) != int(cfg["n_agents"]):
            raise ValueError(
                "cross_play mismatch: "
                f"n_agents primary={cfg['n_agents']} xp={xp_cfg['n_agents']}"
            )
        if bool(xp_cfg.get("comm_enabled", False)) != comm_enabled:
            raise ValueError(
                "cross_play mismatch: "
                f"comm_enabled primary={comm_enabled} xp={bool(xp_cfg.get('comm_enabled', False))}"
            )
        if int(xp_cfg.get("vocab_size", 2)) != vocab_size:
            raise ValueError(
                "cross_play mismatch: "
                f"vocab_size primary={vocab_size} xp={int(xp_cfg.get('vocab_size', 2))}"
            )
        if list(xp_agent_ids) != list(agent_ids):
            raise ValueError(
                "cross_play mismatch: "
                f"agent_ids primary={agent_ids} xp={xp_agent_ids}"
            )
        if list(xp_sender_ids) != list(sender_ids):
            raise ValueError(
                "cross_play mismatch: "
                f"sender_ids primary={sender_ids} xp={xp_sender_ids}"
            )
        if int(xp_wrapper.obs_dim) != int(wrapper.obs_dim):
            raise ValueError(
                "cross_play mismatch: "
                f"obs_dim primary={wrapper.obs_dim} xp={xp_wrapper.obs_dim}"
            )
        msg_agents = xp_agents
        msg_cfg = xp_cfg
        cross_play_label = os.path.basename(cross_play_checkpoint)
        try:
            xp_env.close()
        except Exception:
            pass
    if cross_play_label != "none" and not (comm_enabled and len(sender_ids) > 0):
        raise ValueError("cross_play_checkpoint requires communication-enabled checkpoints")
    return (
        cfg,
        env,
        wrapper,
        agents,
        agent_ids,
        sender_ids,
        value_time_feature,
        msg_agents,
        msg_cfg,
        cross_play_label,
    )


def _collect_sender_shuffle_state(
    cfg: Dict,
    env,
    wrapper: ObservationWrapper,
    agents: Dict[str, PPOAgentV2],
    msg_cfg: Dict,
    msg_agents: Dict[str, PPOAgentV2],
    agent_ids: List[str],
    sender_ids: List[str],
    value_time_feature: bool,
    n_eval_episodes: int,
    eval_seed: int,
    greedy: bool,
    history_intervention: str = "none",
    observability_mode: str = "private",
    public_signal_sigma: Optional[float] = None,
) -> Dict[str, Dict[str, object]]:
    _seed_everything(eval_seed)
    T = int(cfg["T"])
    comm_enabled = bool(cfg.get("comm_enabled", False))
    tokens_by_sender = {sender_id: [] for sender_id in sender_ids}
    history_intervention_label = _canonical_history_intervention(history_intervention)

    with torch.no_grad():
        for _episode in range(n_eval_episodes):
            raw_obs = env.reset()
            wrapper.reset(agent_ids)
            current_messages = None
            done = False
            steps = 0

            while (not done) and (steps < T):
                current_true_f = float(_as_float(env.current_multiplier))
                observed_f_hat = _observed_f_hat_by_agent(
                    raw_obs=raw_obs,
                    agent_ids=agent_ids,
                    true_f=current_true_f,
                    observability_mode=observability_mode,
                    public_signal_sigma=public_signal_sigma,
                )
                aug_obs = _build_aug_obs(
                    wrapper=wrapper,
                    raw_obs=raw_obs,
                    agent_ids=agent_ids,
                    current_messages=current_messages,
                    observed_f_hat_by_agent=observed_f_hat,
                    history_intervention=history_intervention_label,
                )

                proposed = {}
                if comm_enabled and len(sender_ids) > 0:
                    proposed, _policy_entropy = _propose_eval_messages(
                        msg_cfg=msg_cfg,
                        msg_agents=msg_agents,
                        sender_ids=sender_ids,
                        aug_obs=aug_obs,
                        greedy=greedy,
                    )
                    delivered = wrapper.apply_msg_dropout(proposed)
                    for sender_id, msg in proposed.items():
                        wrapper.update_msg_marginals(sender_id, msg)
                    for sender_id in sender_ids:
                        tokens_by_sender[sender_id].append(int(delivered[sender_id]))
                    current_messages = delivered
                    aug_obs = _build_aug_obs(
                        wrapper=wrapper,
                        raw_obs=raw_obs,
                        agent_ids=agent_ids,
                        current_messages=current_messages,
                        observed_f_hat_by_agent=observed_f_hat,
                        history_intervention=history_intervention_label,
                    )

                intended_actions = {}
                t_frac = float(steps) / float(max(1, T - 1))
                for agent_id in agent_ids:
                    value_obs = (
                        np.concatenate(
                            [aug_obs[agent_id], np.array([t_frac], dtype=np.float32)], axis=0
                        ).astype(np.float32)
                        if value_time_feature
                        else aug_obs[agent_id]
                    )
                    if greedy:
                        obs_t = torch.tensor(
                            aug_obs[agent_id],
                            dtype=torch.float32,
                            device=agents[agent_id].action_actor.net[0].weight.device,
                        )
                        logits = agents[agent_id].action_actor(obs_t)
                        action = int(torch.argmax(logits).item())
                    else:
                        action, _lp, _value, _ent, _probs = agents[agent_id].sample_action(
                            aug_obs[agent_id], value_obs=value_obs
                        )
                    intended_actions[agent_id] = int(action)

                raw_next, _rewards, done, infos = env.step(intended_actions)
                executed_actions = infos.get("executed_actions", intended_actions)
                wrapper.update(executed_actions)
                raw_obs = raw_next
                steps += 1

    rng = np.random.default_rng(int(eval_seed) + 2718)
    out: Dict[str, Dict[str, object]] = {}
    for sender_id in sender_ids:
        tokens = list(tokens_by_sender.get(sender_id, []))
        if len(tokens) > 1:
            perm = rng.permutation(len(tokens))
            tokens = [tokens[int(idx)] for idx in perm]
        out[sender_id] = {"tokens": tokens, "cursor": 0}
    return out


def _parse_sender_flip_map(raw_json: str, sender_ids: List[str]) -> Dict[str, bool]:
    text = str(raw_json or "").strip()
    if text == "":
        return {}
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("sender_flip_map_json must decode to an object")
    out: Dict[str, bool] = {}
    valid = set(sender_ids)
    for key, value in payload.items():
        sender_id = str(key)
        if sender_id not in valid:
            raise ValueError(f"sender_flip_map_json contains unknown sender_id: {sender_id}")
        out[sender_id] = bool(int(value)) if isinstance(value, (int, float, bool)) else bool(value)
    return out


def _parse_sender_remap(raw_json: str, sender_ids: List[str]) -> Dict[str, str]:
    text = str(raw_json or "").strip()
    if text == "":
        return {sender_id: sender_id for sender_id in sender_ids}
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("sender_remap_json must decode to an object")
    valid = set(sender_ids)
    keys = {str(key) for key in payload.keys()}
    vals = {str(value) for value in payload.values()}
    if keys != valid:
        raise ValueError(
            "sender_remap_json must provide a complete source->target mapping for sender_ids"
        )
    if vals != valid:
        raise ValueError(
            "sender_remap_json target values must form a complete permutation of sender_ids"
        )
    out: Dict[str, str] = {}
    for key, value in payload.items():
        src = str(key)
        dst = str(value)
        if src not in valid:
            raise ValueError(f"sender_remap_json contains unknown sender_id: {src}")
        if dst not in valid:
            raise ValueError(f"sender_remap_json contains unknown target sender_id: {dst}")
        out[src] = dst
    return out


def _apply_sender_remap(
    delivered: Dict[str, int],
    sender_remap: Dict[str, str],
) -> Dict[str, int]:
    if len(sender_remap) == 0:
        return {sender_id: int(token) for sender_id, token in delivered.items()}
    out: Dict[str, int] = {}
    for sender_id, token in delivered.items():
        target_id = str(sender_remap.get(sender_id, sender_id))
        out[target_id] = int(token)
    if len(out) != len(delivered):
        raise ValueError("sender_remap_json produced a non-bijective delivered-message mapping")
    return out


def _apply_sender_flip_map(
    delivered: Dict[str, int],
    sender_flip_map: Dict[str, bool],
    vocab_size: int,
) -> Dict[str, int]:
    if len(sender_flip_map) == 0:
        return {sender_id: int(token) for sender_id, token in delivered.items()}
    if int(vocab_size) != 2:
        raise ValueError("sender_flip_map_json currently requires vocab_size == 2")
    out = {sender_id: int(token) for sender_id, token in delivered.items()}
    for sender_id, should_flip in sender_flip_map.items():
        if should_flip and sender_id in out:
            out[sender_id] = 1 - int(out[sender_id])
    return out


def _posterior_bin_label(f_hat: float) -> str:
    if f_hat < 1.5:
        return "fhat<1.5"
    if f_hat < 2.5:
        return "1.5<=fhat<2.5"
    if f_hat < 3.5:
        return "2.5<=fhat<3.5"
    if f_hat < 4.5:
        return "3.5<=fhat<4.5"
    return "fhat>=4.5"


def _delivered_msg_key(sender_id: str) -> str:
    return f"delivered_msg_{sender_id}"


def _obs_msg_key(sender_id: str, token_idx: int) -> str:
    return f"obs_msg_{sender_id}_tok{int(token_idx)}"


def _build_received_pattern(
    agent_id: str,
    sender_ids: List[str],
    delivered_messages: Optional[Dict[str, int]],
) -> Tuple[str, int, int]:
    if delivered_messages is None or len(sender_ids) == 0:
        return "", 0, 0

    parts = []
    recv_any_m0 = 0
    recv_any_m1 = 0
    for sender_id in sender_ids:
        if sender_id == agent_id:
            continue
        if sender_id not in delivered_messages:
            continue
        token = int(delivered_messages[sender_id])
        parts.append(f"{sender_id}:{token}")
        if token == 0:
            recv_any_m0 = 1
        elif token == 1:
            recv_any_m1 = 1
    return ("|".join(parts) if len(parts) > 0 else "none", recv_any_m0, recv_any_m1)


_HISTORY_INTERVENTION_CHOICES = (
    "none",
    "zero_temporal",
    "clamp_temporal_high",
    "clamp_temporal_low",
    "zero_last_coop",
    "zero_last_action",
    "zero_ewma",
    "clamp_ewma_high",
)


def _canonical_history_intervention(history_intervention: str) -> str:
    mode = str(history_intervention or "none").strip().lower()
    if mode == "":
        mode = "none"
    if mode not in _HISTORY_INTERVENTION_CHOICES:
        raise ValueError(
            "unknown history_intervention="
            f"{history_intervention!r}; expected one of: {','.join(_HISTORY_INTERVENTION_CHOICES)}"
        )
    return mode


def _apply_history_intervention(
    obs: np.ndarray,
    wrapper: ObservationWrapper,
    history_intervention: str,
) -> np.ndarray:
    mode = _canonical_history_intervention(history_intervention)
    out = np.asarray(obs, dtype=np.float32).copy()
    if mode == "none":
        return out
    if mode == "zero_temporal":
        out[wrapper.temporal_feature_slice] = 0.0
        return out
    if mode == "clamp_temporal_high":
        out[wrapper.last_coop_idx] = 1.0
        out[wrapper.own_last_action_idx] = 1.0
        out[wrapper.ewma_coop_idx] = 1.0
        return out
    if mode == "clamp_temporal_low":
        out[wrapper.last_coop_idx] = 0.0
        out[wrapper.own_last_action_idx] = 0.0
        out[wrapper.ewma_coop_idx] = 0.0
        return out
    if mode == "zero_last_coop":
        out[wrapper.last_coop_idx] = 0.0
        return out
    if mode == "zero_last_action":
        out[wrapper.own_last_action_idx] = 0.0
        return out
    if mode == "zero_ewma":
        out[wrapper.ewma_coop_idx] = 0.0
        return out
    if mode == "clamp_ewma_high":
        out[wrapper.ewma_coop_idx] = 1.0
        return out
    raise AssertionError(f"unhandled history_intervention={mode}")


def _history_feature_record(obs: np.ndarray, wrapper: ObservationWrapper) -> Dict[str, float]:
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    return {
        "obs_last_coop_fraction": float(arr[wrapper.last_coop_idx]),
        "obs_own_last_action": float(arr[wrapper.own_last_action_idx]),
        "obs_ewma_coop": float(arr[wrapper.ewma_coop_idx]),
    }


def _derive_sender_semantics_rows(trace_rows: List[Dict]) -> List[Dict]:
    if len(trace_rows) == 0:
        return []

    by_fhat = defaultdict(lambda: {"n_obs": 0, "msg1_sum": 0.0})
    by_action = defaultdict(lambda: {"n_obs": 0, "msg1_sum": 0.0})
    for row in trace_rows:
        own_sent = row.get("own_sent_msg", "")
        if own_sent == "" or own_sent is None:
            continue
        sender_id = str(row["agent_id"])
        meta = (
            row["checkpoint"],
            row["condition"],
            int(row["train_seed"]),
            int(row["eval_seed"]),
            row["eval_policy"],
            row["ablation"],
            row.get("history_intervention", "none"),
            row.get("sender_remap", "none"),
            row["cross_play"],
            sender_id,
        )
        msg_is_one = float(int(own_sent) == 1)
        fhat_bin = _posterior_bin_label(float(row["f_hat"]))
        by_fhat[meta + (fhat_bin,)]["n_obs"] += 1
        by_fhat[meta + (fhat_bin,)]["msg1_sum"] += msg_is_one

        action = int(row["action"])
        by_action[meta + (action,)]["n_obs"] += 1
        by_action[meta + (action,)]["msg1_sum"] += msg_is_one

    out = []
    for key, acc in sorted(by_fhat.items()):
        checkpoint, condition, train_seed, eval_seed, eval_policy, ablation, history_intervention, sender_remap, cross_play, sender_id, fhat_bin = key
        n_obs = int(acc["n_obs"])
        out.append(
            {
                "checkpoint": checkpoint,
                "condition": condition,
                "train_seed": int(train_seed),
                "eval_seed": int(eval_seed),
                "eval_policy": eval_policy,
                "ablation": ablation,
                "history_intervention": history_intervention,
                "sender_remap": sender_remap,
                "cross_play": cross_play,
                "sender_id": sender_id,
                "summary": "p_msg1_given_fhat",
                "fhat_bin": fhat_bin,
                "action": "",
                "n_obs": n_obs,
                "p_message_1": float(acc["msg1_sum"] / max(1, n_obs)),
            }
        )
    for key, acc in sorted(by_action.items()):
        checkpoint, condition, train_seed, eval_seed, eval_policy, ablation, history_intervention, sender_remap, cross_play, sender_id, action = key
        n_obs = int(acc["n_obs"])
        out.append(
            {
                "checkpoint": checkpoint,
                "condition": condition,
                "train_seed": int(train_seed),
                "eval_seed": int(eval_seed),
                "eval_policy": eval_policy,
                "ablation": ablation,
                "history_intervention": history_intervention,
                "sender_remap": sender_remap,
                "cross_play": cross_play,
                "sender_id": sender_id,
                "summary": "p_msg1_given_action",
                "fhat_bin": "",
                "action": int(action),
                "n_obs": n_obs,
                "p_message_1": float(acc["msg1_sum"] / max(1, n_obs)),
            }
        )
    return out


def _derive_receiver_semantics_rows(trace_rows: List[Dict]) -> List[Dict]:
    if len(trace_rows) == 0:
        return []

    by_pattern = defaultdict(lambda: {"n_obs": 0, "coop_sum": 0.0})
    by_any_token = defaultdict(lambda: {"n_obs": 0, "coop_sum": 0.0})
    by_sender_token = defaultdict(lambda: {"n_obs": 0, "coop_sum": 0.0})
    delivered_cols = sorted(
        {key for row in trace_rows for key in row.keys() if key.startswith("delivered_msg_")}
    )
    for row in trace_rows:
        recv_pattern = str(row.get("recv_pattern", ""))
        if recv_pattern == "":
            continue
        meta = (
            row["checkpoint"],
            row["condition"],
            int(row["train_seed"]),
            int(row["eval_seed"]),
            row["eval_policy"],
            row["ablation"],
            row.get("history_intervention", "none"),
            row.get("sender_remap", "none"),
            row["cross_play"],
        )
        fhat_bin = _posterior_bin_label(float(row["f_hat"]))
        action = float(int(row["action"]))
        by_pattern[meta + (recv_pattern, fhat_bin)]["n_obs"] += 1
        by_pattern[meta + (recv_pattern, fhat_bin)]["coop_sum"] += action

        for token in (0, 1):
            if int(row.get(f"recv_any_m{token}", 0)) != 1:
                continue
            by_any_token[meta + (token, fhat_bin)]["n_obs"] += 1
            by_any_token[meta + (token, fhat_bin)]["coop_sum"] += action

        receiver_id = str(row.get("agent_id", ""))
        for delivered_col in delivered_cols:
            sender_id = str(delivered_col).replace("delivered_msg_", "", 1)
            if sender_id == receiver_id:
                continue
            delivered_value = row.get(delivered_col, "")
            if delivered_value in ("", None):
                continue
            token = int(float(delivered_value))
            by_sender_token[meta + (receiver_id, sender_id, token, fhat_bin)]["n_obs"] += 1
            by_sender_token[meta + (receiver_id, sender_id, token, fhat_bin)]["coop_sum"] += action

    out = []
    for key, acc in sorted(by_pattern.items()):
        checkpoint, condition, train_seed, eval_seed, eval_policy, ablation, history_intervention, sender_remap, cross_play, recv_pattern, fhat_bin = key
        n_obs = int(acc["n_obs"])
        out.append(
            {
                "checkpoint": checkpoint,
                "condition": condition,
                "train_seed": int(train_seed),
                "eval_seed": int(eval_seed),
                "eval_policy": eval_policy,
                "ablation": ablation,
                "history_intervention": history_intervention,
                "sender_remap": sender_remap,
                "cross_play": cross_play,
                "receiver_id": "all_agents",
                "sender_id": "",
                "summary": "p_coop_given_recv_pattern_fhat",
                "recv_pattern": recv_pattern,
                "any_token": "",
                "sender_token": "",
                "fhat_bin": fhat_bin,
                "n_obs": n_obs,
                "p_cooperate": float(acc["coop_sum"] / max(1, n_obs)),
            }
        )
    for key, acc in sorted(by_any_token.items()):
        checkpoint, condition, train_seed, eval_seed, eval_policy, ablation, history_intervention, sender_remap, cross_play, any_token, fhat_bin = key
        n_obs = int(acc["n_obs"])
        out.append(
            {
                "checkpoint": checkpoint,
                "condition": condition,
                "train_seed": int(train_seed),
                "eval_seed": int(eval_seed),
                "eval_policy": eval_policy,
                "ablation": ablation,
                "history_intervention": history_intervention,
                "sender_remap": sender_remap,
                "cross_play": cross_play,
                "receiver_id": "all_agents",
                "sender_id": "",
                "summary": "p_coop_given_any_token_fhat",
                "recv_pattern": "",
                "any_token": int(any_token),
                "sender_token": "",
                "fhat_bin": fhat_bin,
                "n_obs": n_obs,
                "p_cooperate": float(acc["coop_sum"] / max(1, n_obs)),
            }
        )
    for key, acc in sorted(by_sender_token.items()):
        (
            checkpoint,
            condition,
            train_seed,
            eval_seed,
            eval_policy,
            ablation,
            history_intervention,
            sender_remap,
            cross_play,
            receiver_id,
            sender_id,
            sender_token,
            fhat_bin,
        ) = key
        n_obs = int(acc["n_obs"])
        out.append(
            {
                "checkpoint": checkpoint,
                "condition": condition,
                "train_seed": int(train_seed),
                "eval_seed": int(eval_seed),
                "eval_policy": eval_policy,
                "ablation": ablation,
                "history_intervention": history_intervention,
                "sender_remap": sender_remap,
                "cross_play": cross_play,
                "receiver_id": receiver_id,
                "sender_id": sender_id,
                "summary": "p_coop_given_sender_token_fhat",
                "recv_pattern": "",
                "any_token": "",
                "sender_token": int(sender_token),
                "fhat_bin": fhat_bin,
                "n_obs": n_obs,
                "p_cooperate": float(acc["coop_sum"] / max(1, n_obs)),
            }
        )
    return out


def _sender_causal_summary_rows(
    sender_causal_acc: Dict[Tuple[str, str, str], Dict[str, float]],
    *,
    checkpoint_path: str,
    condition: str,
    train_seed: int,
    eval_seed: int,
    greedy: bool,
    ablation_label: str,
    history_intervention_label: str,
    sender_remap_label: str,
    cross_play_label: str,
    n_agents: int,
) -> List[Dict]:
    out: List[Dict] = []
    for (sender_id, receiver_id, f_key), acc in sorted(
        sender_causal_acc.items(), key=lambda item: (float(item[0][2]), item[0][0], item[0][1])
    ):
        n_obs = max(1, int(acc["n_obs"]))
        true_f = float(f_key)
        out.append(
            {
                "checkpoint": checkpoint_path,
                "condition": condition,
                "train_seed": int(train_seed),
                "eval_seed": int(eval_seed),
                "eval_policy": "greedy" if greedy else "sample",
                "ablation": ablation_label,
                "history_intervention": history_intervention_label,
                "sender_remap": sender_remap_label,
                "cross_play": cross_play_label,
                "sender_id": str(sender_id),
                "receiver_id": str(receiver_id),
                "receiver_is_sender": int(str(sender_id) == str(receiver_id)),
                "regime": _regime_label(true_f, n_agents=n_agents),
                "true_f": true_f,
                "n_obs": int(acc["n_obs"]),
                "natural_token0_rate": float(acc["natural_token0_n"] / n_obs),
                "natural_token1_rate": float(acc["natural_token1_n"] / n_obs),
                "natural_p_cooperate": float(acc["natural_p_cooperate_sum"] / n_obs),
                "natural_logit_margin": float(acc["natural_logit_margin_sum"] / n_obs),
                "natural_greedy_cooperate": float(acc["natural_greedy_cooperate_sum"] / n_obs),
                "p_cooperate_force0": float(acc["p_cooperate_force0_sum"] / n_obs),
                "p_cooperate_force1": float(acc["p_cooperate_force1_sum"] / n_obs),
                "delta_p_cooperate_1_minus_0": float(acc["delta_p_cooperate_sum"] / n_obs),
                "logit_margin_force0": float(acc["logit_margin_force0_sum"] / n_obs),
                "logit_margin_force1": float(acc["logit_margin_force1_sum"] / n_obs),
                "delta_logit_margin_1_minus_0": float(acc["delta_logit_margin_sum"] / n_obs),
                "greedy_cooperate_force0": float(acc["greedy_cooperate_force0_sum"] / n_obs),
                "greedy_cooperate_force1": float(acc["greedy_cooperate_force1_sum"] / n_obs),
                "delta_greedy_cooperate_1_minus_0": float(acc["delta_greedy_cooperate_sum"] / n_obs),
                "action_flip_rate_force0_vs_natural": float(acc["action_flip_force0_n"] / n_obs),
                "action_flip_rate_force1_vs_natural": float(acc["action_flip_force1_n"] / n_obs),
            }
        )
    return out


def _expected_action_dist_under_marginals(
    agent: PPOAgentV2,
    obs: np.ndarray,
    wrapper: ObservationWrapper,
    sender_ids: List[str],
    vocab_size: int,
    msg_combos: List[Tuple[int, ...]],
) -> np.ndarray:
    if len(sender_ids) == 0:
        obs_t = torch.tensor(
            obs,
            dtype=torch.float32,
            device=agent.action_actor.net[0].weight.device,
        ).unsqueeze(0)
        return agent.action_distribution(obs_t).squeeze(0).detach().cpu().numpy()

    obs_batch = np.repeat(np.asarray(obs, dtype=np.float32)[None, :], len(msg_combos), axis=0)
    weights = np.ones((len(msg_combos),), dtype=np.float64)
    msg_start = int(wrapper.message_start_idx)
    for s_idx, sender_id in enumerate(sender_ids):
        start = msg_start + s_idx * int(vocab_size)
        end = start + int(vocab_size)
        probs = wrapper.msg_marginals.get(sender_id)
        if probs is None:
            probs = np.ones((vocab_size,), dtype=np.float32) / float(vocab_size)
        obs_batch[:, start:end] = 0.0
        for c_idx, combo in enumerate(msg_combos):
            token = int(combo[s_idx])
            obs_batch[c_idx, start + token] = 1.0
            weights[c_idx] *= float(probs[token])

    w_sum = float(np.sum(weights))
    if w_sum <= 0.0:
        weights[:] = 1.0 / float(len(weights))
    else:
        weights /= w_sum
    obs_t = torch.tensor(
        obs_batch,
        dtype=torch.float32,
        device=agent.action_actor.net[0].weight.device,
    )
    probs = agent.action_distribution(obs_t).detach().cpu().numpy()
    expected = np.average(probs, axis=0, weights=weights)
    expected = expected / np.maximum(np.sum(expected), 1e-12)
    return expected.astype(np.float64)


def _eval_checkpoint(
    checkpoint_path: str,
    n_eval_episodes: int,
    eval_seed: int,
    greedy: bool = False,
    eval_sigmas: Optional[Sequence[float]] = None,
    observability_mode: str = "private",
    public_signal_sigma: Optional[float] = None,
    msg_intervention: str = "none",
    history_intervention: str = "none",
    mi_null_perms: int = 200,
    mi_alpha: float = 0.05,
    cross_play_checkpoint: str = "",
    sender_flip_map_json: str = "",
    sender_remap_json: str = "",
    sender_remap_label: str = "none",
    posterior_strat: bool = False,
    collect_semantics: bool = False,
    collect_sender_causal: bool = False,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], List[Dict], List[Dict], List[Dict]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    ablation_label = str(msg_intervention or "none").strip().lower()
    history_intervention_label = _canonical_history_intervention(history_intervention)
    observability_mode_label = _canonical_observability_mode(observability_mode)
    sender_remap_label = str(sender_remap_label or "none").strip().lower() or "none"
    (
        cfg,
        env,
        wrapper,
        agents,
        agent_ids,
        sender_ids,
        value_time_feature,
        msg_agents,
        msg_cfg,
        cross_play_label,
    ) = _resolve_eval_objects(
        payload=payload,
        greedy=greedy,
        cross_play_checkpoint=str(cross_play_checkpoint or ""),
        eval_sigmas=eval_sigmas,
    )
    resolved_public_signal_sigma: Optional[float] = None
    if observability_mode_label == "public_noisy":
        resolved_public_signal_sigma = _resolve_public_signal_sigma(
            cfg=cfg,
            eval_sigmas=eval_sigmas,
            public_signal_sigma=public_signal_sigma,
        )
    sender_shuffle_state: Optional[Dict[str, Dict[str, object]]] = None
    if ablation_label == "sender_shuffle":
        sender_shuffle_state = _collect_sender_shuffle_state(
            cfg=cfg,
            env=env,
            wrapper=wrapper,
            agents=agents,
            msg_cfg=msg_cfg,
            msg_agents=msg_agents,
            agent_ids=agent_ids,
            sender_ids=sender_ids,
            value_time_feature=value_time_feature,
            n_eval_episodes=int(n_eval_episodes),
            eval_seed=int(eval_seed),
            greedy=bool(greedy),
            history_intervention=history_intervention_label,
            observability_mode=observability_mode_label,
            public_signal_sigma=resolved_public_signal_sigma,
        )
        try:
            env.close()
        except Exception:
            pass
        (
            cfg,
            env,
            wrapper,
            agents,
            agent_ids,
            sender_ids,
            value_time_feature,
            msg_agents,
            msg_cfg,
            cross_play_label,
        ) = _resolve_eval_objects(
            payload=payload,
            greedy=greedy,
            cross_play_checkpoint=str(cross_play_checkpoint or ""),
            eval_sigmas=eval_sigmas,
        )

    _seed_everything(eval_seed)

    n_agents = int(cfg["n_agents"])
    T = int(cfg["T"])
    comm_enabled = bool(cfg.get("comm_enabled", False))
    condition, train_seed = _condition_seed_from_path(checkpoint_path)
    vocab_size = int(cfg.get("vocab_size", 2))
    f_keys_sorted = sorted({f"{float(v):.3f}" for v in cfg["F"]}, key=float)
    f_key_to_idx = {key: idx for idx, key in enumerate(f_keys_sorted)}
    msg_combos = _build_message_combos(vocab_size=vocab_size, n_senders=len(sender_ids))
    mi_rng = np.random.default_rng(int(eval_seed) + 1729)
    sender_flip_map = _parse_sender_flip_map(sender_flip_map_json, sender_ids)
    sender_remap = _parse_sender_remap(sender_remap_json, sender_ids)

    regime_acc = defaultdict(
        lambda: {
            "n_rounds": 0,
            "coop_sum": 0.0,
            "reward_sum": 0.0,
            "welfare_sum": 0.0,
            "all_coop_sum": 0.0,
        }
    )
    f_acc = defaultdict(
        lambda: {
            "n_rounds": 0,
            "coop_sum": 0.0,
            "reward_sum": 0.0,
            "welfare_sum": 0.0,
            "all_coop_sum": 0.0,
            "msg0_n": 0,
            "msg0_coop": 0.0,
            "msg1_n": 0,
            "msg1_coop": 0.0,
        }
    )
    comm_counts = {
        sender_id: {
            "msg_f": np.zeros((vocab_size, len(f_keys_sorted)), dtype=np.float64),
            "msg_action": np.zeros((vocab_size, 2), dtype=np.float64),
        }
        for sender_id in sender_ids
    }
    responsiveness = {agent_id: [] for agent_id in agent_ids}
    policy_entropy = {sender_id: [] for sender_id in sender_ids}
    posterior_records = []
    trace_rows: List[Dict] = []
    sender_causal_acc = defaultdict(
        lambda: {
            "n_obs": 0,
            "natural_token0_n": 0,
            "natural_token1_n": 0,
            "natural_p_cooperate_sum": 0.0,
            "natural_logit_margin_sum": 0.0,
            "natural_greedy_cooperate_sum": 0.0,
            "p_cooperate_force0_sum": 0.0,
            "p_cooperate_force1_sum": 0.0,
            "delta_p_cooperate_sum": 0.0,
            "logit_margin_force0_sum": 0.0,
            "logit_margin_force1_sum": 0.0,
            "delta_logit_margin_sum": 0.0,
            "greedy_cooperate_force0_sum": 0.0,
            "greedy_cooperate_force1_sum": 0.0,
            "delta_greedy_cooperate_sum": 0.0,
            "action_flip_force0_n": 0,
            "action_flip_force1_n": 0,
        }
    )

    with torch.no_grad():
        for _ in range(n_eval_episodes):
            raw_obs = env.reset()
            wrapper.reset(agent_ids)
            current_messages = None
            done = False
            steps = 0

            while (not done) and (steps < T):
                current_true_f = float(_as_float(env.current_multiplier))
                f_hat_by_agent = _observed_f_hat_by_agent(
                    raw_obs=raw_obs,
                    agent_ids=agent_ids,
                    true_f=current_true_f,
                    observability_mode=observability_mode_label,
                    public_signal_sigma=resolved_public_signal_sigma,
                )
                aug_obs = _build_aug_obs(
                    wrapper=wrapper,
                    raw_obs=raw_obs,
                    agent_ids=agent_ids,
                    current_messages=current_messages,
                    observed_f_hat_by_agent=f_hat_by_agent,
                    history_intervention=history_intervention_label,
                )

                proposed = {}
                if comm_enabled and len(sender_ids) > 0:
                    proposed, policy_entropy_step = _propose_eval_messages(
                        msg_cfg=msg_cfg,
                        msg_agents=msg_agents,
                        sender_ids=sender_ids,
                        aug_obs=aug_obs,
                        greedy=greedy,
                    )
                    for sender_id, msg_entropy in policy_entropy_step.items():
                        policy_entropy[sender_id].append(float(msg_entropy))
                    remapped = _apply_sender_remap(proposed, sender_remap=sender_remap)
                    dropped = wrapper.apply_msg_dropout(remapped)
                    for sender_id, msg in remapped.items():
                        wrapper.update_msg_marginals(sender_id, msg)
                    delivered, force_zero_message_slice = _apply_message_intervention(
                        intervention=ablation_label,
                        delivered=dropped,
                        wrapper=wrapper,
                        sender_ids=sender_ids,
                        vocab_size=vocab_size,
                        sender_shuffle_state=sender_shuffle_state,
                    )
                    delivered = _apply_sender_flip_map(
                        delivered=delivered,
                        sender_flip_map=sender_flip_map,
                        vocab_size=vocab_size,
                    )
                    current_messages = delivered
                    aug_obs = _build_aug_obs(
                        wrapper=wrapper,
                        raw_obs=raw_obs,
                        agent_ids=agent_ids,
                        current_messages=current_messages,
                        observed_f_hat_by_agent=f_hat_by_agent,
                        history_intervention=history_intervention_label,
                    )
                    if force_zero_message_slice:
                        msg_start = int(wrapper.message_start_idx)
                        for agent_id in agent_ids:
                            aug_obs[agent_id][msg_start:] = 0.0

                intended_actions = {}
                policy_stats_by_agent = {}
                history_features_by_agent = {}
                t_frac = float(steps) / float(max(1, T - 1))
                for agent_id in agent_ids:
                    history_features_by_agent[agent_id] = _history_feature_record(
                        aug_obs[agent_id],
                        wrapper=wrapper,
                    )
                    value_obs = (
                        np.concatenate(
                            [aug_obs[agent_id], np.array([t_frac], dtype=np.float32)], axis=0
                        ).astype(np.float32)
                        if value_time_feature
                        else aug_obs[agent_id]
                    )
                    if greedy:
                        stats = _binary_action_policy_stats(agents[agent_id], aug_obs[agent_id])
                        probs_with = np.array(
                            [1.0 - float(stats["p_cooperate"]), float(stats["p_cooperate"])],
                            dtype=np.float64,
                        )
                        action = int(stats["greedy_action"])
                    else:
                        action, _lp, _value, _ent, _probs = agents[agent_id].sample_action(
                            aug_obs[agent_id], value_obs=value_obs
                        )
                        probs_with = _probs.detach().cpu().numpy()
                        stats = {
                            "p_cooperate": float(probs_with[1]),
                            "logit_margin": float(
                                np.log(np.maximum(probs_with[1], 1e-12))
                                - np.log(np.maximum(probs_with[0], 1e-12))
                            ),
                            "greedy_action": int(np.argmax(probs_with)),
                        }
                    intended_actions[agent_id] = int(action)
                    policy_stats_by_agent[agent_id] = stats
                    if comm_enabled and len(sender_ids) > 0:
                        probs_marginal = _expected_action_dist_under_marginals(
                            agent=agents[agent_id],
                            obs=aug_obs[agent_id],
                            wrapper=wrapper,
                            sender_ids=sender_ids,
                            vocab_size=vocab_size,
                            msg_combos=msg_combos,
                        )
                        eps = 1e-8
                        kl = float(
                            np.sum(
                                probs_with
                                * (
                                    np.log(np.maximum(probs_with, eps))
                                    - np.log(np.maximum(probs_marginal, eps))
                                )
                            )
                        )
                        responsiveness[agent_id].append(max(0.0, kl))

                if (
                    collect_sender_causal
                    and comm_enabled
                    and len(sender_ids) > 0
                    and current_messages is not None
                    and int(vocab_size) >= 2
                ):
                    for sender_id in sender_ids:
                        natural_token = int(current_messages.get(sender_id, 0))
                        edited_messages = {
                            0: dict(current_messages),
                            1: dict(current_messages),
                        }
                        edited_messages[0][sender_id] = 0
                        edited_messages[1][sender_id] = 1
                        for receiver_id in agent_ids:
                            natural_stats = policy_stats_by_agent[receiver_id]
                            forced_stats = {}
                            for forced_token in (0, 1):
                                if natural_token == forced_token:
                                    forced_stats[forced_token] = natural_stats
                                    continue
                                edited_obs = wrapper.build_obs(
                                    receiver_id,
                                    raw_obs[receiver_id],
                                    edited_messages[forced_token],
                                    f_hat_override=f_hat_by_agent.get(receiver_id),
                                )
                                edited_obs = _apply_history_intervention(
                                    edited_obs,
                                    wrapper=wrapper,
                                    history_intervention=history_intervention_label,
                                )
                                forced_stats[forced_token] = _binary_action_policy_stats(
                                    agents[receiver_id], edited_obs
                                )
                            key = (str(sender_id), str(receiver_id), f"{float(current_true_f):.3f}")
                            acc = sender_causal_acc[key]
                            acc["n_obs"] += 1
                            if natural_token == 0:
                                acc["natural_token0_n"] += 1
                            elif natural_token == 1:
                                acc["natural_token1_n"] += 1
                            acc["natural_p_cooperate_sum"] += float(natural_stats["p_cooperate"])
                            acc["natural_logit_margin_sum"] += float(natural_stats["logit_margin"])
                            acc["natural_greedy_cooperate_sum"] += float(
                                int(natural_stats["greedy_action"]) == 1
                            )
                            for forced_token in (0, 1):
                                stats = forced_stats[forced_token]
                                acc[f"p_cooperate_force{forced_token}_sum"] += float(
                                    stats["p_cooperate"]
                                )
                                acc[f"logit_margin_force{forced_token}_sum"] += float(
                                    stats["logit_margin"]
                                )
                                acc[f"greedy_cooperate_force{forced_token}_sum"] += float(
                                    int(stats["greedy_action"]) == 1
                                )
                                if int(stats["greedy_action"]) != int(
                                    natural_stats["greedy_action"]
                                ):
                                    acc[f"action_flip_force{forced_token}_n"] += 1
                            acc["delta_p_cooperate_sum"] += float(
                                forced_stats[1]["p_cooperate"] - forced_stats[0]["p_cooperate"]
                            )
                            acc["delta_logit_margin_sum"] += float(
                                forced_stats[1]["logit_margin"] - forced_stats[0]["logit_margin"]
                            )
                            acc["delta_greedy_cooperate_sum"] += float(
                                int(forced_stats[1]["greedy_action"]) == 1
                            ) - float(int(forced_stats[0]["greedy_action"]) == 1)

                raw_next, rewards, done, infos = env.step(intended_actions)
                executed_actions = infos.get("executed_actions", intended_actions)
                true_f = float(infos.get("true_f", _as_float(env.current_multiplier)))

                wrapper.update(executed_actions)

                coop_step = float(np.mean([int(v) for v in executed_actions.values()]))
                reward_step = float(np.mean([_as_float(v) for v in rewards.values()]))
                welfare_step = float(np.sum([_as_float(v) for v in rewards.values()]))
                all_coop_step = float(
                    all(int(executed_actions.get(agent_id, 0)) == 1 for agent_id in agent_ids)
                )
                regime = _regime_label(true_f, n_agents=n_agents)
                f_key = f"{true_f:.3f}"

                regime_acc[regime]["n_rounds"] += 1
                regime_acc[regime]["coop_sum"] += coop_step
                regime_acc[regime]["reward_sum"] += reward_step
                regime_acc[regime]["welfare_sum"] += welfare_step
                regime_acc[regime]["all_coop_sum"] += all_coop_step

                f_acc[f_key]["n_rounds"] += 1
                f_acc[f_key]["coop_sum"] += coop_step
                f_acc[f_key]["reward_sum"] += reward_step
                f_acc[f_key]["welfare_sum"] += welfare_step
                f_acc[f_key]["all_coop_sum"] += all_coop_step
                if comm_enabled and len(sender_ids) > 0:
                    f_idx = f_key_to_idx.get(f_key)
                    if f_idx is not None:
                        for sender_id in sender_ids:
                            msg = int(proposed.get(sender_id, 0))
                            act = int(executed_actions.get(sender_id, 0))
                            if 0 <= msg < vocab_size and 0 <= act <= 1:
                                comm_counts[sender_id]["msg_f"][msg, f_idx] += 1.0
                                comm_counts[sender_id]["msg_action"][msg, act] += 1.0
                    if current_messages is not None:
                        for agent_id in agent_ids:
                            received_tokens = set()
                            for sender_id in sender_ids:
                                if sender_id == agent_id:
                                    continue
                                token = int(current_messages.get(sender_id, 0))
                                if token in (0, 1):
                                    received_tokens.add(token)
                            action_i = int(executed_actions.get(agent_id, 0))
                            if 0 in received_tokens:
                                f_acc[f_key]["msg0_n"] += 1
                                f_acc[f_key]["msg0_coop"] += float(action_i)
                            if 1 in received_tokens:
                                f_acc[f_key]["msg1_n"] += 1
                                f_acc[f_key]["msg1_coop"] += float(action_i)

                if posterior_strat:
                    for agent_id in agent_ids:
                        posterior_records.append(
                            {
                                "condition": condition,
                                "seed": int(train_seed),
                                "checkpoint": checkpoint_path,
                                "eval_policy": "greedy" if greedy else "sample",
                                "msg_intervention": ablation_label,
                                "history_intervention": history_intervention_label,
                                "sender_remap": sender_remap_label,
                                "cross_play": cross_play_label,
                                "true_f": float(true_f),
                                "f_hat": float(f_hat_by_agent.get(agent_id, 0.0)),
                                "action": int(executed_actions.get(agent_id, 0)),
                            }
                        )

                if collect_semantics:
                    for agent_id in agent_ids:
                        recv_pattern, recv_any_m0, recv_any_m1 = _build_received_pattern(
                            agent_id=agent_id,
                            sender_ids=sender_ids,
                            delivered_messages=current_messages,
                        )
                        row = {
                            "checkpoint": checkpoint_path,
                            "condition": condition,
                            "train_seed": int(train_seed),
                            "eval_seed": int(eval_seed),
                            "eval_policy": "greedy" if greedy else "sample",
                            "ablation": ablation_label,
                            "history_intervention": history_intervention_label,
                            "sender_remap": sender_remap_label,
                            "cross_play": cross_play_label,
                            "episode": int(_),
                            "t": int(steps),
                            "agent_id": str(agent_id),
                            "true_f": float(true_f),
                            "f_hat": float(f_hat_by_agent.get(agent_id, 0.0)),
                            "intended_action": int(intended_actions.get(agent_id, 0)),
                            "action": int(executed_actions.get(agent_id, 0)),
                            "reward": float(_as_float(rewards.get(agent_id, 0.0))),
                            "round_welfare": float(welfare_step),
                            "own_sent_msg": (
                                int(proposed[agent_id])
                                if (comm_enabled and agent_id in proposed)
                                else ""
                            ),
                            "recv_any_m0": int(recv_any_m0),
                            "recv_any_m1": int(recv_any_m1),
                            "recv_pattern": recv_pattern,
                        }
                        row.update(history_features_by_agent[agent_id])
                        msg_start = int(wrapper.message_start_idx)
                        for sender_id in sender_ids:
                            sender_idx = int(sender_ids.index(sender_id))
                            start = msg_start + sender_idx * int(vocab_size)
                            row[_delivered_msg_key(sender_id)] = (
                                int(current_messages.get(sender_id, 0))
                                if current_messages is not None
                                else ""
                            )
                            for token_idx in range(int(vocab_size)):
                                row[_obs_msg_key(sender_id, token_idx)] = float(
                                    aug_obs[agent_id][start + token_idx]
                                )
                        trace_rows.append(row)

                raw_obs = raw_next
                steps += 1

    rows = []
    for regime, acc in sorted(regime_acc.items()):
        n = max(1, int(acc["n_rounds"]))
        rows.append(
            {
                "checkpoint": checkpoint_path,
                "condition": condition,
                "train_seed": train_seed,
                "comm_enabled": int(comm_enabled),
                "eval_seed": eval_seed,
                "eval_policy": "greedy" if greedy else "sample",
                "ablation": ablation_label,
                "history_intervention": history_intervention_label,
                "sender_remap": sender_remap_label,
                "cross_play": cross_play_label,
                "scope": "regime",
                "key": regime,
                "n_rounds": int(acc["n_rounds"]),
                "coop_rate": float(acc["coop_sum"] / n),
                "avg_reward": float(acc["reward_sum"] / n),
                "p_all_cooperate": float(acc["all_coop_sum"] / n),
                "avg_welfare": float(acc["welfare_sum"] / n),
                "p_coop_given_m0": "",
                "p_coop_given_m1": "",
            }
        )

    for f_key, acc in sorted(f_acc.items(), key=lambda x: float(x[0])):
        n = max(1, int(acc["n_rounds"]))
        if comm_enabled and len(sender_ids) > 0:
            p_m0 = (
                float(acc["msg0_coop"] / float(acc["msg0_n"]))
                if int(acc["msg0_n"]) > 0
                else ""
            )
            p_m1 = (
                float(acc["msg1_coop"] / float(acc["msg1_n"]))
                if int(acc["msg1_n"]) > 0
                else ""
            )
        else:
            p_m0 = ""
            p_m1 = ""
        rows.append(
            {
                "checkpoint": checkpoint_path,
                "condition": condition,
                "train_seed": train_seed,
                "comm_enabled": int(comm_enabled),
                "eval_seed": eval_seed,
                "eval_policy": "greedy" if greedy else "sample",
                "ablation": ablation_label,
                "history_intervention": history_intervention_label,
                "sender_remap": sender_remap_label,
                "cross_play": cross_play_label,
                "scope": "f_value",
                "key": f_key,
                "n_rounds": int(acc["n_rounds"]),
                "coop_rate": float(acc["coop_sum"] / n),
                "avg_reward": float(acc["reward_sum"] / n),
                "p_all_cooperate": float(acc["all_coop_sum"] / n),
                "avg_welfare": float(acc["welfare_sum"] / n),
                "p_coop_given_m0": p_m0,
                "p_coop_given_m1": p_m1,
            }
        )
    comm_rows: List[Dict] = []
    if comm_enabled and len(sender_ids) > 0:
        all_msg_f = np.zeros((vocab_size, len(f_keys_sorted)), dtype=np.float64)
        all_msg_action = np.zeros((vocab_size, 2), dtype=np.float64)
        for sender_id in sender_ids:
            msg_f_counts = comm_counts[sender_id]["msg_f"]
            msg_action_counts = comm_counts[sender_id]["msg_action"]
            all_msg_f += msg_f_counts
            all_msg_action += msg_action_counts

            mi_f = _mi_null_independence_stats(
                msg_f_counts, n_perms=mi_null_perms, alpha=mi_alpha, rng=mi_rng
            )
            mi_a = _mi_null_independence_stats(
                msg_action_counts, n_perms=mi_null_perms, alpha=mi_alpha, rng=mi_rng
            )
            msg_entropy = _entropy_from_counts_1d(np.sum(msg_f_counts, axis=1))
            msg_entropy_max = float(np.log2(max(1, vocab_size)))
            base = {
                "checkpoint": checkpoint_path,
                "condition": condition,
                "train_seed": train_seed,
                "comm_enabled": int(comm_enabled),
                "eval_seed": eval_seed,
                "eval_policy": "greedy" if greedy else "sample",
                "ablation": ablation_label,
                "history_intervention": history_intervention_label,
                "sender_remap": sender_remap_label,
                "cross_play": cross_play_label,
                "scope": "comm",
                "key": sender_id,
                "h_message": float(msg_entropy),
                "h_message_max": float(msg_entropy_max),
            }
            comm_rows.append(
                {
                    **base,
                    "metric": "mi_message_f",
                    "mi": float(mi_f["mi_observed"]),
                    "mi_unit": "bits",
                    "mi_perm_p95": float(mi_f["mi_perm_p95"]),
                    "mi_p_value": float(mi_f["mi_p_value"]),
                    "mi_significant": bool(mi_f["mi_significant"]),
                    "mi_null_method": "independence_multinomial",
                    "mi_null_perms": int(mi_null_perms),
                    "n_pairs": int(np.sum(msg_f_counts)),
                }
            )
            comm_rows.append(
                {
                    **base,
                    "metric": "mi_message_action",
                    "mi": float(mi_a["mi_observed"]),
                    "mi_unit": "bits",
                    "mi_perm_p95": float(mi_a["mi_perm_p95"]),
                    "mi_p_value": float(mi_a["mi_p_value"]),
                    "mi_significant": bool(mi_a["mi_significant"]),
                    "mi_null_method": "independence_multinomial",
                    "mi_null_perms": int(mi_null_perms),
                    "n_pairs": int(np.sum(msg_action_counts)),
                }
            )

        mi_f_all = _mi_null_independence_stats(
            all_msg_f, n_perms=mi_null_perms, alpha=mi_alpha, rng=mi_rng
        )
        mi_a_all = _mi_null_independence_stats(
            all_msg_action, n_perms=mi_null_perms, alpha=mi_alpha, rng=mi_rng
        )
        all_entropy = _entropy_from_counts_1d(np.sum(all_msg_f, axis=1))
        all_entropy_max = float(np.log2(max(1, vocab_size)))
        comm_rows.append(
            {
                "checkpoint": checkpoint_path,
                "condition": condition,
                "train_seed": train_seed,
                "comm_enabled": int(comm_enabled),
                "eval_seed": eval_seed,
                "eval_policy": "greedy" if greedy else "sample",
                "ablation": ablation_label,
                "history_intervention": history_intervention_label,
                "sender_remap": sender_remap_label,
                "cross_play": cross_play_label,
                "scope": "comm",
                "key": "all_senders",
                "metric": "mi_message_f",
                "mi": float(mi_f_all["mi_observed"]),
                "mi_unit": "bits",
                "mi_perm_p95": float(mi_f_all["mi_perm_p95"]),
                "mi_p_value": float(mi_f_all["mi_p_value"]),
                "mi_significant": bool(mi_f_all["mi_significant"]),
                "mi_null_method": "independence_multinomial",
                "mi_null_perms": int(mi_null_perms),
                "h_message": float(all_entropy),
                "h_message_max": float(all_entropy_max),
                "n_pairs": int(np.sum(all_msg_f)),
            }
        )
        comm_rows.append(
            {
                "checkpoint": checkpoint_path,
                "condition": condition,
                "train_seed": train_seed,
                "comm_enabled": int(comm_enabled),
                "eval_seed": eval_seed,
                "eval_policy": "greedy" if greedy else "sample",
                "ablation": ablation_label,
                "history_intervention": history_intervention_label,
                "sender_remap": sender_remap_label,
                "cross_play": cross_play_label,
                "scope": "comm",
                "key": "all_senders",
                "metric": "mi_message_action",
                "mi": float(mi_a_all["mi_observed"]),
                "mi_unit": "bits",
                "mi_perm_p95": float(mi_a_all["mi_perm_p95"]),
                "mi_p_value": float(mi_a_all["mi_p_value"]),
                "mi_significant": bool(mi_a_all["mi_significant"]),
                "mi_null_method": "independence_multinomial",
                "mi_null_perms": int(mi_null_perms),
                "h_message": float(all_entropy),
                "h_message_max": float(all_entropy_max),
                "n_pairs": int(np.sum(all_msg_action)),
            }
        )
        all_resp = np.array(
            [x for values in responsiveness.values() for x in values], dtype=np.float64
        )
        if all_resp.size > 0:
            comm_rows.append(
                {
                    "checkpoint": checkpoint_path,
                    "condition": condition,
                    "train_seed": train_seed,
                    "comm_enabled": int(comm_enabled),
                    "eval_seed": eval_seed,
                    "eval_policy": "greedy" if greedy else "sample",
                    "ablation": ablation_label,
                    "history_intervention": history_intervention_label,
                    "sender_remap": sender_remap_label,
                    "cross_play": cross_play_label,
                    "scope": "comm",
                    "key": "all_agents",
                    "metric": "responsiveness_kl",
                    "value": float(np.mean(all_resp)),
                    "value_std": float(np.std(all_resp)),
                    "n_pairs": int(all_resp.size),
                }
            )
            for agent_id in agent_ids:
                arr = np.asarray(responsiveness.get(agent_id, []), dtype=np.float64)
                if arr.size == 0:
                    continue
                comm_rows.append(
                    {
                        "checkpoint": checkpoint_path,
                        "condition": condition,
                        "train_seed": train_seed,
                        "comm_enabled": int(comm_enabled),
                        "eval_seed": eval_seed,
                        "eval_policy": "greedy" if greedy else "sample",
                        "ablation": ablation_label,
                        "history_intervention": history_intervention_label,
                        "sender_remap": sender_remap_label,
                        "cross_play": cross_play_label,
                        "scope": "comm",
                        "key": str(agent_id),
                        "metric": "responsiveness_kl",
                        "value": float(np.mean(arr)),
                        "value_std": float(np.std(arr)),
                        "n_pairs": int(arr.size),
                    }
                )
        all_policy_entropy = np.array(
            [x for values in policy_entropy.values() for x in values],
            dtype=np.float64,
        )
        if all_policy_entropy.size > 0:
            comm_rows.append(
                {
                    "checkpoint": checkpoint_path,
                    "condition": condition,
                    "train_seed": train_seed,
                    "comm_enabled": int(comm_enabled),
                    "eval_seed": eval_seed,
                    "eval_policy": "greedy" if greedy else "sample",
                    "ablation": ablation_label,
                    "history_intervention": history_intervention_label,
                    "sender_remap": sender_remap_label,
                    "cross_play": cross_play_label,
                    "scope": "comm",
                    "key": "all_senders",
                    "metric": "policy_entropy_message",
                    "value": float(np.mean(all_policy_entropy)),
                    "value_std": float(np.std(all_policy_entropy)),
                    "n_pairs": int(all_policy_entropy.size),
                }
            )
            for sender_id in sender_ids:
                arr = np.asarray(policy_entropy.get(sender_id, []), dtype=np.float64)
                if arr.size == 0:
                    continue
                comm_rows.append(
                    {
                        "checkpoint": checkpoint_path,
                        "condition": condition,
                        "train_seed": train_seed,
                        "comm_enabled": int(comm_enabled),
                        "eval_seed": eval_seed,
                        "eval_policy": "greedy" if greedy else "sample",
                        "ablation": ablation_label,
                        "history_intervention": history_intervention_label,
                        "sender_remap": sender_remap_label,
                        "cross_play": cross_play_label,
                        "scope": "comm",
                        "key": str(sender_id),
                        "metric": "policy_entropy_message",
                        "value": float(np.mean(arr)),
                        "value_std": float(np.std(arr)),
                        "n_pairs": int(arr.size),
                    }
                )

    posterior_rows = []
    if posterior_strat:
        acc = defaultdict(lambda: {"n_obs": 0, "coop_sum": 0.0})
        for rec in posterior_records:
            true_f = float(rec["true_f"])
            if abs(true_f - 3.5) > 1e-6 and abs(true_f - 5.0) > 1e-6:
                continue
            b = _posterior_bin_label(float(rec["f_hat"]))
            key = (f"{true_f:.3f}", b)
            acc[key]["n_obs"] += 1
            acc[key]["coop_sum"] += float(rec["action"])
        for (true_f, bin_label), item in sorted(
            acc.items(), key=lambda x: (float(x[0][0]), x[0][1])
        ):
            n_obs = int(item["n_obs"])
            posterior_rows.append(
                {
                    "condition": condition,
                    "seed": int(train_seed),
                    "checkpoint": checkpoint_path,
                    "eval_policy": "greedy" if greedy else "sample",
                    "msg_intervention": ablation_label,
                    "history_intervention": history_intervention_label,
                    "sender_remap": sender_remap_label,
                    "cross_play": cross_play_label,
                    "true_f": true_f,
                    "fhat_bin": bin_label,
                    "n_obs": n_obs,
                    "p_cooperate": float(item["coop_sum"] / max(1, n_obs)),
                }
            )
    sender_semantics_rows = (
        _derive_sender_semantics_rows(trace_rows) if collect_semantics else []
    )
    receiver_semantics_rows = (
        _derive_receiver_semantics_rows(trace_rows) if collect_semantics else []
    )
    sender_causal_rows = (
        _sender_causal_summary_rows(
            sender_causal_acc,
            checkpoint_path=checkpoint_path,
            condition=condition,
            train_seed=train_seed,
            eval_seed=eval_seed,
            greedy=greedy,
            ablation_label=ablation_label,
            history_intervention_label=history_intervention_label,
            sender_remap_label=sender_remap_label,
            cross_play_label=cross_play_label,
            n_agents=n_agents,
        )
        if collect_sender_causal
        else []
    )
    return (
        rows,
        comm_rows,
        posterior_rows,
        trace_rows,
        sender_semantics_rows,
        receiver_semantics_rows,
        sender_causal_rows,
    )


def _write_csv(path: str, rows: List[Dict]):
    if len(rows) == 0:
        return
    fieldnames = [
        "checkpoint",
        "condition",
        "train_seed",
        "comm_enabled",
        "eval_seed",
        "eval_policy",
        "ablation",
        "history_intervention",
        "sender_remap",
        "cross_play",
        "scope",
        "key",
        "n_rounds",
        "coop_rate",
        "avg_reward",
        "p_all_cooperate",
        "avg_welfare",
        "p_coop_given_m0",
        "p_coop_given_m1",
    ]
    atomic_write_rows(path, rows, fieldnames)


def _write_comm_csv(path: str, rows: List[Dict]):
    if len(rows) == 0:
        return
    fieldnames = [
        "checkpoint",
        "condition",
        "train_seed",
        "comm_enabled",
        "eval_seed",
        "eval_policy",
        "ablation",
        "history_intervention",
        "sender_remap",
        "cross_play",
        "scope",
        "key",
        "metric",
        "mi",
        "mi_unit",
        "mi_perm_p95",
        "mi_p_value",
        "mi_significant",
        "mi_null_method",
        "mi_null_perms",
        "h_message",
        "h_message_max",
        "value",
        "value_std",
        "n_pairs",
    ]
    atomic_write_rows(path, rows, fieldnames)


def _write_posterior_strat_csv(path: str, rows: List[Dict]):
    if len(rows) == 0:
        return
    fieldnames = [
        "condition",
        "seed",
        "checkpoint",
        "eval_policy",
        "msg_intervention",
        "history_intervention",
        "sender_remap",
        "cross_play",
        "true_f",
        "fhat_bin",
        "n_obs",
        "p_cooperate",
    ]
    atomic_write_rows(path, rows, fieldnames)


def _write_trace_csv(path: str, rows: List[Dict]):
    if len(rows) == 0:
        return
    delivered_cols = sorted(
        {key for row in rows for key in row.keys() if key.startswith("delivered_msg_")}
    )
    obs_msg_cols = sorted(
        {key for row in rows for key in row.keys() if key.startswith("obs_msg_")}
    )
    fieldnames = [
        "checkpoint",
        "condition",
        "train_seed",
        "eval_seed",
        "eval_policy",
        "ablation",
        "history_intervention",
        "sender_remap",
        "cross_play",
        "episode",
        "t",
        "agent_id",
        "true_f",
        "f_hat",
        "intended_action",
        "action",
        "reward",
        "round_welfare",
        "obs_last_coop_fraction",
        "obs_own_last_action",
        "obs_ewma_coop",
        "own_sent_msg",
        *delivered_cols,
        *obs_msg_cols,
        "recv_any_m0",
        "recv_any_m1",
        "recv_pattern",
    ]
    atomic_write_rows(path, rows, fieldnames)


def _write_sender_semantics_csv(path: str, rows: List[Dict]):
    if len(rows) == 0:
        return
    fieldnames = [
        "checkpoint",
        "condition",
        "train_seed",
        "eval_seed",
        "eval_policy",
        "ablation",
        "history_intervention",
        "sender_remap",
        "cross_play",
        "sender_id",
        "summary",
        "fhat_bin",
        "action",
        "n_obs",
        "p_message_1",
    ]
    atomic_write_rows(path, rows, fieldnames)


def _write_receiver_semantics_csv(path: str, rows: List[Dict]):
    if len(rows) == 0:
        return
    fieldnames = [
        "checkpoint",
        "condition",
        "train_seed",
        "eval_seed",
        "eval_policy",
        "ablation",
        "history_intervention",
        "sender_remap",
        "cross_play",
        "receiver_id",
        "sender_id",
        "summary",
        "recv_pattern",
        "any_token",
        "sender_token",
        "fhat_bin",
        "n_obs",
        "p_cooperate",
    ]
    atomic_write_rows(path, rows, fieldnames)


def _write_sender_causal_csv(path: str, rows: List[Dict]):
    if len(rows) == 0:
        return
    fieldnames = [
        "checkpoint",
        "condition",
        "train_seed",
        "eval_seed",
        "eval_policy",
        "ablation",
        "history_intervention",
        "sender_remap",
        "cross_play",
        "sender_id",
        "receiver_id",
        "receiver_is_sender",
        "regime",
        "true_f",
        "n_obs",
        "natural_token0_rate",
        "natural_token1_rate",
        "natural_p_cooperate",
        "natural_logit_margin",
        "natural_greedy_cooperate",
        "p_cooperate_force0",
        "p_cooperate_force1",
        "delta_p_cooperate_1_minus_0",
        "logit_margin_force0",
        "logit_margin_force1",
        "delta_logit_margin_1_minus_0",
        "greedy_cooperate_force0",
        "greedy_cooperate_force1",
        "delta_greedy_cooperate_1_minus_0",
        "action_flip_rate_force0_vs_natural",
        "action_flip_rate_force1_vs_natural",
    ]
    atomic_write_rows(path, rows, fieldnames)


def _condition_summary(rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(lambda: {"n_rounds": 0, "coop_weighted": 0.0, "reward_weighted": 0.0})
    for row in rows:
        if row["scope"] != "regime":
            continue
        key = (
            row["condition"],
            row["key"],
            row.get("eval_policy", "sample"),
            row.get("ablation", "none"),
            row.get("history_intervention", "none"),
            row.get("sender_remap", "none"),
            row.get("cross_play", "none"),
        )
        n = int(row["n_rounds"])
        grouped[key]["n_rounds"] += n
        grouped[key]["coop_weighted"] += float(row["coop_rate"]) * n
        grouped[key]["reward_weighted"] += float(row["avg_reward"]) * n

    out = []
    for (
        condition,
        regime,
        eval_policy,
        ablation,
        history_intervention,
        sender_remap,
        cross_play,
    ), acc in sorted(grouped.items()):
        n = max(1, int(acc["n_rounds"]))
        out.append(
            {
                "condition": condition,
                "regime": regime,
                "eval_policy": eval_policy,
                "ablation": ablation,
                "history_intervention": history_intervention,
                "sender_remap": sender_remap,
                "cross_play": cross_play,
                "n_rounds": int(acc["n_rounds"]),
                "coop_rate": float(acc["coop_weighted"] / n),
                "avg_reward": float(acc["reward_weighted"] / n),
            }
        )
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints_glob", type=str, default="outputs/train/grid/*.pt")
    p.add_argument("--n_eval_episodes", type=int, default=300)
    p.add_argument("--eval_seed", type=int, default=9001)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--eval_sigmas", nargs="*", type=float, default=None)
    p.add_argument(
        "--observability_mode",
        type=str,
        default="private",
        choices=list(_OBSERVABILITY_MODE_CHOICES),
        help="Evaluation-only regime observation control. 'private' preserves the base family; public_* modes inject a shared multiplier signal for all agents.",
    )
    p.add_argument(
        "--public_signal_sigma",
        type=float,
        default=None,
        help="Shared public multiplier noise scale for --observability_mode=public_noisy. Defaults to the scalar eval sigma when available.",
    )
    p.add_argument(
        "--msg_intervention",
        type=str,
        default="none",
        choices=[
            "none",
            "marginal",
            "zeros",
            "fixed0",
            "fixed1",
            "flip",
            "uniform",
            "indep_random",
            "public_random",
            "public_marginal",
            "sender_shuffle",
            "permute_slots",
        ],
    )
    p.add_argument(
        "--history_intervention",
        type=str,
        default="none",
        choices=list(_HISTORY_INTERVENTION_CHOICES),
    )
    p.add_argument("--mi_null_perms", type=int, default=200)
    p.add_argument("--mi_alpha", type=float, default=0.05)
    p.add_argument("--cross_play_checkpoint", type=str, default="")
    p.add_argument("--sender_flip_map_json", type=str, default="")
    p.add_argument("--sender_remap_json", type=str, default="")
    p.add_argument("--sender_remap_label", type=str, default="none")
    p.add_argument("--posterior_strat", action="store_true")
    p.add_argument("--out_csv", type=str, default="outputs/train/grid/regime_eval.csv")
    p.add_argument("--out_comm_csv", type=str, default="")
    p.add_argument("--out_trace_csv", type=str, default="")
    p.add_argument("--out_sender_semantics_csv", type=str, default="")
    p.add_argument("--out_receiver_semantics_csv", type=str, default="")
    p.add_argument("--out_sender_causal_csv", type=str, default="")
    p.add_argument(
        "--out_condition_csv",
        type=str,
        default="outputs/train/grid/regime_eval_condition_summary.csv",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ckpts = sorted(glob.glob(args.checkpoints_glob))
    if len(ckpts) == 0:
        raise FileNotFoundError(f"no checkpoints matched: {args.checkpoints_glob}")
    if args.mi_null_perms <= 0:
        raise ValueError("mi_null_perms must be > 0")
    if not (0.0 < float(args.mi_alpha) < 1.0):
        raise ValueError("mi_alpha must be in (0, 1)")
    out_comm_csv = str(args.out_comm_csv or "").strip()
    if out_comm_csv == "":
        root, ext = os.path.splitext(args.out_csv)
        out_comm_csv = f"{root}_comm{ext or '.csv'}"
    out_trace_csv = str(args.out_trace_csv or "").strip()
    out_sender_semantics_csv = str(args.out_sender_semantics_csv or "").strip()
    out_receiver_semantics_csv = str(args.out_receiver_semantics_csv or "").strip()
    out_sender_causal_csv = str(args.out_sender_causal_csv or "").strip()
    collect_semantics = any(
        len(val) > 0
        for val in (out_trace_csv, out_sender_semantics_csv, out_receiver_semantics_csv)
    )
    collect_sender_causal = len(out_sender_causal_csv) > 0
    if collect_semantics:
        root, ext = os.path.splitext(args.out_csv)
        if out_trace_csv == "":
            out_trace_csv = f"{root}_trace{ext or '.csv'}"
        if out_sender_semantics_csv == "":
            out_sender_semantics_csv = f"{root}_sender_semantics{ext or '.csv'}"
        if out_receiver_semantics_csv == "":
            out_receiver_semantics_csv = f"{root}_receiver_semantics{ext or '.csv'}"
    if collect_sender_causal:
        root, ext = os.path.splitext(args.out_csv)
        if out_sender_causal_csv == "":
            out_sender_causal_csv = f"{root}_sender_causal{ext or '.csv'}"
    posterior_out_csv = ""
    if bool(args.posterior_strat):
        root, ext = os.path.splitext(args.out_csv)
        posterior_out_csv = f"{root}_posterior_strat{ext or '.csv'}"

    all_rows: List[Dict] = []
    all_comm_rows: List[Dict] = []
    all_posterior_rows: List[Dict] = []
    all_trace_rows: List[Dict] = []
    all_sender_semantics_rows: List[Dict] = []
    all_receiver_semantics_rows: List[Dict] = []
    all_sender_causal_rows: List[Dict] = []
    for idx, ckpt in enumerate(ckpts):
        run_seed = int(args.eval_seed + idx)
        print(f"[eval] {idx + 1}/{len(ckpts)} -> {ckpt} (seed={run_seed})")
        rows, comm_rows, posterior_rows, trace_rows, sender_semantics_rows, receiver_semantics_rows, sender_causal_rows = _eval_checkpoint(
            checkpoint_path=ckpt,
            n_eval_episodes=args.n_eval_episodes,
            eval_seed=run_seed,
            greedy=bool(args.greedy),
            eval_sigmas=args.eval_sigmas,
            observability_mode=str(args.observability_mode),
            public_signal_sigma=args.public_signal_sigma,
            msg_intervention=str(args.msg_intervention),
            history_intervention=str(args.history_intervention),
            mi_null_perms=int(args.mi_null_perms),
            mi_alpha=float(args.mi_alpha),
            cross_play_checkpoint=str(args.cross_play_checkpoint or ""),
            sender_flip_map_json=str(args.sender_flip_map_json or ""),
            sender_remap_json=str(args.sender_remap_json or ""),
            sender_remap_label=str(args.sender_remap_label or "none"),
            posterior_strat=bool(args.posterior_strat),
            collect_semantics=collect_semantics,
            collect_sender_causal=collect_sender_causal,
        )
        all_rows.extend(rows)
        all_comm_rows.extend(comm_rows)
        all_posterior_rows.extend(posterior_rows)
        all_trace_rows.extend(trace_rows)
        all_sender_semantics_rows.extend(sender_semantics_rows)
        all_receiver_semantics_rows.extend(receiver_semantics_rows)
        all_sender_causal_rows.extend(sender_causal_rows)

    _write_csv(args.out_csv, all_rows)
    _write_comm_csv(out_comm_csv, all_comm_rows)
    if bool(args.posterior_strat):
        _write_posterior_strat_csv(posterior_out_csv, all_posterior_rows)
    if collect_semantics:
        _write_trace_csv(out_trace_csv, all_trace_rows)
        _write_sender_semantics_csv(out_sender_semantics_csv, all_sender_semantics_rows)
        _write_receiver_semantics_csv(out_receiver_semantics_csv, all_receiver_semantics_rows)
    if collect_sender_causal:
        _write_sender_causal_csv(out_sender_causal_csv, all_sender_causal_rows)
    cond_rows = _condition_summary(all_rows)
    atomic_write_rows(
        args.out_condition_csv,
        cond_rows,
        [
            "condition",
            "regime",
            "eval_policy",
            "ablation",
            "history_intervention",
            "sender_remap",
            "cross_play",
            "n_rounds",
            "coop_rate",
            "avg_reward",
        ],
    )

    print(f"[eval] rows={len(all_rows)} out={args.out_csv}")
    print(f"[eval] comm_rows={len(all_comm_rows)} out={out_comm_csv}")
    if bool(args.posterior_strat):
        print(
            f"[eval] posterior_rows={len(all_posterior_rows)} out={posterior_out_csv}"
        )
    if collect_semantics:
        print(f"[eval] trace_rows={len(all_trace_rows)} out={out_trace_csv}")
        print(
            f"[eval] sender_semantics_rows={len(all_sender_semantics_rows)} "
            f"out={out_sender_semantics_csv}"
        )
        print(
            f"[eval] receiver_semantics_rows={len(all_receiver_semantics_rows)} "
            f"out={out_receiver_semantics_csv}"
        )
    if collect_sender_causal:
        print(
            f"[eval] sender_causal_rows={len(all_sender_causal_rows)} "
            f"out={out_sender_causal_csv}"
        )
    print(f"[eval] condition_summary_rows={len(cond_rows)} out={args.out_condition_csv}")
    for row in cond_rows:
        print(
            "[summary] "
            f"{row['condition']} {row['regime']} "
            f"[{row['eval_policy']}:{row['ablation']}:{row['history_intervention']}:"
            f"{row['sender_remap']}:{row['cross_play']}] "
            f"coop={row['coop_rate']:.3f} reward={row['avg_reward']:.3f} "
            f"n_rounds={row['n_rounds']}"
        )


if __name__ == "__main__":
    main()
