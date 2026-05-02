from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.environments import pgg_parallel_v0


@dataclass
class AuditConfig:
    n_agents: int = 4
    num_game_iterations: int = 100
    mult_fact: tuple = (0.5, 1.5, 2.5, 3.5, 5.0)
    F: tuple = (0.5, 1.5, 2.5, 3.5, 5.0)
    uncertainties: tuple = (0.5, 0.5, 0.5, 0.5)
    fraction: bool = False
    rho: float = 0.05
    epsilon_tremble: float = 0.05
    endowment: float = 4.0


def _transition_matrix(F: List[float], rho: float) -> np.ndarray:
    n = len(F)
    P = np.full((n, n), 0.0, dtype=np.float64)
    if n == 1:
        P[0, 0] = 1.0
        return P
    off_diag = rho / float(n - 1)
    for i in range(n):
        for j in range(n):
            if i == j:
                P[i, j] = 1.0 - rho
            else:
                P[i, j] = off_diag
    return P


def _log_gaussian_pdf(x: np.ndarray, mean: float, sigma: np.ndarray) -> np.ndarray:
    var = sigma * sigma
    return -0.5 * (np.log(2.0 * np.pi * var) + ((x - mean) ** 2) / var)


def _normalize(v: np.ndarray) -> np.ndarray:
    s = np.sum(v)
    if s <= 0.0 or not np.isfinite(s):
        return np.ones_like(v) / float(len(v))
    return v / s


def regime_audit(
    env_config: Dict,
    n_sessions: int = 1000,
    posterior_threshold: float = 0.9,
):
    F = [float(v) for v in env_config.get("F", env_config.get("mult_fact", []))]
    if len(F) == 0:
        raise ValueError("env_config must include non-empty F or mult_fact")
    sigmas = np.array(env_config.get("uncertainties", [0.0] * int(env_config["n_agents"])))
    sigmas = sigmas.astype(np.float64) + 1e-6
    rho = float(env_config.get("rho", 0.05))
    T = int(env_config["num_game_iterations"])
    n_agents = int(env_config["n_agents"])

    P = _transition_matrix(F, rho)
    convergence_times = []

    for _ in range(n_sessions):
        env = pgg_parallel_v0.parallel_env(env_config)
        obs = env.reset()
        actions = {f"agent_{i}": 0 for i in range(n_agents)}

        pi = np.ones((len(F),), dtype=np.float64) / float(len(F))
        previous_true_f = None
        active_timer = None

        for _t in range(T):
            current_true_f = float(env.current_multiplier.item())

            # Predict/update posterior with private f_hat observations at current round.
            if _t > 0:
                pi = P.T @ pi
                pi = _normalize(pi)

            f_hats = np.array([float(np.asarray(obs[a]).reshape(-1)[0]) for a in env.possible_agents])

            log_lik = np.zeros((len(F),), dtype=np.float64)
            for j, f_val in enumerate(F):
                log_lik[j] = np.sum(_log_gaussian_pdf(f_hats, f_val, sigmas))

            ll_shift = np.max(log_lik)
            lik = np.exp(log_lik - ll_shift)
            pi = _normalize(pi * lik)

            if previous_true_f is not None and current_true_f != previous_true_f:
                active_timer = 0

            if active_timer is not None:
                if float(np.max(pi)) > posterior_threshold:
                    convergence_times.append(active_timer)
                    active_timer = None
                else:
                    active_timer += 1

            obs, _rewards, done, _infos = env.step(actions)
            previous_true_f = current_true_f
            if done:
                break

    if len(convergence_times) == 0:
        stats = {"mean": np.nan, "median": np.nan, "p90": np.nan, "n_switches": 0}
    else:
        arr = np.array(convergence_times, dtype=np.float64)
        stats = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "n_switches": int(arr.size),
        }

    stats["recommendation"] = (
        "increase_sigma_or_reduce_F"
        if (np.isfinite(stats["median"]) and stats["median"] <= 2.0)
        else "ok"
    )
    return stats
