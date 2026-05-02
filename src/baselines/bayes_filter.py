from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def build_sticky_transition(f_values: Sequence[float], rho: float) -> np.ndarray:
    """
    Build sticky Markov transition matrix for regimes in `f_values`.
    P[i, j] = P(f_{t+1}=f_j | f_t=f_i)
    """
    f_values = [float(v) for v in f_values]
    n = len(f_values)
    if n == 0:
        raise ValueError("f_values must be non-empty")
    if n == 1:
        return np.array([[1.0]], dtype=np.float64)
    if rho < 0.0 or rho > 1.0:
        raise ValueError("rho must be in [0, 1]")

    P = np.full((n, n), fill_value=float(rho) / float(n - 1), dtype=np.float64)
    np.fill_diagonal(P, 1.0 - float(rho))
    return P


def _normalize(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    s = float(np.sum(p))
    if not np.isfinite(s) or s <= 0.0:
        return np.ones_like(p) / float(len(p))
    return p / s


def _log_gaussian_pdf(x: np.ndarray, mean: float, sigma: np.ndarray) -> np.ndarray:
    var = sigma * sigma
    return -0.5 * (np.log(2.0 * np.pi * var) + ((x - mean) ** 2) / var)


def _log_observation_likelihood(
    f_hats_t: np.ndarray, regime_value: float, sigmas: np.ndarray
) -> float:
    """
    Log p(f_hats_t | f_t = regime_value) under independent Gaussian channels.
    Handles zero-noise channels as exact-observation deltas.
    """
    f_hats_t = np.asarray(f_hats_t, dtype=np.float64).reshape(-1)
    sigmas = np.asarray(sigmas, dtype=np.float64).reshape(-1)
    if f_hats_t.shape != sigmas.shape:
        raise ValueError("f_hats_t and sigmas must have matching shape")

    nonzero = sigmas > 0.0
    ll = 0.0
    if np.any(nonzero):
        ll += float(np.sum(_log_gaussian_pdf(f_hats_t[nonzero], float(regime_value), sigmas[nonzero])))
    if np.any(~nonzero):
        exact = np.isclose(f_hats_t[~nonzero], float(regime_value), atol=1e-8)
        if not bool(np.all(exact)):
            return -1e12
    return float(ll)


def forward_filter_sequence(
    f_hats: np.ndarray,
    f_values: Sequence[float],
    sigmas: Sequence[float],
    rho: float,
    prior: Optional[np.ndarray] = None,
    observed_agent_idx: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Exact HMM forward pass:
    pi_t = P(f_t | f_hat_{1:t})
    """
    f_hats = np.asarray(f_hats, dtype=np.float64)
    if f_hats.ndim != 2:
        raise ValueError("f_hats must have shape (T, n_agents)")
    T, n_agents = f_hats.shape
    if T <= 0:
        raise ValueError("f_hats must have at least one timestep")

    f_values = np.asarray([float(v) for v in f_values], dtype=np.float64)
    K = int(f_values.shape[0])
    if K == 0:
        raise ValueError("f_values must be non-empty")

    sigmas = np.asarray([float(v) for v in sigmas], dtype=np.float64).reshape(-1)
    if sigmas.shape[0] != n_agents:
        raise ValueError("len(sigmas) must match n_agents in f_hats")

    if observed_agent_idx is None:
        observed_agent_idx = list(range(n_agents))
    observed_agent_idx = [int(i) for i in observed_agent_idx]
    if len(observed_agent_idx) == 0:
        raise ValueError("observed_agent_idx must be non-empty")

    P = build_sticky_transition(f_values=f_values, rho=float(rho))
    post = np.zeros((T, K), dtype=np.float64)

    if prior is None:
        pi_prev = np.ones((K,), dtype=np.float64) / float(K)
    else:
        pi_prev = _normalize(np.asarray(prior, dtype=np.float64).reshape(K))

    for t in range(T):
        pi_pred = pi_prev if t == 0 else _normalize(pi_prev @ P)
        obs_t = f_hats[t, observed_agent_idx]
        sig_t = sigmas[observed_agent_idx]
        log_lik = np.array(
            [_log_observation_likelihood(obs_t, regime_value=fv, sigmas=sig_t) for fv in f_values],
            dtype=np.float64,
        )
        log_lik -= np.max(log_lik)
        lik = np.exp(log_lik)
        pi_t = _normalize(pi_pred * lik)
        post[t] = pi_t
        pi_prev = pi_t

    return post


def _to_flat_time_major(true_f: np.ndarray, executed_actions: np.ndarray):
    true_f = np.asarray(true_f, dtype=np.float64)
    executed_actions = np.asarray(executed_actions, dtype=np.float64)

    if true_f.ndim == 1 and executed_actions.ndim == 2:
        return true_f.reshape(-1), executed_actions

    if true_f.ndim == 2 and executed_actions.ndim == 3:
        if true_f.shape[:2] != executed_actions.shape[:2]:
            raise ValueError("true_f and executed_actions must align on (sessions, T)")
        return true_f.reshape(-1), executed_actions.reshape(-1, executed_actions.shape[-1])

    raise ValueError(
        "expected true_f/actions with shapes (T,)/(T,N) or (S,T)/(S,T,N)"
    )


def _nearest_regime_indices(true_f_flat: np.ndarray, f_values: np.ndarray) -> np.ndarray:
    d = np.abs(true_f_flat[:, None] - f_values[None, :])
    return np.argmin(d, axis=1)


def fit_empirical_cooperation_table(
    true_f: np.ndarray,
    executed_actions: np.ndarray,
    f_values: Sequence[float],
    laplace_alpha: float = 1.0,
) -> np.ndarray:
    """
    Estimate cbar[j, k] = mean cooperation of agent j under regime k.
    """
    true_f_flat, actions_flat = _to_flat_time_major(true_f=true_f, executed_actions=executed_actions)
    f_values = np.asarray([float(v) for v in f_values], dtype=np.float64)
    K = int(f_values.shape[0])
    n_rounds, n_agents = actions_flat.shape
    if true_f_flat.shape[0] != n_rounds:
        raise ValueError("true_f and executed_actions must have matching timesteps")
    if K == 0:
        raise ValueError("f_values must be non-empty")
    if laplace_alpha < 0.0:
        raise ValueError("laplace_alpha must be >= 0")

    regime_idx = _nearest_regime_indices(true_f_flat=true_f_flat, f_values=f_values)
    coop_sum = np.zeros((n_agents, K), dtype=np.float64)
    rounds_per_regime = np.zeros((K,), dtype=np.float64)

    for t in range(n_rounds):
        k = int(regime_idx[t])
        rounds_per_regime[k] += 1.0
        coop_sum[:, k] += actions_flat[t]

    if laplace_alpha > 0.0:
        # Bernoulli smoothing: add alpha pseudo-counts to coop and defect.
        denom = rounds_per_regime[None, :] + 2.0 * float(laplace_alpha)
        cbar = (coop_sum + float(laplace_alpha)) / np.maximum(denom, 1e-12)
    else:
        denom = np.maximum(rounds_per_regime[None, :], 1.0)
        cbar = coop_sum / denom

    return cbar.astype(np.float64)


def predict_cooperation_from_posterior(posteriors: np.ndarray, cbar: np.ndarray) -> np.ndarray:
    """
    posteriors:
      - shape (K,) -> returns (N,)
      - shape (T, K) -> returns (T, N)
    cbar: shape (N, K)
    """
    post = np.asarray(posteriors, dtype=np.float64)
    cbar = np.asarray(cbar, dtype=np.float64)
    if cbar.ndim != 2:
        raise ValueError("cbar must have shape (N, K)")
    n_agents, K = cbar.shape

    if post.ndim == 1:
        if post.shape[0] != K:
            raise ValueError("posterior and cbar regime dimensions differ")
        return (post @ cbar.T).astype(np.float64)

    if post.ndim == 2:
        if post.shape[1] != K:
            raise ValueError("posterior and cbar regime dimensions differ")
        return (post @ cbar.T).astype(np.float64)

    raise ValueError("posteriors must have shape (K,) or (T, K)")


def evaluate_action_predictions(pred_probs: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    pred_probs = np.asarray(pred_probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    if pred_probs.shape != y_true.shape:
        raise ValueError("pred_probs and y_true must have identical shape")
    if pred_probs.ndim != 2:
        raise ValueError("expected shape (T, N)")
    p = np.clip(pred_probs, 1e-6, 1.0 - 1e-6)
    y = np.clip(y_true, 0.0, 1.0)

    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    accuracy = float(np.mean((p >= 0.5).astype(np.float64) == y))
    return {"brier": brier, "logloss": logloss, "accuracy@0.5": accuracy}
