from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np

# Allow direct execution: `python src/analysis/run_bayesian_baseline1.py ...`
if __package__ is None or __package__ == "":
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from src.baselines import (
    evaluate_action_predictions,
    fit_empirical_cooperation_table,
    forward_filter_sequence,
    predict_cooperation_from_posterior,
)


def _nearest_regime_idx(true_f: np.ndarray, f_values: np.ndarray) -> np.ndarray:
    d = np.abs(true_f[..., None] - f_values[None, ...])
    return np.argmin(d, axis=-1)


def _ensure_session_layout(
    true_f: np.ndarray, f_hats: np.ndarray, executed_actions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Coerce arrays into shapes:
    - true_f: (S, T)
    - f_hats: (S, T, N)
    - executed_actions: (S, T, N)
    """
    true_f = np.asarray(true_f, dtype=np.float64)
    f_hats = np.asarray(f_hats, dtype=np.float64)
    executed_actions = np.asarray(executed_actions, dtype=np.float64)

    if true_f.ndim == 1 and f_hats.ndim == 2 and executed_actions.ndim == 2:
        return (
            true_f[None, :],
            f_hats[None, :, :],
            executed_actions[None, :, :],
        )
    if true_f.ndim == 2 and f_hats.ndim == 3 and executed_actions.ndim == 3:
        return true_f, f_hats, executed_actions

    raise ValueError(
        "expected (T,)/(T,N)/(T,N) or (S,T)/(S,T,N)/(S,T,N) arrays for "
        "true_f/f_hats/executed_actions"
    )


def _load_triplet(path: str):
    with np.load(path, allow_pickle=False) as data:
        true_f = data["true_f"]
        f_hats = data["f_hats"]
        if "executed_actions" in data.files:
            executed = data["executed_actions"]
        else:
            raise KeyError(f"{path} does not contain `executed_actions`")
    return _ensure_session_layout(true_f=true_f, f_hats=f_hats, executed_actions=executed)


def run_baseline1(
    train_npz: str,
    eval_npz: str,
    f_values,
    sigmas,
    rho: float,
    laplace_alpha: float = 1.0,
) -> Dict:
    f_values = np.asarray([float(v) for v in f_values], dtype=np.float64)
    sigmas = np.asarray([float(v) for v in sigmas], dtype=np.float64)
    if sigmas.ndim != 1:
        raise ValueError("sigmas must be 1D")

    train_true_f, _train_f_hats, train_actions = _load_triplet(train_npz)
    eval_true_f, eval_f_hats, eval_actions = _load_triplet(eval_npz)
    n_sessions, T, n_agents = eval_actions.shape
    if sigmas.shape[0] != n_agents:
        raise ValueError("len(sigmas) must equal n_agents in eval data")

    cbar = fit_empirical_cooperation_table(
        true_f=train_true_f,
        executed_actions=train_actions,
        f_values=f_values,
        laplace_alpha=laplace_alpha,
    )

    pred_all = []
    y_all = []
    regime_acc = []
    for s in range(n_sessions):
        post = forward_filter_sequence(
            f_hats=eval_f_hats[s],
            f_values=f_values,
            sigmas=sigmas,
            rho=float(rho),
        )
        pred_t = predict_cooperation_from_posterior(posteriors=post, cbar=cbar)
        if T >= 2:
            pred_all.append(pred_t[:-1])
            y_all.append(eval_actions[s, 1:, :])

        pred_regime = np.argmax(post, axis=1)
        true_regime = _nearest_regime_idx(eval_true_f[s], f_values=f_values)
        regime_acc.append(float(np.mean(pred_regime == true_regime)))

    if len(pred_all) == 0:
        raise ValueError("evaluation trajectories must have T >= 2")

    pred_mat = np.concatenate(pred_all, axis=0)
    y_mat = np.concatenate(y_all, axis=0)

    overall = evaluate_action_predictions(pred_probs=pred_mat, y_true=y_mat)
    per_agent = {}
    for i in range(n_agents):
        met = evaluate_action_predictions(
            pred_probs=pred_mat[:, i : i + 1],
            y_true=y_mat[:, i : i + 1],
        )
        per_agent[f"agent_{i}"] = met

    return {
        "train_npz": train_npz,
        "eval_npz": eval_npz,
        "f_values": [float(v) for v in f_values.tolist()],
        "sigmas": [float(v) for v in sigmas.tolist()],
        "rho": float(rho),
        "laplace_alpha": float(laplace_alpha),
        "n_sessions_eval": int(n_sessions),
        "T_eval": int(T),
        "overall": overall,
        "per_agent": per_agent,
        "regime_decode_accuracy_mean": float(np.mean(regime_acc)),
        "regime_decode_accuracy_std": float(np.std(regime_acc)),
        "cbar_table": cbar.tolist(),
        "evaluation_protocol": {
            "horizons_supported": [1],
            "alignment": "predict_a_tplus1_from_pi_t",
            "input_mode": "teacher_forced_observations",
            "open_loop_supported": False,
            "notes": (
                "Baseline-1 scaffold currently supports one-step evaluation only. "
                "Spec-required multi-step H={5,10,20} and teacher-forced vs open-loop "
                "modes are pending."
            ),
        },
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_npz", type=str, required=True)
    p.add_argument("--eval_npz", type=str, required=True)
    p.add_argument("--F", nargs="*", type=float, default=[0.5, 1.5, 2.5, 3.5, 5.0])
    p.add_argument("--sigmas", nargs="*", type=float, default=[0.5, 0.5, 0.5, 0.5])
    p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--laplace_alpha", type=float, default=1.0)
    p.add_argument(
        "--out_json",
        type=str,
        default="outputs/baselines/baseline1_summary.json",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out = run_baseline1(
        train_npz=args.train_npz,
        eval_npz=args.eval_npz,
        f_values=args.F,
        sigmas=args.sigmas,
        rho=args.rho,
        laplace_alpha=args.laplace_alpha,
    )
    parent = os.path.dirname(args.out_json)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
