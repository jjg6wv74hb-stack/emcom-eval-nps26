import numpy as np

from src.algos.trajectory_buffer import TrajectoryBuffer
from src.analysis.regime_audit import regime_audit
from src.logging import SessionLogger


def _dummy_transition(t, agent_ids):
    obs = {a: np.array([1.5 + 0.1 * t, 4.0], dtype=np.float32) for a in agent_ids}
    actions = {a: (t + i) % 2 for i, a in enumerate(agent_ids)}
    rewards = {a: float(1.0 + i + t) for i, a in enumerate(agent_ids)}
    values = {a: float(0.5 + i) for i, a in enumerate(agent_ids)}
    log_probs = {a: -0.2 for a in agent_ids}
    executed = actions.copy()
    flips = {a: False for a in agent_ids}
    f_hats = obs
    return obs, actions, rewards, values, log_probs, executed, flips, f_hats


def test_session_logger_and_consolidate(tmp_path):
    agent_ids = ["agent_0", "agent_1"]
    logger = SessionLogger(str(tmp_path), condition_name="condA", seed=7)

    for sess in range(2):
        buffer = TrajectoryBuffer(agent_ids=agent_ids, T=3, obs_dim=2)
        for t in range(3):
            obs, actions, rewards, values, log_probs, executed, flips, f_hats = _dummy_transition(
                t + 10 * sess, agent_ids
            )
            buffer.store(
                obs=obs,
                actions=actions,
                rewards=rewards,
                values=values,
                log_probs=log_probs,
                done=(t == 2),
                executed_actions=executed,
                flips=flips,
                true_f=2.5,
                f_hats=f_hats,
            )
        path = logger.log_session(buffer)
        assert path.endswith(".npz")

    consolidated = logger.consolidate(delete_parts=False)
    with np.load(consolidated, allow_pickle=False) as data:
        assert data["true_f"].shape == (2, 3)
        assert data["f_t"].shape == (2, 3)
        assert data["regime_t"].shape == (2, 3)
        assert np.isin(data["regime_t"], [0, 1, 2]).all()
        assert data["executed_actions"].shape == (2, 3, 2)
        assert data["welfare"].shape == (2, 3)


def test_regime_audit_runs():
    env_config = dict(
        n_agents=4,
        num_game_iterations=20,
        mult_fact=[0.5, 1.5, 2.5, 3.5, 5.0],
        F=[0.5, 1.5, 2.5, 3.5, 5.0],
        uncertainties=[0.5, 0.5, 0.5, 0.5],
        fraction=False,
        rho=0.05,
        epsilon_tremble=0.05,
        endowment=4.0,
    )
    out = regime_audit(env_config=env_config, n_sessions=3)
    assert "mean" in out
    assert "median" in out
    assert "p90" in out
    assert "recommendation" in out
    assert out["recommendation"] in {"ok", "increase_sigma_or_reduce_F"}
