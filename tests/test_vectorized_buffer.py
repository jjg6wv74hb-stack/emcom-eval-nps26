import numpy as np

from src.algos.trajectory_buffer import VectorizedTrajectoryBuffer


def _obs(value):
    return np.array([value, 4.0], dtype=np.float32)


def test_vectorized_buffer_flatten_and_env_views():
    agent_ids = ["agent_0", "agent_1"]
    buffer = VectorizedTrajectoryBuffer(
        agent_ids=agent_ids,
        T=2,
        obs_dim=2,
        value_obs_dim=2,
        n_envs=2,
        comm_enabled=True,
        vocab_size=2,
        sender_ids=["agent_0"],
    )

    buffer.store_step(
        obs_batch=[
            {"agent_0": _obs(1.0), "agent_1": _obs(1.1)},
            {"agent_0": _obs(2.0), "agent_1": _obs(2.1)},
        ],
        actions_batch=[
            {"agent_0": 0, "agent_1": 1},
            {"agent_0": 1, "agent_1": 0},
        ],
        rewards_batch=[
            {"agent_0": 1.0, "agent_1": 2.0},
            {"agent_0": 3.0, "agent_1": 4.0},
        ],
        raw_rewards_batch=[
            {"agent_0": 10.0, "agent_1": 20.0},
            {"agent_0": 30.0, "agent_1": 40.0},
        ],
        values_batch=[
            {"agent_0": 0.1, "agent_1": 0.2},
            {"agent_0": 0.3, "agent_1": 0.4},
        ],
        log_probs_batch=[
            {"agent_0": -0.1, "agent_1": -0.2},
            {"agent_0": -0.3, "agent_1": -0.4},
        ],
        done_batch=[False, False],
        executed_actions_batch=[
            {"agent_0": 0, "agent_1": 1},
            {"agent_0": 1, "agent_1": 0},
        ],
        flips_batch=[
            {"agent_0": False, "agent_1": False},
            {"agent_0": False, "agent_1": True},
        ],
        true_f_batch=[1.5, 2.5],
        f_hats_batch=[
            {"agent_0": _obs(1.0), "agent_1": _obs(1.1)},
            {"agent_0": _obs(2.0), "agent_1": _obs(2.1)},
        ],
        messages_batch=[
            {"agent_0": 0},
            {"agent_0": 1},
        ],
        value_obs_batch=[
            {"agent_0": _obs(1.0), "agent_1": _obs(1.1)},
            {"agent_0": _obs(2.0), "agent_1": _obs(2.1)},
        ],
        message_actions_batch=[
            {"agent_0": 0},
            {"agent_0": 1},
        ],
        message_log_probs_batch=[
            {"agent_0": -0.5},
            {"agent_0": -0.6},
        ],
        listening_bonus_batch=[
            {"agent_0": -0.01, "agent_1": -0.02},
            {"agent_0": -0.03, "agent_1": -0.04},
        ],
    )
    buffer.store_step(
        obs_batch=[
            {"agent_0": _obs(1.2), "agent_1": _obs(1.3)},
            {"agent_0": _obs(2.2), "agent_1": _obs(2.3)},
        ],
        actions_batch=[
            {"agent_0": 1, "agent_1": 1},
            {"agent_0": 0, "agent_1": 0},
        ],
        rewards_batch=[
            {"agent_0": 5.0, "agent_1": 6.0},
            {"agent_0": 7.0, "agent_1": 8.0},
        ],
        raw_rewards_batch=[
            {"agent_0": 50.0, "agent_1": 60.0},
            {"agent_0": 70.0, "agent_1": 80.0},
        ],
        values_batch=[
            {"agent_0": 0.5, "agent_1": 0.6},
            {"agent_0": 0.7, "agent_1": 0.8},
        ],
        log_probs_batch=[
            {"agent_0": -0.7, "agent_1": -0.8},
            {"agent_0": -0.9, "agent_1": -1.0},
        ],
        done_batch=[True, True],
        executed_actions_batch=[
            {"agent_0": 1, "agent_1": 1},
            {"agent_0": 0, "agent_1": 0},
        ],
        flips_batch=[
            {"agent_0": False, "agent_1": False},
            {"agent_0": False, "agent_1": False},
        ],
        true_f_batch=[1.5, 2.5],
        f_hats_batch=[
            {"agent_0": _obs(1.2), "agent_1": _obs(1.3)},
            {"agent_0": _obs(2.2), "agent_1": _obs(2.3)},
        ],
        messages_batch=[
            {"agent_0": 1},
            {"agent_0": 0},
        ],
        value_obs_batch=[
            {"agent_0": _obs(1.2), "agent_1": _obs(1.3)},
            {"agent_0": _obs(2.2), "agent_1": _obs(2.3)},
        ],
        message_actions_batch=[
            {"agent_0": 1},
            {"agent_0": 0},
        ],
        message_log_probs_batch=[
            {"agent_0": -0.7},
            {"agent_0": -0.8},
        ],
        listening_bonus_batch=[
            {"agent_0": -0.05, "agent_1": -0.06},
            {"agent_0": -0.07, "agent_1": -0.08},
        ],
    )

    flat = buffer.flatten()
    assert flat.t == 4
    assert flat.observations.shape == (4, 2, 2)
    assert flat.messages.shape == (4, 1)
    assert np.allclose(flat.true_f[:4], np.array([1.5, 2.5, 1.5, 2.5], dtype=np.float32))

    env0 = buffer.to_single_env_buffer(0)
    env1 = buffer.to_single_env_buffer(1)
    assert env0.t == 2
    assert env1.t == 2
    assert np.allclose(env0.true_f[:2], np.array([1.5, 1.5], dtype=np.float32))
    assert np.allclose(env1.true_f[:2], np.array([2.5, 2.5], dtype=np.float32))
    assert env1.flips[0, 1]


def test_vectorized_buffer_gae_matches_per_env_returns():
    buffer = VectorizedTrajectoryBuffer(
        agent_ids=["agent_0"],
        T=3,
        obs_dim=2,
        n_envs=2,
    )
    buffer.rewards[:3, :, 0] = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ],
        dtype=np.float32,
    )
    buffer.values[:3, :, 0] = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ],
        dtype=np.float32,
    )
    buffer.dones[:3] = np.array(
        [
            [False, False],
            [False, False],
            [True, True],
        ],
        dtype=bool,
    )
    buffer.t = 3

    advantages, returns = buffer.compute_gae(
        last_values=np.zeros((2, 1), dtype=np.float32),
        gamma=0.99,
        lam=0.95,
    )

    assert advantages.shape == (6, 1)
    assert returns.shape == (6, 1)
    assert abs(float(advantages[0, 0]) - 4.7728) < 1e-2
    assert abs(float(advantages[1, 0]) - 47.728) < 1e-1
    assert abs(float(returns[4, 0]) - 3.0) < 1e-4
    assert abs(float(returns[5, 0]) - 30.0) < 1e-4
