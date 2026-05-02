import numpy as np

from src.algos.trajectory_buffer import TrajectoryBuffer


def test_gae_simple_case():
    buffer = TrajectoryBuffer(agent_ids=["agent_0"], T=3, obs_dim=5)
    buffer.rewards[:3] = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    buffer.values[:3] = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    buffer.dones[:3] = np.array([False, False, True])
    buffer.t = 3

    advantages, returns = buffer.compute_gae(
        last_values=np.array([0.0], dtype=np.float32),
        gamma=0.99,
        lam=0.95,
    )

    assert abs(float(advantages[2, 0]) - 0.0) < 1e-4
    assert abs(float(advantages[1, 0]) - 2.97) < 1e-2
    # 1.98 + 0.99 * 0.95 * 2.97 ~= 4.7728
    assert abs(float(advantages[0, 0]) - 4.7728) < 1e-2
    assert abs(float(returns[2, 0]) - 3.0) < 1e-4
