import numpy as np
try:
    from gym.spaces import Box
except ImportError:  # pragma: no cover - compatibility path
    from gymnasium.spaces import Box

from src.environments import pgg_parallel_v0


def make_env(**overrides):
    cfg = dict(
        n_agents=4,
        num_game_iterations=10,
        mult_fact=[0.5, 1.5, 2.5, 3.5, 5.0],
        F=[0.5, 1.5, 2.5, 3.5, 5.0],
        uncertainties=[0.5, 0.5, 0.5, 0.5],
        fraction=False,
        rho=0.05,
        epsilon_tremble=0.05,
        endowment=4.0,
    )
    cfg.update(overrides)
    return pgg_parallel_v0.parallel_env(cfg)


def _f_hat(obs_tensor):
    return float(np.asarray(obs_tensor.detach().cpu()).reshape(-1)[0])


def test_multi_step_observations():
    env = make_env(num_game_iterations=10)
    obs = env.reset()
    assert len(obs) == 4

    for t in range(10):
        actions = {agent: np.random.randint(0, 2) for agent in env.possible_agents}
        obs, rewards, done, infos = env.step(actions)
        assert isinstance(done, bool)
        for agent in env.possible_agents:
            assert agent in obs
            assert obs[agent].shape[0] == 2
            assert agent in rewards
            assert "executed_action" in infos[agent]
        if t < 9:
            assert done is False
    assert done is True


def test_sticky_f_transitions():
    env = make_env(num_game_iterations=2000, rho=0.1, uncertainties=[0, 0, 0, 0])
    env.reset()
    switches = 0
    prev_f = float(env.current_multiplier.item())
    for _ in range(2000):
        env.step({agent: 0 for agent in env.possible_agents})
        cur_f = float(env.current_multiplier.item())
        if cur_f != prev_f:
            switches += 1
        prev_f = cur_f
    # Expected ~200 switches with rho=0.1; keep broad tolerance for stochasticity.
    assert 120 <= switches <= 280


def test_sticky_f_never_self_transitions():
    env = make_env(num_game_iterations=300, rho=1.0, uncertainties=[0, 0, 0, 0])
    env.reset()
    prev_f = float(env.current_multiplier.item())
    for _ in range(299):
        env.step({agent: 0 for agent in env.possible_agents})
        cur_f = float(env.current_multiplier.item())
        assert cur_f != prev_f
        prev_f = cur_f


def test_tremble_rate():
    env = make_env(num_game_iterations=1500, epsilon_tremble=0.05, uncertainties=[0, 0, 0, 0])
    env.reset()
    total = 0
    flipped = 0
    for _ in range(1500):
        _, _, _, infos = env.step({agent: 1 for agent in env.possible_agents})
        for agent in env.possible_agents:
            total += 1
            flipped += int(infos[agent]["flipped"])
    rate = flipped / float(total)
    assert 0.035 <= rate <= 0.065


def test_rewards_match_payoff_formula():
    env = make_env(
        num_game_iterations=1,
        epsilon_tremble=0.0,
        uncertainties=[0, 0, 0, 0],
        F=[2.5],
        mult_fact=[2.5],
    )
    env.reset()
    env._set_current_multiplier(2.5)
    _, rewards, done, _ = env.step({agent: 1 for agent in env.possible_agents})
    assert done is True
    for agent in env.possible_agents:
        assert abs(float(rewards[agent]) - 10.0) < 1e-6


def test_rewards_match_payoff_formula_mixed_actions():
    env = make_env(
        num_game_iterations=1,
        epsilon_tremble=0.0,
        uncertainties=[0, 0, 0, 0],
        F=[2.5],
        mult_fact=[2.5],
    )
    env.reset()
    env._set_current_multiplier(2.5)
    actions = {
        "agent_0": 1,  # cooperate
        "agent_1": 1,  # cooperate
        "agent_2": 0,  # defect
        "agent_3": 0,  # defect
    }
    _, rewards, done, _ = env.step(actions)
    assert done is True
    # common pot = 8, pot/N * f = 8/4*2.5 = 5.0
    # cooperators: 5.0 + (4-4) = 5.0
    # defectors:   5.0 + (4-0) = 9.0
    assert abs(float(rewards["agent_0"]) - 5.0) < 1e-6
    assert abs(float(rewards["agent_1"]) - 5.0) < 1e-6
    assert abs(float(rewards["agent_2"]) - 9.0) < 1e-6
    assert abs(float(rewards["agent_3"]) - 9.0) < 1e-6


def test_no_observation_clamping():
    env = make_env(num_game_iterations=600, uncertainties=[5.0, 5.0, 5.0, 5.0], F=[0.5, 1.5, 2.5])
    obs = env.reset()
    saw_negative = any(_f_hat(obs[a]) < 0.0 for a in env.possible_agents)
    saw_above = any(_f_hat(obs[a]) > max(env.F) for a in env.possible_agents)
    for _ in range(599):
        obs, _, done, _ = env.step({agent: 0 for agent in env.possible_agents})
        saw_negative = saw_negative or any(_f_hat(obs[a]) < 0.0 for a in env.possible_agents)
        saw_above = saw_above or any(_f_hat(obs[a]) > max(env.F) for a in env.possible_agents)
        if done:
            break
    assert saw_negative
    assert saw_above


def test_zero_uncertainty_observations_are_exact():
    env = make_env(
        num_game_iterations=3,
        uncertainties=[0.0, 0.0, 0.0, 0.0],
        F=[2.5],
        mult_fact=[2.5],
    )
    obs = env.reset(mult_in=2.5)
    for agent in env.possible_agents:
        assert abs(_f_hat(obs[agent]) - 2.5) < 1e-9

    for _ in range(2):
        obs, _, done, _ = env.step({agent: 0 for agent in env.possible_agents})
        for agent in env.possible_agents:
            assert abs(_f_hat(obs[agent]) - 2.5) < 1e-9
        if done:
            break


def test_observation_space_is_box():
    env = make_env()
    for agent in env.possible_agents:
        space = env.observation_space(agent)
        assert isinstance(space, Box)
        assert tuple(space.shape) == (2,)
