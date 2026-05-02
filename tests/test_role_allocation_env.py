import numpy as np
try:
    from gym.spaces import Box
except ImportError:  # pragma: no cover
    from gymnasium.spaces import Box

from src.environments.role_allocation import role_allocation_parallel_v0


def make_env(**overrides):
    cfg = dict(
        n_agents=4,
        num_game_iterations=10,
        need_levels=[0.0, 1.0],
        uncertainties=[0.5, 0.5, 0.5, 0.5],
        rho=0.05,
        epsilon_tremble=0.05,
        volunteer_cost=1.0,
        benefit=4.0,
        miss_loss=2.0,
        redundant_penalty=0.0,
        target_volunteers=1,
        fraction=False,
    )
    cfg.update(overrides)
    return role_allocation_parallel_v0.parallel_env(cfg)


def _need_hat(obs_tensor):
    return float(np.asarray(obs_tensor.detach().cpu()).reshape(-1)[0])


def _local_cost(obs_tensor):
    return float(np.asarray(obs_tensor.detach().cpu()).reshape(-1)[1])


def _eligibility(obs_tensor):
    return float(np.asarray(obs_tensor.detach().cpu()).reshape(-1)[2])


def test_multi_step_observations_and_infos():
    env = make_env(num_game_iterations=10)
    obs = env.reset()
    assert len(obs) == 4

    for step in range(10):
        actions = {agent: np.random.randint(0, 2) for agent in env.possible_agents}
        obs, rewards, done, infos = env.step(actions)
        assert isinstance(done, bool)
        for agent in env.possible_agents:
            assert agent in obs
            assert obs[agent].shape[0] == 2
            assert agent in rewards
            assert "intended_action" in infos[agent]
            assert "executed_action" in infos[agent]
            assert "true_need" in infos[agent]
            assert "volunteer_cost" in infos[agent]
            assert "n_volunteers" in infos[agent]
        assert "volunteer_costs" in infos
        if step < 9:
            assert done is False
    assert done is True


def test_sticky_need_transitions():
    env = make_env(num_game_iterations=2000, rho=0.1, uncertainties=[0, 0, 0, 0])
    env.reset()
    switches = 0
    prev_need = float(env.current_need.item())
    for _ in range(2000):
        env.step({agent: 0 for agent in env.possible_agents})
        cur_need = float(env.current_need.item())
        if cur_need != prev_need:
            switches += 1
        prev_need = cur_need
    assert 120 <= switches <= 280


def test_sticky_need_never_self_transitions_when_forced():
    env = make_env(num_game_iterations=100, rho=1.0, uncertainties=[0, 0, 0, 0])
    env.reset()
    prev_need = float(env.current_need.item())
    for _ in range(99):
        env.step({agent: 0 for agent in env.possible_agents})
        cur_need = float(env.current_need.item())
        assert cur_need != prev_need
        prev_need = cur_need


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


def test_need_present_no_volunteers_all_lose():
    env = make_env(
        num_game_iterations=1,
        epsilon_tremble=0.0,
        uncertainties=[0, 0, 0, 0],
        need_levels=[1.0],
        benefit=4.0,
        miss_loss=2.0,
    )
    env.reset(need_in=1.0)
    _, rewards, done, infos = env.step({agent: 0 for agent in env.possible_agents})
    assert done is True
    assert infos["produced"] is False
    for reward in rewards.values():
        assert abs(float(reward) + 2.0) < 1e-6


def test_need_present_one_volunteer_produces_benefit():
    env = make_env(
        num_game_iterations=1,
        epsilon_tremble=0.0,
        uncertainties=[0, 0, 0, 0],
        need_levels=[1.0],
        benefit=4.0,
        volunteer_cost=1.0,
    )
    env.reset(need_in=1.0)
    actions = {agent: 0 for agent in env.possible_agents}
    actions["agent_0"] = 1
    _, rewards, done, infos = env.step(actions)
    assert done is True
    assert infos["produced"] is True
    assert abs(float(rewards["agent_0"]) - 3.0) < 1e-6
    for agent in ["agent_1", "agent_2", "agent_3"]:
        assert abs(float(rewards[agent]) - 4.0) < 1e-6


def test_balanced_costs_are_shuffled_per_round_and_observed_locally():
    np.random.seed(123)
    cost_levels = [0.25, 0.75, 1.25, 1.75]
    env = make_env(
        num_game_iterations=20,
        epsilon_tremble=0.0,
        uncertainties=[0, 0, 0, 0],
        need_levels=[1.0],
        cost_mode="balanced",
        cost_levels=cost_levels,
    )
    obs = env.reset(need_in=1.0)
    seen_assignments = set()

    for _ in range(20):
        observed_costs = {agent: _local_cost(obs[agent]) for agent in env.possible_agents}
        seen_assignments.add(tuple(observed_costs[agent] for agent in env.possible_agents))
        assert np.allclose(sorted(observed_costs.values()), sorted(cost_levels))

        obs, _, done, infos = env.step({agent: 0 for agent in env.possible_agents})

        assert np.allclose(
            [infos["volunteer_costs"][agent] for agent in env.possible_agents],
            [observed_costs[agent] for agent in env.possible_agents],
        )
        for agent in env.possible_agents:
            assert abs(float(infos[agent]["volunteer_cost"]) - observed_costs[agent]) < 1e-6
        if done:
            break

    assert len(seen_assignments) > 1


def test_cost_levels_default_to_balanced_mode():
    cost_levels = [0.25, 0.75, 1.25, 1.75]
    env = make_env(
        uncertainties=[0, 0, 0, 0],
        need_levels=[1.0],
        cost_levels=cost_levels,
    )
    obs = env.reset(need_in=1.0)

    assert env.cost_mode == "balanced"
    assert np.allclose(
        sorted(_local_cost(obs[agent]) for agent in env.possible_agents),
        sorted(cost_levels),
    )


def test_current_local_cost_is_used_for_volunteer_reward():
    np.random.seed(321)
    env = make_env(
        num_game_iterations=1,
        epsilon_tremble=0.0,
        uncertainties=[0, 0, 0, 0],
        need_levels=[1.0],
        benefit=4.0,
        cost_mode="balanced",
        cost_levels=[0.25, 0.75, 1.25, 1.75],
    )
    obs = env.reset(need_in=1.0)
    volunteer = "agent_0"
    step_cost = _local_cost(obs[volunteer])
    actions = {agent: 0 for agent in env.possible_agents}
    actions[volunteer] = 1

    _, rewards, done, infos = env.step(actions)

    assert done is True
    assert infos["produced"] is True
    assert abs(float(rewards[volunteer]) - (4.0 - step_cost)) < 1e-6
    for agent in ["agent_1", "agent_2", "agent_3"]:
        assert abs(float(rewards[agent]) - 4.0) < 1e-6


def test_no_need_volunteering_is_wasteful():
    env = make_env(
        num_game_iterations=1,
        epsilon_tremble=0.0,
        uncertainties=[0, 0, 0, 0],
        need_levels=[0.0],
        volunteer_cost=1.0,
    )
    env.reset(need_in=0.0)
    actions = {agent: 0 for agent in env.possible_agents}
    actions["agent_0"] = 1
    _, rewards, _, infos = env.step(actions)
    assert infos["produced"] is False
    assert abs(float(rewards["agent_0"]) + 1.0) < 1e-6
    for agent in ["agent_1", "agent_2", "agent_3"]:
        assert abs(float(rewards[agent])) < 1e-6


def test_no_observation_clamping():
    env = make_env(num_game_iterations=600, uncertainties=[5.0, 5.0, 5.0, 5.0], need_levels=[0.0, 1.0])
    obs = env.reset()
    saw_negative = any(_need_hat(obs[a]) < 0.0 for a in env.possible_agents)
    saw_above = any(_need_hat(obs[a]) > max(env.need_levels) for a in env.possible_agents)
    for _ in range(599):
        obs, _, done, _ = env.step({agent: 0 for agent in env.possible_agents})
        saw_negative = saw_negative or any(_need_hat(obs[a]) < 0.0 for a in env.possible_agents)
        saw_above = saw_above or any(_need_hat(obs[a]) > max(env.need_levels) for a in env.possible_agents)
        if done:
            break
    assert saw_negative
    assert saw_above


def test_zero_uncertainty_observations_are_exact():
    env = make_env(
        num_game_iterations=3,
        uncertainties=[0.0, 0.0, 0.0, 0.0],
        need_levels=[1.0],
    )
    obs = env.reset(need_in=1.0)
    for agent in env.possible_agents:
        assert abs(_need_hat(obs[agent]) - 1.0) < 1e-9


def test_observation_space_is_box():
    env = make_env()
    for agent in env.possible_agents:
        space = env.observation_space(agent)
        assert isinstance(space, Box)


def test_crossed_mode_observes_only_own_eligibility_and_cost():
    np.random.seed(10)
    env = make_env(
        env_mode="crossed",
        cost_mode="iid",
        cost_levels=[0.5, 0.9, 1.3, 1.7],
        eligibility_prob=0.5,
        uncertainties=[0, 0, 0, 0],
        need_levels=[1.0],
    )
    obs = env.reset(need_in=1.0)

    assert env.obs_dim == 3
    assert any(env.current_eligibility.values())
    for agent in env.possible_agents:
        # Structural check: each agent's raw observation is exactly
        # [need_hat, own_cost, own_eligibility] — three scalars, nothing else.
        # No other agent's cost or eligibility, no volunteer count, no true need,
        # no production label can appear in this tensor.
        assert obs[agent].shape == (3,)
        assert _eligibility(obs[agent]) == float(env.current_eligibility[agent])
        assert _local_cost(obs[agent]) == float(env.current_costs[agent])
    # Value-level leakage guard: the focal agent's tensor must not expose any
    # other agent's private (cost, eligibility) pair at positions 1-2. If costs
    # happen to be identical across agents under iid sampling, the eligibility
    # slot still distinguishes.
    for agent in env.possible_agents:
        own_slot = (
            float(env.current_costs[agent]),
            float(env.current_eligibility[agent]),
        )
        tensor_slot = (_local_cost(obs[agent]), _eligibility(obs[agent]))
        assert tensor_slot == own_slot


def test_crossed_need_present_forces_at_least_one_eligible_agent():
    env = make_env(
        env_mode="crossed",
        eligibility_prob=0.0,
        uncertainties=[0, 0, 0, 0],
        need_levels=[1.0],
    )
    obs = env.reset(need_in=1.0)

    assert sum(env.current_eligibility.values()) == 1
    assert sum(int(_eligibility(obs[agent])) for agent in env.possible_agents) == 1


def test_crossed_ineligible_volunteer_does_not_produce_service():
    env = make_env(
        env_mode="crossed",
        cost_mode="constant",
        num_game_iterations=1,
        epsilon_tremble=0.0,
        uncertainties=[0, 0, 0, 0],
        need_levels=[1.0],
        benefit=4.0,
        miss_loss=2.0,
        invalid_volunteer_penalty=0.5,
    )
    env.reset(need_in=1.0)
    env.current_eligibility = {
        "agent_0": 0,
        "agent_1": 1,
        "agent_2": 0,
        "agent_3": 0,
    }

    actions = {agent: 0 for agent in env.possible_agents}
    actions["agent_0"] = 1
    _, rewards, done, infos = env.step(actions)

    assert done is True
    assert infos["feasible_need"] is True
    assert infos["produced"] is False
    assert infos["n_volunteers"] == 1
    assert infos["n_effective_volunteers"] == 0
    assert infos["n_invalid_volunteers"] == 1
    assert abs(float(rewards["agent_0"]) + 2.5) < 1e-6
    for agent in ["agent_1", "agent_2", "agent_3"]:
        assert abs(float(rewards[agent]) + 2.0) < 1e-6


def test_crossed_eligible_volunteer_produces_service():
    env = make_env(
        env_mode="crossed",
        cost_mode="constant",
        num_game_iterations=1,
        epsilon_tremble=0.0,
        uncertainties=[0, 0, 0, 0],
        need_levels=[1.0],
        benefit=4.0,
        volunteer_cost=1.0,
    )
    env.reset(need_in=1.0)
    env.current_eligibility = {
        "agent_0": 1,
        "agent_1": 0,
        "agent_2": 0,
        "agent_3": 0,
    }

    actions = {agent: 0 for agent in env.possible_agents}
    actions["agent_0"] = 1
    _, rewards, done, infos = env.step(actions)

    assert done is True
    assert infos["produced"] is True
    assert infos["n_effective_volunteers"] == 1
    assert infos["n_invalid_volunteers"] == 0
    assert abs(float(rewards["agent_0"]) - 3.0) < 1e-6
    for agent in ["agent_1", "agent_2", "agent_3"]:
        assert abs(float(rewards[agent]) - 4.0) < 1e-6


def test_crossed_mixed_eligibility_multi_volunteer_rewards():
    # Need is present. One eligible volunteer + one ineligible volunteer.
    # Expectation: the eligible volunteer produces service, so everyone gets
    # the benefit. The eligible volunteer pays their own cost AND the redundant
    # penalty (because n_volunteers > target). The ineligible volunteer pays
    # the invalid-volunteer penalty AND the redundant penalty (since any
    # volunteer above target incurs it). Non-volunteers just get the benefit.
    env = make_env(
        env_mode="crossed",
        cost_mode="constant",
        num_game_iterations=1,
        epsilon_tremble=0.0,
        uncertainties=[0, 0, 0, 0],
        need_levels=[1.0],
        benefit=4.0,
        miss_loss=2.0,
        volunteer_cost=1.0,
        redundant_penalty=0.2,
        invalid_volunteer_penalty=0.5,
    )
    env.reset(need_in=1.0)
    env.current_eligibility = {
        "agent_0": 1,  # eligible volunteer
        "agent_1": 0,  # ineligible volunteer
        "agent_2": 1,  # eligible bystander
        "agent_3": 0,  # ineligible bystander
    }

    actions = {agent: 0 for agent in env.possible_agents}
    actions["agent_0"] = 1
    actions["agent_1"] = 1
    _, rewards, done, infos = env.step(actions)

    assert done is True
    assert infos["feasible_need"] is True
    assert infos["produced"] is True
    assert infos["n_volunteers"] == 2
    assert infos["n_effective_volunteers"] == 1
    assert infos["n_invalid_volunteers"] == 1
    # redundant at n_volunteers=2, target=1 → redundant=1 → -0.2 for each volunteer
    assert abs(float(rewards["agent_0"]) - (4.0 - 1.0 - 0.2)) < 1e-6  # 2.8
    assert abs(float(rewards["agent_1"]) - (4.0 - 0.5 - 0.2)) < 1e-6  # 3.3
    assert abs(float(rewards["agent_2"]) - 4.0) < 1e-6
    assert abs(float(rewards["agent_3"]) - 4.0) < 1e-6


def test_informant_executor_roles_split_information_from_capability():
    np.random.seed(11)
    env = make_env(
        env_mode="informant_executor",
        cost_mode="iid",
        cost_levels=[0.5, 0.9, 1.3, 1.7],
        need_levels=[0.0, 1.0],
        informant_sigma=0.0,
        capable_sigma=0.0,
        neither_sigma=0.0,
        neither_eligibility_prob=0.0,
        prohibitive_cost=8.0,
        uncertainties=[0, 0, 0, 0],
    )

    obs = env.reset(need_in=1.0)
    roles = env.current_roles

    assert sorted(roles.values()) == ["capable", "capable", "informed", "neither"]
    assert env.obs_dim == 3
    for agent in env.possible_agents:
        assert obs[agent].shape == (3,)
        role = roles[agent]
        if role == "informed":
            assert _eligibility(obs[agent]) == 0.0
            assert _local_cost(obs[agent]) == 8.0
            assert _need_hat(obs[agent]) == 1.0
        elif role == "capable":
            assert _eligibility(obs[agent]) == 1.0
            assert any(
                abs(_local_cost(obs[agent]) - cost) < 1e-6
                for cost in [0.5, 0.9, 1.3, 1.7]
            )
            assert _need_hat(obs[agent]) == 0.5
        else:
            assert role == "neither"
            assert _eligibility(obs[agent]) == 0.0
            assert _local_cost(obs[agent]) == 8.0


def test_informant_executor_roles_are_logged_only_in_infos():
    env = make_env(
        env_mode="informant_executor",
        cost_mode="iid",
        cost_levels=[0.5, 0.9, 1.3, 1.7],
        num_game_iterations=1,
        epsilon_tremble=0.0,
        informant_sigma=0.0,
        capable_sigma=0.0,
        neither_sigma=0.0,
        prohibitive_cost=8.0,
        uncertainties=[0, 0, 0, 0],
        need_levels=[1.0],
    )
    obs = env.reset(need_in=1.0)

    _, _, done, infos = env.step({agent: 0 for agent in env.possible_agents})

    assert done is True
    assert "roles" in infos
    assert sorted(infos["roles"].values()) == ["capable", "capable", "informed", "neither"]
    for agent in env.possible_agents:
        assert infos[agent]["role"] == infos["roles"][agent]
        # The role is available only through Set-C logging. Set-A observation
        # remains exactly [need_hat, own_cost, own_eligibility].
        assert obs[agent].shape == (3,)


def test_informant_executor_informed_volunteer_cannot_produce_service():
    env = make_env(
        env_mode="informant_executor",
        cost_mode="iid",
        cost_levels=[0.5, 0.9, 1.3, 1.7],
        num_game_iterations=1,
        epsilon_tremble=0.0,
        need_levels=[1.0],
        benefit=4.0,
        miss_loss=2.0,
        invalid_volunteer_penalty=1.7,
        prohibitive_cost=8.0,
    )
    env.reset(need_in=1.0)
    informed = next(
        agent for agent, role in env.current_roles.items() if role == "informed"
    )
    actions = {agent: 0 for agent in env.possible_agents}
    actions[informed] = 1

    _, rewards, done, infos = env.step(actions)

    assert done is True
    assert infos["produced"] is False
    assert infos["n_effective_volunteers"] == 0
    assert infos["n_invalid_volunteers"] == 1
    assert abs(float(rewards[informed]) + 10.0) < 1e-6


def test_informant_executor_capable_volunteer_produces_service():
    env = make_env(
        env_mode="informant_executor",
        cost_mode="iid",
        cost_levels=[0.5, 0.9, 1.3, 1.7],
        num_game_iterations=1,
        epsilon_tremble=0.0,
        need_levels=[1.0],
        benefit=4.0,
        miss_loss=2.0,
        prohibitive_cost=8.0,
    )
    env.reset(need_in=1.0)
    capable = next(agent for agent, role in env.current_roles.items() if role == "capable")
    cost = float(env.current_costs[capable])
    actions = {agent: 0 for agent in env.possible_agents}
    actions[capable] = 1

    _, rewards, done, infos = env.step(actions)

    assert done is True
    assert infos["produced"] is True
    assert infos["n_effective_volunteers"] == 1
    assert abs(float(rewards[capable]) - (4.0 - cost)) < 1e-6
