import numpy as np
import pytest

from src.experiments_pgg_v0.env_pool import (
    EnvWorkerError,
    SerialParallelEnvPool,
    SubprocParallelEnvPool,
)
from src.experiments_pgg_v0.train_ppo import (
    _build_agents,
    _build_wrapper,
    _collect_vectorized_rollout,
    _make_env_cfg,
    _seed_everything,
    _sender_ids,
    minimal_test_config,
)


def _make_cfg(**overrides):
    cfg = minimal_test_config(
        n_agents=4,
        T=4,
        num_envs=2,
        seed=123,
        save_path="outputs/test_env_backend.pt",
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _assert_nested_equal(lhs, rhs):
    if isinstance(lhs, dict):
        assert set(lhs.keys()) == set(rhs.keys())
        for key in lhs:
            _assert_nested_equal(lhs[key], rhs[key])
        return
    if isinstance(lhs, (list, tuple)):
        assert len(lhs) == len(rhs)
        for left_item, right_item in zip(lhs, rhs):
            _assert_nested_equal(left_item, right_item)
        return
    if isinstance(lhs, np.ndarray):
        assert np.allclose(lhs, rhs)
        return
    if isinstance(lhs, float):
        assert lhs == pytest.approx(rhs)
        return
    assert lhs == rhs


def _fixed_actions(n_envs: int):
    patterns = [
        {"agent_0": 0, "agent_1": 1, "agent_2": 0, "agent_3": 1},
        {"agent_0": 1, "agent_1": 1, "agent_2": 0, "agent_3": 0},
        {"agent_0": 0, "agent_1": 0, "agent_2": 1, "agent_3": 1},
        {"agent_0": 1, "agent_1": 0, "agent_2": 1, "agent_3": 0},
    ]
    out = []
    for step_idx in range(4):
        out.append(
            [
                {
                    agent_id: int((action + env_idx) % 2)
                    for agent_id, action in patterns[step_idx].items()
                }
                for env_idx in range(n_envs)
            ]
        )
    return out


def test_subproc_pool_reset_step_and_close():
    cfg = _make_cfg()
    pool = SubprocParallelEnvPool(
        env_cfg=_make_env_cfg(cfg),
        n_envs=2,
        base_seed=cfg.seed,
        start_method="spawn",
    )
    try:
        obs_batch = pool.reset_all()
        assert len(obs_batch) == 2
        for obs in obs_batch:
            assert set(obs.keys()) == {"agent_0", "agent_1", "agent_2", "agent_3"}
            assert all(tuple(value.shape) == (2,) for value in obs.values())

        next_obs_batch, rewards_batch, done_batch, infos_batch = pool.step_batch(
            _fixed_actions(2)[0]
        )
        assert len(next_obs_batch) == 2
        assert len(rewards_batch) == 2
        assert len(done_batch) == 2
        assert len(infos_batch) == 2
        assert all(isinstance(done, bool) for done in done_batch)
        assert all("executed_actions" in info for info in infos_batch)
    finally:
        pool.close()


def test_subproc_pool_matches_serial_pool():
    cfg = _make_cfg(T=5, num_envs=3, seed=777)
    serial_pool = SerialParallelEnvPool(
        env_cfg=_make_env_cfg(cfg),
        n_envs=cfg.num_envs,
        base_seed=cfg.seed,
    )
    subproc_pool = SubprocParallelEnvPool(
        env_cfg=_make_env_cfg(cfg),
        n_envs=cfg.num_envs,
        base_seed=cfg.seed,
        start_method="spawn",
    )
    action_batches = _fixed_actions(cfg.num_envs)
    try:
        _assert_nested_equal(serial_pool.reset_all(), subproc_pool.reset_all())
        for actions in action_batches:
            serial_step = serial_pool.step_batch(actions)
            subproc_step = subproc_pool.step_batch(actions)
            _assert_nested_equal(serial_step, subproc_step)
    finally:
        serial_pool.close()
        subproc_pool.close()


def test_vectorized_rollout_matches_across_backends():
    cfg = _make_cfg(
        T=4,
        num_envs=2,
        seed=202,
        comm_enabled=True,
        n_senders=2,
        msg_dropout=0.25,
    )
    sender_ids = _sender_ids(cfg)

    def run_rollout(backend: str):
        _seed_everything(cfg.seed)
        wrappers = [_build_wrapper(cfg, sender_ids) for _ in range(cfg.num_envs)]
        agents = _build_agents(cfg, obs_dim=wrappers[0].obs_dim, sender_ids=sender_ids)
        if backend == "serial":
            pool = SerialParallelEnvPool(
                env_cfg=_make_env_cfg(cfg),
                n_envs=cfg.num_envs,
                base_seed=cfg.seed,
            )
        else:
            pool = SubprocParallelEnvPool(
                env_cfg=_make_env_cfg(cfg),
                n_envs=cfg.num_envs,
                base_seed=cfg.seed,
                start_method="spawn",
            )
        try:
            return _collect_vectorized_rollout(
                cfg=cfg,
                env_backend=pool,
                wrappers=wrappers,
                agents=agents,
                agent_ids=[f"agent_{idx}" for idx in range(cfg.n_agents)],
                sender_ids=sender_ids,
            )
        finally:
            pool.close()

    serial_buffer, serial_adv, serial_ret, serial_resp = run_rollout("serial")
    subproc_buffer, subproc_adv, subproc_ret, subproc_resp = run_rollout("subproc")

    for field_name in (
        "observations",
        "value_observations",
        "actions",
        "rewards",
        "values",
        "log_probs",
        "dones",
        "executed_actions",
        "flips",
        "true_f",
        "f_hats",
        "agent_rewards",
        "messages",
        "message_actions",
        "message_log_probs",
        "listening_bonus",
    ):
        serial_value = getattr(serial_buffer, field_name)
        subproc_value = getattr(subproc_buffer, field_name)
        if serial_value is None or subproc_value is None:
            assert serial_value is subproc_value
            continue
        assert np.allclose(serial_value[: serial_buffer.t], subproc_value[: subproc_buffer.t])

    assert serial_buffer.t == subproc_buffer.t
    assert np.allclose(serial_adv, subproc_adv)
    assert np.allclose(serial_ret, subproc_ret)
    _assert_nested_equal(serial_resp, subproc_resp)


def test_subproc_pool_propagates_worker_exception():
    cfg = _make_cfg()
    pool = SubprocParallelEnvPool(
        env_cfg=_make_env_cfg(cfg),
        n_envs=1,
        base_seed=cfg.seed,
        start_method="spawn",
    )
    try:
        pool.reset_all()
        with pytest.raises(EnvWorkerError, match="worker 0 failed during step"):
            pool.step_batch([{"agent_0": 1}])
    finally:
        pool.close()
