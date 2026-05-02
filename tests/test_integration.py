import os
import platform

import numpy as np
import pytest

from src.experiments_pgg_v0.train_ppo import minimal_test_config, train


RUN_SUBPROC_TESTS = os.environ.get("EPGG_RUN_SUBPROC_TESTS") == "1"
SKIP_DARWIN_SUBPROC = platform.system() == "Darwin" and not RUN_SUBPROC_TESTS


def test_full_loop_runs(tmp_path):
    cfg = minimal_test_config(
        n_episodes=5,
        T=8,
        save_path=str(tmp_path / "agents.pt"),
        seed=123,
    )
    metrics = train(cfg)
    assert len(metrics) == 5
    assert (tmp_path / "agents.pt").exists()


def test_cooperation_changes(tmp_path):
    cfg = minimal_test_config(
        n_episodes=30,
        T=8,
        save_path=str(tmp_path / "agents2.pt"),
        seed=123,
    )
    metrics = train(cfg)
    coop_rates = np.array([m["coop_rate"] for m in metrics], dtype=np.float32)
    assert np.std(coop_rates) > 1e-3


def test_train_with_session_logging(tmp_path):
    session_dir = tmp_path / "sessions"
    cfg = minimal_test_config(
        n_episodes=3,
        T=6,
        save_path=str(tmp_path / "agents3.pt"),
        seed=123,
        log_sessions=True,
        session_log_dir=str(session_dir),
        condition_name="ci",
        consolidate_sessions=True,
    )
    _ = train(cfg)
    parts = list(session_dir.glob("data_ci_123_*.npz"))
    assert len(parts) >= 3
    consolidated = session_dir / "data_ci_123_consolidated.npz"
    assert consolidated.exists()


def test_full_loop_runs_vectorized(tmp_path):
    cfg = minimal_test_config(
        n_episodes=3,
        T=6,
        num_envs=2,
        save_path=str(tmp_path / "agents_vec.pt"),
        seed=321,
    )
    metrics = train(cfg)
    assert len(metrics) == 3
    assert (tmp_path / "agents_vec.pt").exists()
    assert all(int(row["num_envs"]) == 2 for row in metrics)
    assert all(int(row["steps"]) == 12 for row in metrics)


@pytest.mark.skipif(
    SKIP_DARWIN_SUBPROC,
    reason="set EPGG_RUN_SUBPROC_TESTS=1 to force macOS subprocess-backend tests",
)
def test_full_loop_runs_vectorized_subproc(tmp_path):
    cfg = minimal_test_config(
        n_episodes=2,
        T=5,
        num_envs=2,
        env_backend="subproc",
        env_start_method="spawn",
        save_path=str(tmp_path / "agents_vec_subproc.pt"),
        seed=654,
    )
    metrics = train(cfg)
    assert len(metrics) == 2
    assert (tmp_path / "agents_vec_subproc.pt").exists()
    assert all(int(row["num_envs"]) == 2 for row in metrics)
    assert all(int(row["steps"]) == 10 for row in metrics)


def test_train_exposes_timing_metrics(tmp_path):
    cfg = minimal_test_config(
        n_episodes=2,
        T=5,
        num_envs=2,
        env_backend="serial",
        save_path=str(tmp_path / "agents_timing.pt"),
        seed=777,
    )
    metrics = train(cfg)

    timing_keys = [
        "episode_wall_s",
        "episode_other_s",
        "session_log_wall_s",
        "update_wall_s",
        "rollout_wall_s",
        "rollout_other_s",
        "rollout_reset_s",
        "rollout_obs_build_s",
        "rollout_message_policy_s",
        "rollout_message_postprocess_s",
        "rollout_action_policy_s",
        "rollout_diag_s",
        "rollout_env_step_s",
        "rollout_buffer_store_s",
        "rollout_bootstrap_value_s",
        "rollout_gae_s",
        "rollout_flatten_s",
        "episode_steps_per_s",
        "rollout_steps_per_s",
        "update_steps_per_s",
        "episode_agent_steps_per_s",
    ]

    assert len(metrics) == 2
    for row in metrics:
        for key in timing_keys:
            assert key in row
            assert np.isfinite(float(row[key]))
            assert float(row[key]) >= 0.0
        assert float(row["episode_steps_per_s"]) > 0.0
        assert float(row["episode_agent_steps_per_s"]) > 0.0
        assert float(row["episode_wall_s"]) + 1e-9 >= (
            float(row["rollout_wall_s"])
            + float(row["update_wall_s"])
            + float(row["session_log_wall_s"])
        )
