from pathlib import Path

import pytest

from src.experiments_pgg_v0.train_ppo import _scheduled_value, minimal_test_config, train


def test_scheduled_value_supports_linear_and_cosine():
    assert _scheduled_value(1.0, 0.0, "none", 0.5) == 1.0
    assert _scheduled_value(1.0, 0.0, "linear", 0.5) == 0.5
    cosine_mid = _scheduled_value(1.0, 0.0, "cosine", 0.5)
    assert 0.49 < cosine_mid < 0.51


def test_train_logs_entropy_schedule_progress(tmp_path: Path):
    ckpt = tmp_path / "sched_seed321.pt"
    cfg = minimal_test_config(
        n_agents=4,
        n_episodes=3,
        T=4,
        comm_enabled=True,
        n_senders=4,
        seed=321,
        save_path=str(ckpt),
        condition_name="cond1",
        entropy_coeff=0.02,
        entropy_schedule="linear",
        entropy_coeff_final=0.0,
        msg_entropy_coeff=0.03,
        msg_entropy_coeff_final=0.01,
        lr_schedule="cosine",
        min_lr=1e-5,
    )
    metrics = train(cfg)
    assert len(metrics) == 3
    assert metrics[0]["entropy_coeff_current"] == 0.02
    assert metrics[-1]["entropy_coeff_current"] == 0.0
    assert metrics[0]["msg_entropy_coeff_current"] == 0.03
    assert metrics[-1]["msg_entropy_coeff_current"] == pytest.approx(0.01)
    assert metrics[0]["lr_current"] == cfg.lr
    assert metrics[-1]["lr_current"] == pytest.approx(cfg.min_lr)


def test_schedule_progress_respects_episode_offset(tmp_path: Path):
    ckpt = tmp_path / "sched_seed654.pt"
    cfg = minimal_test_config(
        n_agents=4,
        n_episodes=2,
        T=4,
        comm_enabled=False,
        n_senders=0,
        seed=654,
        save_path=str(ckpt),
        condition_name="cond2",
        entropy_coeff=0.02,
        entropy_schedule="linear",
        entropy_coeff_final=0.0,
        lr_schedule="linear",
        min_lr=1e-5,
        episode_offset=2,
        schedule_total_episodes=4,
    )
    metrics = train(cfg)
    assert len(metrics) == 2
    assert metrics[0]["episode"] == 3
    assert metrics[0]["entropy_coeff_current"] == pytest.approx(0.02 / 3.0)
    assert metrics[-1]["episode"] == 4
    assert metrics[-1]["entropy_coeff_current"] == 0.0
