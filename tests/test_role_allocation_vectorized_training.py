import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.experiments_role_allocation.train_ppo import _make_agents, _make_wrapper
from src.experiments_role_allocation.train_ppo_vec import VectorizedTrainConfig, train_vec
from src.experiments_role_allocation.vectorized_rollout import VectorizedRoleRunner


def test_vectorized_rollout_collects_batched_learned_messages():
    cfg = VectorizedTrainConfig(
        condition="learned",
        seed=321,
        total_episodes=2,
        num_envs=2,
        rollout_len=4,
        horizon=4,
        hidden_size=8,
        ppo_epochs=1,
        mini_batch_size=4,
    )
    template_wrapper = _make_wrapper(cfg, eval_mode=False)
    agents = _make_agents(cfg, obs_dim=template_wrapper.obs_dim)
    runner = VectorizedRoleRunner(
        cfg,
        num_envs=2,
        rng=np.random.default_rng(777),
    )

    result = runner.collect(agents, rollout_len=4, trace_episodes=1)

    assert result.buffer.t == 4
    assert result.buffer.n_envs == 2
    assert result.metrics["rounds"] == 8.0
    assert result.buffer.messages is not None
    assert result.buffer.message_actions is not None
    assert np.any(result.buffer.message_actions[: result.buffer.t] >= 0)
    assert len(result.trace_rows) > 0


def test_minimal_vectorized_trainer_writes_outputs(tmp_path: Path):
    result = train_vec(
        VectorizedTrainConfig(
            condition="no_comm",
            seed=456,
            total_episodes=2,
            num_envs=2,
            rollout_len=4,
            horizon=4,
            eval_episodes=1,
            out_dir=str(tmp_path),
            hidden_size=8,
            ppo_epochs=1,
            mini_batch_size=4,
            train_trace_episodes=1,
        )
    )

    out_dir = Path(result["out_dir"])
    assert (out_dir / "checkpoint.pt").exists()
    assert (out_dir / "metrics.jsonl").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "train_trace.csv").exists()
    assert (out_dir / "eval_trace.csv").exists()

    with (out_dir / "eval_trace.csv").open("r", encoding="utf-8") as f:
        row = next(csv.DictReader(f))

    assert "local_cost" in row
    assert "lowest_cost_agent" in row


def test_vectorized_trainer_writes_slot_permutation_eval(tmp_path: Path):
    result = train_vec(
        VectorizedTrainConfig(
            condition="no_comm",
            seed=654,
            total_episodes=2,
            num_envs=2,
            rollout_len=4,
            horizon=4,
            eval_episodes=1,
            out_dir=str(tmp_path),
            hidden_size=8,
            ppo_epochs=1,
            mini_batch_size=4,
            eval_slot_permutation=True,
        )
    )

    out_dir = Path(result["out_dir"])
    slot_trace_path = out_dir / "eval_slot_permuted_trace.csv"
    assert slot_trace_path.exists()
    assert result["slot_permutation_eval_metrics"] is not None

    with slot_trace_path.open("r", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["phase"] == "eval_slot_permuted"

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["slot_permutation_policy_map"] == {
        "agent_0": "agent_1",
        "agent_1": "agent_2",
        "agent_2": "agent_3",
        "agent_3": "agent_0",
    }
    assert manifest["slot_permutation_eval_metrics"] is not None


def test_vectorized_trainer_writes_message_shuffle_eval(tmp_path: Path):
    result = train_vec(
        VectorizedTrainConfig(
            condition="learned",
            seed=655,
            total_episodes=2,
            num_envs=2,
            rollout_len=4,
            horizon=4,
            eval_episodes=2,
            out_dir=str(tmp_path),
            hidden_size=8,
            ppo_epochs=1,
            mini_batch_size=4,
            eval_message_shuffle=True,
        )
    )

    out_dir = Path(result["out_dir"])
    shuffle_trace_path = out_dir / "eval_message_shuffled_trace.csv"
    assert shuffle_trace_path.exists()
    assert result["message_shuffle_eval_metrics"] is not None

    with shuffle_trace_path.open("r", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["phase"] == "eval_message_shuffled"

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["message_shuffle_eval_metrics"] is not None
    assert manifest["message_shuffle_eval_trace_csv"] == str(shuffle_trace_path)


def test_vectorized_trainer_supports_informant_executor_mode(tmp_path: Path):
    result = train_vec(
        VectorizedTrainConfig(
            condition="learned",
            seed=656,
            total_episodes=2,
            num_envs=2,
            rollout_len=4,
            horizon=4,
            eval_episodes=2,
            out_dir=str(tmp_path),
            env_mode="informant_executor",
            cost_mode="iid",
            cost_levels=(0.5, 0.9, 1.3, 1.7),
            hidden_size=8,
            ppo_epochs=1,
            mini_batch_size=4,
            train_trace_episodes=1,
            eval_message_shuffle=True,
        )
    )

    out_dir = Path(result["out_dir"])
    assert result["message_shuffle_eval_metrics"] is not None

    with (out_dir / "train_trace.csv").open("r", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["local_role"] in {"informed", "capable", "neither"}
    assert "role_agent_0" in row

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["env_config"]["env_mode"] == "informant_executor"
    assert manifest["env_config"]["informant_sigma"] == 0.3
    assert manifest["message_shuffle_eval_metrics"] is not None


def test_vectorized_trainer_rejects_non_terminal_bootstrap_case(tmp_path: Path):
    cfg = VectorizedTrainConfig(
        condition="no_comm",
        seed=789,
        total_episodes=2,
        num_envs=2,
        rollout_len=2,
        horizon=4,
        eval_episodes=1,
        out_dir=str(tmp_path),
        hidden_size=8,
        ppo_epochs=1,
        mini_batch_size=4,
    )

    with pytest.raises(ValueError, match="rollout_len == horizon"):
        train_vec(cfg)


def test_vectorized_trainer_applies_linear_schedules(tmp_path: Path):
    result = train_vec(
        VectorizedTrainConfig(
            condition="no_comm",
            seed=790,
            total_episodes=6,
            num_envs=2,
            rollout_len=4,
            horizon=4,
            eval_episodes=1,
            out_dir=str(tmp_path),
            hidden_size=8,
            ppo_epochs=1,
            mini_batch_size=4,
            entropy_coeff=0.01,
            entropy_coeff_end=0.004,
            entropy_decay_end_fraction=1.0,
            msg_entropy_coeff=0.02,
            msg_entropy_coeff_end=0.002,
            msg_entropy_decay_end_fraction=0.5,
            lr=3e-4,
            lr_end=3e-5,
            lr_decay_end_fraction=1.0,
        )
    )

    metrics_path = Path(result["out_dir"]) / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3

    assert rows[0]["scheduled_entropy_coeff"] == pytest.approx(0.008, abs=1e-8)
    assert rows[1]["scheduled_entropy_coeff"] == pytest.approx(0.006, abs=1e-8)
    assert rows[2]["scheduled_entropy_coeff"] == pytest.approx(0.004, abs=1e-8)

    assert rows[0]["scheduled_msg_entropy_coeff"] == pytest.approx(0.008, abs=1e-8)
    assert rows[1]["scheduled_msg_entropy_coeff"] == pytest.approx(0.002, abs=1e-8)
    assert rows[2]["scheduled_msg_entropy_coeff"] == pytest.approx(0.002, abs=1e-8)

    assert rows[0]["scheduled_lr"] == pytest.approx(2.1e-4, abs=1e-12)
    assert rows[1]["scheduled_lr"] == pytest.approx(1.2e-4, abs=1e-12)
    assert rows[2]["scheduled_lr"] == pytest.approx(3e-5, abs=1e-12)
