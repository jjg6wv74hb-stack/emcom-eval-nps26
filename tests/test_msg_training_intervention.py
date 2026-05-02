from pathlib import Path
from collections import deque
import json

import pytest
import torch

from src.algos.PPO import PPOAgentV2
from src.algos.trajectory_buffer import TrajectoryBuffer
from src.experiments_pgg_v0 import train_ppo
from src.experiments_pgg_v0.train_ppo import minimal_test_config, train
from src.wrappers.observation_wrapper import ObservationWrapper


def test_uniform_training_intervention_updates_marginals_with_delivered_tokens(
    tmp_path: Path, monkeypatch
):
    ckpt = tmp_path / "cond1_seed777.pt"
    recorded = []
    orig_update = ObservationWrapper.update_msg_marginals

    def fake_apply(intervention, delivered, vocab_size):
        return {sender_id: 1 for sender_id in delivered.keys()}

    def record_update(self, sender_id, message):
        recorded.append(int(message))
        return orig_update(self, sender_id, message)

    monkeypatch.setattr(train_ppo, "_apply_training_message_intervention", fake_apply)
    monkeypatch.setattr(ObservationWrapper, "update_msg_marginals", record_update)

    cfg = minimal_test_config(
        n_agents=4,
        n_episodes=2,
        T=4,
        comm_enabled=True,
        n_senders=4,
        seed=777,
        save_path=str(ckpt),
        condition_name="cond1",
        sign_lambda=0.0,
        list_lambda=0.0,
        msg_training_intervention="uniform",
    )
    train(cfg)

    assert len(recorded) > 0
    assert set(recorded) == {1}
    payload = torch.load(ckpt, map_location="cpu")
    assert payload["config"]["msg_training_intervention"] == "uniform"


def test_public_random_training_intervention_shares_one_token_across_senders():
    delivered = {"agent_0": 0, "agent_1": 1, "agent_2": 0, "agent_3": 1}
    out = train_ppo._apply_training_message_intervention(
        intervention="public_random",
        delivered=delivered,
        vocab_size=2,
    )
    assert set(out.keys()) == set(delivered.keys())
    assert len(set(out.values())) == 1


def test_sender_shuffle_training_intervention_uses_history_when_present():
    delivered = {"agent_0": 0, "agent_1": 1}
    sender_history = {
        "agent_0": deque([1, 1, 1]),
        "agent_1": deque([0, 0, 0]),
    }
    out = train_ppo._apply_training_message_intervention(
        intervention="sender_shuffle",
        delivered=delivered,
        vocab_size=2,
        sender_history=sender_history,
    )
    assert out == {"agent_0": 1, "agent_1": 0}


def test_sender_shuffle_training_intervention_falls_back_to_delivered_if_history_empty():
    delivered = {"agent_0": 0, "agent_1": 1}
    sender_history = {
        "agent_0": deque([]),
        "agent_1": deque([]),
    }
    out = train_ppo._apply_training_message_intervention(
        intervention="sender_shuffle",
        delivered=delivered,
        vocab_size=2,
        sender_history=sender_history,
    )
    assert out == delivered


def test_public_random_msg_source_records_shared_delivered_messages(
    tmp_path: Path, monkeypatch
):
    ckpt = tmp_path / "cond1_public_random_seed700.pt"
    recorded_messages = []
    orig_store = TrajectoryBuffer.store

    def record_store(self, *args, **kwargs):
        messages = kwargs.get("messages")
        if messages is not None:
            recorded_messages.append(dict(messages))
        return orig_store(self, *args, **kwargs)

    monkeypatch.setattr(TrajectoryBuffer, "store", record_store)

    cfg = minimal_test_config(
        n_agents=4,
        n_episodes=2,
        T=4,
        comm_enabled=True,
        n_senders=4,
        seed=700,
        save_path=str(ckpt),
        condition_name="cond1",
        sign_lambda=0.0,
        list_lambda=0.0,
        msg_dropout=0.0,
        msg_source_mode="public_random",
    )

    train(cfg)

    assert len(recorded_messages) > 0
    assert all(len(set(messages.values())) == 1 for messages in recorded_messages)
    payload = torch.load(ckpt, map_location="cpu")
    assert payload["config"]["msg_source_mode"] == "public_random"
    assert all(saved["message_actor"] is None for saved in payload["agents"].values())


def test_vectorized_exogenous_msg_source_bypasses_sender_policy(monkeypatch, tmp_path: Path):
    ckpt = tmp_path / "cond1_uniform_seed701.pt"

    def fail_if_called(*args, **kwargs):
        raise AssertionError("sample_message_batch should not be called for exogenous msg_source_mode")

    monkeypatch.setattr(PPOAgentV2, "sample_message_batch", fail_if_called)

    cfg = minimal_test_config(
        n_agents=4,
        n_episodes=4,
        T=4,
        num_envs=2,
        count_env_episodes=True,
        env_backend="serial",
        comm_enabled=True,
        n_senders=4,
        seed=701,
        save_path=str(ckpt),
        condition_name="cond1",
        sign_lambda=0.0,
        list_lambda=0.0,
        msg_source_mode="uniform",
    )

    train(cfg)

    payload = torch.load(ckpt, map_location="cpu")
    assert payload["config"]["msg_source_mode"] == "uniform"
    assert all(saved["message_actor"] is None for saved in payload["agents"].values())


def test_exogenous_msg_source_updates_marginals_from_source_tokens(
    tmp_path: Path, monkeypatch
):
    ckpt = tmp_path / "cond1_uniform_seed702.pt"
    recorded = []
    orig_update = ObservationWrapper.update_msg_marginals

    def fake_sample(source_mode, sender_ids, vocab_size):
        return {sender_id: 1 for sender_id in sender_ids}

    def fake_dropout(self, messages):
        return {sender_id: 0 for sender_id in messages.keys()}

    def record_update(self, sender_id, message):
        recorded.append(int(message))
        return orig_update(self, sender_id, message)

    monkeypatch.setattr(train_ppo, "_sample_exogenous_messages", fake_sample)
    monkeypatch.setattr(ObservationWrapper, "apply_msg_dropout", fake_dropout)
    monkeypatch.setattr(ObservationWrapper, "update_msg_marginals", record_update)

    cfg = minimal_test_config(
        n_agents=4,
        n_episodes=2,
        T=4,
        comm_enabled=True,
        n_senders=4,
        seed=702,
        save_path=str(ckpt),
        condition_name="cond1",
        sign_lambda=0.0,
        list_lambda=0.0,
        msg_source_mode="uniform",
    )

    train(cfg)

    assert len(recorded) > 0
    assert set(recorded) == {1}


def test_exogenous_msg_source_trainer_logs_comm_counts(tmp_path: Path):
    metrics_path = tmp_path / "exog_comm_metrics.jsonl"
    ckpt = tmp_path / "cond1_public_random_seed703.pt"
    cfg = minimal_test_config(
        n_agents=4,
        n_episodes=2,
        T=6,
        comm_enabled=True,
        n_senders=4,
        seed=703,
        save_path=str(ckpt),
        condition_name="cond1",
        sign_lambda=0.0,
        list_lambda=0.0,
        regime_log_interval=1,
        metrics_jsonl_path=str(metrics_path),
        msg_dropout=0.0,
        msg_source_mode="public_random",
    )

    train(cfg)

    rows = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    comm_rows = [
        row
        for row in rows
        if row.get("scope") == "comm"
        and row.get("metric") in {"mi_message_f", "mi_message_action"}
    ]
    assert len(comm_rows) > 0
    assert any(int(row.get("n_pairs", 0)) > 0 for row in comm_rows)


def test_msg_source_mode_cannot_mix_with_training_intervention():
    args = train_ppo.parse_args(
        [
            "--comm_enabled",
            "--n_senders",
            "4",
            "--msg_source_mode",
            "public_random",
            "--msg_training_intervention",
            "uniform",
        ]
    )

    with pytest.raises(
        ValueError,
        match="msg_source_mode != learned cannot be combined with msg_training_intervention",
    ):
        train_ppo.args_to_config(args)


def test_episode_offset_drives_absolute_checkpoint_numbering(tmp_path: Path):
    ckpt = tmp_path / "cond2_seed888.pt"
    cfg = minimal_test_config(
        n_agents=4,
        n_episodes=3,
        T=4,
        comm_enabled=False,
        n_senders=0,
        seed=888,
        save_path=str(ckpt),
        condition_name="cond2",
        checkpoint_interval=1,
        episode_offset=10,
        schedule_total_episodes=13,
    )
    train(cfg)

    assert (tmp_path / "cond2_seed888_ep11.pt").exists()
    assert (tmp_path / "cond2_seed888_ep12.pt").exists()
    payload = torch.load(ckpt, map_location="cpu")
    assert payload["config"]["episode_offset"] == 10
    assert payload["config"]["schedule_total_episodes"] == 13


def test_reduced_history_training_smoke_persists_history_mode(tmp_path: Path):
    ckpt = tmp_path / "cond1_seed999.pt"
    cfg = minimal_test_config(
        n_agents=4,
        n_episodes=2,
        T=4,
        comm_enabled=True,
        n_senders=4,
        seed=999,
        save_path=str(ckpt),
        condition_name="cond1",
        history_mode="reduced",
    )

    train(cfg)

    payload = torch.load(ckpt, map_location="cpu")
    assert payload["config"]["history_mode"] == "reduced"
