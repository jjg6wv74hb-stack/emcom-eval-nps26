from pathlib import Path

import pytest
import torch

from src.experiments_pgg_v0.train_ppo import (
    _build_agents,
    _maybe_resume_agents,
    _sender_ids,
    minimal_test_config,
    train,
)
from src.wrappers.observation_wrapper import ObservationWrapper


def _assert_optimizer_state_matches(restored: dict, saved: dict):
    assert restored["param_groups"] == saved["param_groups"]
    assert set(restored["state"].keys()) == set(saved["state"].keys())
    for param_id, restored_state in restored["state"].items():
        saved_state = saved["state"][param_id]
        assert set(restored_state.keys()) == set(saved_state.keys())
        for key, value in restored_state.items():
            saved_value = saved_state[key]
            if torch.is_tensor(value):
                assert torch.allclose(value.cpu(), saved_value.cpu())
            else:
                assert value == saved_value


def test_checkpoint_saves_training_state_and_optimizer(tmp_path: Path):
    ckpt = tmp_path / "cond2_seed901.pt"
    cfg = minimal_test_config(
        n_agents=4,
        n_episodes=2,
        T=4,
        comm_enabled=False,
        n_senders=0,
        seed=901,
        save_path=str(ckpt),
        condition_name="cond2",
        checkpoint_interval=1,
    )
    train(cfg)

    ep1 = tmp_path / "cond2_seed901_ep1.pt"
    assert ep1.exists()
    payload = torch.load(ep1, map_location="cpu")
    assert payload["checkpoint_state_version"] == 2
    assert payload["training_state"]["abs_episode"] == 1
    assert payload["training_state"]["local_episode"] == 1
    for agent_state in payload["agents"].values():
        assert "optimizer" in agent_state
        assert len(agent_state["optimizer"]["state"]) > 0


def test_resume_ckpt_restores_optimizer_and_episode_offset(tmp_path: Path):
    ckpt = tmp_path / "cond2_seed902.pt"
    cfg = minimal_test_config(
        n_agents=4,
        n_episodes=2,
        T=4,
        comm_enabled=False,
        n_senders=0,
        seed=902,
        save_path=str(ckpt),
        condition_name="cond2",
        checkpoint_interval=1,
    )
    train(cfg)

    ep1 = tmp_path / "cond2_seed902_ep1.pt"
    saved_payload = torch.load(ep1, map_location="cpu")

    wrapper = ObservationWrapper(n_agents=4, comm_enabled=False, n_senders=0)
    sender_ids = _sender_ids(cfg)
    agents = _build_agents(cfg, obs_dim=wrapper.obs_dim, sender_ids=sender_ids)
    restored_payload = _maybe_resume_agents(agents, str(ep1))
    assert restored_payload["training_state"]["abs_episode"] == 1
    for agent_id, agent in agents.items():
        _assert_optimizer_state_matches(
            agent.optimizer.state_dict(),
            saved_payload["agents"][agent_id]["optimizer"],
        )

    resumed_ckpt = tmp_path / "cond2_seed902_resumed.pt"
    resumed_cfg = minimal_test_config(
        n_agents=4,
        n_episodes=1,
        T=4,
        comm_enabled=False,
        n_senders=0,
        seed=902,
        save_path=str(resumed_ckpt),
        condition_name="cond2",
        resume_ckpt=str(ep1),
        lr_schedule="linear",
        min_lr=1e-5,
    )
    train(resumed_cfg)

    resumed_payload = torch.load(resumed_ckpt, map_location="cpu")
    assert resumed_payload["config"]["episode_offset"] == 1
    assert resumed_payload["training_state"]["abs_episode"] == 2


def test_resume_ckpt_rejects_legacy_weight_only_checkpoint(tmp_path: Path):
    ckpt = tmp_path / "cond2_seed904.pt"
    cfg = minimal_test_config(
        n_agents=4,
        n_episodes=2,
        T=4,
        comm_enabled=False,
        n_senders=0,
        seed=904,
        save_path=str(ckpt),
        condition_name="cond2",
        checkpoint_interval=1,
    )
    train(cfg)

    ep1 = tmp_path / "cond2_seed904_ep1.pt"
    legacy_payload = torch.load(ep1, map_location="cpu")
    legacy_payload.pop("checkpoint_state_version", None)
    legacy_payload.pop("training_state", None)
    for agent_state in legacy_payload["agents"].values():
        agent_state.pop("optimizer", None)

    legacy_ckpt = tmp_path / "cond2_seed904_legacy.pt"
    torch.save(legacy_payload, legacy_ckpt)

    resumed_ckpt = tmp_path / "cond2_seed904_resumed.pt"
    resumed_cfg = minimal_test_config(
        n_agents=4,
        n_episodes=1,
        T=4,
        comm_enabled=False,
        n_senders=0,
        seed=904,
        save_path=str(resumed_ckpt),
        condition_name="cond2",
        resume_ckpt=str(legacy_ckpt),
    )
    with pytest.raises(ValueError, match="checkpoint_state_version >= 2"):
        train(resumed_cfg)


def test_vectorized_count_env_episodes_preserves_episode_budget(tmp_path: Path):
    ckpt = tmp_path / "cond2_seed903.pt"
    cfg = minimal_test_config(
        n_agents=4,
        n_episodes=8,
        num_envs=4,
        count_env_episodes=True,
        T=4,
        comm_enabled=False,
        n_senders=0,
        seed=903,
        save_path=str(ckpt),
        condition_name="cond2",
        checkpoint_interval=4,
        log_interval=4,
        regime_log_interval=4,
    )
    train(cfg)

    ep4 = tmp_path / "cond2_seed903_ep4.pt"
    assert ep4.exists()

    mid_payload = torch.load(ep4, map_location="cpu")
    assert mid_payload["config"]["count_env_episodes"] is True
    assert mid_payload["config"]["num_envs"] == 4
    assert mid_payload["training_state"]["abs_episode"] == 4
    assert mid_payload["training_state"]["local_episode"] == 4

    final_payload = torch.load(ckpt, map_location="cpu")
    assert final_payload["training_state"]["abs_episode"] == 8
    assert final_payload["training_state"]["local_episode"] == 8
