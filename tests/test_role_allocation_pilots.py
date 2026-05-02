import csv
import json
from pathlib import Path

from scripts.reproduce.run_role_cost_profile_grid import _expand_conditions
from src.experiments_role_allocation.common import (
    RoleMetrics,
    config_for_run,
    cost_profile,
    resolve_invalid_volunteer_penalty,
    write_json,
)
from src.experiments_role_allocation.train_ppo import TrainConfig, train


def test_role_metrics_primary_rates():
    metrics = RoleMetrics()
    infos = {
        "true_need": 1.0,
        "executed_actions": {
            "agent_0": 0,
            "agent_1": 1,
            "agent_2": 0,
            "agent_3": 0,
        },
        "volunteer_costs": {
            "agent_0": 1.75,
            "agent_1": 0.25,
            "agent_2": 0.75,
            "agent_3": 1.25,
        },
        "n_volunteers": 1,
        "produced": True,
    }
    rewards = {
        "agent_0": 4.0,
        "agent_1": 3.75,
        "agent_2": 4.0,
        "agent_3": 4.0,
    }

    metrics.update(infos, rewards)
    out = metrics.summary()

    assert out["p_any_volunteer_given_need"] == 1.0
    assert out["p_exactly_one_given_need"] == 1.0
    assert out["missed_need_rate"] == 0.0
    assert out["redundant_volunteer_rate"] == 0.0
    assert out["lowest_cost_volunteer_rate_given_need"] == 1.0
    assert out["mean_volunteer_cost_when_need_produced"] == 0.25


def test_role_metrics_crossed_effective_service_rates():
    metrics = RoleMetrics()
    infos = {
        "true_need": 1.0,
        "need_present": True,
        "feasible_need": True,
        "executed_actions": {
            "agent_0": 1,
            "agent_1": 0,
            "agent_2": 0,
            "agent_3": 0,
        },
        "eligibility": {
            "agent_0": 0,
            "agent_1": 1,
            "agent_2": 0,
            "agent_3": 0,
        },
        "volunteer_costs": {
            "agent_0": 0.5,
            "agent_1": 0.9,
            "agent_2": 1.3,
            "agent_3": 1.7,
        },
        "n_volunteers": 1,
        "n_effective_volunteers": 0,
        "n_invalid_volunteers": 1,
        "target_volunteers": 1,
        "produced": False,
    }
    rewards = {
        "agent_0": -2.5,
        "agent_1": -2.0,
        "agent_2": -2.0,
        "agent_3": -2.0,
    }

    metrics.update(infos, rewards)
    out = metrics.summary()

    assert out["p_exactly_one_given_need"] == 1.0
    assert out["p_exactly_one_effective_given_need"] == 0.0
    assert out["missed_need_rate"] == 0.0
    assert out["missed_service_need_rate"] == 1.0
    assert out["missed_feasible_need_rate"] == 1.0
    assert out["invalid_volunteer_round_rate"] == 1.0
    assert out["lowest_eligible_cost_volunteer_rate_given_need"] == 0.0


def test_json_writer_converts_nan_to_null(tmp_path: Path):
    path = tmp_path / "payload.json"
    write_json(path, {"x": float("nan")})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["x"] is None


def test_cost_profile_grid_profiles_and_fixed_alias():
    assert cost_profile("equal") == [1.0, 1.0, 1.0, 1.0]
    assert cost_profile("narrow") == [0.75, 0.9, 1.1, 1.25]
    assert cost_profile("moderate") == [0.5, 0.9, 1.3, 1.7]
    assert _expand_conditions(["no_comm", "fixed"], "both") == [
        "no_comm",
        "fixed0",
        "fixed1",
    ]


def test_invalid_volunteer_penalty_defaults_to_max_cost_level():
    levels = [0.5, 0.9, 1.3, 1.7]
    cfg = config_for_run(
        horizon=4,
        uncertainty=0.5,
        cost_levels=levels,
        miss_loss=2.0,
        redundant_penalty=0.0,
    )

    assert resolve_invalid_volunteer_penalty(levels) == 1.7
    assert cfg["invalid_volunteer_penalty"] == 1.7


def test_invalid_volunteer_penalty_explicit_override_is_preserved():
    cfg = config_for_run(
        horizon=4,
        uncertainty=0.5,
        cost_levels=[0.5, 0.9, 1.3, 1.7],
        miss_loss=2.0,
        redundant_penalty=0.0,
        invalid_volunteer_penalty=0.25,
    )

    assert cfg["invalid_volunteer_penalty"] == 0.25


def test_minimal_no_comm_trainer_writes_role_trace(tmp_path: Path):
    result = train(
        TrainConfig(
            condition="no_comm",
            seed=123,
            episodes=2,
            horizon=4,
            update_episodes=1,
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
    assert (out_dir / "train_trace.csv").exists()
    assert (out_dir / "eval_trace.csv").exists()

    with (out_dir / "eval_trace.csv").open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)

    for column in [
        "need_hat",
        "local_cost",
        "true_need",
        "cost_agent_0",
        "intended_action",
        "executed_action",
        "reward",
        "n_volunteers",
        "lowest_cost_agent",
    ]:
        assert column in row
