import json
from pathlib import Path

from src.experiments_pgg_v0.train_ppo import minimal_test_config, train


def test_trainer_logs_responsiveness_metric(tmp_path: Path):
    metrics_path = tmp_path / "comm_metrics.jsonl"
    cfg = minimal_test_config(
        n_agents=4,
        n_episodes=2,
        T=6,
        comm_enabled=True,
        n_senders=4,
        seed=321,
        save_path=str(tmp_path / "cond1_seed321.pt"),
        condition_name="cond1",
        regime_log_interval=1,
        metrics_jsonl_path=str(metrics_path),
    )
    train(cfg)
    assert metrics_path.exists()

    rows = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    resp_rows = [r for r in rows if r.get("scope") == "comm" and r.get("metric") == "responsiveness_kl"]
    assert len(resp_rows) > 0
    assert all(float(r.get("value", 0.0)) >= 0.0 for r in resp_rows)
