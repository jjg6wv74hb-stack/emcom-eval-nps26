import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.analysis import evaluate_regime_conditional


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_write_trace_csv_includes_intended_action(tmp_path: Path):
    out_csv = tmp_path / "trace.csv"
    evaluate_regime_conditional._write_trace_csv(
        str(out_csv),
        [
            {
                "checkpoint": "ckpt.pt",
                "condition": "cond1",
                "train_seed": 101,
                "eval_seed": 9001,
                "eval_policy": "greedy",
                "ablation": "none",
                "history_intervention": "none",
                "sender_remap": "none",
                "cross_play": "none",
                "episode": 0,
                "t": 0,
                "agent_id": "agent_0",
                "true_f": 3.5,
                "f_hat": 3.2,
                "intended_action": 1,
                "action": 0,
                "reward": 0.0,
                "round_welfare": 0.0,
                "obs_last_coop_fraction": 0.0,
                "obs_own_last_action": 0.0,
                "obs_ewma_coop": 0.0,
                "own_sent_msg": 1,
                "delivered_msg_agent_0": 1,
                "recv_any_m0": 0,
                "recv_any_m1": 1,
                "recv_pattern": "agent_1:1|agent_2:0|agent_3:1",
            }
        ],
    )

    with out_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert "intended_action" in reader.fieldnames
        row = next(reader)
    assert row["intended_action"] == "1"
    assert row["action"] == "0"


def _make_sender_encoding_trace(path: Path) -> None:
    rows = []
    for train_seed in (101, 202, 303):
        for agent_id in ("agent_0", "agent_1"):
            for intended_action in (0, 1):
                for f_hat, true_f in ((2.0, 0.5), (4.0, 5.0)):
                    for own_last_action, obs_ewma_coop in ((0, 0.2), (1, 0.8)):
                        for rep in range(6):
                            if agent_id == "agent_0":
                                own_sent_msg = int(f_hat >= 3.5)
                            else:
                                own_sent_msg = int(intended_action == 1)
                            rows.append(
                                {
                                    "checkpoint": "cond1_seed101.pt",
                                    "condition": "cond1",
                                    "train_seed": train_seed,
                                    "eval_seed": 9001 + (train_seed % 10),
                                    "eval_policy": "greedy",
                                    "ablation": "none",
                                    "history_intervention": "none",
                                    "sender_remap": "none",
                                    "cross_play": "none",
                                    "episode": rep,
                                    "t": rep,
                                    "agent_id": agent_id,
                                    "true_f": true_f,
                                    "f_hat": f_hat,
                                    "intended_action": intended_action,
                                    # Deliberately invert the executed action so the script
                                    # must prefer intended_action when present.
                                    "action": 1 - intended_action,
                                    "reward": 0.0,
                                    "round_welfare": 0.0,
                                    "obs_last_coop_fraction": 0.25 if rep % 2 == 0 else 0.75,
                                    "obs_own_last_action": own_last_action,
                                    "obs_ewma_coop": obs_ewma_coop,
                                    "own_sent_msg": own_sent_msg,
                                    "delivered_msg_agent_0": 0,
                                    "delivered_msg_agent_1": 0,
                                    "delivered_msg_agent_2": 0,
                                    "delivered_msg_agent_3": 0,
                                    "recv_any_m0": 1,
                                    "recv_any_m1": 0,
                                    "recv_pattern": "agent_1:0|agent_2:0|agent_3:0",
                                    "checkpoint_episode": 150000,
                                    "suite_kind": "comm",
                                }
                            )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_sender_encoding_summary_prefers_intended_action_and_labels_dominance(tmp_path: Path):
    trace_csv = tmp_path / "trace.csv"
    out_dir = tmp_path / "sender_encoding"
    _make_sender_encoding_trace(trace_csv)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.analysis.summarize_phase3_sender_encoding",
            "--trace_csv",
            str(trace_csv),
            "--out_dir",
            str(out_dir),
            "--min_cell_obs",
            "2",
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )

    effect_csv = out_dir / "sender_effect_summary.csv"
    assert effect_csv.exists()
    effect_df = pd.read_csv(effect_csv)
    row_a0 = effect_df[(effect_df["train_seed"] == 101) & (effect_df["sender_id"] == "agent_0")].iloc[0]
    row_a1 = effect_df[(effect_df["train_seed"] == 101) & (effect_df["sender_id"] == "agent_1")].iloc[0]
    assert row_a0["dominance_label"] == "regime_leaning"
    assert row_a1["dominance_label"] == "action_leaning"
    assert row_a0["dominance_regime_abs_pp"] > 90.0
    assert row_a1["dominance_action_abs_pp"] > 90.0

    summary_md = out_dir / "sender_encoding_summary.md"
    assert summary_md.exists()
    text = summary_md.read_text(encoding="utf-8")
    assert "action_column_used: `intended_action`" in text

    model_csv = out_dir / "sender_encoding_model_summary.csv"
    assert model_csv.exists()
    model_df = pd.read_csv(model_csv)
    assert set(model_df["model_name"]) == {
        "history_only",
        "history_fhat",
        "history_action",
        "history_fhat_action",
    }
