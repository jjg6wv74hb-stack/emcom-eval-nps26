import csv
import os
import subprocess
import sys
from pathlib import Path

from src.experiments_pgg_v0.train_ppo import minimal_test_config, train


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_checkpoint(tmp_path: Path, condition: str, seed: int) -> Path:
    ckpt = tmp_path / f"{condition}_seed{seed}.pt"
    cfg = minimal_test_config(
        n_agents=4,
        n_episodes=2,
        T=4,
        comm_enabled=True,
        n_senders=4,
        seed=seed,
        save_path=str(ckpt),
        checkpoint_interval=1,
        regime_log_interval=1,
        condition_name=condition,
    )
    train(cfg)
    assert ckpt.exists()
    assert (tmp_path / f"{condition}_seed{seed}_ep1.pt").exists()
    return ckpt


def test_base_checkpoint_suite_runner_explicit_paths(tmp_path: Path):
    final_ckpt = _make_checkpoint(tmp_path, "cond1", 123)
    mid_ckpt = tmp_path / "cond1_seed123_ep1.pt"
    out_dir = tmp_path / "suite_out"
    env = os.environ.copy()
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.analysis.run_base_checkpoint_suite",
            "--checkpoint",
            str(mid_ckpt),
            "--checkpoint",
            str(final_ckpt),
            "--out_dir",
            str(out_dir),
            "--n_eval_episodes",
            "1",
            "--max_workers",
            "1",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    )

    main_csv = out_dir / "checkpoint_suite_main.csv"
    condition_csv = out_dir / "checkpoint_suite_condition.csv"
    assert main_csv.exists()
    assert condition_csv.exists()

    with main_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    assert {"1", "2"} <= {row["checkpoint_episode"] for row in rows}
    assert {"none"} == {row["ablation"] for row in rows}
    assert {"none"} == {row["sender_remap"] for row in rows}
    assert {"none"} == {row["cross_play"] for row in rows}
    assert {"comm"} == {row["suite_kind"] for row in rows}
