import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _base_row(
    *,
    condition: str,
    seed: int,
    episode: int,
    f_value: str,
    coop_rate: float,
) -> dict:
    return {
        "checkpoint": f"/tmp/{condition}_seed{seed}_ep{episode}.pt",
        "condition": condition,
        "train_seed": str(seed),
        "comm_enabled": "1" if condition == "cond1" else "0",
        "eval_seed": "9001",
        "eval_policy": "greedy",
        "ablation": "none",
        "history_intervention": "none",
        "sender_remap": "none",
        "cross_play": "none",
        "scope": "f_value",
        "key": f_value,
        "n_rounds": "100",
        "coop_rate": f"{coop_rate:.6f}",
        "avg_reward": "1.0",
        "avg_welfare": "4.0",
        "checkpoint_episode": str(episode),
        "suite_kind": "comm" if condition == "cond1" else "baseline",
    }


def test_sameckpt_continuation_summary_reports_paired_stats(tmp_path: Path):
    reference_csv = tmp_path / "reference_suite.csv"
    branch_csv = tmp_path / "branch_suite.csv"
    out_dir = tmp_path / "out"

    reference_rows = []
    branch_rows = []
    seeds = [101, 202, 303, 404, 505]
    for seed in seeds:
        reference_rows.extend(
            [
                _base_row(condition="cond1", seed=seed, episode=100000, f_value="3.500", coop_rate=0.40),
                _base_row(condition="cond1", seed=seed, episode=100000, f_value="5.000", coop_rate=0.60),
                _base_row(condition="cond1", seed=seed, episode=150000, f_value="3.500", coop_rate=0.55),
                _base_row(condition="cond1", seed=seed, episode=150000, f_value="5.000", coop_rate=0.75),
                _base_row(condition="cond2", seed=seed, episode=150000, f_value="3.500", coop_rate=0.35),
                _base_row(condition="cond2", seed=seed, episode=150000, f_value="5.000", coop_rate=0.50),
            ]
        )
        branch_rows.extend(
            [
                _base_row(condition="cond1", seed=seed, episode=150000, f_value="3.500", coop_rate=0.45),
                _base_row(condition="cond1", seed=seed, episode=150000, f_value="5.000", coop_rate=0.60),
            ]
        )

    _write_rows(reference_csv, reference_rows)
    _write_rows(branch_csv, branch_rows)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.analysis.summarize_phase3_sameckpt_continuations",
            "--reference_suite_csv",
            str(reference_csv),
            "--branch_suite",
            "fixed0_100k",
            str(branch_csv),
            "--bootstrap_samples",
            "2000",
            "--out_dir",
            str(out_dir),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )

    paired_csv = out_dir / "sameckpt_continuation_paired_stats.csv"
    summary_md = out_dir / "sameckpt_continuation_summary.md"
    assert paired_csv.exists()
    assert summary_md.exists()

    with paired_csv.open("r", encoding="utf-8") as f:
        paired_rows = list(csv.DictReader(f))

    assert len(paired_rows) == 2
    for row in paired_rows:
        assert row["mode"] == "fixed0_100k"
        assert int(row["checkpoint_episode"]) == 150000
        assert int(row["n_pairs"]) == 5
        assert float(row["mean_delta_mode_minus_reference"]) < 0.0
        assert float(row["sign_flip_p_value"]) <= 0.0625 + 1e-12

    text = summary_md.read_text(encoding="utf-8")
    assert "fixed0_100k" in text
    assert "Delta vs learned" in text
