from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import shlex
import subprocess
import time
from typing import Dict, List, Optional, Tuple


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
LOG_DIR = os.path.join(OUT_DIR, "overnight_logs")
MASTER_LOG = os.path.join(LOG_DIR, "overnight_runner.log")
CHECKPOINT_TXT = os.path.join(REPO_ROOT, "outputs", "train", "curriculum", "stage2_progress_checkpoints.txt")
SUMMARY_MD = os.path.join(OUT_DIR, "OVERNIGHT_SUMMARY.md")


def _ts() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CHECKPOINT_TXT), exist_ok=True)


def _log(msg: str):
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)
    with open(MASTER_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _run_ps() -> str:
    cmd = "ps -ax -o pid=,command="
    proc = subprocess.run(
        cmd,
        shell=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout


def _matching_pids(pattern: str) -> List[int]:
    out = _run_ps()
    rx = re.compile(pattern)
    pids: List[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        if not rx.search(line):
            continue
        parts = line.split(maxsplit=1)
        try:
            pid = int(parts[0])
        except Exception:
            continue
        pids.append(pid)
    return sorted(set(pids))


def _kill_pid(pid: int):
    try:
        os.kill(pid, 15)
        _log(f"sent TERM to pid={pid}")
    except ProcessLookupError:
        _log(f"pid={pid} already exited")


def _wait_for_processes(pattern: str, poll_sec: int = 30):
    while True:
        pids = _matching_pids(pattern)
        if len(pids) == 0:
            return
        _log(f"waiting for processes to finish pattern={pattern!r}, pids={pids}")
        _update_stage2_checkpoints()
        time.sleep(poll_sec)


def _spawn_and_wait(cmd: List[str], log_path: str) -> int:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    _log("run: " + " ".join(shlex.quote(x) for x in cmd))
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# started {_ts()}\n")
        logf.write("# cmd: " + " ".join(cmd) + "\n\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
        while True:
            rc = proc.poll()
            if rc is not None:
                _log(f"finished rc={rc}: {os.path.basename(log_path)}")
                return int(rc)
            _update_stage2_checkpoints()
            time.sleep(30)


def _count_active_trainers() -> int:
    out = _run_ps()
    n = 0
    for line in out.splitlines():
        if "train_ppo.py" in line:
            n += 1
    return n


def _stage2_log_path() -> str:
    return os.path.join(REPO_ROOT, "outputs", "train", "curriculum", "logs", "stage2_switch_seed101_200k.log")


def _parse_episode_lines(log_path: str) -> Dict[int, Dict[str, float]]:
    if not os.path.exists(log_path):
        return {}
    ep_map: Dict[int, Dict[str, float]] = {}
    rx_ep = re.compile(r"\[episode\s+(\d+)\]\s+coop=([0-9.]+)\s+avg_reward=([0-9.]+)")
    rx_reg = re.compile(
        r"\[regime @ episode\s+(\d+)\]\s+comp=([0-9.]+)\(n=(\d+)\)\s+mixe=([0-9.]+)\(n=(\d+)\)\s+coop=([0-9.]+)\(n=(\d+)\)"
    )
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = rx_ep.search(line)
            if m:
                ep = int(m.group(1))
                ep_map.setdefault(ep, {})
                ep_map[ep]["coop_rate"] = float(m.group(2))
                ep_map[ep]["avg_reward"] = float(m.group(3))
                continue
            m = rx_reg.search(line)
            if m:
                ep = int(m.group(1))
                ep_map.setdefault(ep, {})
                ep_map[ep]["reg_comp"] = float(m.group(2))
                ep_map[ep]["reg_comp_n"] = int(m.group(3))
                ep_map[ep]["reg_mixed"] = float(m.group(4))
                ep_map[ep]["reg_mixed_n"] = int(m.group(5))
                ep_map[ep]["reg_coop"] = float(m.group(6))
                ep_map[ep]["reg_coop_n"] = int(m.group(7))
    return ep_map


def _load_written_milestones() -> set[int]:
    if not os.path.exists(CHECKPOINT_TXT):
        return set()
    out = set()
    rx = re.compile(r"milestone=(\d+)")
    with open(CHECKPOINT_TXT, "r", encoding="utf-8") as f:
        for line in f:
            m = rx.search(line)
            if m:
                out.add(int(m.group(1)))
    return out


def _append_checkpoint_line(line: str):
    with open(CHECKPOINT_TXT, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    _log(line)


def _update_stage2_checkpoints():
    log_path = _stage2_log_path()
    ep_map = _parse_episode_lines(log_path)
    if len(ep_map) == 0:
        return
    milestones = [50000, 100000, 150000, 200000]
    written = _load_written_milestones()
    for ms in milestones:
        if ms in written:
            continue
        if ms not in ep_map:
            continue
        row = ep_map[ms]
        comp = row.get("reg_comp")
        mixed = row.get("reg_mixed")
        coop = row.get("reg_coop")
        regime_str = (
            f"reg_comp={comp:.3f} reg_mixed={mixed:.3f} reg_coop={coop:.3f}"
            if (comp is not None and mixed is not None and coop is not None)
            else "regime_split=unavailable"
        )
        line = (
            f"{_ts()} stage2 milestone={ms} "
            f"coop={row.get('coop_rate', float('nan')):.3f} "
            f"avg_reward={row.get('avg_reward', float('nan')):.3f} "
            f"{regime_str}"
        )
        _append_checkpoint_line(line)


def _latest_jsonl_regime(path: str) -> Dict[str, Dict]:
    latest: Dict[str, Dict] = {}
    if not os.path.exists(path):
        return latest
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("scope") != "regime" or row.get("window") != "cumulative":
                continue
            key = str(row.get("key", ""))
            if key == "":
                continue
            latest[key] = row
    return latest


def _estimate_remaining_from_log(log_path: str, target_eps: int) -> Optional[str]:
    if not os.path.exists(log_path):
        return None
    ep_map = _parse_episode_lines(log_path)
    if len(ep_map) < 2:
        return None
    eps = sorted(ep_map.keys())
    e0, e1 = eps[0], eps[-1]
    if e1 <= e0:
        return None
    mtime = os.path.getmtime(log_path)
    # Use log age over covered episode span as rough rate.
    covered = e1 - e0
    if covered <= 0:
        return None
    # Fallback rough estimate: assume 1000 eps per ~15 min if unknown.
    # Here we use a conservative constant to avoid over-claiming precision.
    eps_per_hour = 3000.0
    rem = max(0, target_eps - e1)
    hours = rem / eps_per_hour
    return f"~{hours:.1f}h remaining (rough)"


def _run_train_if_missing(
    save_path_rel: str,
    cmd: List[str],
    log_name: str,
    failures: List[str],
):
    save_path = os.path.join(REPO_ROOT, save_path_rel)
    if os.path.exists(save_path):
        _log(f"skip existing checkpoint: {save_path_rel}")
        return
    rc = _spawn_and_wait(cmd=cmd, log_path=os.path.join(LOG_DIR, log_name))
    if rc != 0:
        failures.append(f"{log_name} failed rc={rc}")


def _stage2_is_running() -> bool:
    pids = _matching_pids(r"stage2_switch_seed101_200k\.pt")
    return len(pids) > 0


def _run_fixedf_sweep_batch(
    seeds: List[int],
    failures: List[str],
    fixedf_workers_with_stage2: int,
    fixedf_workers_without_stage2: int,
    max_total_trainers: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
):
    base_workers = (
        int(fixedf_workers_with_stage2) if _stage2_is_running() else int(fixedf_workers_without_stage2)
    )
    active_now = _count_active_trainers()
    available = max(1, int(max_total_trainers) - int(active_now))
    max_workers = max(1, min(int(base_workers), int(available)))
    _log(
        f"fixed-f batch seeds={seeds} active_trainers={active_now} "
        f"base_workers={base_workers} available={available} using_workers={max_workers}"
    )
    cmd = [
        "python3",
        "src/experiments_pgg_v0/run_fixed_f_sweep.py",
        "--f_values",
        "0.5",
        "1.5",
        "2.5",
        "3.5",
        "5.0",
        "--seeds",
        *[str(s) for s in seeds],
        "--n_episodes",
        "50000",
        "--max_workers",
        str(max_workers),
        "--log_interval",
        "500",
        "--regime_log_interval",
        "500",
        "--reward_scale",
        "20.0",
        "--early_stop_patience",
        str(early_stop_patience),
        "--early_stop_min_delta",
        str(early_stop_min_delta),
        "--skip_existing",
    ]
    rc = _spawn_and_wait(
        cmd=cmd,
        log_path=os.path.join(LOG_DIR, f"fixedf_batch_{'_'.join(str(s) for s in seeds)}.log"),
    )
    if rc != 0:
        failures.append(f"fixedf_batch_{seeds} failed rc={rc}")


def _run_condition_loop(condition: str, seeds: List[int], failures: List[str]):
    for seed in seeds:
        save_path = f"outputs/train/grid/{condition}_seed{seed}.pt"
        metrics_jsonl = f"outputs/train/grid/metrics/{condition}_seed{seed}.jsonl"
        common = [
            "python3",
            "src/experiments_pgg_v0/train_ppo.py",
            "--n_episodes",
            "2000",
            "--T",
            "100",
            "--n_agents",
            "4",
            "--F",
            "0.5",
            "1.5",
            "2.5",
            "3.5",
            "5.0",
            "--rho",
            "0.05",
            "--seed",
            str(seed),
            "--save_path",
            save_path,
            "--gamma",
            "0.99",
            "--log_interval",
            "100",
            "--regime_log_interval",
            "200",
            "--metrics_jsonl_path",
            metrics_jsonl,
            "--condition_name",
            condition,
            "--log_sessions",
            "--session_log_dir",
            "outputs/train/grid/sessions",
            "--consolidate_sessions",
        ]
        if condition == "cond6":
            cmd = common + [
                "--sigmas",
                "0",
                "0",
                "0",
                "0",
                "--epsilon_tremble",
                "0.0",
            ]
        elif condition == "cond2":
            cmd = common + [
                "--sigmas",
                "0.5",
                "0.5",
                "0.5",
                "0.5",
                "--epsilon_tremble",
                "0.05",
            ]
        else:
            raise ValueError(condition)

        _run_train_if_missing(
            save_path_rel=save_path,
            cmd=cmd,
            log_name=f"{condition}_seed{seed}.log",
            failures=failures,
        )


def _collect_run_status() -> Dict:
    status: Dict[str, Dict] = {}
    # Fixed-f grid metrics.
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "outputs", "train", "fixed_f_grid", "metrics", "*.jsonl"))):
        run = os.path.splitext(os.path.basename(path))[0]
        status[run] = {"type": "fixed_f", "metrics_jsonl": path, "regime": _latest_jsonl_regime(path)}
    # Grid metrics cond6/cond2.
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "outputs", "train", "grid", "metrics", "*.jsonl"))):
        run = os.path.splitext(os.path.basename(path))[0]
        status[run] = {"type": "grid", "metrics_jsonl": path, "regime": _latest_jsonl_regime(path)}
    # Stage2 from log.
    stage2_log = _stage2_log_path()
    ep_map = _parse_episode_lines(stage2_log)
    if len(ep_map) > 0:
        last_ep = max(ep_map.keys())
        status["stage2_switch_seed101_200k"] = {
            "type": "stage2",
            "log_path": stage2_log,
            "last_episode": last_ep,
            "last_metrics": ep_map[last_ep],
            "remaining_estimate": _estimate_remaining_from_log(stage2_log, 200000),
            "running": _stage2_is_running(),
        }
    return status


def _write_summary(failures: List[str]):
    status = _collect_run_status()
    lines: List[str] = []
    lines.append("# Overnight Summary")
    lines.append("")
    lines.append(f"- Generated: `{_ts()}`")
    lines.append("")
    lines.append("## Run Status")
    for run, row in sorted(status.items()):
        if row["type"] in {"fixed_f", "grid"}:
            reg = row.get("regime", {})
            comp = reg.get("competitive", {}).get("coop_rate")
            mixed = reg.get("mixed", {}).get("coop_rate")
            coop = reg.get("cooperative", {}).get("coop_rate")
            lines.append(
                f"- `{run}`: regime_coop(comp={_fmt_num(comp)}, mixed={_fmt_num(mixed)}, coop={_fmt_num(coop)})"
            )
        elif row["type"] == "stage2":
            lm = row.get("last_metrics", {})
            lines.append(
                f"- `{run}`: last_ep={row.get('last_episode')} coop={_fmt_num(lm.get('coop_rate'))} "
                f"avg_reward={_fmt_num(lm.get('avg_reward'))} running={row.get('running')} "
                f"remaining={row.get('remaining_estimate') or 'unknown'}"
            )
            if all(k in lm for k in ("reg_comp", "reg_mixed", "reg_coop")):
                lines.append(
                    f"  - regime_split(comp={_fmt_num(lm.get('reg_comp'))}, "
                    f"mixed={_fmt_num(lm.get('reg_mixed'))}, coop={_fmt_num(lm.get('reg_coop'))})"
                )
            else:
                lines.append("  - regime_split: unavailable in current stage2 log format")
    lines.append("")
    lines.append("## Errors")
    if len(failures) == 0:
        lines.append("- none")
    else:
        for x in failures:
            lines.append(f"- {x}")
    lines.append("")
    lines.append("## Stage2 Regime-Separation Status")
    stage2 = status.get("stage2_switch_seed101_200k")
    if stage2 is None:
        lines.append("- stage2 run not found.")
    else:
        lm = stage2.get("last_metrics", {})
        if all(k in lm for k in ("reg_comp", "reg_mixed", "reg_coop")):
            lines.append(
                f"- latest split: comp={_fmt_num(lm['reg_comp'])}, mixed={_fmt_num(lm['reg_mixed'])}, coop={_fmt_num(lm['reg_coop'])}"
            )
        else:
            lines.append("- regime split not yet available from current long-running log.")
    lines.append("")
    lines.append("## Recommendation")
    lines.append(
        "- If mixed/cooperative regime cooperation is still flat at low values by 50k+, increase episode budget and inspect reward/advantage signal quality before changing algorithms."
    )
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    _log(f"wrote summary: {SUMMARY_MD}")


def _fmt_num(v) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):.3f}"
    except Exception:
        return "NA"


def _phase2_gate_status(min_episode: int, min_gap: float) -> Tuple[bool, str]:
    ep_map = _parse_episode_lines(_stage2_log_path())
    if len(ep_map) == 0:
        return False, "stage2 log unavailable"

    eligible = sorted([ep for ep in ep_map.keys() if ep >= int(min_episode)])
    if len(eligible) == 0:
        return False, f"stage2 below gate episode threshold ({max(ep_map.keys())} < {min_episode})"

    ep = int(eligible[-1])
    row = ep_map[ep]
    req = ("reg_comp", "reg_mixed", "reg_coop")
    if not all(k in row for k in req):
        return False, f"stage2 ep={ep} has no regime-split metrics yet"

    comp = float(row["reg_comp"])
    mixed = float(row["reg_mixed"])
    coop = float(row["reg_coop"])
    gap = coop - comp
    ordered = (coop > mixed) and (mixed > comp)
    separated = gap >= float(min_gap)
    ok = bool(ordered and separated)
    reason = (
        f"stage2 ep={ep} comp={comp:.3f} mixed={mixed:.3f} coop={coop:.3f} "
        f"gap={gap:.3f} ordered={ordered} separated={separated}"
    )
    return ok, reason


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll_sec", type=int, default=30)
    parser.add_argument("--run_grid_after_phase2_gate", action="store_true")
    parser.add_argument("--phase2_gate_min_episode", type=int, default=50000)
    parser.add_argument("--phase2_gate_min_gap", type=float, default=0.20)
    parser.add_argument("--max_total_trainers", type=int, default=9)
    parser.add_argument("--fixedf_workers_with_stage2", type=int, default=4)
    parser.add_argument("--fixedf_workers_without_stage2", type=int, default=6)
    parser.add_argument("--fixedf_early_stop_patience", type=int, default=5000)
    parser.add_argument("--fixedf_early_stop_min_delta", type=float, default=1e-3)
    args = parser.parse_args()

    _ensure_dirs()
    _log("overnight plan start")
    failures: List[str] = []

    # Task 2: ensure any broken gamma=0 stage1 run is stopped.
    broken_pids = _matching_pids(r"stage1_fixedf5_seed101_50k\.pt.*--gamma 0\.0")
    if len(broken_pids) > 0:
        _log(f"found broken stage1 gamma0 run pids={broken_pids}; stopping")
        for pid in broken_pids:
            _kill_pid(pid)
        time.sleep(2)
    else:
        _log("no broken stage1 gamma=0.0 run found")

    # Wait for all currently active fixed-f sweep workers to finish before follow-up tasks.
    _wait_for_processes(r"run_fixed_f_sweep\.py")
    _wait_for_processes(r"train_ppo\.py.*outputs/train/fixed_f_grid/fixedf_")

    # Task 1 follow-up: explicit f=5.0 seed101 run.
    f5_cmd = [
        "python3",
        "src/experiments_pgg_v0/train_ppo.py",
        "--n_episodes",
        "50000",
        "--T",
        "100",
        "--n_agents",
        "4",
        "--F",
        "5.0",
        "--rho",
        "0.0",
        "--sigmas",
        "0",
        "0",
        "0",
        "0",
        "--epsilon_tremble",
        "0.0",
        "--gamma",
        "0.99",
        "--lam",
        "0.95",
        "--seed",
        "101",
        "--save_path",
        "outputs/train/fixed_f_grid/fixedf_5p0_seed101.pt",
        "--reward_scale",
        "20.0",
        "--lr_schedule",
        "linear",
        "--min_lr",
        "1e-5",
        "--log_interval",
        "500",
        "--regime_log_interval",
        "500",
        "--metrics_jsonl_path",
        "outputs/train/fixed_f_grid/metrics/fixedf_5p0_seed101.jsonl",
        "--condition_name",
        "fixedf_5p0",
        "--early_stop_patience",
        str(args.fixedf_early_stop_patience),
        "--early_stop_min_delta",
        str(args.fixedf_early_stop_min_delta),
    ]
    _run_train_if_missing(
        save_path_rel="outputs/train/fixed_f_grid/fixedf_5p0_seed101.pt",
        cmd=f5_cmd,
        log_name="fixedf_5p0_seed101_followup.log",
        failures=failures,
    )

    # Task 3: multi-seed fixed-f sweeps.
    _run_fixedf_sweep_batch(
        [202, 303],
        failures,
        fixedf_workers_with_stage2=args.fixedf_workers_with_stage2,
        fixedf_workers_without_stage2=args.fixedf_workers_without_stage2,
        max_total_trainers=args.max_total_trainers,
        early_stop_patience=args.fixedf_early_stop_patience,
        early_stop_min_delta=args.fixedf_early_stop_min_delta,
    )
    _run_fixedf_sweep_batch(
        [404, 505],
        failures,
        fixedf_workers_with_stage2=args.fixedf_workers_with_stage2,
        fixedf_workers_without_stage2=args.fixedf_workers_without_stage2,
        max_total_trainers=args.max_total_trainers,
        early_stop_patience=args.fixedf_early_stop_patience,
        early_stop_min_delta=args.fixedf_early_stop_min_delta,
    )

    # Task 4: update stage2 checkpoints opportunistically.
    _update_stage2_checkpoints()

    # Task 5: cond6 + cond2 baseline loops.
    if args.run_grid_after_phase2_gate:
        ok, gate_reason = _phase2_gate_status(
            min_episode=args.phase2_gate_min_episode,
            min_gap=args.phase2_gate_min_gap,
        )
        _log(f"phase2 gate: {gate_reason}")
        if ok:
            _run_condition_loop("cond6", [101, 202, 303, 404, 505], failures)
            _run_condition_loop("cond2", [101, 202, 303, 404, 505], failures)
        else:
            msg = "phase2 gate not met; skipped cond6/cond2 grids"
            failures.append(msg)
            _log(msg)
    else:
        _log("cond6/cond2 grids disabled by default; use --run_grid_after_phase2_gate to enable.")

    _update_stage2_checkpoints()
    _write_summary(failures)
    _log("overnight plan done")


if __name__ == "__main__":
    main()
