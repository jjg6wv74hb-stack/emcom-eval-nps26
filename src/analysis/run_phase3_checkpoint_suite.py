from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from src.analysis.checkpoint_artifacts import (
    atomic_write_json,
    atomic_write_rows,
    csv_has_data_rows,
    resolve_checkpoint_path,
    resolve_manifest_checkpoint_path,
)


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _checkpoint_path(
    checkpoint_dir: str,
    checkpoint_manifest: str,
    condition: str,
    seed: int,
    episode: int,
) -> str:
    manifest = str(checkpoint_manifest or "").strip()
    if manifest != "":
        return resolve_manifest_checkpoint_path(manifest, condition=condition, seed=seed, episode=episode)
    return resolve_checkpoint_path(checkpoint_dir, condition=condition, seed=seed, episode=episode)


def _task_name(
    condition: str,
    seed: int,
    episode: int,
    intervention: str,
    history_intervention: str = "none",
    observability_mode: str = "private",
    public_signal_sigma: float | None = None,
) -> str:
    base = f"{condition}_seed{int(seed)}_ep{int(episode)}_{intervention}"
    history_label = str(history_intervention or "none").strip().lower() or "none"
    if history_label != "none":
        base = f"{base}_hist_{history_label}"
    obs_label = str(observability_mode or "private").strip().lower() or "private"
    if obs_label != "private":
        base = f"{base}_obs_{obs_label}"
        if public_signal_sigma is not None and obs_label == "public_noisy":
            sigma_label = str(public_signal_sigma).replace(".", "p")
            base = f"{base}_psig_{sigma_label}"
    return base


def _run_task(task: Dict, raw_dir: str, log_dir: str, skip_existing: bool):
    out_csv = os.path.join(raw_dir, f"{task['name']}.csv")
    out_condition_csv = os.path.join(raw_dir, f"{task['name']}_condition.csv")
    out_comm_csv = os.path.join(raw_dir, f"{task['name']}_comm.csv")
    out_trace_csv = os.path.join(raw_dir, f"{task['name']}_trace.csv")
    out_sender_csv = os.path.join(raw_dir, f"{task['name']}_sender_semantics.csv")
    out_receiver_csv = os.path.join(raw_dir, f"{task['name']}_receiver_semantics.csv")
    out_posterior_csv = os.path.join(raw_dir, f"{task['name']}_posterior_strat.csv")
    log_path = os.path.join(log_dir, f"{task['name']}.log")

    expected = [out_csv, out_condition_csv]
    if str(task.get("suite_kind", "")).strip().lower() != "baseline":
        expected.append(out_comm_csv)
    if bool(task.get("posterior_strat")):
        expected.append(out_posterior_csv)
    if bool(task.get("collect_semantics")):
        expected.extend([out_trace_csv, out_sender_csv, out_receiver_csv])
    if skip_existing and all(csv_has_data_rows(path) for path in expected):
        return {
            **task,
            "out_csv": out_csv,
            "out_condition_csv": out_condition_csv,
            "out_comm_csv": out_comm_csv,
            "out_trace_csv": out_trace_csv if bool(task.get("collect_semantics")) else "",
            "out_sender_csv": out_sender_csv if bool(task.get("collect_semantics")) else "",
            "out_receiver_csv": out_receiver_csv if bool(task.get("collect_semantics")) else "",
            "out_posterior_csv": out_posterior_csv if bool(task.get("posterior_strat")) else "",
            "skipped": True,
        }

    cmd = [
        sys.executable,
        "-m",
        "src.analysis.evaluate_regime_conditional",
        "--checkpoints_glob",
        task["checkpoint"],
        "--n_eval_episodes",
        str(int(task["n_eval_episodes"])),
        "--eval_seed",
        str(int(task["eval_seed"])),
        "--msg_intervention",
        str(task["intervention"]),
        "--history_intervention",
        str(task.get("history_intervention", "none")),
        "--out_csv",
        out_csv,
        "--out_comm_csv",
        out_comm_csv,
        "--out_condition_csv",
        out_condition_csv,
    ]
    if bool(task.get("greedy", True)):
        cmd.append("--greedy")
    if bool(task.get("posterior_strat")):
        cmd.append("--posterior_strat")
    eval_sigmas = task.get("eval_sigmas")
    if eval_sigmas is not None and len(eval_sigmas) > 0:
        cmd.extend(["--eval_sigmas", *[str(float(v)) for v in eval_sigmas]])
    cmd.extend(["--observability_mode", str(task.get("observability_mode", "private"))])
    public_signal_sigma = task.get("public_signal_sigma")
    if public_signal_sigma is not None:
        cmd.extend(["--public_signal_sigma", str(float(public_signal_sigma))])
    if bool(task.get("collect_semantics")):
        cmd.extend(
            [
                "--out_trace_csv",
                out_trace_csv,
                "--out_sender_semantics_csv",
                out_sender_csv,
                "--out_receiver_semantics_csv",
                out_receiver_csv,
            ]
        )

    env = os.environ.copy()
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("MPLBACKEND", "Agg")
    with open(log_path, "w", encoding="utf-8") as log_f:
        subprocess.run(
            cmd,
            cwd=_ROOT,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return {
        **task,
        "out_csv": out_csv,
        "out_condition_csv": out_condition_csv,
        "out_comm_csv": out_comm_csv,
        "out_trace_csv": out_trace_csv if bool(task.get("collect_semantics")) else "",
        "out_sender_csv": out_sender_csv if bool(task.get("collect_semantics")) else "",
        "out_receiver_csv": out_receiver_csv if bool(task.get("collect_semantics")) else "",
        "out_posterior_csv": out_posterior_csv if bool(task.get("posterior_strat")) else "",
        "skipped": False,
    }


def _read_csv_rows(path: str) -> List[Dict]:
    if path == "" or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: str, rows: List[Dict]):
    if len(rows) == 0:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    atomic_write_rows(path, rows, fieldnames)


def _aggregate_results(results: List[Dict], out_dir: str):
    main_rows: List[Dict] = []
    comm_rows: List[Dict] = []
    condition_rows: List[Dict] = []
    posterior_rows: List[Dict] = []
    trace_rows: List[Dict] = []
    sender_rows: List[Dict] = []
    receiver_rows: List[Dict] = []

    for result in results:
        extra = {
            "checkpoint_episode": int(result["episode"]),
            "suite_kind": str(result["suite_kind"]),
            "observability_mode": str(result.get("observability_mode", "private")),
            "public_signal_sigma": (
                ""
                if result.get("public_signal_sigma") is None
                else float(result["public_signal_sigma"])
            ),
        }
        for row in _read_csv_rows(result["out_csv"]):
            main_rows.append({**row, **extra})
        for row in _read_csv_rows(result["out_comm_csv"]):
            comm_rows.append({**row, **extra})
        for row in _read_csv_rows(result["out_condition_csv"]):
            condition_rows.append({**row, **extra})
        for row in _read_csv_rows(result["out_posterior_csv"]):
            posterior_rows.append({**row, **extra})
        for row in _read_csv_rows(result["out_trace_csv"]):
            trace_rows.append({**row, **extra})
        for row in _read_csv_rows(result["out_sender_csv"]):
            sender_rows.append({**row, **extra})
        for row in _read_csv_rows(result["out_receiver_csv"]):
            receiver_rows.append({**row, **extra})

    _write_rows(os.path.join(out_dir, "checkpoint_suite_main.csv"), main_rows)
    _write_rows(os.path.join(out_dir, "checkpoint_suite_comm.csv"), comm_rows)
    _write_rows(os.path.join(out_dir, "checkpoint_suite_condition.csv"), condition_rows)
    _write_rows(os.path.join(out_dir, "checkpoint_suite_posterior_strat.csv"), posterior_rows)
    _write_rows(os.path.join(out_dir, "checkpoint_suite_trace.csv"), trace_rows)
    _write_rows(os.path.join(out_dir, "checkpoint_suite_sender_semantics.csv"), sender_rows)
    _write_rows(os.path.join(out_dir, "checkpoint_suite_receiver_semantics.csv"), receiver_rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=str, default="")
    p.add_argument("--comm_checkpoint_manifest", type=str, default="")
    p.add_argument("--baseline_checkpoint_manifest", type=str, default="")
    p.add_argument("--out_dir", type=str, default="outputs/eval/phase3/checkpoint_suite")
    p.add_argument("--comm_condition", type=str, default="cond1")
    p.add_argument("--baseline_condition", type=str, default="cond2")
    p.add_argument("--seeds", nargs="*", type=int, default=[101, 202])
    p.add_argument("--milestones", nargs="*", type=int, default=[50000, 100000, 150000, 200000])
    p.add_argument(
        "--interventions",
        nargs="*",
        type=str,
        default=[
            "none",
            "zeros",
            "marginal",
            "fixed0",
            "fixed1",
            "indep_random",
            "public_random",
            "public_marginal",
            "sender_shuffle",
            "permute_slots",
        ],
    )
    p.add_argument("--history_intervention", type=str, default="none")
    p.add_argument("--history_interventions", nargs="*", type=str, default=None)
    p.add_argument("--n_eval_episodes", type=int, default=300)
    p.add_argument("--eval_seed", type=int, default=9001)
    p.add_argument("--max_workers", type=int, default=4)
    p.add_argument("--eval_sigmas", nargs="*", type=float, default=None)
    p.add_argument(
        "--observability_mode",
        type=str,
        default="private",
        choices=["private", "public_noisy", "public_exact"],
    )
    p.add_argument("--public_signal_sigma", type=float, default=None)
    p.add_argument("--sample_policy", action="store_true")
    p.add_argument("--collect_posterior_strat", action="store_true")
    p.add_argument("--skip_semantics", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    raw_dir = os.path.join(out_dir, "raw")
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    history_interventions = (
        [str(v) for v in args.history_interventions]
        if args.history_interventions is not None and len(args.history_interventions) > 0
        else [str(args.history_intervention or "none")]
    )

    tasks = []
    for seed in args.seeds:
        for episode in args.milestones:
            comm_ckpt = _checkpoint_path(
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_manifest=args.comm_checkpoint_manifest,
                condition=args.comm_condition,
                seed=int(seed),
                episode=int(episode),
            )
            for history_intervention in history_interventions:
                history_label = str(history_intervention or "none")
                for intervention in args.interventions:
                    tasks.append(
                        {
                            "name": _task_name(
                                args.comm_condition,
                                seed,
                                episode,
                                intervention,
                                history_intervention=history_label,
                                observability_mode=str(args.observability_mode),
                                public_signal_sigma=args.public_signal_sigma,
                            ),
                            "history_intervention": history_label,
                            "suite_kind": "comm",
                            "checkpoint": comm_ckpt,
                            "episode": int(episode),
                            "intervention": str(intervention),
                            "observability_mode": str(args.observability_mode),
                            "public_signal_sigma": (
                                None
                                if args.public_signal_sigma is None
                                else float(args.public_signal_sigma)
                            ),
                            "posterior_strat": (
                                (str(intervention) == "none")
                                and bool(args.collect_posterior_strat)
                            ),
                            "collect_semantics": (
                                (str(intervention) == "none") and (not bool(args.skip_semantics))
                            ),
                            "n_eval_episodes": int(args.n_eval_episodes),
                            "eval_seed": int(args.eval_seed),
                            "greedy": not bool(args.sample_policy),
                            "eval_sigmas": (
                                [float(v) for v in args.eval_sigmas]
                                if args.eval_sigmas is not None and len(args.eval_sigmas) > 0
                                else None
                            ),
                        }
                    )
                if str(args.baseline_condition or "").strip() != "":
                    baseline_ckpt = _checkpoint_path(
                        checkpoint_dir=args.checkpoint_dir,
                        checkpoint_manifest=args.baseline_checkpoint_manifest,
                        condition=args.baseline_condition,
                        seed=int(seed),
                        episode=int(episode),
                    )
                    tasks.append(
                        {
                            "name": _task_name(
                                args.baseline_condition,
                                seed,
                                episode,
                                "none",
                                history_intervention=history_label,
                                observability_mode=str(args.observability_mode),
                                public_signal_sigma=args.public_signal_sigma,
                            ),
                            "history_intervention": history_label,
                            "suite_kind": "baseline",
                            "checkpoint": baseline_ckpt,
                            "episode": int(episode),
                            "intervention": "none",
                            "observability_mode": str(args.observability_mode),
                            "public_signal_sigma": (
                                None
                                if args.public_signal_sigma is None
                                else float(args.public_signal_sigma)
                            ),
                            "posterior_strat": bool(args.collect_posterior_strat),
                            "collect_semantics": False,
                            "n_eval_episodes": int(args.n_eval_episodes),
                            "eval_seed": int(args.eval_seed),
                            "greedy": not bool(args.sample_policy),
                            "eval_sigmas": (
                                [float(v) for v in args.eval_sigmas]
                                if args.eval_sigmas is not None and len(args.eval_sigmas) > 0
                                else None
                            ),
                        }
                    )

    results = []
    total_tasks = len(tasks)
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
        future_to_task = {
            ex.submit(_run_task, task, raw_dir, log_dir, bool(args.skip_existing)): task
            for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            result = future.result()
            results.append(result)
            print(f"[suite] done {task['name']} skipped={bool(result.get('skipped', False))}")
            print(f"[suite] progress current={len(results)} total={total_tasks}")

    results = sorted(results, key=lambda item: item["name"])
    _aggregate_results(results=results, out_dir=out_dir)
    atomic_write_json(os.path.join(out_dir, "checkpoint_suite_manifest.json"), results)
    print(f"[suite] tasks={len(results)} out_dir={out_dir}")


if __name__ == "__main__":
    main()
