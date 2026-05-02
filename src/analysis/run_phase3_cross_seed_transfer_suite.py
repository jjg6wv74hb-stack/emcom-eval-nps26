from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from src.analysis.checkpoint_artifacts import atomic_write_json, atomic_write_rows, csv_has_data_rows


_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINT_RE = re.compile(r"(?P<condition>cond[0-9]+)_seed(?P<seed>[0-9]+)")


def _seed_from_checkpoint(path: str) -> int:
    m = _CHECKPOINT_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"could not infer seed from checkpoint path: {path}")
    return int(m.group("seed"))


def _condition_from_checkpoint(path: str) -> str:
    m = _CHECKPOINT_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"could not infer condition from checkpoint path: {path}")
    return str(m.group("condition"))


def _collect_from_suite_manifest(
    suite_manifest_json: str,
    *,
    condition: str,
    episode: int,
) -> List[str]:
    payload = json.loads(Path(suite_manifest_json).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("suite manifest must decode to a list of task objects")
    by_seed: Dict[int, str] = {}
    for task in payload:
        if not isinstance(task, dict):
            continue
        if str(task.get("suite_kind", "")) != "comm":
            continue
        if str(task.get("intervention", "")) != "none":
            continue
        if int(task.get("episode", -1)) != int(episode):
            continue
        checkpoint = os.path.abspath(str(task.get("checkpoint", "")))
        if checkpoint == "":
            continue
        if _condition_from_checkpoint(checkpoint) != str(condition):
            continue
        by_seed[_seed_from_checkpoint(checkpoint)] = checkpoint
    return [by_seed[seed] for seed in sorted(by_seed)]


def _collect_from_text_manifest(checkpoint_manifest: str) -> List[str]:
    out: List[str] = []
    seen = set()
    with open(checkpoint_manifest, "r", encoding="utf-8") as f:
        for line in f:
            path = str(line).strip()
            if path == "" or path.startswith("#"):
                continue
            norm = os.path.abspath(path)
            if norm in seen:
                continue
            seen.add(norm)
            out.append(norm)
    return sorted(out)


def _collect_checkpoints(
    *,
    suite_manifest_json: str,
    checkpoint_manifest: str,
    condition: str,
    episode: int,
) -> List[str]:
    if str(suite_manifest_json or "").strip():
        return _collect_from_suite_manifest(
            suite_manifest_json=str(suite_manifest_json),
            condition=str(condition),
            episode=int(episode),
        )
    if str(checkpoint_manifest or "").strip():
        checkpoints = _collect_from_text_manifest(str(checkpoint_manifest))
        filtered = [path for path in checkpoints if _condition_from_checkpoint(path) == str(condition)]
        return filtered
    raise ValueError("one of --suite_manifest_json or --checkpoint_manifest is required")


def _infer_sender_ids(checkpoint_path: str) -> List[str]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    cfg = payload.get("config", {})
    if not isinstance(cfg, dict):
        raise ValueError(f"checkpoint missing config dict: {checkpoint_path}")
    n_senders = int(cfg.get("n_senders", cfg.get("n_agents", 0)))
    if n_senders <= 0:
        raise ValueError(f"checkpoint does not appear communication-enabled: {checkpoint_path}")
    return [f"agent_{idx}" for idx in range(n_senders)]


def _global_flip_map_json(sender_ids: Sequence[str], enabled: bool) -> str:
    if not enabled:
        return ""
    return json.dumps({sender_id: 1 for sender_id in sender_ids}, sort_keys=True)


def _perm_label(sender_ids: Sequence[str], perm: Sequence[str]) -> str:
    idx = {sender_id: i for i, sender_id in enumerate(sender_ids)}
    return "perm_" + "".join(str(idx[sender_id]) for sender_id in perm)


def _alignment_specs(sender_ids: Sequence[str], alignment_mode: str) -> List[Dict[str, str]]:
    sender_ids = list(sender_ids)
    specs: List[Dict[str, str]] = []
    permutations: List[Sequence[str]]
    if str(alignment_mode) == "identity":
        permutations = [tuple(sender_ids)]
        flip_opts = [False]
    elif str(alignment_mode) == "flip":
        permutations = [tuple(sender_ids)]
        flip_opts = [False, True]
    elif str(alignment_mode) == "permute_flip":
        permutations = list(itertools.permutations(sender_ids))
        flip_opts = [False, True]
    else:
        raise ValueError(f"unknown alignment_mode: {alignment_mode}")

    for perm in permutations:
        remap = {src: dst for src, dst in zip(sender_ids, perm)}
        is_identity = all(src == dst for src, dst in remap.items())
        remap_json = "" if is_identity else json.dumps(remap, sort_keys=True)
        remap_label = "identity" if is_identity else _perm_label(sender_ids, perm)
        for flip_enabled in flip_opts:
            flip_label = "flipall" if flip_enabled else "noflip"
            alignment_label = f"{remap_label}__{flip_label}"
            specs.append(
                {
                    "sender_remap_json": remap_json,
                    "sender_remap_label": remap_label,
                    "sender_flip_map_json": _global_flip_map_json(sender_ids, flip_enabled),
                    "flip_label": flip_label,
                    "alignment_label": alignment_label,
                }
            )
    return specs


def _task_name(receiver_seed: int, donor_seed: int, alignment_label: str) -> str:
    return f"recv_seed{int(receiver_seed)}_from_seed{int(donor_seed)}_{alignment_label}"


def _run_task(task: Dict[str, object], raw_dir: Path, log_dir: Path, skip_existing: bool) -> Dict[str, object]:
    out_csv = raw_dir / f"{task['name']}.csv"
    out_condition_csv = raw_dir / f"{task['name']}_condition.csv"
    out_comm_csv = raw_dir / f"{task['name']}_comm.csv"
    log_path = log_dir / f"{task['name']}.log"

    expected = [out_csv, out_condition_csv, out_comm_csv]
    if skip_existing and all(csv_has_data_rows(path) for path in expected):
        return {
            **task,
            "out_csv": str(out_csv),
            "out_condition_csv": str(out_condition_csv),
            "out_comm_csv": str(out_comm_csv),
            "skipped": True,
        }

    cmd = [
        sys.executable,
        "-m",
        "src.analysis.evaluate_regime_conditional",
        "--checkpoints_glob",
        str(task["receiver_checkpoint"]),
        "--cross_play_checkpoint",
        str(task["donor_checkpoint"]),
        "--sender_flip_map_json",
        str(task["sender_flip_map_json"]),
        "--sender_remap_json",
        str(task["sender_remap_json"]),
        "--sender_remap_label",
        str(task["sender_remap_label"]),
        "--n_eval_episodes",
        str(int(task["n_eval_episodes"])),
        "--eval_seed",
        str(int(task["eval_seed"])),
        "--msg_intervention",
        "none",
        "--history_intervention",
        "none",
        "--out_csv",
        str(out_csv),
        "--out_comm_csv",
        str(out_comm_csv),
        "--out_condition_csv",
        str(out_condition_csv),
    ]
    if bool(task.get("greedy", True)):
        cmd.append("--greedy")

    env = os.environ.copy()
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("MPLBACKEND", "Agg")
    with log_path.open("w", encoding="utf-8") as log_f:
        subprocess.run(
            cmd,
            cwd=str(_ROOT),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return {
        **task,
        "out_csv": str(out_csv),
        "out_condition_csv": str(out_condition_csv),
        "out_comm_csv": str(out_comm_csv),
        "skipped": False,
    }


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(str(key))
    atomic_write_rows(path, rows, fieldnames)


def _aggregate_results(results: Sequence[Dict[str, object]], out_dir: Path) -> None:
    main_rows: List[Dict[str, object]] = []
    comm_rows: List[Dict[str, object]] = []
    condition_rows: List[Dict[str, object]] = []
    for result in results:
        extra = {
            "receiver_seed": int(result["receiver_seed"]),
            "donor_seed": int(result["donor_seed"]),
            "receiver_checkpoint": str(result["receiver_checkpoint"]),
            "donor_checkpoint": str(result["donor_checkpoint"]),
            "alignment_label": str(result["alignment_label"]),
            "flip_label": str(result["flip_label"]),
            "transfer_kind": "self"
            if int(result["receiver_seed"]) == int(result["donor_seed"])
            else "foreign",
        }
        for row in _read_csv_rows(str(result["out_csv"])):
            main_rows.append({**row, **extra})
        # Flip-aligned comm diagnostics are not receiver-facing semantics because the
        # merged comm artifact does not distinguish pre- and post-flip token meaning.
        if str(result.get("flip_label", "")) == "noflip":
            for row in _read_csv_rows(str(result["out_comm_csv"])):
                comm_rows.append({**row, **extra})
        for row in _read_csv_rows(str(result["out_condition_csv"])):
            condition_rows.append({**row, **extra})

    _write_rows(out_dir / "cross_seed_transfer_main.csv", main_rows)
    _write_rows(out_dir / "cross_seed_transfer_comm.csv", comm_rows)
    _write_rows(out_dir / "cross_seed_transfer_condition.csv", condition_rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--suite_manifest_json", type=str, default="")
    p.add_argument("--checkpoint_manifest", type=str, default="")
    p.add_argument("--condition", type=str, default="cond1")
    p.add_argument("--episode", type=int, default=150000)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument(
        "--alignment_mode",
        type=str,
        default="flip",
        choices=["identity", "flip", "permute_flip"],
    )
    p.add_argument("--include_self", action="store_true")
    p.add_argument("--n_eval_episodes", type=int, default=300)
    p.add_argument("--eval_seed", type=int, default=7001)
    p.add_argument("--max_workers", type=int, default=4)
    p.add_argument("--sample_policy", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = _collect_checkpoints(
        suite_manifest_json=str(args.suite_manifest_json or ""),
        checkpoint_manifest=str(args.checkpoint_manifest or ""),
        condition=str(args.condition),
        episode=int(args.episode),
    )
    if not checkpoints:
        raise FileNotFoundError("no checkpoints matched the requested cross-seed transfer slice")

    sender_ids = _infer_sender_ids(checkpoints[0])
    alignments = _alignment_specs(sender_ids, alignment_mode=str(args.alignment_mode))

    out_dir = Path(args.out_dir).resolve()
    raw_dir = out_dir / "raw"
    log_dir = out_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    tasks: List[Dict[str, object]] = []
    pair_counter = 0
    for receiver_checkpoint in checkpoints:
        receiver_seed = _seed_from_checkpoint(receiver_checkpoint)
        for donor_checkpoint in checkpoints:
            donor_seed = _seed_from_checkpoint(donor_checkpoint)
            if receiver_seed == donor_seed and not bool(args.include_self):
                continue
            pair_eval_seed = int(args.eval_seed) + pair_counter
            for alignment in alignments:
                tasks.append(
                    {
                        "name": _task_name(
                            receiver_seed=receiver_seed,
                            donor_seed=donor_seed,
                            alignment_label=str(alignment["alignment_label"]),
                        ),
                        "receiver_seed": int(receiver_seed),
                        "donor_seed": int(donor_seed),
                        "receiver_checkpoint": str(receiver_checkpoint),
                        "donor_checkpoint": str(donor_checkpoint),
                        "sender_remap_json": str(alignment["sender_remap_json"]),
                        "sender_remap_label": str(alignment["sender_remap_label"]),
                        "sender_flip_map_json": str(alignment["sender_flip_map_json"]),
                        "flip_label": str(alignment["flip_label"]),
                        "alignment_label": str(alignment["alignment_label"]),
                        "n_eval_episodes": int(args.n_eval_episodes),
                        "eval_seed": int(pair_eval_seed),
                        "greedy": not bool(args.sample_policy),
                    }
                )
            pair_counter += 1

    results: List[Dict[str, object]] = []
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
            print(
                f"[xseed] done {task['name']} skipped={bool(result.get('skipped', False))}"
            )
            print(f"[xseed] progress current={len(results)} total={total_tasks}")

    results = sorted(results, key=lambda item: str(item["name"]))
    _aggregate_results(results=results, out_dir=out_dir)
    atomic_write_json(out_dir / "cross_seed_transfer_manifest.json", results)
    print(f"[xseed] tasks={len(results)} out_dir={out_dir}")


if __name__ == "__main__":
    main()
