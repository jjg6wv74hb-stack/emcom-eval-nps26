#!/usr/bin/env python3
"""Build the anonymized NeurIPS review artifact under dist/.

The zip contains source, docs, paper files, and the artifact subset needed by
paper/neurips2026_comm_vecstraight/main.qmd.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import zipfile
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dist" / "neurips2026_review_artifact"
DEFAULT_ARCHIVE = REPO_ROOT / "dist" / "neurips2026_review_artifact.zip"

SOURCE_PATHS = (
    "README.md",
    "requirements.txt",
    "requirements_locked.txt",
    "setup.py",
    "configs",
    "docs",
    "paper/neurips2026_comm_vecstraight",
    "scripts/reproduce",
    "scripts/release",
    "src",
    "tests",
)

EXCLUDE_REL_PATHS = {
    "docs/SUBMISSION_RUNBOOK.md",
    "scripts/release/prepare_ed_submission.py",
}

OPTIONAL_SOURCE_PATHS = (
    "LICENSE",
    "LICENSE.md",
    "THIRD_PARTY_NOTICES.md",
    "CITATION.cff",
)

EXCLUDE_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".quarto",
    "_output",
    "tmp",
}

EXCLUDE_DIR_PATTERNS = (
    "overleaf_upload_*",
    "overleaf_source_*",
)

EXCLUDE_FILE_PATTERNS = (
    ".DS_Store",
    ".Rhistory",
    ".gitignore",
    "*_COMMENT_RESPONSES.md",
    "*PRIVATE*.md",
    "*private*.md",
    "*REVIEWER*.md",
    "*_gk.pdf",
    "*.aux",
    "*.fdb_latexmk",
    "*.fls",
    "*.ipynb",
    "*.log",
    "*.pyc",
    "last_version_IG.pdf",
    "main.html",
    "main.tex",
    "main_edits_*.tex",
    "overleaf_source_*.zip",
    "overleaf_upload_*.zip",
)

ARTIFACT_TREES = (
    "artifacts/paper/role_allocation/cfg07_30k_framework_validation",
)

ARTIFACT_FILES = (
    "artifacts/paper/base_gap/report/exact_f_gap_table.csv",
    "artifacts/paper/base_gap/suite/checkpoint_suite_main.csv",
    "artifacts/paper/crossover/analysis/crossover_matrix_summary.csv",
    "artifacts/paper/endpoint/frozen50k_expanded/report/intervention_suite_summary.csv",
    "artifacts/paper/endpoint/frozen50k_expanded/suite/checkpoint_suite_main.csv",
    "artifacts/paper/endpoint/frozen50k_expanded/suite/checkpoint_suite_receiver_semantics.csv",
    "artifacts/paper/endpoint/frozen50k_expanded/suite/checkpoint_suite_sender_semantics.csv",
    "artifacts/paper/endpoint/frozen150k_expanded/report/intervention_suite_paired_stats.csv",
    "artifacts/paper/endpoint/frozen150k_expanded/report/intervention_suite_summary.csv",
    "artifacts/paper/endpoint/frozen150k_expanded/report/lowdim_mechanism/any_token_response_summary.csv",
    "artifacts/paper/endpoint/frozen150k_expanded/report/lowdim_mechanism/count_response_summary.csv",
    "artifacts/paper/endpoint/frozen150k_expanded/report/lowdim_mechanism/pattern_seed_summary.csv",
    "artifacts/paper/endpoint/frozen150k_expanded/report/lowdim_mechanism/surrogate_model_summary.csv",
    "artifacts/paper/endpoint/frozen150k_expanded/suite/checkpoint_suite_main.csv",
    "artifacts/paper/endpoint/frozen150k_expanded/suite/checkpoint_suite_receiver_semantics.csv",
    "artifacts/paper/endpoint/frozen150k_expanded/suite/checkpoint_suite_sender_semantics.csv",
    "artifacts/paper/factorial/endpoint_reduced/suite/checkpoint_suite_main.csv",
    "artifacts/paper/factorial/with_comm_reduced_history/status/progress.log",
    "artifacts/paper/factorial/without_comm_full_history/status/progress.log",
    "artifacts/paper/factorial/without_comm_reduced_history/status/progress.log",
    "artifacts/paper/message_source/status/report/channel_control_raw.csv",
    "artifacts/paper/message_source/status/report/channel_control_summary.csv",
    "artifacts/paper/message_source/train/fixed0/status/progress.log",
    "artifacts/paper/message_source/train/fixed1/status/progress.log",
    "artifacts/paper/message_source/train/learned/status/progress.log",
    "artifacts/paper/message_source/train/public_random/status/progress.log",
    "artifacts/paper/message_source/train/uniform/status/progress.log",
    "artifacts/paper/sender_causal/150k/report/sender_causal_overall_summary.csv",
    "artifacts/paper/sender_causal/150k/report/sender_causal_top_pairs.csv",
    "artifacts/paper/sender_causal/150k/sender_causal_matrix.csv",
    "artifacts/paper/sender_encoding/natural_intended_150k/report/sender_encoding_decomposition/encoding_effects.csv",
    "artifacts/paper/sender_encoding/natural_intended_150k/report/sender_encoding_decomposition/encoding_scatter.pdf",
    "artifacts/paper/sender_encoding/natural_intended_150k/report/sender_encoding_decomposition/sender_dominance_summary.csv",
    "artifacts/paper/sender_encoding/natural_intended_150k/report/sender_encoding_decomposition/sensitivity_table.csv",
    "artifacts/paper/transfer/cross_seed_flip_matched/summary/best_alignment_usage.csv",
    "artifacts/paper/transfer/cross_seed_flip_matched/summary/nonoracle_alignment_receiver_summary.csv",
    "artifacts/paper/transfer/cross_seed_flip_matched/summary/nonoracle_alignment_summary_by_f.csv",
    "artifacts/paper/transfer/cross_seed_flip_matched/summary/summary_by_f.csv",
)

ARTIFACT_GLOBS = (
    "artifacts/paper/crossover/eval/phase3_vecstraight_zeroaux_crossover_train_*_test_*_15seeds_*/report/intervention_suite_paired_stats.csv",
    "artifacts/paper/noise_sweep/private_sigma*/report/intervention_suite_summary.csv",
)

METRIC_GLOBS = (
    "artifacts/paper/message_source/train/learned/train/metrics/cond*_seed*.jsonl",
    "artifacts/paper/factorial/with_comm_reduced_history/train/metrics/cond*_seed*.jsonl",
)


def relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def should_skip_file(path: Path) -> bool:
    if relative(path) in EXCLUDE_REL_PATHS:
        return True
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in EXCLUDE_FILE_PATTERNS)


def should_skip_parts(parts: Iterable[str]) -> bool:
    return any(
        part in EXCLUDE_DIR_NAMES
        or any(fnmatch.fnmatch(part, pattern) for pattern in EXCLUDE_DIR_PATTERNS)
        for part in parts
    )


def iter_source_files(rel_root: str) -> Iterable[Path]:
    root = REPO_ROOT / rel_root
    if root.is_file():
        if not should_skip_file(root):
            yield root
        return
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if should_skip_parts(path.relative_to(root).parts):
            continue
        if should_skip_file(path):
            continue
        yield path


def copy_file(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.stat().st_size


def copy_rel_file(rel_path: str, output_dir: Path) -> int:
    src = REPO_ROOT / rel_path
    if not src.is_file():
        raise FileNotFoundError(rel_path)
    return copy_file(src, output_dir / rel_path)


def copy_source_files(output_dir: Path) -> tuple[int, int]:
    file_count = 0
    byte_count = 0
    for rel_root in SOURCE_PATHS:
        src = REPO_ROOT / rel_root
        if not src.exists():
            continue
        for path in iter_source_files(rel_root):
            byte_count += copy_file(path, output_dir / relative(path))
            file_count += 1
    for rel_path in OPTIONAL_SOURCE_PATHS:
        src = REPO_ROOT / rel_path
        if src.is_file():
            byte_count += copy_rel_file(rel_path, output_dir)
            file_count += 1
    return file_count, byte_count


def copy_artifact_files(output_dir: Path) -> tuple[int, int]:
    files = set(ARTIFACT_FILES)
    for pattern in ARTIFACT_GLOBS:
        files.update(relative(path) for path in sorted(REPO_ROOT.glob(pattern)))
    for tree in ARTIFACT_TREES:
        root = REPO_ROOT / tree
        if not root.exists():
            raise FileNotFoundError(tree)
        files.update(relative(path) for path in sorted(root.rglob("*")) if path.is_file())

    file_count = 0
    byte_count = 0
    for rel_path in sorted(files):
        byte_count += copy_rel_file(rel_path, output_dir)
        file_count += 1
    return file_count, byte_count


def metric_row_needed(row: dict[str, object]) -> bool:
    return (
        int(row.get("episode", 0)) == 150000
        and row.get("scope") == "comm"
        and row.get("window") == "window"
        and row.get("key") == "all_agents"
        and row.get("metric") == "responsiveness_kl"
    )


def copy_filtered_metrics(output_dir: Path) -> tuple[int, int, int]:
    file_count = 0
    row_count = 0
    byte_count = 0
    for pattern in METRIC_GLOBS:
        for src in sorted(REPO_ROOT.glob(pattern)):
            dst = output_dir / relative(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            rows_kept = 0
            with src.open() as in_f, dst.open("w") as out_f:
                for line in in_f:
                    row = json.loads(line)
                    if metric_row_needed(row):
                        out_f.write(json.dumps(row, sort_keys=True) + "\n")
                        rows_kept += 1
            if rows_kept == 0:
                raise ValueError(f"No matching responsiveness rows in {relative(src)}")
            file_count += 1
            row_count += rows_kept
            byte_count += dst.stat().st_size
    return file_count, row_count, byte_count


def write_manifest(
    output_dir: Path,
    source_files: int,
    source_bytes: int,
    artifact_files: int,
    artifact_bytes: int,
    metric_files: int,
    metric_rows: int,
    metric_bytes: int,
) -> None:
    manifest = {
        "description": "Anonymized NeurIPS review artifact.",
        "source_files": source_files,
        "source_bytes": source_bytes,
        "artifact_files": artifact_files,
        "artifact_bytes": artifact_bytes,
        "filtered_metric_jsonl_files": metric_files,
        "filtered_metric_jsonl_rows": metric_rows,
        "filtered_metric_jsonl_bytes": metric_bytes,
        "notes": [
            "Built under dist/.",
            "Includes the artifact files read by main.qmd.",
            "Large training checkpoints, raw traces, and logs are not included.",
            "Metric JSONL files are filtered to the rows consumed by the manuscript render.",
            "Smoke-run and full-training command templates are documented in docs/COMMANDS.md.",
        ],
    }
    (output_dir / "REVIEW_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def create_zip(output_dir: Path, archive_path: Path) -> int:
    if archive_path.exists():
        archive_path.unlink()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir.parent))
    return archive_path.stat().st_size


def build(output_dir: Path, archive_path: Path | None, dry_run: bool) -> dict[str, int | str | None]:
    if dry_run:
        return {
            "output_dir": str(output_dir),
            "archive": str(archive_path) if archive_path else None,
            "dry_run": 1,
        }

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    source_files, source_bytes = copy_source_files(output_dir)
    artifact_files, artifact_bytes = copy_artifact_files(output_dir)
    metric_files, metric_rows, metric_bytes = copy_filtered_metrics(output_dir)
    write_manifest(
        output_dir,
        source_files,
        source_bytes,
        artifact_files,
        artifact_bytes,
        metric_files,
        metric_rows,
        metric_bytes,
    )
    archive_bytes = create_zip(output_dir, archive_path) if archive_path else 0
    return {
        "output_dir": str(output_dir),
        "archive": str(archive_path) if archive_path else None,
        "source_files": source_files,
        "source_bytes": source_bytes,
        "artifact_files": artifact_files,
        "artifact_bytes": artifact_bytes,
        "filtered_metric_jsonl_files": metric_files,
        "filtered_metric_jsonl_rows": metric_rows,
        "filtered_metric_jsonl_bytes": metric_bytes,
        "archive_bytes": archive_bytes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = None if args.no_archive else args.archive
    result = build(args.output_dir, archive, args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
