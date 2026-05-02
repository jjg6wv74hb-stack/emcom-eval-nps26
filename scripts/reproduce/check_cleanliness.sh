#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

python3 - <<'PYSCAN'
from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path

SOURCE_ROOTS = [
    'README.md',
    'LICENSE',
    'THIRD_PARTY_NOTICES.md',
    'docs',
    'src',
    'tests',
    'configs',
    'paper/neurips2026_comm_vecstraight/main.qmd',
    'paper/neurips2026_comm_vecstraight/_quarto.yml',
    'scripts',
]

SKIP_DIRS = {
    '.git',
    'artifacts',
    'dist',
    '__pycache__',
    '.pytest_cache',
    '.mypy_cache',
    '.ruff_cache',
    '.quarto',
    '_output',
    '.venv',
    'venv',
}

TEXT_SUFFIXES = {
    '.bib',
    '.cfg',
    '.ini',
    '.json',
    '.md',
    '.py',
    '.qmd',
    '.sh',
    '.tex',
    '.toml',
    '.txt',
    '.yaml',
    '.yml',
}
TEXT_NAMES = {'LICENSE', 'Makefile'}

STALE_PATTERNS = [
    'LEG' + 'ACY_',
    'LOCAL_' + 'EVAL_ROOT',
    'LOCAL_' + 'HETZNER',
    'LOCAL_' + 'IWR',
    'first_' + 'existing_path',
    'hetzner' + '-results',
    'iwr' + '-results',
]

PACKAGE_FORBIDDEN_NAMES = {
    '.DS_Store',
    '.Rhistory',
    'AGENTS.md',
    'LOCAL_AGENT_NOTES.md',
    'checkpoint_suite_trace.csv',
}
PACKAGE_FORBIDDEN_GLOBS = [
    '*.pyc',
    '*.pyo',
    '*.ipynb',
    '*.pt',
    '*.pth',
    '*.ckpt',
    '*COMMENT_RESPONSES*',
    '*REVIEWER*',
]
PACKAGE_FORBIDDEN_DIRS = {
    '__pycache__',
    '.pytest_cache',
    '.quarto',
    '_output',
    'raw',
    'logs',
}


def run_git_ls(args: list[str]) -> list[Path]:
    result = subprocess.run(
        ['git', 'ls-files', *args, '--', *SOURCE_ROOTS],
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line]


def has_git_checkout() -> bool:
    return Path('.git').exists()


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if parts & SKIP_DIRS:
        return True
    return any(part.startswith(('overleaf_upload_', 'overleaf_source_')) for part in path.parts)


def iter_source_files() -> list[Path]:
    if has_git_checkout():
        files = set(run_git_ls([]))
        files.update(run_git_ls(['--others', '--exclude-standard']))
        return sorted(path for path in files if path.is_file() and not should_skip(path))

    files: list[Path] = []
    for root_name in SOURCE_ROOTS:
        root = Path(root_name)
        if not root.exists():
            continue
        if root.is_file():
            candidates = [root]
        else:
            candidates = [path for path in root.rglob('*') if path.is_file()]
        files.extend(path for path in candidates if not should_skip(path))
    return sorted(set(files))


def is_text_file(path: Path) -> bool:
    return path.name in TEXT_NAMES or path.suffix in TEXT_SUFFIXES


def scan_text_files() -> list[str]:
    issues: list[str] = []
    for path in iter_source_files():
        if not is_text_file(path):
            continue
        text = path.read_text(errors='ignore')
        for pattern in STALE_PATTERNS:
            if pattern in text:
                issues.append(f'{path}: contains stale pattern {pattern}')
    return issues


def package_targets() -> list[Path]:
    if has_git_checkout():
        target = Path('dist/neurips2026_review_artifact')
        return [target] if target.exists() else []
    return [Path('.')]


def scan_package_files() -> list[str]:
    issues: list[str] = []
    for target in package_targets():
        for path in target.rglob('*'):
            rel = path.relative_to(target)
            if any(part in PACKAGE_FORBIDDEN_DIRS for part in rel.parts):
                issues.append(f'{path}: generated or raw directory is present')
                continue
            if not path.is_file():
                continue
            if path.name in PACKAGE_FORBIDDEN_NAMES:
                issues.append(f'{path}: forbidden file is present')
                continue
            if any(fnmatch.fnmatch(path.name, pattern) for pattern in PACKAGE_FORBIDDEN_GLOBS):
                issues.append(f'{path}: forbidden file type is present')
                continue
    return issues


issues = scan_text_files() + scan_package_files()
if issues:
    for issue in issues:
        print(issue)
    raise SystemExit('Cleanliness scan found files that should not be in the release.')

print('Cleanliness scan passed.')
PYSCAN
