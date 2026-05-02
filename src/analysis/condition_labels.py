from __future__ import annotations


_ALIASES = {
    "cond1": "comm_symm",
    "cond2": "no_comm_symm",
    "cond6": "no_comm_no_unc",
}

_DISPLAY = {
    "cond1": "Comm-On (Symmetric Uncertainty)",
    "cond2": "No-Comm (Symmetric Uncertainty)",
    "cond6": "No-Comm (No Uncertainty)",
}


def condition_alias(condition: str) -> str:
    key = str(condition or "")
    return _ALIASES.get(key, key)


def condition_display(condition: str) -> str:
    key = str(condition or "")
    return _DISPLAY.get(key, condition_alias(key))
