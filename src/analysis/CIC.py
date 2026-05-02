"""Deprecated legacy CIC prototype.

This module previously executed hardcoded local-path loading at import time and
implemented an outdated CIC calculation path that is not used by the current
phase-3 evaluation pipeline.

Use `src.analysis.evaluate_regime_conditional` for current communication
analyses.
"""

from __future__ import annotations


def _deprecated(*_args, **_kwargs):
    raise RuntimeError(
        "src.analysis.CIC is deprecated and disabled. "
        "Use src.analysis.evaluate_regime_conditional for maintained "
        "communication analyses."
    )


calc_model_cic = _deprecated
calc_cic = _deprecated
get_p_a_given_do_c = _deprecated
