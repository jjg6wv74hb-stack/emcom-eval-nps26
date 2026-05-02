import numpy as np

from src.experiments_pgg_v0.train_ppo import _mi_null_independence_stats


def test_mi_significance_detects_dependence():
    counts = np.array([[90, 10], [10, 90]], dtype=np.float64)
    out = _mi_null_independence_stats(
        counts=counts,
        n_perms=400,
        alpha=0.05,
        rng=np.random.default_rng(0),
    )
    assert out["mi_observed"] > 0.0
    assert out["mi_p_value"] < 0.05
    assert out["mi_significant"] is True


def test_mi_significance_rejects_independence_table():
    counts = np.array([[50, 50], [50, 50]], dtype=np.float64)
    out = _mi_null_independence_stats(
        counts=counts,
        n_perms=400,
        alpha=0.05,
        rng=np.random.default_rng(1),
    )
    assert out["mi_observed"] < 1e-6
    assert out["mi_significant"] is False
