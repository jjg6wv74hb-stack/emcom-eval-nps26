import numpy as np

from src.experiments_pgg_v0.train_ppo import _entropy_from_counts_1d


def test_entropy_uniform_binary():
    h = _entropy_from_counts_1d(np.array([50.0, 50.0], dtype=np.float64))
    assert abs(h - 1.0) < 1e-6


def test_entropy_deterministic_binary():
    h = _entropy_from_counts_1d(np.array([100.0, 0.0], dtype=np.float64))
    assert abs(h - 0.0) < 1e-6


def test_entropy_bounds():
    counts = np.array([3.0, 7.0], dtype=np.float64)
    h = _entropy_from_counts_1d(counts)
    assert 0.0 <= h <= 1.0
