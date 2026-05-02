import numpy as np

from src.baselines.bayes_filter import (
    _to_flat_time_major,
    build_sticky_transition,
    evaluate_action_predictions,
    fit_empirical_cooperation_table,
    forward_filter_sequence,
    predict_cooperation_from_posterior,
)


def test_transition_matrix_rows_sum_to_one():
    P = build_sticky_transition([0.5, 1.5, 2.5, 3.5, 5.0], rho=0.05)
    assert P.shape == (5, 5)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.all(np.diag(P) > 0.9)


def test_forward_filter_with_zero_noise_collapses_to_true_regime():
    f_values = [0.5, 1.5, 2.5]
    f_hats = np.array(
        [
            [2.5, 2.5],
            [2.5, 2.5],
            [2.5, 2.5],
        ],
        dtype=np.float64,
    )
    post = forward_filter_sequence(
        f_hats=f_hats,
        f_values=f_values,
        sigmas=[0.0, 0.0],
        rho=0.05,
    )
    assert post.shape == (3, 3)
    assert np.allclose(post[:, 2], 1.0)


def test_forward_filter_with_nonzero_noise_concentrates_on_true_regime():
    rng = np.random.default_rng(7)
    f_values = [0.5, 2.5, 5.0]
    true_f = 2.5
    f_hats = rng.normal(loc=true_f, scale=0.35, size=(25, 4))
    post = forward_filter_sequence(
        f_hats=f_hats,
        f_values=f_values,
        sigmas=[0.5, 0.5, 0.5, 0.5],
        rho=0.05,
    )
    assert post.shape == (25, 3)
    assert int(np.argmax(post[-1])) == 1
    assert float(post[-1, 1]) > 0.8


def test_forward_filter_tracks_regime_switch():
    rng = np.random.default_rng(19)
    f_values = [0.5, 2.5, 5.0]
    pre = rng.normal(loc=0.5, scale=0.25, size=(8, 4))
    post_switch = rng.normal(loc=5.0, scale=0.25, size=(12, 4))
    f_hats = np.vstack([pre, post_switch])
    post = forward_filter_sequence(
        f_hats=f_hats,
        f_values=f_values,
        sigmas=[0.5, 0.5, 0.5, 0.5],
        rho=0.20,
    )
    assert int(np.argmax(post[5])) == 0
    assert int(np.argmax(post[-1])) == 2
    assert float(post[-1, 2]) > 0.8


def test_empirical_table_and_posterior_weighted_prediction():
    true_f = np.array([0.5, 0.5, 1.5, 1.5], dtype=np.float64)
    actions = np.array(
        [
            [0, 1],
            [0, 1],
            [1, 0],
            [1, 0],
        ],
        dtype=np.float64,
    )
    cbar = fit_empirical_cooperation_table(
        true_f=true_f,
        executed_actions=actions,
        f_values=[0.5, 1.5],
        laplace_alpha=0.0,
    )
    assert cbar.shape == (2, 2)
    assert np.allclose(cbar, np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64))

    post = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    pred = predict_cooperation_from_posterior(posteriors=post, cbar=cbar)
    assert pred.shape == (2, 2)
    assert np.allclose(pred, np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64))


def test_to_flat_time_major_supports_multi_session_shapes():
    true_f = np.array(
        [
            [0.5, 0.5, 1.5],
            [1.5, 1.5, 0.5],
        ],
        dtype=np.float64,
    )
    actions = np.array(
        [
            [[0, 1], [0, 1], [1, 0]],
            [[1, 0], [1, 0], [0, 1]],
        ],
        dtype=np.float64,
    )
    tf_flat, a_flat = _to_flat_time_major(true_f=true_f, executed_actions=actions)
    assert tf_flat.shape == (6,)
    assert a_flat.shape == (6, 2)


def test_fit_empirical_table_multi_session_input():
    true_f = np.array(
        [
            [0.5, 0.5, 1.5],
            [1.5, 1.5, 0.5],
        ],
        dtype=np.float64,
    )
    actions = np.array(
        [
            [[0, 1], [0, 1], [1, 0]],
            [[1, 0], [1, 0], [0, 1]],
        ],
        dtype=np.float64,
    )
    cbar = fit_empirical_cooperation_table(
        true_f=true_f,
        executed_actions=actions,
        f_values=[0.5, 1.5],
        laplace_alpha=0.0,
    )
    assert cbar.shape == (2, 2)
    assert np.allclose(cbar, np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64))


def test_eval_metrics_perfect_predictions():
    y = np.array([[0, 1], [1, 0], [1, 1]], dtype=np.float64)
    p = y.astype(np.float64)
    metrics = evaluate_action_predictions(pred_probs=p, y_true=y)
    assert metrics["accuracy@0.5"] == 1.0
    assert metrics["brier"] < 1e-6


def test_forward_filter_rejects_shape_mismatch():
    f_hats = np.zeros((4, 3), dtype=np.float64)
    try:
        forward_filter_sequence(
            f_hats=f_hats,
            f_values=[0.5, 1.5],
            sigmas=[0.5, 0.5],
            rho=0.05,
        )
    except ValueError as exc:
        assert "len(sigmas)" in str(exc)
    else:
        raise AssertionError("expected ValueError for mismatched sigmas length")
