"""
Uncertainty-quantification tests.

The claim under test is specific and falsifiable: conformal calibration makes
interval coverage match the nominal level, where raw quantile intervals
systematically under-cover. The suite checks both the mechanism (CQR scores,
monotone rearrangement, exceedance probabilities) and the outcome (empirical
coverage on held-out data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import uncertainty


def _toy_problem(n: int = 900, seed: int = 0):
    """Heteroscedastic regression: noise grows with x, so quantiles differ."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, size=n)
    noise = rng.normal(scale=0.5 + 0.4 * x)
    y = 2.0 * x + noise
    frame = pd.DataFrame({"x": x, "x_sq": x ** 2, "const": 1.0})
    return frame, y


@pytest.fixture(scope="module")
def fitted():
    from lightgbm import LGBMRegressor

    X, y = _toy_problem()
    split = len(X) // 2
    X_train, X_rest = X.iloc[:split], X.iloc[split:]
    y_train, y_rest = y[:split], y[split:]
    half = len(X_rest) // 2
    X_calib, X_test = X_rest.iloc[:half], X_rest.iloc[half:]
    y_calib, y_test = y_rest[:half], y_rest[half:]

    models = {
        level: LGBMRegressor(objective="quantile", alpha=level, n_estimators=120,
                             num_leaves=15, learning_rate=0.1, verbose=-1,
                             random_state=0, n_jobs=2)
        for level in uncertainty.QUANTILE_LEVELS
    }
    uncertainty.fit_quantile_models(models, X_train, y_train, verbose=False)
    calibration = uncertainty.calibrate(models, X_calib, y_calib)
    return models, calibration, X_test, y_test


def test_quantile_predictions_are_monotone(fitted):
    models, _, X_test, _ = fitted
    qmat = uncertainty.predict_quantiles(models, X_test)
    assert qmat.shape == (len(X_test), len(models))
    # Rearrangement guarantees non-decreasing quantiles across levels.
    assert (np.diff(qmat, axis=1) >= -1e-9).all()


def test_calibration_produces_a_non_negative_offset_per_level(fitted):
    _, calibration, _, _ = fitted
    assert set(calibration) == set(uncertainty.COVERAGE_LEVELS)
    for specs in calibration.values():
        assert specs["q_hat"] >= 0
        assert specs["n_calibration"] > 0
        assert specs["lower_level"] < specs["upper_level"]


def test_conformal_intervals_are_wider_than_raw_quantile_intervals(fitted):
    models, calibration, X_test, _ = fitted
    lo_c, hi_c = uncertainty.intervals(models, calibration, X_test, 0.9, use_conformal=True)
    lo_r, hi_r = uncertainty.intervals(models, calibration, X_test, 0.9, use_conformal=False)
    assert (hi_c - lo_c).mean() >= (hi_r - lo_r).mean()


def test_conformal_coverage_is_close_to_nominal(fitted):
    """The headline claim: empirical coverage ≈ nominal after calibration."""
    models, calibration, X_test, y_test = fitted
    report = uncertainty.evaluate(models, calibration, X_test, y_test)
    for row in report["intervals"].itertuples():
        assert abs(row.picp_conformal - row.coverage_nominal) <= 0.12, (
            f"{row.coverage_nominal:.0%} interval covered {row.picp_conformal:.1%}")
    # And calibration must beat the raw quantile intervals on the 90% level.
    row = report["intervals"][report["intervals"]["coverage_nominal"] == 0.9].iloc[0]
    assert row["picp_conformal"] > row["picp_raw_quantile"]


def test_pinball_loss_is_minimised_by_the_true_quantile():
    rng = np.random.default_rng(0)
    y = rng.normal(size=4000)
    good = uncertainty.pinball_loss(y, np.full_like(y, np.quantile(y, 0.9)), 0.9)
    bad = uncertainty.pinball_loss(y, np.full_like(y, np.quantile(y, 0.5)), 0.9)
    assert good < bad


def test_exceedance_probabilities_are_bounded_and_ordered(fitted):
    models, _, X_test, _ = fitted
    qmat = uncertainty.predict_quantiles(models, X_test)
    probabilities = uncertainty.exceedance_probabilities(models, qmat, [5, 15, 30])
    assert set(probabilities) == {5, 15, 30}
    for values in probabilities.values():
        assert values.shape == (len(X_test),)
        assert ((values > 0) & (values <= 1)).all()
    # A higher threshold must not be more likely than a lower one.
    assert (probabilities[30] <= probabilities[5] + 1e-9).all()


def test_reliability_figure_writes_a_file(fitted, tmp_path):
    models, _, X_test, y_test = fitted
    X, y = _toy_problem(seed=5)
    out = uncertainty.reliability_figure(models, X, y, X_test, y_test,
                                         tmp_path / "reliability.png")
    assert out is not None
    assert (tmp_path / "reliability.png").exists()
