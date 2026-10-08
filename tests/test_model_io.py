"""
Model-persistence tests.

The published model must survive a dependency upgrade. That is only true if it is
stored in each library's own format rather than as a pickle — and only if the
rebuilt artefact produces *identical* predictions. Both properties are asserted
here, together with the mathematical equivalence of the fused ensemble to the
scikit-learn ``VotingRegressor`` it replaces.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_io import (FusedEnsemble, load_ensemble, load_model, save_ensemble,
                      save_model)


@pytest.fixture(scope="module")
def toy_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(240, 5)), columns=[f"f{i}" for i in range(5)])
    y = 1.5 * X["f0"] - 0.7 * X["f1"] + rng.normal(scale=0.2, size=240)
    return X, y


def test_xgboost_is_stored_in_its_native_format(toy_data, tmp_path):
    xgb = pytest.importorskip("xgboost")
    X, y = toy_data
    model = xgb.XGBRegressor(n_estimators=40, max_depth=3, n_jobs=2,
                             verbosity=0, random_state=0).fit(X, y)
    path = save_model(model, tmp_path / "xgb")
    assert path.suffix == ".json", "XGBoost must not fall back to a pickle"
    reloaded = load_model(path)
    assert np.allclose(reloaded.predict(xgb.DMatrix(X)), model.predict(X), atol=1e-5)


def test_lightgbm_is_stored_in_its_native_format(toy_data, tmp_path):
    lgb = pytest.importorskip("lightgbm")
    X, y = toy_data
    model = lgb.LGBMRegressor(n_estimators=40, num_leaves=15, n_jobs=2,
                              verbose=-1, random_state=0).fit(X, y)
    path = save_model(model, tmp_path / "lgb")
    assert path.suffix == ".txt"
    reloaded = load_model(path)
    assert np.allclose(reloaded.predict(X), model.predict(X), atol=1e-6)


def test_catboost_is_stored_in_its_native_format(toy_data, tmp_path):
    catboost = pytest.importorskip("catboost")
    X, y = toy_data
    model = catboost.CatBoostRegressor(iterations=40, depth=3, verbose=0,
                                       random_seed=0, allow_writing_files=False)
    model.fit(X, y)
    path = save_model(model, tmp_path / "cat")
    assert path.suffix == ".cbm"
    reloaded = load_model(path)
    assert np.allclose(reloaded.predict(X), model.predict(X), atol=1e-6)


def test_sklearn_models_fall_back_to_joblib(toy_data, tmp_path):
    from sklearn.ensemble import ExtraTreesRegressor

    X, y = toy_data
    model = ExtraTreesRegressor(n_estimators=20, random_state=0, n_jobs=2).fit(X, y)
    path = save_model(model, tmp_path / "etr")
    assert path.suffix == ".joblib"
    assert np.allclose(load_model(path).predict(X), model.predict(X), atol=1e-9)


def test_fused_ensemble_matches_voting_regressor(toy_data):
    """The portable wrapper must be mathematically identical to the original."""
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor
    from xgboost import XGBRegressor

    from train import _voting_components

    X, y = toy_data
    voting = VotingRegressor([
        ("xgb", XGBRegressor(n_estimators=40, max_depth=3, n_jobs=2, verbosity=0,
                             random_state=1)),
        ("lgb", LGBMRegressor(n_estimators=40, num_leaves=15, n_jobs=2, verbose=-1,
                              random_state=1)),
        ("etr", ExtraTreesRegressor(n_estimators=20, n_jobs=2, random_state=1)),
    ]).fit(X, y)

    fused = FusedEnsemble(_voting_components(voting))
    assert fused.component_names == ["xgb", "lgb", "etr"]
    assert np.allclose(fused.predict(X), voting.predict(X), atol=1e-9)


def test_ensemble_survives_a_save_load_round_trip(toy_data, tmp_path):
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor
    from xgboost import XGBRegressor

    from train import _voting_components

    X, y = toy_data
    voting = VotingRegressor([
        ("xgb", XGBRegressor(n_estimators=30, max_depth=3, n_jobs=2, verbosity=0,
                             random_state=3)),
        ("lgb", LGBMRegressor(n_estimators=30, num_leaves=15, n_jobs=2, verbose=-1,
                              random_state=3)),
        ("etr", ExtraTreesRegressor(n_estimators=15, n_jobs=2, random_state=3)),
    ]).fit(X, y)

    manifest = save_ensemble(_voting_components(voting), tmp_path / "ensemble")
    assert [c["kind"] for c in manifest["components"]] == ["xgboost", "lightgbm", "sklearn"]

    rebuilt = load_ensemble(manifest, tmp_path / "ensemble")
    assert rebuilt is not None
    # Reloading from disk must reproduce the original predictions exactly.
    assert np.allclose(rebuilt.predict(X), voting.predict(X), atol=1e-5)


def test_load_ensemble_reports_missing_dependency_instead_of_crashing(toy_data, tmp_path):
    manifest = {"kind": "fused_mean",
                "components": [{"name": "gone", "file": "nope.json", "kind": "xgboost"}]}
    assert load_ensemble(manifest, tmp_path) is None


# ---------------------------------------------------------------------------
# Regression: dotted model names must not collide
# ---------------------------------------------------------------------------
def test_dotted_stems_do_not_overwrite_each_other(toy_data, tmp_path):
    """``Path.with_suffix`` ate the decimals, so every quantile level was served
    from one file. Distinct stems must produce distinct files."""
    from lightgbm import LGBMRegressor

    X, y = toy_data
    written = []
    for level in (0.025, 0.5, 0.975):
        model = LGBMRegressor(objective="quantile", alpha=level, n_estimators=20,
                              num_leaves=7, verbose=-1, random_state=0,
                              n_jobs=2).fit(X, y)
        written.append(save_model(model, tmp_path / f"q{level}"))
    assert len({p.name for p in written}) == 3, [p.name for p in written]
    for path in written:
        assert path.exists()
    # Three genuinely different levels must reload as three different models —
    # the failure mode was every level reloading whichever file was written last.
    predictions = [load_model(p).predict(X) for p in written]
    assert not np.allclose(predictions[0], predictions[1])
    assert not np.allclose(predictions[1], predictions[2])
    # And the ordering must survive the round trip: a higher quantile predicts higher.
    assert predictions[0].mean() < predictions[1].mean() < predictions[2].mean()


def test_quantile_levels_survive_a_save_load_round_trip(toy_data, tmp_path):
    """Every level must reload as itself, not as whichever file was written last."""
    from lightgbm import LGBMRegressor

    X, y = toy_data
    levels = [0.05, 0.5, 0.95]
    fitted = {}
    for level in levels:
        model = LGBMRegressor(objective="quantile", alpha=level, n_estimators=30,
                              num_leaves=15, verbose=-1, random_state=0, n_jobs=2)
        fitted[level] = model.fit(X, y)
        save_model(model, tmp_path / f"q{int(level * 1000):04d}")

    reloaded = {level: load_model(tmp_path / f"q{int(level * 1000):04d}.txt")
                for level in levels}
    means = [float(reloaded[level].predict(X.to_numpy()).mean()) for level in levels]
    # A quantile model's mean prediction must grow with tau.
    assert means[0] < means[1] < means[2], means
    for level in levels:
        assert np.allclose(reloaded[level].predict(X.to_numpy()),
                           fitted[level].predict(X), atol=1e-6)
