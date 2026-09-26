"""Unit Tests for Multi-Seed Ensemble Module (Phase 13 / Exp 10)."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.ensemble import EnsemblePairwiseScorer, EnsembleStage2Rescorer
from src.context_features import G5_FEATURE_NAMES


@pytest.fixture
def synthetic_stage1_data():
    """Generates synthetic pair feature matrices for Stage-1 ensemble testing."""
    np.random.seed(42)
    n_train, n_es, n_calib, n_test = 200, 50, 50, 60
    n_features = 10
    feature_names = [f"feat_{i}" for i in range(n_features)]

    X_train = np.random.randn(n_train, n_features)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

    X_es = np.random.randn(n_es, n_features)
    y_es = (X_es[:, 0] + X_es[:, 1] > 0).astype(int)

    X_cal = np.random.randn(n_calib, n_features)
    y_cal = (X_cal[:, 0] + X_cal[:, 1] > 0).astype(int)

    X_test = np.random.randn(n_test, n_features)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

    return {
        "feature_names": feature_names,
        "X_train": X_train,
        "y_train": y_train,
        "X_es": X_es,
        "y_es": y_es,
        "X_cal": X_cal,
        "y_cal": y_cal,
        "X_test": X_test,
        "y_test": y_test,
    }


@pytest.fixture
def synthetic_stage2_data():
    """Generates synthetic G.5 feature dataframes for Stage-2 ensemble testing."""
    np.random.seed(42)
    n_train, n_es, n_calib, n_test = 150, 40, 40, 50

    def make_df(n):
        data = {col: np.random.rand(n) for col in G5_FEATURE_NAMES}
        return pd.DataFrame(data)

    df_train = make_df(n_train)
    y_train = (df_train["stage1_p1"] > 0.5).astype(int).values

    df_es = make_df(n_es)
    y_es = (df_es["stage1_p1"] > 0.5).astype(int).values

    df_cal = make_df(n_calib)
    y_cal = (df_cal["stage1_p1"] > 0.5).astype(int).values

    df_test = make_df(n_test)
    y_test = (df_test["stage1_p1"] > 0.5).astype(int).values

    return {
        "df_train": df_train,
        "y_train": y_train,
        "df_es": df_es,
        "y_es": y_es,
        "df_cal": df_cal,
        "y_cal": y_cal,
        "df_test": df_test,
        "y_test": y_test,
    }


def test_stage1_ensemble_fit_and_predict(synthetic_stage1_data):
    """Verifies that Stage-1 ensemble trains across seeds and yields calibrated probabilities."""
    d = synthetic_stage1_data
    seeds = [10, 20, 30]

    ensemble = EnsemblePairwiseScorer(
        seeds=seeds,
        feature_names=d["feature_names"],
        n_estimators=20,
        max_depth=3,
        num_leaves=7,
    )

    ensemble.fit(
        X_train=d["X_train"],
        y_train=d["y_train"],
        X_earlystop=d["X_es"],
        y_earlystop=d["y_es"],
        X_calib=d["X_cal"],
        y_calib=d["y_cal"],
        early_stopping_rounds=10,
    )

    assert len(ensemble.models) == 3
    assert ensemble.calibrator is not None

    raw_probs = ensemble.predict_raw_proba(d["X_test"])
    assert len(raw_probs) == len(d["X_test"])
    assert np.all(raw_probs >= 0.0) and np.all(raw_probs <= 1.0)

    calib_probs = ensemble.predict_proba(d["X_test"])
    assert len(calib_probs) == len(d["X_test"])
    assert np.all(calib_probs >= 0.0) and np.all(calib_probs <= 1.0)


def test_stage1_ensemble_serialization(synthetic_stage1_data, tmp_path):
    """Verifies saving and loading the Stage-1 ensemble."""
    d = synthetic_stage1_data
    ensemble = EnsemblePairwiseScorer(seeds=[42, 43], n_estimators=10, max_depth=2, num_leaves=4)
    ensemble.fit(d["X_train"], d["y_train"], X_calib=d["X_cal"], y_calib=d["y_cal"])

    preds_orig = ensemble.predict_proba(d["X_test"])
    save_path = tmp_path / "stage1_ensemble.joblib"
    ensemble.save(save_path)

    loaded = EnsemblePairwiseScorer.load(save_path)
    preds_loaded = loaded.predict_proba(d["X_test"])

    np.testing.assert_allclose(preds_orig, preds_loaded, rtol=1e-5)


def test_stage2_ensemble_fit_and_predict(synthetic_stage2_data):
    """Verifies that Stage-2 ensemble trains across seeds and yields calibrated probabilities."""
    d = synthetic_stage2_data
    seeds = [101, 102, 103]

    ensemble = EnsembleStage2Rescorer(
        seeds=seeds,
        max_depth=2,
        num_leaves=4,
        n_estimators=20,
    )

    ensemble.fit(
        X_train=d["df_train"],
        y_train=d["y_train"],
        X_earlystop=d["df_es"],
        y_earlystop=d["y_es"],
        X_calib=d["df_cal"],
        y_calib=d["y_cal"],
    )

    assert len(ensemble.models) == 3
    assert ensemble.calibrator is not None

    probs = ensemble.predict_proba(d["df_test"])
    assert len(probs) == len(d["df_test"])
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_stage2_ensemble_serialization(synthetic_stage2_data, tmp_path):
    """Verifies saving and loading Stage-2 ensemble."""
    d = synthetic_stage2_data
    ensemble = EnsembleStage2Rescorer(seeds=[77, 88], n_estimators=10, max_depth=2, num_leaves=4)
    ensemble.fit(d["df_train"], d["y_train"], X_calib=d["df_cal"], y_calib=d["y_cal"])

    preds_orig = ensemble.predict_proba(d["df_test"])
    save_path = tmp_path / "stage2_ensemble.joblib"
    ensemble.save(save_path)

    loaded = EnsembleStage2Rescorer.load(save_path)
    preds_loaded = loaded.predict_proba(d["df_test"])

    np.testing.assert_allclose(preds_orig, preds_loaded, rtol=1e-5)
