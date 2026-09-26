"""Unit tests for pairwise model training and inference."""

import pytest
import numpy as np
from src.model import PairwiseScorer, PairwiseModelMetrics
from src.pair_features import FEATURE_NAMES


def test_lightgbm_fit_and_predict():
    np.random.seed(42)
    n_samples = 200
    n_features = len(FEATURE_NAMES)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Simple linear decision boundary
    y = ((X[:, 0] + X[:, 1] * 2.0 + np.random.randn(n_samples) * 0.1) > 0).astype(np.int32)

    X_train, y_train = X[:150], y[:150]
    X_val, y_val = X[150:], y[150:]

    scorer = PairwiseScorer(
        model_type="lightgbm",
        feature_names=FEATURE_NAMES,
        n_estimators=50,
        random_state=42,
    )
    scorer.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=10)

    probs = scorer.predict_proba(X_val)
    assert len(probs) == 50
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    metrics = scorer.evaluate_pairs(X_val, y_val)
    assert metrics.roc_auc >= 0.70
    assert metrics.pr_auc >= 0.50

    importances = scorer.get_feature_importances(importance_type="gain")
    assert len(importances) == n_features
    assert importances[FEATURE_NAMES[0]] >= 0.0


def test_xgboost_fit_and_predict():
    np.random.seed(42)
    n_samples = 200
    n_features = len(FEATURE_NAMES)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = ((X[:, 0] + X[:, 1] * 2.0 + np.random.randn(n_samples) * 0.1) > 0).astype(np.int32)

    X_train, y_train = X[:150], y[:150]
    X_val, y_val = X[150:], y[150:]

    scorer = PairwiseScorer(
        model_type="xgboost",
        feature_names=FEATURE_NAMES,
        n_estimators=50,
        random_state=42,
    )
    scorer.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=10)

    probs = scorer.predict_proba(X_val)
    assert len(probs) == 50
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    metrics = scorer.evaluate_pairs(X_val, y_val)
    assert metrics.roc_auc >= 0.70
