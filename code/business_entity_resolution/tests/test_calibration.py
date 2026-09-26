"""Unit tests for probability calibration."""

import pytest
import numpy as np
from src.calibration import ProbabilityCalibrator, compute_ece


def test_sigmoid_and_isotonic_calibration():
    np.random.seed(42)
    n_samples = 300
    # Overconfident uncalibrated scores
    raw_probs = np.random.uniform(0.01, 0.99, n_samples)
    y_true = (np.random.rand(n_samples) < raw_probs * 0.8).astype(int)

    # Sigmoid calibration
    sig_cal = ProbabilityCalibrator(method="sigmoid")
    sig_cal.fit(raw_probs, y_true)
    cal_probs_sig = sig_cal.predict_proba(raw_probs)
    assert len(cal_probs_sig) == n_samples
    assert np.all(cal_probs_sig >= 0.0) and np.all(cal_probs_sig <= 1.0)

    sig_metrics = sig_cal.evaluate(raw_probs, y_true)
    assert sig_metrics.brier_score_calibrated <= sig_metrics.brier_score_raw + 0.05

    # Isotonic calibration
    iso_cal = ProbabilityCalibrator(method="isotonic")
    iso_cal.fit(raw_probs, y_true)
    cal_probs_iso = iso_cal.predict_proba(raw_probs)
    assert len(cal_probs_iso) == n_samples
    assert np.all(cal_probs_iso >= 0.0) and np.all(cal_probs_iso <= 1.0)

    # Test ECE
    ece = compute_ece(cal_probs_sig, y_true)
    assert 0.0 <= ece <= 1.0
