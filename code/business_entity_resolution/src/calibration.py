"""Probability Calibration Module for Business Entity Resolution.

Supports Platt scaling (Sigmoid / Logistic Regression) and Isotonic Regression
to ensure model confidence outputs reflect true empirical posterior probabilities
P(match=1 | pair), reducing false merge risk under F0.5 precision weighting.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss


@dataclass
class CalibrationMetrics:
    """Diagnostic metrics before and after probability calibration."""
    method: str
    brier_score_raw: float
    brier_score_calibrated: float
    log_loss_raw: float
    log_loss_calibrated: float
    ece_raw: float
    ece_calibrated: float
    num_samples: int


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        else:
            in_bin = (probs >= bin_lower) & (probs < bin_upper)
            
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(probs[in_bin])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)
            
    return float(ece)


class ProbabilityCalibrator:
    """Calibrator for post-hoc pairwise match probabilities."""

    def __init__(self, method: str = "sigmoid"):
        """Initialize calibrator.
        
        Args:
            method: 'sigmoid' (Platt scaling) or 'isotonic' (Isotonic regression).
        """
        self.method = method.lower()
        if self.method not in ("sigmoid", "isotonic"):
            raise ValueError(f"Unknown calibration method: {method}. Must be 'sigmoid' or 'isotonic'.")

        self.calibrator = None
        self.fitted_ = False

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        """Fit calibration curve on held-out calibration partition."""
        raw_probs_clipped = np.clip(raw_probs, 1e-7, 1.0 - 1e-7)

        if self.method == "sigmoid":
            # Platt scaling: fit logistic regression on log-odds
            log_odds = np.log(raw_probs_clipped / (1.0 - raw_probs_clipped)).reshape(-1, 1)
            self.calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            self.calibrator.fit(log_odds, y_true)
        elif self.method == "isotonic":
            # Isotonic regression: non-decreasing fit
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(raw_probs, y_true)

        self.fitted_ = True
        return self

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        """Calibrate raw probability scores."""
        if not self.fitted_:
            raise RuntimeError("Calibrator has not been fitted yet. Call fit() first.")

        raw_probs_clipped = np.clip(raw_probs, 1e-7, 1.0 - 1e-7)

        if self.method == "sigmoid":
            log_odds = np.log(raw_probs_clipped / (1.0 - raw_probs_clipped)).reshape(-1, 1)
            # Probability for class 1
            return self.calibrator.predict_proba(log_odds)[:, 1]
        elif self.method == "isotonic":
            return self.calibrator.predict(raw_probs)

    def evaluate(self, raw_probs: np.ndarray, y_true: np.ndarray) -> CalibrationMetrics:
        """Evaluate calibration effectiveness."""
        cal_probs = self.predict_proba(raw_probs)
        
        raw_clip = np.clip(raw_probs, 1e-15, 1.0 - 1e-15)
        cal_clip = np.clip(cal_probs, 1e-15, 1.0 - 1e-15)

        brier_raw = float(brier_score_loss(y_true, raw_clip))
        brier_cal = float(brier_score_loss(y_true, cal_clip))

        ll_raw = float(log_loss(y_true, raw_clip))
        ll_cal = float(log_loss(y_true, cal_clip))

        ece_raw = compute_ece(raw_clip, y_true)
        ece_cal = compute_ece(cal_clip, y_true)

        return CalibrationMetrics(
            method=self.method,
            brier_score_raw=brier_raw,
            brier_score_calibrated=brier_cal,
            log_loss_raw=ll_raw,
            log_loss_calibrated=ll_cal,
            ece_raw=ece_raw,
            ece_calibrated=ece_cal,
            num_samples=len(y_true),
        )
