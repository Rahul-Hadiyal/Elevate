"""Stage-2 Re-Scoring Model (Exp 7 / Phase 10).

Consumes Stage-1 calibrated probabilities and G.5 context/reverse-rank features
to produce refined pairwise probabilities p2(match | pair, entity context).
"""

from pathlib import Path
import os
import joblib
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

from .context_features import G5_FEATURE_NAMES
from .calibration import ProbabilityCalibrator

logger = logging.getLogger(__name__)


class Stage2Rescorer:
    """Low-capacity Stage-2 LightGBM model for context-aware match re-scoring."""

    def __init__(
        self,
        max_depth: int = 3,
        num_leaves: int = 8,
        min_child_samples: int = 50,
        learning_rate: float = 0.05,
        n_estimators: int = 150,
        random_state: int = 42,
    ):
        """Initializes the Stage-2 re-scoring model parameters."""
        self.params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        self.model: Optional[lgb.LGBMClassifier] = None
        self.calibrator: Optional[ProbabilityCalibrator] = None
        self.feature_names: List[str] = list(G5_FEATURE_NAMES)
        self.best_iteration_: Optional[int] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_earlystop: Optional[pd.DataFrame] = None,
        y_earlystop: Optional[np.ndarray] = None,
        X_calib: Optional[pd.DataFrame] = None,
        y_calib: Optional[np.ndarray] = None,
        calibration_method: str = "sigmoid",
    ) -> "Stage2Rescorer":
        """Trains the Stage-2 LightGBM model with optional early stopping and calibration.
        
        Args:
            X_train: Training features DataFrame containing G5_FEATURE_NAMES.
            y_train: Binary labels for training pairs.
            X_earlystop: Validation features for early stopping.
            y_earlystop: Binary labels for early stopping.
            X_calib: Calibration features.
            y_calib: Binary labels for calibration.
            calibration_method: 'sigmoid' (Platt) or 'isotonic'.
            
        Returns:
            self
        """
        X_tr = X_train[self.feature_names].values
        y_tr = np.asarray(y_train, dtype=np.int32)

        self.model = lgb.LGBMClassifier(**self.params)

        callbacks = []
        eval_set = None
        if X_earlystop is not None and y_earlystop is not None:
            X_es = X_earlystop[self.feature_names].values
            y_es = np.asarray(y_earlystop, dtype=np.int32)
            eval_set = [(X_es, y_es)]
            callbacks.append(lgb.early_stopping(stopping_rounds=20, verbose=False))

        logger.info(f"Training Stage-2 LightGBM on {len(X_tr):,d} pairs...")
        self.model.fit(
            X_tr,
            y_tr,
            eval_set=eval_set,
            callbacks=callbacks,
        )

        if hasattr(self.model, "best_iteration_"):
            self.best_iteration_ = self.model.best_iteration_
            logger.info(f"Stage-2 training converged at iteration {self.best_iteration_}.")

        # Fit Calibrator on dedicated calibration fold if provided
        if X_calib is not None and y_calib is not None:
            logger.info(f"Fitting Stage-2 {calibration_method} calibrator on {len(X_calib):,d} pairs...")
            raw_calib_probs = self.predict_raw(X_calib)
            self.calibrator = ProbabilityCalibrator(method=calibration_method)
            self.calibrator.fit(raw_calib_probs, y_calib)

        return self

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Computes uncalibrated probabilities from the Stage-2 LightGBM model."""
        if self.model is None:
            raise RuntimeError("Stage2Rescorer must be fitted before predict_raw.")
        X_mat = X[self.feature_names].values
        return self.model.predict_proba(X_mat)[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Computes calibrated Stage-2 probabilities p2."""
        raw_probs = self.predict_raw(X)
        if self.calibrator is not None:
            return self.calibrator.predict_proba(raw_probs)
        return raw_probs

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """Evaluates Stage-2 discrimination and calibration metrics."""
        probs = self.predict_proba(X)
        y_arr = np.asarray(y, dtype=np.int32)

        roc_auc = float(roc_auc_score(y_arr, probs))
        pr_auc = float(average_precision_score(y_arr, probs))
        brier = float(brier_score_loss(y_arr, probs))

        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier_score": brier,
        }

    def get_feature_importances(self) -> Dict[str, float]:
        """Returns normalized gain feature importances."""
        if self.model is None:
            return {}
        gains = self.model.booster_.feature_importance(importance_type="gain")
        total_gain = float(np.sum(gains)) + 1e-9
        return {name: float(gain / total_gain) for name, gain in zip(self.feature_names, gains)}

    def save(self, file_path: Path) -> None:
        """Saves the Stage-2 model and calibrator to disk."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "calibrator": self.calibrator,
                "feature_names": self.feature_names,
                "best_iteration_": self.best_iteration_,
                "params": self.params,
            },
            file_path,
        )
        logger.info(f"Saved Stage2Rescorer to {file_path}")

    @classmethod
    def load(cls, file_path: Path) -> "Stage2Rescorer":
        """Loads a saved Stage2Rescorer artifact."""
        data = joblib.load(file_path)
        rescorer = cls()
        rescorer.model = data["model"]
        rescorer.calibrator = data.get("calibrator")
        rescorer.feature_names = data.get("feature_names", list(G5_FEATURE_NAMES))
        rescorer.best_iteration_ = data.get("best_iteration_")
        rescorer.params = data.get("params", {})
        return rescorer
