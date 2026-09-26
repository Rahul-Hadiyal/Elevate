"""Ensembling and Multi-Seed Averaging Module for Business Entity Resolution (Phase 13 / Exp 10).

Provides multi-seed LightGBM ensemble wrappers for Stage-1 and Stage-2 models,
averaging predicted probabilities and fitting a dedicated post-ensemble calibrator.
"""

from pathlib import Path
import os
import joblib
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd

from .model import PairwiseScorer
from .calibration import ProbabilityCalibrator
from .stage2_model import Stage2Rescorer
from .context_features import G5_FEATURE_NAMES

logger = logging.getLogger(__name__)


class EnsemblePairwiseScorer:
    """Multi-seed ensemble of Stage-1 gradient boosted pairwise classifiers."""

    def __init__(
        self,
        model_type: str = "lightgbm",
        seeds: Optional[List[int]] = None,
        feature_names: Optional[List[str]] = None,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_weight: float = 1e-3,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = 1.0,
        n_jobs: int = -1,
    ):
        self.model_type = model_type
        self.seeds = seeds or [42, 43, 44, 45, 46]
        self.feature_names = feature_names
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.n_jobs = n_jobs

        self.models: List[PairwiseScorer] = []
        self.calibrator: Optional[ProbabilityCalibrator] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_earlystop: Optional[np.ndarray] = None,
        y_earlystop: Optional[np.ndarray] = None,
        X_calib: Optional[np.ndarray] = None,
        y_calib: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 30,
        calibration_method: str = "sigmoid",
    ) -> "EnsemblePairwiseScorer":
        """Fits all base models across random seeds and fits post-ensemble calibrator."""
        self.models = []
        logger.info(f"Fitting Ensemble of {len(self.seeds)} Stage-1 models with seeds: {self.seeds}")

        for seed in self.seeds:
            scorer = PairwiseScorer(
                model_type=self.model_type,
                feature_names=self.feature_names,
                random_state=seed,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                min_child_weight=self.min_child_weight,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                scale_pos_weight=self.scale_pos_weight,
                n_jobs=self.n_jobs,
            )
            scorer.fit(
                X_train,
                y_train,
                X_val=X_earlystop,
                y_val=y_earlystop,
                early_stopping_rounds=early_stopping_rounds,
            )
            self.models.append(scorer)

        # Post-ensemble calibration on the dedicated calibration fold
        if X_calib is not None and y_calib is not None:
            raw_calib_probs = self.predict_raw_proba(X_calib)
            self.calibrator = ProbabilityCalibrator(method=calibration_method)
            self.calibrator.fit(raw_calib_probs, y_calib)
            logger.info(f"Post-ensemble calibrator fitted using '{calibration_method}' method.")

        return self

    def predict_raw_proba(self, X: np.ndarray) -> np.ndarray:
        """Averages raw output probabilities across all ensemble members."""
        if not self.models:
            raise ValueError("Ensemble has not been fitted yet.")
        
        all_preds = np.zeros(len(X), dtype=np.float64)
        for scorer in self.models:
            all_preds += scorer.predict_proba(X)
        
        return all_preds / len(self.models)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns calibrated ensemble probabilities."""
        raw_probs = self.predict_raw_proba(X)
        if self.calibrator is not None:
            return self.calibrator.predict_proba(raw_probs)
        return raw_probs

    def save(self, file_path: Union[str, Path]) -> None:
        """Saves the ensemble model to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Ensemble saved to {path}")

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "EnsemblePairwiseScorer":
        """Loads the ensemble model from disk."""
        return joblib.load(file_path)


class EnsembleStage2Rescorer:
    """Multi-seed ensemble of Stage-2 context re-scoring models."""

    def __init__(
        self,
        seeds: Optional[List[int]] = None,
        max_depth: int = 3,
        num_leaves: int = 8,
        min_child_samples: int = 50,
        learning_rate: float = 0.05,
        n_estimators: int = 150,
    ):
        self.seeds = seeds or [42, 43, 44, 45, 46]
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators

        self.models: List[Stage2Rescorer] = []
        self.calibrator: Optional[ProbabilityCalibrator] = None
        self.feature_names: List[str] = list(G5_FEATURE_NAMES)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_earlystop: Optional[pd.DataFrame] = None,
        y_earlystop: Optional[np.ndarray] = None,
        X_calib: Optional[pd.DataFrame] = None,
        y_calib: Optional[np.ndarray] = None,
        calibration_method: str = "sigmoid",
    ) -> "EnsembleStage2Rescorer":
        """Fits all Stage-2 base models across seeds and calibrates the ensemble."""
        self.models = []
        logger.info(f"Fitting Ensemble of {len(self.seeds)} Stage-2 models with seeds: {self.seeds}")

        for seed in self.seeds:
            rescorer = Stage2Rescorer(
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                random_state=seed,
            )
            # Fit without internal calibrator (we calibrate the ensemble output)
            rescorer.fit(
                X_train=X_train,
                y_train=y_train,
                X_earlystop=X_earlystop,
                y_earlystop=y_earlystop,
                calibration_method="none",
            )
            self.models.append(rescorer)

        if X_calib is not None and y_calib is not None:
            raw_calib_probs = self.predict_raw_proba(X_calib)
            self.calibrator = ProbabilityCalibrator(method=calibration_method)
            self.calibrator.fit(raw_calib_probs, y_calib)
            logger.info(f"Post-ensemble Stage-2 calibrator fitted using '{calibration_method}' method.")

        return self

    def predict_raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Averages raw probabilities across Stage-2 models."""
        if not self.models:
            raise ValueError("Stage-2 Ensemble has not been fitted yet.")
        
        all_preds = np.zeros(len(X), dtype=np.float64)
        for rescorer in self.models:
            all_preds += rescorer.predict_raw(X)
        
        return all_preds / len(self.models)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns calibrated Stage-2 ensemble probabilities."""
        raw_probs = self.predict_raw_proba(X)
        if self.calibrator is not None:
            return self.calibrator.predict_proba(raw_probs)
        return raw_probs

    def save(self, file_path: Union[str, Path]) -> None:
        """Saves the Stage-2 ensemble model to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Stage-2 Ensemble saved to {path}")

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "EnsembleStage2Rescorer":
        """Loads the Stage-2 ensemble model from disk."""
        return joblib.load(file_path)
