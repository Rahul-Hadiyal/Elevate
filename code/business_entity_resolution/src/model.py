"""Pairwise Scoring Model Module for Business Entity Resolution.

Provides model wrappers for LightGBM and XGBoost gradient boosting classifiers,
designed for CPU-efficient, high-precision pairwise matching with class imbalance
and probability estimation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import time
import logging
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss

logger = logging.getLogger(__name__)


@dataclass
class PairwiseModelMetrics:
    """Pair-level classification metrics."""
    loss: float
    brier: float
    roc_auc: float
    pr_auc: float
    num_samples: int
    pos_rate: float


class PairwiseScorer:
    """Gradient boosted tree wrapper for pairwise entity matching scoring."""

    def __init__(
        self,
        model_type: str = "lightgbm",
        feature_names: Optional[List[str]] = None,
        random_state: int = 42,
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
        self.model_type = model_type.lower()
        if self.model_type not in ("lightgbm", "xgboost"):
            raise ValueError(f"Unsupported model_type: {model_type}. Must be 'lightgbm' or 'xgboost'.")

        self.feature_names = feature_names
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.n_jobs = n_jobs

        self.model = None
        self.best_iteration_: Optional[int] = None
        self.fit_time_: float = 0.0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 30,
        verbose: bool = False,
    ) -> "PairwiseScorer":
        """Fit pairwise classification model with optional early stopping."""
        t0 = time.time()

        if self.model_type == "lightgbm":
            lgb_params = {
                "objective": "binary",
                "metric": ["binary_logloss", "auc"],
                "boosting_type": "gbdt",
                "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves,
                "max_depth": self.max_depth,
                "min_child_weight": self.min_child_weight,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "scale_pos_weight": self.scale_pos_weight,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
                "verbose": -1,
            }

            callbacks = []
            if X_val is not None and y_val is not None and early_stopping_rounds > 0:
                callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=verbose))
            if verbose:
                callbacks.append(lgb.log_evaluation(period=50))

            dtrain = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
            valid_sets = [dtrain]
            valid_names = ["train"]

            if X_val is not None and y_val is not None:
                dval = lgb.Dataset(X_val, label=y_val, feature_name=self.feature_names, reference=dtrain)
                valid_sets.append(dval)
                valid_names.append("val")

            self.model = lgb.train(
                lgb_params,
                dtrain,
                num_boost_round=self.n_estimators,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks,
            )
            self.best_iteration_ = self.model.best_iteration

        elif self.model_type == "xgboost":
            xgb_params = {
                "objective": "binary:logistic",
                "eval_metric": ["logloss", "auc"],
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "scale_pos_weight": self.scale_pos_weight,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs if self.n_jobs > 0 else 4,
                "tree_method": "hist",
            }

            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
            evals = [(dtrain, "train")]

            if X_val is not None and y_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
                evals.append((dval, "val"))

            self.model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=self.n_estimators,
                evals=evals,
                early_stopping_rounds=early_stopping_rounds if (X_val is not None and early_stopping_rounds > 0) else None,
                verbose_eval=verbose,
            )
            self.best_iteration_ = getattr(self.model, "best_iteration", self.n_estimators)

        self.fit_time_ = time.time() - t0
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability P(match=1 | pair) for input feature matrix."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        if self.model_type == "lightgbm":
            iteration = self.best_iteration_ if self.best_iteration_ and self.best_iteration_ > 0 else None
            return self.model.predict(X, num_iteration=iteration)
        elif self.model_type == "xgboost":
            dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
            iteration_range = (0, self.best_iteration_ + 1) if (self.best_iteration_ and self.best_iteration_ > 0) else (0, 0)
            return self.model.predict(dmatrix, iteration_range=iteration_range)

    def evaluate_pairs(self, X: np.ndarray, y: np.ndarray) -> PairwiseModelMetrics:
        """Compute pair-level classification metrics."""
        probs = self.predict_proba(X)
        # Numerical clipping for safe logloss
        probs_clipped = np.clip(probs, 1e-15, 1.0 - 1e-15)

        loss = float(log_loss(y, probs_clipped))
        brier = float(brier_score_loss(y, probs))
        
        # Check if single class
        if len(np.unique(y)) > 1:
            roc_auc = float(roc_auc_score(y, probs))
            pr_auc = float(average_precision_score(y, probs))
        else:
            roc_auc = 0.5
            pr_auc = float(np.mean(y))

        return PairwiseModelMetrics(
            loss=loss,
            brier=brier,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            num_samples=len(y),
            pos_rate=float(np.mean(y)),
        )

    def get_feature_importances(self, importance_type: str = "gain") -> Dict[str, float]:
        """Extract feature importances sorted in descending order."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        feature_names = self.feature_names or [f"feat_{i}" for i in range(self.model.num_feature() if hasattr(self.model, 'num_feature') else 0)]

        if self.model_type == "lightgbm":
            # importance_type: 'split' or 'gain'
            scores = self.model.feature_importance(importance_type=importance_type)
            imp_dict = {name: float(score) for name, score in zip(feature_names, scores)}
        elif self.model_type == "xgboost":
            # xgb get_score
            xgb_imp_type = "gain" if importance_type == "gain" else "weight"
            score_map = self.model.get_score(importance_type=xgb_imp_type)
            imp_dict = {name: float(score_map.get(name, 0.0)) for name in feature_names}

        return dict(sorted(imp_dict.items(), key=lambda item: item[1], reverse=True))
