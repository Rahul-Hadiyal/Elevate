"""High-level pairwise feature engineering and quality audit pipeline.

Orchestrates lookup table generation, feature matrix computation, quality audits,
label assignment, and data preparation across partitions.
"""

from typing import Dict, List, Any, Tuple, Optional, Set
import logging
import numpy as np
import pandas as pd

from src.pair_features import FEATURE_NAMES, compute_single_pair_features
from src.feature_store import FeatureBatch, FeatureExtractor
from src.hard_negatives import HardNegativeAnalyzer

logger = logging.getLogger(__name__)


def build_entity_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Build fast entity attribute lookup map from a normalized DataFrame.

    Converts DataFrame rows into compact dictionary lookups for fast retrieval.
    """
    lookup: Dict[str, Dict[str, Any]] = {}
    if df is None or df.empty:
        return lookup

    eids = df["entity_id"].astype(str).str.strip().tolist()
    name_raws = df["business_name"].fillna("").astype(str).tolist() if "business_name" in df.columns else [""] * len(eids)
    name_norms = df["name_norm"].fillna("").astype(str).tolist() if "name_norm" in df.columns else [""] * len(eids)
    name_sorteds = df["name_tokens_sorted"].fillna("").astype(str).tolist() if "name_tokens_sorted" in df.columns else [""] * len(eids)
    name_degens = df["name_is_degenerate"].tolist() if "name_is_degenerate" in df.columns else [False] * len(eids)

    addr_raws = df["business_address"].fillna("").astype(str).tolist() if "business_address" in df.columns else [""] * len(eids)
    addr_norms = df["addr_norm"].fillna("").astype(str).tolist() if "addr_norm" in df.columns else [""] * len(eids)
    addr_lms = df["addr_landmark"].fillna("").astype(str).tolist() if "addr_landmark" in df.columns else [""] * len(eids)
    addr_nums = df["addr_numbers"].tolist() if "addr_numbers" in df.columns else [[]] * len(eids)

    country_norms = df["country_norm"].fillna("unknown").astype(str).tolist() if "country_norm" in df.columns else ["unknown"] * len(eids)

    for i in range(len(eids)):
        lookup[eids[i]] = {
            "entity_id": eids[i],
            "business_name": name_raws[i],
            "name_norm": name_norms[i],
            "name_tokens_sorted": name_sorteds[i],
            "name_is_degenerate": name_degens[i],
            "business_address": addr_raws[i],
            "addr_norm": addr_norms[i],
            "addr_landmark": addr_lms[i],
            "addr_numbers": addr_nums[i],
            "country_norm": country_norms[i],
        }

    return lookup


def audit_feature_quality(
    features_matrix: np.ndarray,
    feature_names: List[str] = FEATURE_NAMES,
) -> Dict[str, Any]:
    """Perform comprehensive quality audit of computed feature matrix.

    Measures:
    - Missing/NaN rates
    - Zero/near-constant features
    - Mean, std, min, max distributions
    - Top pairwise correlations
    """
    n_rows, n_cols = features_matrix.shape
    if n_rows == 0:
        return {"num_rows": 0, "num_features": n_cols}

    feature_stats: List[Dict[str, Any]] = []
    constant_features: List[str] = []
    nan_count = int(np.isnan(features_matrix).sum())

    for col_idx, feat_name in enumerate(feature_names):
        col_vals = features_matrix[:, col_idx]
        col_nan = int(np.isnan(col_vals).sum())
        col_min = float(np.nanmin(col_vals)) if col_nan < n_rows else 0.0
        col_max = float(np.nanmax(col_vals)) if col_nan < n_rows else 0.0
        col_mean = float(np.nanmean(col_vals)) if col_nan < n_rows else 0.0
        col_std = float(np.nanstd(col_vals)) if col_nan < n_rows else 0.0
        unique_cnt = len(np.unique(col_vals[~np.isnan(col_vals)]))

        if unique_cnt <= 1:
            constant_features.append(feat_name)

        feature_stats.append({
            "feature_name": feat_name,
            "nan_count": col_nan,
            "nan_pct": float(col_nan / n_rows) * 100.0,
            "min": round(col_min, 4),
            "max": round(col_max, 4),
            "mean": round(col_mean, 4),
            "std": round(col_std, 4),
            "unique_values": unique_cnt,
            "is_constant": unique_cnt <= 1,
        })

    # Correlation analysis (sample if > 50,000 for CPU performance)
    sample_size = min(n_rows, 50000)
    sample_indices = np.random.choice(n_rows, size=sample_size, replace=False) if n_rows > sample_size else np.arange(n_rows)
    sub_matrix = features_matrix[sample_indices, :]

    # Replace NaNs with 0 for correlation
    clean_sub = np.nan_to_num(sub_matrix, nan=0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_matrix = np.corrcoef(clean_sub, rowvar=False)

    high_correlations: List[Dict[str, Any]] = []
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            val = corr_matrix[i, j]
            if not np.isnan(val) and abs(val) >= 0.95:
                high_correlations.append({
                    "feature_1": feature_names[i],
                    "feature_2": feature_names[j],
                    "pearson_r": round(float(val), 4),
                })

    return {
        "num_rows": n_rows,
        "num_features": n_cols,
        "total_nan_count": nan_count,
        "constant_feature_count": len(constant_features),
        "constant_features": constant_features,
        "high_correlation_pairs_gte_0_95": high_correlations,
        "feature_stats": feature_stats,
    }
