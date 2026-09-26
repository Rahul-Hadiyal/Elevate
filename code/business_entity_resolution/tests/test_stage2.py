"""Unit Tests for Phase 10: Stage-2 Context Features and Re-Scoring Model."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.context_features import ContextFeatureExtractor, G5_FEATURE_NAMES
from src.stage2_model import Stage2Rescorer


def test_context_feature_extraction():
    """Verifies within-entity and reverse-rank context feature calculations."""
    pairs = [
        ("s1_1", "s2_100", 0.90),
        ("s1_1", "s2_101", 0.70),
        ("s1_1", "s2_102", 0.20),
        ("s1_2", "s2_100", 0.85),
        ("s1_2", "s2_103", 0.60),
    ]

    cluster_map = {"s2_100": "c_1", "s2_101": "c_1"}
    extractor = ContextFeatureExtractor(cluster_map=cluster_map)
    df = extractor.extract_context_features(pairs)

    assert len(df) == 5
    for col in G5_FEATURE_NAMES:
        assert col in df.columns

    # Verify S1_1 top candidate properties
    row_top = df[(df["s1_id"] == "s1_1") & (df["cand_id"] == "s2_100")].iloc[0]
    assert row_top["candidate_count_for_s1"] == 3
    assert row_top["score_rank_within_s1"] == 1.0
    assert row_top["score_percentile_within_s1"] == 1.0
    assert pytest.approx(row_top["score_margin_to_second"], 1e-4) == 0.20  # 0.90 - 0.70

    # Verify reverse rank on s2_100 (claimants: s1_1 @ 0.90, s1_2 @ 0.85)
    assert row_top["n_s1_claimants"] == 2
    assert row_top["reverse_rank_of_cand"] == 1.0
    assert row_top["is_mutual_best_match"] == 1.0
    assert pytest.approx(row_top["max_competing_score"], 1e-4) == 0.85
    assert pytest.approx(row_top["score_lead_over_competitor"], 1e-4) == 0.05

    # Verify cluster consistency
    assert row_top["cand_in_near_dup_cluster"] == 1.0
    assert row_top["cluster_size"] == 2


def test_stage2_rescorer_training_and_serialization():
    """Verifies training, prediction, evaluation, and persistence of Stage2Rescorer."""
    np.random.seed(42)
    n_samples = 300
    
    # Generate synthetic G5 features
    data = {col: np.random.uniform(0.0, 1.0, n_samples).astype(np.float32) for col in G5_FEATURE_NAMES}
    data["stage1_p1"] = np.random.uniform(0.0, 1.0, n_samples).astype(np.float32)
    
    # Target correlated with stage1_p1 and mutual match
    logits = 4.0 * data["stage1_p1"] + 2.0 * data["is_mutual_best_match"] - 2.5
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs > 0.5).astype(np.int32)
    
    df = pd.DataFrame(data)
    
    rescorer = Stage2Rescorer(n_estimators=30, max_depth=3)
    rescorer.fit(
        X_train=df.iloc[:200],
        y_train=y[:200],
        X_calib=df.iloc[200:],
        y_calib=y[200:],
        calibration_method="sigmoid",
    )
    
    eval_metrics = rescorer.evaluate(df.iloc[200:], y[200:])
    assert "roc_auc" in eval_metrics
    assert eval_metrics["roc_auc"] > 0.70
    assert eval_metrics["brier_score"] < 0.25
    
    importances = rescorer.get_feature_importances()
    assert len(importances) == len(G5_FEATURE_NAMES)
    assert sum(importances.values()) == pytest.approx(1.0, rel=1e-3)
    
    # Test persistence
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "stage2_model.joblib"
        rescorer.save(save_path)
        assert save_path.exists()
        
        loaded = Stage2Rescorer.load(save_path)
        p_orig = rescorer.predict_proba(df.iloc[:10])
        p_loaded = loaded.predict_proba(df.iloc[:10])
        np.testing.assert_allclose(p_orig, p_loaded, rtol=1e-5)
