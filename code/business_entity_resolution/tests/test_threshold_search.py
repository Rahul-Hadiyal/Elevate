"""Unit tests for threshold optimization."""

import pytest
from src.threshold_search import ThresholdOptimizer


def test_threshold_optimizer_evaluation():
    candidate_probs = {
        "S1-1": [("S2-1", 0.95), ("S3-1", 0.90), ("S2-99", 0.15)],
        "S1-2": [("S2-2", 0.40)],  # Lower score
        "S1-3": [],  # Singleton
    }
    ground_truth = {
        "S1-1": {"S2-1", "S3-1"},
        "S1-2": {"S2-2"},
        "S1-3": set(),  # True singleton
    }

    optimizer = ThresholdOptimizer(
        candidate_probs=candidate_probs,
        ground_truth=ground_truth,
        all_s1_ids=["S1-1", "S1-2", "S1-3"],
    )

    # Threshold 0.50
    summary_050 = optimizer.evaluate_threshold(0.50)
    assert summary_050.correct_singletons == 1
    # S1-1 is perfect (TP=2, FP=0, FN=0 -> 1.0), S1-2 is FN (0.0), S1-3 is singleton (1.0)
    # macro F0.5 = (1.0 + 0.0 + 1.0) / 3 = 2/3 ~ 0.6667
    assert abs(summary_050.macro_f05 - (2.0 / 3.0)) < 1e-4

    # Grid search
    best_t, best_sum, df = optimizer.grid_search(thresholds=[0.30, 0.50, 0.80])
    assert len(df) == 3
    # At t=0.30, S1-2 is also captured -> macro F0.5 = 1.0
    assert best_t == 0.30
    assert abs(best_sum.macro_f05 - 1.0) < 1e-4
