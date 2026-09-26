"""Unit tests for metrics.py."""

import pytest
import numpy as np
from src.metrics import (
    f05_per_entity_reference,
    evaluate_entity_detailed,
    compute_macro_f05,
    BETA,
    BETA2,
)


def test_f05_worked_example():
    """Verify exact match with the Amazon ML Challenge official worked example.
    
    Example from problem statement:
    - Pred: [S2-00047, S2-00193, S3-00812] (TP=2, FP=1)
    - True: [S2-00047, S3-00812] (TP=2, FN=0)
    - Precision = 2/3, Recall = 2/2 = 1.0
    - F0.5 = (1.25 * 0.667 * 1.0) / (0.25 * 0.667 + 1.0) = 5/7 ~= 0.7142857... ~= 0.714
    """
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    true = {"S2-00047", "S3-00812"}

    score = f05_per_entity_reference(pred, true)
    expected = 5.0 / 7.0
    assert abs(score - expected) < 1e-6, f"Expected {expected}, got {score}"

    res = evaluate_entity_detailed("S1-00001", pred, true)
    assert abs(res.f05 - expected) < 1e-6
    assert res.tp == 2
    assert res.fp == 1
    assert res.fn == 0
    assert abs(res.precision - (2.0 / 3.0)) < 1e-6
    assert res.recall == 1.0


def test_singleton_cases():
    """Verify singleton rules:
    - True singleton + empty pred -> 1.0
    - True singleton + non-empty pred (false merge) -> 0.0
    - True non-singleton + empty pred -> 0.0
    """
    # 1. Correctly predicted singleton
    assert f05_per_entity_reference(set(), set()) == 1.0
    res_correct = evaluate_entity_detailed("S1-00001", set(), set())
    assert res_correct.f05 == 1.0
    assert res_correct.is_true_singleton is True
    assert res_correct.is_pred_singleton is True

    # 2. False merge on singleton
    assert f05_per_entity_reference({"S2-00001"}, set()) == 0.0
    assert f05_per_entity_reference({"S2-00001", "S3-00002"}, set()) == 0.0
    res_false_merge = evaluate_entity_detailed("S1-00002", {"S2-00001"}, set())
    assert res_false_merge.f05 == 0.0
    assert res_false_merge.is_true_singleton is True
    assert res_false_merge.is_pred_singleton is False

    # 3. Missed non-singleton (empty prediction for true matches)
    assert f05_per_entity_reference(set(), {"S2-00001"}) == 0.0
    res_miss = evaluate_entity_detailed("S1-00003", set(), {"S2-00001"})
    assert res_miss.f05 == 0.0
    assert res_miss.is_true_singleton is False
    assert res_miss.is_pred_singleton is True


def test_single_match_variations():
    """Test 1-to-1 match variations."""
    # Exact single match
    assert f05_per_entity_reference({"S2-00001"}, {"S2-00001"}) == 1.0
    
    # Completely incorrect single match
    assert f05_per_entity_reference({"S2-00002"}, {"S2-00001"}) == 0.0


def test_multi_match_variations():
    """Test multi-match scenarios (TP, FP, FN combinations)."""
    # Multiple exact matches
    true_multi = {"S2-00001", "S2-00002", "S3-00003"}
    assert f05_per_entity_reference(true_multi, true_multi) == 1.0

    # Partial match: TP=1, FP=0, FN=1 -> prec=1.0, rec=0.5
    # num = 1.25 * 1 = 1.25, den = 1.25 * 1 + 0.25 * 1 + 0 = 1.50 -> 1.25 / 1.50 = 5/6 ~= 0.8333
    score_partial = f05_per_entity_reference({"S2-00001"}, {"S2-00001", "S2-00002"})
    assert abs(score_partial - (5.0 / 6.0)) < 1e-6

    # Precision-heavy penalty comparison:
    # 1 FP with 1 TP: TP=1, FP=1, FN=0 -> 1.25 / (1.25 + 0 + 1) = 1.25 / 2.25 = 5/9 ~= 0.5555
    # Notice: 1 FP hurts more (0.5555) than 1 FN (0.8333), demonstrating precision weighting!
    score_with_fp = f05_per_entity_reference({"S2-00001", "S2-99999"}, {"S2-00001"})
    assert abs(score_with_fp - (5.0 / 9.0)) < 1e-6
    assert score_partial > score_with_fp, "F0.5 must penalize FP more severely than FN"


def test_macro_averaging():
    """Test macro average across a multi-entity ground truth."""
    gt = {
        "S1-1": {"S2-1"},                          # 1.0
        "S1-2": set(),                             # singleton: pred empty -> 1.0
        "S1-3": set(),                             # singleton: pred false -> 0.0
        "S1-4": {"S2-4", "S3-4"},                  # pred 1 correct out of 2 -> 5/6 ~= 0.8333
    }
    preds = {
        "S1-1": {"S2-1"},                          # exact -> 1.0
        "S1-2": set(),                             # exact singleton -> 1.0
        "S1-3": {"S2-99"},                         # false merge -> 0.0
        "S1-4": {"S2-4"},                          # partial -> 5/6
    }

    summary = compute_macro_f05(preds, gt, include_detailed_results=True)
    
    expected_scores = [1.0, 1.0, 0.0, 5.0 / 6.0]
    expected_macro = sum(expected_scores) / 4.0
    
    assert abs(summary.macro_f05 - expected_macro) < 1e-6
    assert summary.total_entities == 4
    assert summary.true_singletons == 2
    assert summary.correct_singletons == 1
    assert summary.singleton_accuracy == 0.5
    assert summary.false_merge_count == 1
    assert summary.missed_non_singletons == 0
    assert summary.exact_match_entities == 2


def test_duplicate_predicted_ids():
    """Verify that duplicate IDs in predicted list are handled cleanly via set semantics."""
    pred_with_dups = ["S2-00047", "S2-00047", "S3-00812"]
    true = {"S2-00047", "S3-00812"}
    score = f05_per_entity_reference(pred_with_dups, true)
    assert score == 1.0


def test_extra_false_matches_and_completely_incorrect():
    """Verify extra false matches and completely disjoint sets."""
    # Disjoint sets: TP=0 -> 0.0
    true = {"S2-00001", "S3-00002"}
    pred_disjoint = {"S2-99991", "S3-99992"}
    assert f05_per_entity_reference(pred_disjoint, true) == 0.0

    # 1 correct, 2 extra false, 1 missed: TP=1, FP=2, FN=1 -> 1.25 / (1.25 + 0.25*1 + 2) = 1.25 / 3.5 = 5/14 ~= 0.3571
    pred_extra_false = {"S2-00001", "S2-99991", "S3-99992"}
    score_extra = f05_per_entity_reference(pred_extra_false, true)
    assert abs(score_extra - (5.0 / 14.0)) < 1e-6


def test_programmatic_randomized_equivalence():
    """Generates 1,000 random entity match pairs and tests equivalence of implementations."""
    rng = np.random.RandomState(42)
    id_pool = [f"S2-{i:05d}" for i in range(1, 20)] + [f"S3-{i:05d}" for i in range(1, 20)]

    for _ in range(500):
        # Random truth size (0 to 5)
        n_true = rng.choice([0, 0, 1, 2, 3, 4, 5])
        true_set = set(rng.choice(id_pool, size=n_true, replace=False)) if n_true > 0 else set()

        # Random pred size (0 to 5)
        n_pred = rng.choice([0, 0, 1, 2, 3, 4, 5])
        pred_set = set(rng.choice(id_pool, size=n_pred, replace=False)) if n_pred > 0 else set()

        ref_score = f05_per_entity_reference(pred_set, true_set)
        detailed_res = evaluate_entity_detailed("S1-TEST", pred_set, true_set)

        assert abs(ref_score - detailed_res.f05) < 1e-12
        assert 0.0 <= ref_score <= 1.0

