"""Unit tests for Official Validation Evaluator Module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.evaluator import ValidationEvaluator, FullEvaluationReport



@pytest.fixture
def synthetic_eval_data():
    """Generates ground truth and predictions with known performance characteristics."""
    gt = {
        # True Singletons
        "S1-000": set(),  # Pred empty -> Correct Singleton (1.0)
        "S1-001": set(),  # Pred non-empty -> False Merge (0.0)
        
        # 1-Match entities
        "S1-002": {"S2-002"},  # Exact match (1.0)
        "S1-003": {"S2-003"},  # Missed / Pred empty (0.0)
        "S1-004": {"S2-004"},  # Wrong match (0.0)
        
        # Multi-match entities
        "S1-005": {"S2-005", "S3-005"},  # Exact match (1.0)
        "S1-006": {"S2-006", "S3-006"},  # Partial match (1 TP, 0 FP, 1 FN)
        "S1-007": {"S2-007", "S3-007"},  # Partial with FP (1 TP, 1 FP, 1 FN)
    }

    pred = {
        "S1-000": set(),
        "S1-001": {"S2-999"},  # False merge
        "S1-002": {"S2-002"},  # Perfect
        "S1-003": set(),       # Missed
        "S1-004": {"S2-888"},  # Wrong
        "S1-005": {"S2-005", "S3-005"}, # Perfect
        "S1-006": {"S2-006"},  # Partial
        "S1-007": {"S2-007", "S3-999"}, # Partial + FP
    }

    metadata = pd.DataFrame({
        "entity_id": [f"S1-{i:03d}" for i in range(8)],
        "country": ["US", "US", "US", "US", "IN", "IN", "IN", "IN"],
    })

    return gt, pred, metadata


def test_validation_evaluator_metrics(synthetic_eval_data):
    gt, pred, metadata = synthetic_eval_data
    evaluator = ValidationEvaluator(ground_truth=gt)
    report = evaluator.evaluate(predictions=pred, s1_metadata=metadata)

    assert report.total_entities == 8
    assert report.true_singletons == 2
    assert report.correct_singletons == 1
    assert report.false_merge_count == 1
    assert report.singleton_accuracy == 0.5
    assert report.false_merge_rate == 0.5

    # Check F0.5 scores per entity:
    # S1-000: 1.0 (singleton)
    # S1-001: 0.0 (false merge)
    # S1-002: 1.0 (exact match)
    # S1-003: 0.0 (missed)
    # S1-004: 0.0 (wrong)
    # S1-005: 1.0 (exact match)
    # S1-006: 1.25*1 / (1.25*1 + 0.25*1 + 0) = 1.25 / 1.5 = 0.833333
    # S1-007: 1.25*1 / (1.25*1 + 0.25*1 + 1) = 1.25 / 2.5 = 0.50
    # Mean F0.5 = (1.0 + 0.0 + 1.0 + 0.0 + 0.0 + 1.0 + (1.25/1.5) + 0.5) / 8 = 4.333333 / 8 = 0.5416666
    expected_f05 = (1.0 + 0.0 + 1.0 + 0.0 + 0.0 + 1.0 + (1.25 / 1.5) + 0.5) / 8.0
    assert np.isclose(report.macro_f05, expected_f05, atol=1e-5)

    assert report.exact_match_entities == 3  # S1-000, S1-002, S1-005
    assert np.isclose(report.exact_match_rate, 3 / 8)


def test_subgroup_diagnostics(synthetic_eval_data):
    gt, pred, metadata = synthetic_eval_data
    evaluator = ValidationEvaluator(ground_truth=gt)
    report = evaluator.evaluate(predictions=pred, s1_metadata=metadata)

    # Subgroups should contain 'cardinality', 'match_pattern', 'country'
    assert "cardinality" in report.subgroups
    assert "match_pattern" in report.subgroups
    assert "country" in report.subgroups

    # Check country subgroup
    country_metrics = {m.group_value: m for m in report.subgroups["country"]}
    assert "US" in country_metrics
    assert "IN" in country_metrics
    assert country_metrics["US"].entity_count == 4
    assert country_metrics["IN"].entity_count == 4


def test_report_export(synthetic_eval_data):
    gt, pred, metadata = synthetic_eval_data
    evaluator = ValidationEvaluator(ground_truth=gt)
    report = evaluator.evaluate(predictions=pred, s1_metadata=metadata)

    # Test Markdown formatting
    md = report.to_markdown()
    assert "# Official F0.5 Validation Report" in md
    assert "Macro F0.5" in md
    assert "Subgroup: Country" in md

    # Test JSON and MD save
    with tempfile.TemporaryDirectory() as tmpdir:
        p_json = Path(tmpdir) / "report.json"
        p_md = Path(tmpdir) / "report.md"
        report.save(p_json)
        report.save(p_md)

        assert p_json.exists()
        assert p_md.exists()
