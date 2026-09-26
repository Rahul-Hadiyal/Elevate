"""Unit tests for h1_gate.py."""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.h1_gate import evaluate_h1_gate, H1GateResult


def test_h1_gate_clean_confirmation(tmp_path: Path):
    """Test H1 gate confirmation when all matched IDs are uniquely claimed."""
    gt_df = pd.DataFrame([
        {"source1_entity_id": "S1-00001", "matched_entity_ids": "S2-00047,S3-00812"},
        {"source1_entity_id": "S1-00002", "matched_entity_ids": "S2-00048"},
        {"source1_entity_id": "S1-00003", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-00004", "matched_entity_ids": "S3-00813"},
    ])

    json_path = tmp_path / "h1_gate.json"
    result = evaluate_h1_gate(gt_df, output_json_path=json_path)

    assert result.is_h1_confirmed is True
    assert result.conflict_resolution_enabled is True
    assert result.invariant_8_enforced is True
    assert result.total_distinct_matched_ids == 4
    assert result.total_single_claimant_ids == 4
    assert result.total_multi_claimant_ids == 0
    assert result.overall_violation_rate == 0.0

    # Source-specific stats
    assert result.s2_stats.total_distinct_matched_ids == 2
    assert result.s2_stats.single_claimant_count == 2
    assert result.s2_stats.multiple_claimants_count == 0
    assert result.s2_stats.violation_rate == 0.0

    assert result.s3_stats.total_distinct_matched_ids == 2
    assert result.s3_stats.single_claimant_count == 2
    assert result.s3_stats.multiple_claimants_count == 0
    assert result.s3_stats.violation_rate == 0.0

    # Check JSON serialization
    assert json_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["is_h1_confirmed"] is True


def test_h1_gate_violation_s2_and_s3(tmp_path: Path):
    """Test H1 gate rejection when S2 and S3 IDs have multiple claimants."""
    gt_df = pd.DataFrame([
        {"source1_entity_id": "S1-00001", "matched_entity_ids": "S2-SHARED,S3-SHARED"},
        {"source1_entity_id": "S1-00002", "matched_entity_ids": "S2-SHARED"},
        {"source1_entity_id": "S1-00003", "matched_entity_ids": "S3-SHARED,S3-UNIQUE"},
        {"source1_entity_id": "S1-00004", "matched_entity_ids": "S3-SHARED"},
    ])

    json_path = tmp_path / "h1_gate_violated.json"
    result = evaluate_h1_gate(gt_df, output_json_path=json_path)

    assert result.is_h1_confirmed is False
    assert result.conflict_resolution_enabled is False
    assert result.invariant_8_enforced is False

    # 3 distinct matched IDs: S2-SHARED, S3-SHARED, S3-UNIQUE
    assert result.total_distinct_matched_ids == 3
    assert result.total_multi_claimant_ids == 2  # S2-SHARED and S3-SHARED
    assert result.total_single_claimant_ids == 1  # S3-UNIQUE

    # Check S2 stats
    assert result.s2_stats.total_distinct_matched_ids == 1
    assert result.s2_stats.multiple_claimants_count == 1
    assert result.s2_stats.violation_rate == 1.0
    assert result.s2_stats.violating_examples["S2-SHARED"] == ["S1-00001", "S1-00002"]

    # Check S3 stats
    assert result.s3_stats.total_distinct_matched_ids == 2
    assert result.s3_stats.single_claimant_count == 1
    assert result.s3_stats.multiple_claimants_count == 1
    assert result.s3_stats.violation_rate == 0.5
    assert result.s3_stats.violating_examples["S3-SHARED"] == ["S1-00001", "S1-00003", "S1-00004"]
