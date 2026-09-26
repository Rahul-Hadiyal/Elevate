"""Unit tests for Post-Processing and Constraint Enforcement."""

import pytest
from src.post_processor import PostProcessor, CandidatePrediction


def test_post_processor_1_to_1_constraint():
    pairs = [
        CandidatePrediction(s1_id="S1-1", cand_id="S2-1", prob=0.90, cand_source="S2"),
        CandidatePrediction(s1_id="S1-1", cand_id="S2-2", prob=0.75, cand_source="S2"),
        CandidatePrediction(s1_id="S1-1", cand_id="S3-1", prob=0.85, cand_source="S3"),
    ]

    pp = PostProcessor(base_threshold=0.60, max_cands_per_source=1)
    preds = pp.filter_and_assign(pairs, all_s1_ids=["S1-1"])

    # Should select top S2 (S2-1) and top S3 (S3-1), discarding S2-2
    assert preds["S1-1"] == {"S2-1", "S3-1"}


def test_post_processor_missing_address_guard():
    pairs = [
        CandidatePrediction(s1_id="S1-1", cand_id="S2-1", prob=0.70, cand_source="S2", has_empty_addr=True),
        CandidatePrediction(s1_id="S1-2", cand_id="S2-2", prob=0.90, cand_source="S2", has_empty_addr=True),
    ]

    pp = PostProcessor(base_threshold=0.60, empty_addr_threshold=0.85)
    preds = pp.filter_and_assign(pairs, all_s1_ids=["S1-1", "S1-2"])

    # S1-1 has prob 0.70 < 0.85 -> rejected
    # S1-2 has prob 0.90 >= 0.85 -> accepted
    assert preds["S1-1"] == set()
    assert preds["S1-2"] == {"S2-2"}


def test_post_processor_vetoes():
    pairs = [
        CandidatePrediction(s1_id="S1-1", cand_id="S2-1", prob=0.95, cand_source="S2", has_numeric_disagreement=True),
        CandidatePrediction(s1_id="S1-2", cand_id="S2-2", prob=0.95, cand_source="S2", has_country_disagreement=True),
    ]

    pp = PostProcessor(base_threshold=0.60, enforce_numeric_veto=True, enforce_country_veto=True)
    preds = pp.filter_and_assign(pairs, all_s1_ids=["S1-1", "S1-2"])

    assert preds["S1-1"] == set()
    assert preds["S1-2"] == set()
