"""Unit tests for profiler.py."""

from pathlib import Path
import pandas as pd
import pytest

from src.profiler import (
    detect_script,
    compute_distribution_stats,
    profile_entity_dataframe,
    profile_ground_truth,
    build_corpus_profile,
)


def test_detect_script():
    """Test script classification for various alphabets and characters."""
    assert detect_script("Starbucks Coffee") == "LATIN"
    assert detect_script("12345") == "NUMERIC_ONLY"
    assert detect_script("!!!???") == "PUNCTUATION_ONLY"
    assert detect_script("") == "EMPTY"
    # Devanagari test: टाटा कंसल्टेंसी (Tata Consultancy)
    assert detect_script("टाटा कंसल्टेंसी") == "DEVANAGARI"
    # Mixed test
    assert detect_script("Tata टाटा") == "MIXED_LATIN_DEVANAGARI"


def test_distribution_stats():
    """Test quantile and moment calculations."""
    stats = compute_distribution_stats([10, 20, 30, 40, 50])
    assert stats["mean"] == 30.0
    assert stats["median"] == 30.0
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0


def test_profile_entity_dataframe():
    """Test profiling on an entity table."""
    df = pd.DataFrame([
        {"entity_id": "S1-1", "business_name": "Acme Corp", "business_address": "123 Main St", "country": "US"},
        {"entity_id": "S1-2", "business_name": "Beta Ltd", "business_address": "456 Oak Rd", "country": "India"},
        {"entity_id": "S1-3", "business_name": "", "business_address": "789 Pine Ave", "country": "US"},
    ])
    prof = profile_entity_dataframe(df, "S1")
    assert prof.total_rows == 3
    assert prof.unique_ids == 3
    assert prof.duplicate_id_count == 0
    assert prof.null_name_count == 1
    assert prof.null_address_count == 0
    assert prof.country_distribution["US"] == 2
    assert prof.country_distribution["India"] == 1
    assert prof.name_scripts["LATIN"] == 2
    assert prof.name_scripts["EMPTY"] == 1


def test_profile_ground_truth():
    """Test ground truth profiling with match statistics."""
    gt_df = pd.DataFrame([
        {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-10,S3-20"},
        {"source1_entity_id": "S1-2", "matched_entity_ids": "S2-11"},
        {"source1_entity_id": "S1-3", "matched_entity_ids": ""},
    ])
    gt_prof = profile_ground_truth(gt_df)
    assert gt_prof.total_s1_entities == 3
    assert gt_prof.singleton_count == 1
    assert gt_prof.singleton_rate == 1.0 / 3.0
    assert gt_prof.one_match_count == 1
    assert gt_prof.multi_match_count == 1
    assert gt_prof.both_s2_s3_count == 1
    assert gt_prof.s2_only_count == 1
    assert gt_prof.s3_only_count == 0
    assert gt_prof.distinct_s2_matched == 2
    assert gt_prof.distinct_s3_matched == 1
    assert gt_prof.h1_gate_result["is_h1_confirmed"] is True


def test_build_corpus_profile(tmp_path: Path):
    """Test end-to-end corpus profile report generation and JSON saving."""
    s1 = pd.DataFrame([{"entity_id": "S1-1", "business_name": "Apollo Hospitals", "business_address": "Road 1", "country": "India"}])
    s2 = pd.DataFrame([{"entity_id": "S2-1", "business_name": "Apollo Pharmacy", "business_address": "Road 1", "country": "India"}])
    s3 = pd.DataFrame([{"entity_id": "S3-1", "business_name": "Apollo Clinic", "business_address": "Road 2", "country": "India"}])
    gt = pd.DataFrame([{"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1,S3-1"}])

    report = build_corpus_profile(s1, s2, s3, gt, dataset_name="TestCorpus")
    assert "S1" in report.sources
    assert "S2" in report.sources
    assert "S3" in report.sources
    assert report.ground_truth is not None

    json_path = tmp_path / "corpus_profile.json"
    report.save(json_path)
    assert json_path.exists()
