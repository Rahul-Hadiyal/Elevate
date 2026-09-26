"""Unit tests for data_loader.py."""

import pytest
import pandas as pd
from pathlib import Path
from src.data_loader import (
    validate_entity_table,
    validate_ground_truth_table,
    load_tsv_file,
    load_entity_source,
    load_ground_truth,
    parse_ground_truth_to_dict,
    extract_reverse_mapping,
    compute_h1_statistics,
    DataValidationError,
    SchemaMismatchError,
    DuplicateIdError,
    InvalidIdPrefixError,
)


def test_valid_entity_table(tmp_path: Path):
    """Test loading and validating a compliant entity table."""
    data = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-00001\tStarbucks Corp\t123 Main St, Seattle, WA\tUS\n"
        "S1-00002\tTata Consultancy\tPark Street, Kolkata\tIndia\n"
        "S1-00003\tCafé de Paris\t10 Rue de Rivoli, Paris\tFrance\n"
    )
    tsv_file = tmp_path / "train_source1.tsv"
    tsv_file.write_text(data, encoding="utf-8")

    df, report = load_entity_source(tsv_file, source_tag="S1", strict=True)
    assert report.is_valid is True
    assert len(df) == 3
    assert report.stats["total_records"] == 3
    assert report.stats["unique_ids"] == 3
    assert "US" in report.stats["country_distribution"]
    # Verify raw values preserved (commas, accented chars)
    assert df.loc[0, "business_address"] == "123 Main St, Seattle, WA"
    assert df.loc[2, "business_name"] == "Café de Paris"


def test_missing_column_error(tmp_path: Path):
    """Test schema mismatch when a required column is missing."""
    data = (
        "entity_id\tbusiness_name\tcountry\n"  # missing business_address
        "S1-00001\tStarbucks Corp\tUS\n"
    )
    tsv_file = tmp_path / "bad_schema.tsv"
    tsv_file.write_text(data, encoding="utf-8")

    with pytest.raises(SchemaMismatchError):
        load_entity_source(tsv_file, source_tag="S1", strict=True)


def test_duplicate_id_error(tmp_path: Path):
    """Test rejection when duplicate entity_id entries exist."""
    data = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-00001\tStarbucks Corp\t123 Main St\tUS\n"
        "S1-00001\tStarbucks Store 2\t456 Market St\tUS\n"
    )
    tsv_file = tmp_path / "dup_ids.tsv"
    tsv_file.write_text(data, encoding="utf-8")

    with pytest.raises(DuplicateIdError):
        load_entity_source(tsv_file, source_tag="S1", strict=True)


def test_invalid_prefix_error(tmp_path: Path):
    """Test rejection when ID prefix does not match source."""
    data = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-00001\tStarbucks Corp\t123 Main St\tUS\n"  # S2 prefix in S1 file
    )
    tsv_file = tmp_path / "bad_prefix.tsv"
    tsv_file.write_text(data, encoding="utf-8")

    with pytest.raises(InvalidIdPrefixError):
        load_entity_source(tsv_file, source_tag="S1", strict=True)


def test_valid_ground_truth(tmp_path: Path):
    """Test loading and parsing a compliant ground truth table."""
    data = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-00001\tS2-00047,S3-00812\n"
        "S1-00002\t\n"  # singleton
        "S1-00003\tS3-00004\n"
    )
    tsv_file = tmp_path / "train_ground_truth.tsv"
    tsv_file.write_text(data, encoding="utf-8")

    df, report = load_ground_truth(tsv_file, strict=True)
    assert report.is_valid is True
    assert report.stats["total_s1_entities"] == 3
    assert report.stats["singleton_count"] == 1
    assert report.stats["multi_match_count"] == 1

    gt_dict = parse_ground_truth_to_dict(df)
    assert gt_dict["S1-00001"] == {"S2-00047", "S3-00812"}
    assert gt_dict["S1-00002"] == set()
    assert gt_dict["S1-00003"] == {"S3-00004"}


def test_ground_truth_invalid_prefix(tmp_path: Path):
    """Test rejection when matched_entity_ids contains S1 reference (self-match)."""
    data = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-00001\tS1-00002\n"  # invalid matched prefix
    )
    tsv_file = tmp_path / "bad_gt.tsv"
    tsv_file.write_text(data, encoding="utf-8")

    with pytest.raises(InvalidIdPrefixError):
        load_ground_truth(tsv_file, strict=True)


def test_h1_reverse_mapping_and_statistics():
    """Test Hypothesis H1 gate preparation logic."""
    # Case 1: H1 Confirmed (No S2/S3 ID claimed by >1 S1)
    df_clean = pd.DataFrame([
        {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-10,S3-10"},
        {"source1_entity_id": "S1-2", "matched_entity_ids": "S2-20"},
        {"source1_entity_id": "S1-3", "matched_entity_ids": ""},
    ])
    rev_clean = extract_reverse_mapping(df_clean)
    assert rev_clean["S2-10"] == {"S1-1"}
    assert rev_clean["S3-10"] == {"S1-1"}
    assert rev_clean["S2-20"] == {"S1-2"}
    
    stats_clean = compute_h1_statistics(df_clean)
    assert stats_clean["is_h1_confirmed"] is True
    assert stats_clean["multiple_claimants_count"] == 0
    assert stats_clean["violation_rate"] == 0.0

    # Case 2: H1 Violated (S2-10 claimed by both S1-1 and S1-4)
    df_violated = pd.DataFrame([
        {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-10,S3-10"},
        {"source1_entity_id": "S1-4", "matched_entity_ids": "S2-10"},
    ])
    rev_violated = extract_reverse_mapping(df_violated)
    assert rev_violated["S2-10"] == {"S1-1", "S1-4"}

    stats_violated = compute_h1_statistics(df_violated)
    assert stats_violated["is_h1_confirmed"] is False
    assert stats_violated["multiple_claimants_count"] == 1
    assert "S2-10" in stats_violated["violation_examples"]
    assert stats_violated["violation_examples"]["S2-10"] == ["S1-1", "S1-4"]
