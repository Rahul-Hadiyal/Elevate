"""Unit tests for Leakage-Safe Validation Splitting Module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.split import (
    SplitConfig,
    SplitManifest,
    create_stratified_split,
    get_cardinality_bucket,
    get_match_pattern,
    DEFAULT_SPLIT_RATIOS,
)



@pytest.fixture
def synthetic_s1_and_gt():
    """Generates a representative synthetic dataset for split testing."""
    n_entities = 1000
    eids = [f"S1-{i:06d}" for i in range(n_entities)]
    
    # 60% US, 40% India
    countries = ["US" if i % 10 < 6 else "INDIA" for i in range(n_entities)]
    
    s1_df = pd.DataFrame({
        "entity_id": eids,
        "business_name": [f"Company {i}" for i in range(n_entities)],
        "business_address": [f"Address {i}" for i in range(n_entities)],
        "country": countries,
    })

    # Ground truth with various patterns:
    # 5% singletons (0 matches)
    # 10% 1 match (S2-only or S3-only)
    # 85% multi match (both S2 and S3)
    gt: dict[str, list[str]] = {}
    for i, eid in enumerate(eids):
        if i % 20 == 0:
            # Singleton (0 matches)
            gt[eid] = []
        elif i % 10 == 1:
            # S2 only (1 match)
            gt[eid] = [f"S2-{i:06d}"]
        elif i % 10 == 2:
            # S3 only (1 match)
            gt[eid] = [f"S3-{i:06d}"]
        elif i % 10 == 3:
            # Multi match (6+ matches)
            gt[eid] = [f"S2-{i}_{j}" for j in range(4)] + [f"S3-{i}_{j}" for j in range(4)]
        else:
            # Standard multi match (2 matches)
            gt[eid] = [f"S2-{i:06d}", f"S3-{i:06d}"]

    return s1_df, gt


def test_cardinality_bucket_helper():
    assert get_cardinality_bucket(0) == "0_singleton"
    assert get_cardinality_bucket(1) == "1_single"
    assert get_cardinality_bucket(2) == "2_to_5_multi"
    assert get_cardinality_bucket(5) == "2_to_5_multi"
    assert get_cardinality_bucket(6) == "6_plus_multi"
    assert get_cardinality_bucket(50) == "6_plus_multi"


def test_match_pattern_helper():
    assert get_match_pattern([]) == "none"
    assert get_match_pattern(["S2-001"]) == "s2_only"
    assert get_match_pattern(["S3-001", "S3-002"]) == "s3_only"
    assert get_match_pattern(["S2-001", "S3-001"]) == "both_s2_s3"


def test_invalid_split_config():
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        SplitConfig(ratios={"train": 0.5, "val": 0.3})

    with pytest.raises(ValueError, match="cannot be negative"):
        SplitConfig(ratios={"train": 1.2, "val": -0.2})


def test_split_mutual_exclusivity_and_completeness(synthetic_s1_and_gt):
    s1_df, gt = synthetic_s1_and_gt
    manifest = create_stratified_split(s1_df, gt)

    assert manifest.total_entities == 1000

    partition_sets = {
        p: manifest.get_entity_set(p)
        for p in DEFAULT_SPLIT_RATIOS.keys()
    }

    # Verify mutual exclusivity
    partitions = list(partition_sets.keys())
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            p1, p2 = partitions[i], partitions[j]
            intersection = partition_sets[p1] & partition_sets[p2]
            assert len(intersection) == 0, f"Leakage detected between {p1} and {p2}: {intersection}"

    # Verify complete union
    all_eids = set()
    for p_set in partition_sets.values():
        all_eids.update(p_set)
    assert all_eids == set(s1_df["entity_id"])


def test_split_determinism(synthetic_s1_and_gt):
    s1_df, gt = synthetic_s1_and_gt
    manifest1 = create_stratified_split(s1_df, gt, SplitConfig(random_seed=42))
    manifest2 = create_stratified_split(s1_df, gt, SplitConfig(random_seed=42))

    assert manifest1.entity_to_partition == manifest2.entity_to_partition


def test_stratification_proportions(synthetic_s1_and_gt):
    s1_df, gt = synthetic_s1_and_gt
    manifest = create_stratified_split(s1_df, gt, SplitConfig(random_seed=42))

    # Check partition sizes are within +/- 1% of expected target ratios
    for part, target_ratio in DEFAULT_SPLIT_RATIOS.items():
        actual_count = manifest.partition_counts[part]
        expected_count = int(1000 * target_ratio)
        assert abs(actual_count - expected_count) <= 5, f"Size divergence in {part}: {actual_count} vs {expected_count}"

    # Check country distribution in each partition is balanced (~60% US, ~40% IN)
    for part, stats in manifest.partition_stats.items():
        cdist = stats["country_distribution"]
        total_p = stats["entity_count"]
        us_ratio = cdist.get("US", 0) / total_p
        in_ratio = cdist.get("IN", 0) / total_p
        assert 0.55 <= us_ratio <= 0.65, f"US ratio skewed in {part}: {us_ratio:.3f}"
        assert 0.35 <= in_ratio <= 0.45, f"IN ratio skewed in {part}: {in_ratio:.3f}"


def test_manifest_filtering(synthetic_s1_and_gt):
    s1_df, gt = synthetic_s1_and_gt
    manifest = create_stratified_split(s1_df, gt)

    val_a_s1 = manifest.filter_s1_dataframe(s1_df, "val_a")
    assert len(val_a_s1) == manifest.partition_counts["val_a"]
    assert set(val_a_s1["entity_id"]) == manifest.get_entity_set("val_a")

    val_a_gt = manifest.filter_ground_truth(gt, "val_a")
    assert len(val_a_gt) == manifest.partition_counts["val_a"]
    for eid in val_a_gt:
        assert eid in manifest.get_entity_set("val_a")


def test_manifest_serialization(synthetic_s1_and_gt):
    s1_df, gt = synthetic_s1_and_gt
    manifest = create_stratified_split(s1_df, gt)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Test JSON
        json_path = tmp_path / "manifest.json"
        manifest.save(json_path)
        loaded_json = SplitManifest.load(json_path)
        assert loaded_json.entity_to_partition == manifest.entity_to_partition
        assert loaded_json.partition_counts == manifest.partition_counts

        # Test TSV
        tsv_path = tmp_path / "manifest.tsv"
        manifest.save(tsv_path)
        loaded_tsv = SplitManifest.load(tsv_path)
        assert loaded_tsv.entity_to_partition == manifest.entity_to_partition

        # Test TSV.GZ
        tsvg_path = tmp_path / "manifest.tsv.gz"
        manifest.save(tsvg_path)
        loaded_tsvg = SplitManifest.load(tsvg_path)
        assert loaded_tsvg.entity_to_partition == manifest.entity_to_partition

        # Test Parquet if pyarrow/fastparquet is installed
        try:
            import pyarrow
            has_parquet = True
        except ImportError:
            has_parquet = False

        if has_parquet:
            parquet_path = tmp_path / "manifest.parquet"
            manifest.save(parquet_path)
            loaded_parquet = SplitManifest.load(parquet_path)
            assert loaded_parquet.entity_to_partition == manifest.entity_to_partition
            assert loaded_parquet.partition_counts == manifest.partition_counts

