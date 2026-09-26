"""Unit tests for pairwise feature computation."""

import pytest
import numpy as np

from src.pair_features import (
    FEATURE_NAMES,
    compute_single_pair_features,
    compute_char_ngrams,
    jaccard_set_similarity,
)


def test_feature_names_integrity():
    """Verify feature names list is non-empty, unique, and well-formed."""
    assert len(FEATURE_NAMES) > 40
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    for name in FEATURE_NAMES:
        assert name.startswith("feat_")


def test_exact_name_and_address_match():
    """Test feature extraction when two records are identical."""
    s1 = {
        "entity_id": "S1_001",
        "business_name": "Acme Corp",
        "name_norm": "acme corp",
        "name_tokens_sorted": "acme corp",
        "name_is_degenerate": False,
        "business_address": "123 Main St, New York, NY 10001",
        "addr_norm": "123 main st new york ny 10001",
        "addr_landmark": "",
        "addr_numbers": ["123", "10001"],
        "country_norm": "us",
    }
    cand = {
        "entity_id": "S2_001",
        "business_name": "Acme Corp",
        "name_norm": "acme corp",
        "name_tokens_sorted": "acme corp",
        "name_is_degenerate": False,
        "business_address": "123 Main St, New York, NY 10001",
        "addr_norm": "123 main st new york ny 10001",
        "addr_landmark": "",
        "addr_numbers": ["123", "10001"],
        "country_norm": "us",
    }

    vec = compute_single_pair_features(s1, cand, channel_count=2)
    assert len(vec) == len(FEATURE_NAMES)
    assert not np.isnan(vec).any()

    # Exact name norm match
    feat_map = dict(zip(FEATURE_NAMES, vec))
    assert feat_map["feat_name_exact_raw"] == 1.0
    assert feat_map["feat_name_exact_norm"] == 1.0
    assert feat_map["feat_name_exact_sorted"] == 1.0
    assert feat_map["feat_name_fuzz_ratio"] == 100.0
    assert feat_map["feat_name_tok_jaccard"] == 1.0
    assert feat_map["feat_addr_exact_norm"] == 1.0
    assert feat_map["feat_addr_num_exact_match"] == 1.0
    assert feat_map["feat_country_exact_match"] == 1.0
    assert feat_map["feat_cand_is_s2"] == 1.0
    assert feat_map["feat_cand_is_s3"] == 0.0
    assert feat_map["feat_cand_channel_count"] == 2.0


def test_phonetic_transliteration_similarity():
    """Test phonetic variation (e.g. Laxmi vs Lakshmi)."""
    s1 = {
        "entity_id": "S1_002",
        "business_name": "Shree Laxmi Medicals",
        "name_norm": "shree laxmi medicals",
        "name_tokens_sorted": "laxmi medicals shree",
        "country_norm": "india",
    }
    cand = {
        "entity_id": "S3_002",
        "business_name": "Shri Lakshmi Medicals",
        "name_norm": "shri lakshmi medicals",
        "name_tokens_sorted": "lakshmi medicals shri",
        "country_norm": "india",
    }

    vec = compute_single_pair_features(s1, cand, channel_count=1)
    feat_map = dict(zip(FEATURE_NAMES, vec))

    assert feat_map["feat_name_phonetic_exact"] == 1.0
    assert feat_map["feat_name_fuzz_token_sort"] > 80.0
    assert feat_map["feat_cand_is_s3"] == 1.0
    assert feat_map["feat_cand_is_s2"] == 0.0


def test_numeric_mismatch_protection():
    """Verify 1 Main St does NOT match 11 Main St in exact numeric anchor."""
    s1 = {
        "entity_id": "S1_003",
        "name_norm": "coffee shop",
        "business_address": "1 main street",
        "addr_norm": "1 main street",
        "addr_numbers": ["1"],
        "country_norm": "us",
    }
    cand = {
        "entity_id": "S2_003",
        "name_norm": "coffee shop",
        "business_address": "11 main street",
        "addr_norm": "11 main street",
        "addr_numbers": ["11"],
        "country_norm": "us",
    }

    vec = compute_single_pair_features(s1, cand)
    feat_map = dict(zip(FEATURE_NAMES, vec))

    assert feat_map["feat_addr_num_overlap_count"] == 0.0
    assert feat_map["feat_addr_num_exact_match"] == 0.0
    assert feat_map["feat_addr_num_disagreement"] == 1.0


def test_missingness_and_degenerate_handling():
    """Verify degenerate/empty records have zero similarities and correct flags."""
    s1 = {
        "entity_id": "S1_004",
        "name_norm": "",
        "name_is_degenerate": True,
        "addr_norm": "",
        "country_norm": "unknown",
    }
    cand = {
        "entity_id": "S2_004",
        "name_norm": "valid name",
        "addr_norm": "valid addr",
        "country_norm": "us",
    }

    vec = compute_single_pair_features(s1, cand)
    assert not np.isnan(vec).any()

    feat_map = dict(zip(FEATURE_NAMES, vec))
    assert feat_map["feat_s1_name_is_degenerate"] == 1.0
    assert feat_map["feat_s1_addr_is_empty"] == 1.0
    assert feat_map["feat_country_s1_missing"] == 1.0
    assert feat_map["feat_name_fuzz_ratio"] == 0.0
