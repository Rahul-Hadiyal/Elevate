"""Unit tests for exact label creation and multi-match/zero-match handling."""

import pytest
import numpy as np

from src.feature_store import FeatureExtractor


def test_label_creation_exact_ground_truth():
    """Verify ground truth pairs receive label 1 and non-ground truth receive label 0."""
    s1_lookup = {
        "S1_1": {"entity_id": "S1_1", "name_norm": "apple store", "addr_norm": "5th ave", "country_norm": "us"},
        "S1_2": {"entity_id": "S1_2", "name_norm": "zero match store", "addr_norm": "main st", "country_norm": "us"},
    }
    cand_lookup = {
        "S2_1": {"entity_id": "S2_1", "name_norm": "apple store", "addr_norm": "5th ave", "country_norm": "us"},
        "S3_1": {"entity_id": "S3_1", "name_norm": "apple store nyc", "addr_norm": "5th ave", "country_norm": "us"},
        "S2_2": {"entity_id": "S2_2", "name_norm": "random other store", "addr_norm": "elm st", "country_norm": "us"},
    }

    # S1_1 has 2 true matches (multi-match). S1_2 has 0 true matches.
    ground_truth = {
        "S1_1": {"S2_1", "S3_1"},
        "S1_2": set(),
    }

    extractor = FeatureExtractor(s1_lookup, cand_lookup, ground_truth)

    pairs = [
        ("S1_1", "S2_1"),  # True match -> 1
        ("S1_1", "S3_1"),  # True match -> 1
        ("S1_1", "S2_2"),  # False candidate -> 0
        ("S1_2", "S2_1"),  # False candidate -> 0
        ("S1_2", "S2_2"),  # False candidate -> 0
    ]

    batch = extractor.extract_pair_batch(pairs)
    assert batch.labels.tolist() == [1, 1, 0, 0, 0]
    assert len(batch.pair_ids) == 5


def test_no_duplicate_pairs():
    """Verify pair IDs are distinct and well preserved."""
    s1_lookup = {"S1_1": {"entity_id": "S1_1", "name_norm": "abc"}}
    cand_lookup = {"S2_1": {"entity_id": "S2_1", "name_norm": "abc"}}
    extractor = FeatureExtractor(s1_lookup, cand_lookup, {"S1_1": {"S2_1"}})

    batch = extractor.extract_pair_batch([("S1_1", "S2_1")])
    assert len(batch.pair_ids) == 1
    assert batch.pair_ids[0] == ("S1_1", "S2_1")
    assert batch.labels[0] == 1
