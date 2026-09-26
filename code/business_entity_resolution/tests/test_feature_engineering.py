"""Unit tests for feature engineering, store, and quality auditing."""

import pytest
import numpy as np
import pandas as pd

from src.feature_engineer import build_entity_lookup, audit_feature_quality
from src.feature_store import FeatureBatch, FeatureExtractor
from src.pair_features import FEATURE_NAMES


def test_build_entity_lookup():
    """Verify DataFrame conversion to fast lookup dictionary."""
    df = pd.DataFrame([
        {
            "entity_id": "S1_101",
            "business_name": "Target Store",
            "name_norm": "target store",
            "name_tokens_sorted": "store target",
            "name_is_degenerate": False,
            "business_address": "500 Broadway",
            "addr_norm": "500 broadway",
            "addr_landmark": "",
            "addr_numbers": ["500"],
            "country_norm": "us",
        }
    ])

    lookup = build_entity_lookup(df)
    assert "S1_101" in lookup
    assert lookup["S1_101"]["name_norm"] == "target store"
    assert lookup["S1_101"]["addr_numbers"] == ["500"]


def test_audit_feature_quality():
    """Test feature quality audit calculations and constant feature detection."""
    np.random.seed(42)
    n_rows = 100
    n_cols = len(FEATURE_NAMES)
    mat = np.random.uniform(0.0, 1.0, size=(n_rows, n_cols)).astype(np.float32)

    # Set one feature to constant
    mat[:, 0] = 1.0

    audit = audit_feature_quality(mat, FEATURE_NAMES)
    assert audit["num_rows"] == 100
    assert audit["num_features"] == n_cols
    assert audit["total_nan_count"] == 0
    assert audit["constant_feature_count"] == 1
    assert audit["constant_features"] == [FEATURE_NAMES[0]]


def test_feature_batch_and_dataframe():
    """Verify FeatureBatch conversions and properties."""
    pairs = [("S1_1", "S2_1"), ("S1_2", "S2_2")]
    features = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32)
    labels = np.array([1, 0], dtype=np.int8)

    batch = FeatureBatch(pair_ids=pairs, features=features, labels=labels)
    assert batch.num_pairs == 2
    assert batch.num_features == len(FEATURE_NAMES)
    assert batch.memory_mb > 0.0

    df = batch.to_dataframe()
    assert len(df) == 2
    assert "s1_entity_id" in df.columns
    assert "cand_entity_id" in df.columns
    assert "label" in df.columns
    assert df["label"].tolist() == [1, 0]
