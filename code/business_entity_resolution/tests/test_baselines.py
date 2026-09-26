"""Unit tests for Baseline Matchers Module."""

import pytest
import pandas as pd
from src.baselines import (
    ExactRawNameMatcher,
    ExactNormalizedNameMatcher,
    ExactNormalizedNameCountryMatcher,
    ExactNormalizedNameAddressMatcher,
)



@pytest.fixture
def synthetic_corpus_data():
    s1_df = pd.DataFrame({
        "entity_id": ["S1-001", "S1-002", "S1-003", "S1-004"],
        "business_name": ["Acme Corp", "Global Tech Inc.", "Single Entity", "Pizzeria Napoli"],
        "business_address": ["100 Main St, Ste 4", "200 Market Ave", "300 Oak Rd", "400 Pine St"],
        "country": ["US", "India", "US", "France"],
    })

    s2_df = pd.DataFrame({
        "entity_id": ["S2-001", "S2-002", "S2-004"],
        "business_name": ["ACME CORP", "GLOBAL TECH INC", "Pizzeria Napoli"],
        "business_address": ["100 Main Street #4", "200 Market Avenue", "999 Other St"],
        "country": ["USA", "IND", "France"],
    })

    s3_df = pd.DataFrame({
        "entity_id": ["S3-001", "S3-002"],
        "business_name": ["Acme Corporation", "Global Tech Incorporated"],
        "business_address": ["100 Main St Suite 4", "200 Market Ave"],
        "country": ["US", "India"],
    })

    return s1_df, s2_df, s3_df



def test_exact_raw_name_matcher(synthetic_corpus_data):
    s1_df, s2_df, s3_df = synthetic_corpus_data
    matcher = ExactRawNameMatcher()
    matcher.fit(s2_df, s3_df)
    preds = matcher.predict(s1_df)

    # "Pizzeria Napoli" matches S2-004 exactly (case-insensitive)
    assert preds["S1-004"] == ["S2-004"]
    # "Single Entity" does not exist in S2/S3 -> empty list
    assert preds["S1-003"] == []


def test_exact_normalized_name_matcher(synthetic_corpus_data):
    s1_df, s2_df, s3_df = synthetic_corpus_data
    matcher = ExactNormalizedNameMatcher()
    matcher.fit(s2_df, s3_df)
    preds = matcher.predict(s1_df)

    # "Acme Corp LLC" -> name_norm: "acme"
    # S2-001 "ACME CORP" -> "acme"
    # S3-001 "Acme Corporation" -> "acme"
    assert "S2-001" in preds["S1-001"]
    assert "S3-001" in preds["S1-001"]

    # "Global Tech Inc." -> name_norm: "global tech"
    # S2-002 "GLOBAL TECH" -> "global tech"
    # S3-002 "Global Tech Private Limited" -> "global tech"
    assert "S2-002" in preds["S1-002"]
    assert "S3-002" in preds["S1-002"]


def test_exact_normalized_name_country_matcher(synthetic_corpus_data):
    s1_df, s2_df, s3_df = synthetic_corpus_data
    matcher = ExactNormalizedNameCountryMatcher()
    matcher.fit(s2_df, s3_df)
    preds = matcher.predict(s1_df)

    # S1-001 (US) matches S2-001 (USA -> US) and S3-001 (US -> US)
    assert set(preds["S1-001"]) == {"S2-001", "S3-001"}

    # S1-002 (India -> IN) matches S2-002 (IND -> IN) and S3-002 (India -> IN)
    assert set(preds["S1-002"]) == {"S2-002", "S3-002"}


def test_exact_normalized_name_address_matcher(synthetic_corpus_data):
    s1_df, s2_df, s3_df = synthetic_corpus_data
    matcher = ExactNormalizedNameAddressMatcher()
    matcher.fit(s2_df, s3_df)
    preds = matcher.predict(s1_df)

    # S1-004 "Pizzeria Napoli" has address "400 Pine St", but S2-004 has "999 Other St" -> address mismatch
    assert preds["S1-004"] == []
