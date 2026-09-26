"""Unit tests for Phase 5.1 Blocking Recall Recovery Channels (I, J, K, L, M)."""

import pytest
import pandas as pd
import numpy as np

from src.normalizer import EntityNormalizer
from src.index_builder import (
    BlockingIndex,
    canonicalize_phonetic,
    extract_postal_code,
)
from src.candidate_store import CandidateStore
from src.blocker import MultiChannelBlocker
from src.blocking_metrics import compute_blocking_metrics


def test_canonicalize_phonetic():
    assert canonicalize_phonetic("shree ganesh") == "sri ganesh"
    assert canonicalize_phonetic("shri ram") == "sri ram"
    assert canonicalize_phonetic("laxmi trading") == "lakshmi trading"
    assert canonicalize_phonetic("choudhary enterprise") == "chaudhary enterprise"
    assert canonicalize_phonetic("aggarwal brothers") == "agarwal bros"
    assert canonicalize_phonetic("national centre") == "national center"


def test_extract_postal_code():
    assert extract_postal_code("Hyderabad, Telangana 500081", "") == "500081"
    assert extract_postal_code("New Delhi 110001", "") == "110001"
    assert extract_postal_code("Baker, MT 59313, USA", "") == "59313"
    assert extract_postal_code("No postal code here", "") == ""


@pytest.fixture
def synthetic_recovery_corpus():
    normalizer = EntityNormalizer()

    s1_raw = pd.DataFrame({
        "entity_id": ["S1-101", "S1-102", "S1-103", "S1-104", "S1-105"],
        "business_name": [
            "Shree Laxmi Enterprises",        # Phonetic test
            "Alpha Beta Gamma Solutions",       # Core token pair test
            "Dr. John Miller Dental",          # Address Number + Street test
            "Apollo Pharmacy",                 # PIN code test
            "Zypheron Technologies",           # Rare 3-gram test
        ],
        "business_address": [
            "100 Market Rd",
            "200 Tech Park",
            "404 Oak Ave",
            "Shop 5, Jubilee Hills, Hyderabad 500081",
            "700 Innovation Way",
        ],
        "country": ["India", "US", "US", "India", "US"],
    })

    s2_raw = pd.DataFrame({
        "entity_id": ["S2-101", "S2-102", "S2-103", "S2-104", "S2-105"],
        "business_name": [
            "Sri Lakshmi Enterprise",          # Phonetic match
            "Gamma Alpha Solutions",           # Word permutation & subset
            "John Miller Dentistry",           # Name initial mismatch ("dr" vs "joh")
            "Apollo Drugs",                    # Name variation with same PIN
            "Zypheron Tech",                   # Shared rare 3-grams
        ],
        "business_address": [
            "100 Market Road",
            "200 Tech Park Way",
            "404 Oak Avenue",                  # Same number 404 + street "oak"
            "Plot 12, Hyderabad 500081",       # Same PIN 500081
            "700 Innovation Blvd",
        ],
        "country": ["India", "USA", "US", "IND", "US"],
    })

    s3_raw = pd.DataFrame({
        "entity_id": ["S3-101"],
        "business_name": ["Shri Lakshmi Ent"],
        "business_address": ["100 Market Rd"],
        "country": ["India"],
    })

    s1_norm = normalizer.normalize_dataframe(s1_raw)
    s2_norm = normalizer.normalize_dataframe(s2_raw)
    s3_norm = normalizer.normalize_dataframe(s3_raw)

    gt = {
        "S1-101": {"S2-101", "S3-101"},
        "S1-102": {"S2-102"},
        "S1-103": {"S2-103"},
        "S1-104": {"S2-104"},
        "S1-105": {"S2-105"},
    }

    return s1_norm, s2_norm, s3_norm, gt


def test_channel_i_phonetic_blocking(synthetic_recovery_corpus):
    s1_df, s2_df, s3_df, gt = synthetic_recovery_corpus
    blocker = MultiChannelBlocker().fit(s2_df, s3_df)
    cands_i = blocker.generate_channel_i(s1_df)

    # "Shree Laxmi Enterprises" vs "Sri Lakshmi Enterprise" & "Shri Lakshmi Ent"
    assert "S2-101" in cands_i["S1-101"]
    assert "S3-101" in cands_i["S1-101"]


def test_channel_j_core_token_pairs(synthetic_recovery_corpus):
    s1_df, s2_df, s3_df, gt = synthetic_recovery_corpus
    blocker = MultiChannelBlocker().fit(s2_df, s3_df)
    cands_j = blocker.generate_channel_j(s1_df)

    # "Alpha Beta Gamma Solutions" vs "Gamma Alpha Solutions" -> shares pair ("alpha", "gamma")
    assert "S2-102" in cands_j["S1-102"]


def test_channel_k_addr_num_street(synthetic_recovery_corpus):
    s1_df, s2_df, s3_df, gt = synthetic_recovery_corpus
    blocker = MultiChannelBlocker().fit(s2_df, s3_df)
    cands_k = blocker.generate_channel_k(s1_df)

    # Number "404" + street token "oak" -> matches S2-103 despite name starting with "dr." vs "john"
    assert "S2-103" in cands_k["S1-103"]


def test_channel_l_pin_code_anchor(synthetic_recovery_corpus):
    s1_df, s2_df, s3_df, gt = synthetic_recovery_corpus
    blocker = MultiChannelBlocker().fit(s2_df, s3_df)
    cands_l = blocker.generate_channel_l(s1_df)

    # PIN 500081 + name token "apollo" -> matches S2-104
    assert "S2-104" in cands_l["S1-104"]


def test_channel_m_rare_3grams(synthetic_recovery_corpus):
    s1_df, s2_df, s3_df, gt = synthetic_recovery_corpus
    blocker = MultiChannelBlocker().fit(s2_df, s3_df)
    cands_m = blocker.generate_channel_m(s1_df)

    # Rare 3-grams "zyp", "yph" in "zypheron" -> matches S2-105
    assert "S2-105" in cands_m["S1-105"]
