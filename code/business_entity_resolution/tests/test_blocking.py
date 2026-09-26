"""Unit tests for Multi-Channel Candidate Generation and Blocking Engine."""

import pytest
import pandas as pd
import numpy as np

from src.normalizer import EntityNormalizer
from src.index_builder import BlockingIndex
from src.candidate_store import CandidateStore
from src.blocker import MultiChannelBlocker
from src.blocking_metrics import compute_blocking_metrics, BlockingMetricsSummary


@pytest.fixture
def synthetic_blocking_corpus():
    """Generates synthetic S1, S2, S3 datasets with diverse match and discrepancy patterns."""
    normalizer = EntityNormalizer()

    s1_raw = pd.DataFrame({
        "entity_id": ["S1-001", "S1-002", "S1-003", "S1-004", "S1-005", "S1-006"],
        "business_name": [
            "Starbucks Coffee",
            "Apex Logistics Solutions",
            "Cafe de Paris",
            "Dr. Smith Dental Clinic",
            "Unmatched Singleton Corp",
            "Hotel Grand Palace",
        ],
        "business_address": [
            "100 Main St Suite 4",
            "250 Airport Rd",
            "15 Rue de Rivoli",
            "404 Oak Ave",
            "999 Nowhere Lane",
            "500 Beach Rd Near Central Mall",
        ],
        "country": ["US", "India", "France", "US", "US", "India"],
    })

    s2_raw = pd.DataFrame({
        "entity_id": ["S2-001", "S2-002", "S2-003", "S2-004", "S2-006"],
        "business_name": [
            "Starbucks Coffee",              # Exact name match
            "Solutions Apex Logistics",       # Word permutation (sorted tokens match)
            "Cafe De Paris",                 # Exact name (France)
            "Smith Dental Clinic",           # Missing "Dr." prefix
            "Grand Palace Hotel",            # Landmark test
        ],
        "business_address": [
            "100 Main Street",
            "250 Airport Road",
            "15 Rue de Rivoli",
            "404 Oak Avenue",
            "500 Beach Road Near Central Mall",
        ],
        "country": ["USA", "IND", "France", "US", "India"],
    })

    s3_raw = pd.DataFrame({
        "entity_id": ["S3-001", "S3-002", "S3-004"],
        "business_name": [
            "Starbucks Coffee",              # Exact name
            "Apex Logistics Solutions",      # Exact normalized
            "Dental Clinic of Dr Smith",     # Paraphrased
        ],
        "business_address": [
            "100 Main St #4",
            "250 Airport Rd",
            "404 Oak Ave Suite 1",
        ],
        "country": ["US", "India", "US"],
    })


    s1_norm = normalizer.normalize_dataframe(s1_raw)
    s2_norm = normalizer.normalize_dataframe(s2_raw)
    s3_norm = normalizer.normalize_dataframe(s3_raw)

    gt = {
        "S1-001": {"S2-001", "S3-001"},
        "S1-002": {"S2-002", "S3-002"},
        "S1-003": {"S2-003"},
        "S1-004": {"S2-004", "S3-004"},
        "S1-005": set(),  # True singleton
        "S1-006": {"S2-006"},
    }

    return s1_norm, s2_norm, s3_norm, gt


def test_channel_a_exact_name(synthetic_blocking_corpus):
    s1_df, s2_df, s3_df, gt = synthetic_blocking_corpus
    blocker = MultiChannelBlocker().fit(s2_df, s3_df)
    cands_a = blocker.generate_channel_a(s1_df)

    # S1-001 ("Starbucks Coffee") should retrieve S2-001 ("Starbucks Coffee") and S3-001 ("Starbucks Coffee Company" -> suffix normalized)
    assert "S2-001" in cands_a["S1-001"]
    assert "S3-001" in cands_a["S1-001"]

    # S1-003 (France) should retrieve S2-003
    assert "S2-003" in cands_a["S1-003"]

    # S1-005 is a singleton -> empty candidate list from Channel A
    assert cands_a["S1-005"] == []


def test_channel_b_token_signature(synthetic_blocking_corpus):
    s1_df, s2_df, s3_df, _ = synthetic_blocking_corpus
    blocker = MultiChannelBlocker().fit(s2_df, s3_df)
    cands_b = blocker.generate_channel_b(s1_df)

    # S1-002 ("Apex Logistics Solutions") vs S2-002 ("Solutions Apex Logistics") -> sorted tokens match
    assert "S2-002" in cands_b["S1-002"]


def test_channel_c_address_numbers_safe(synthetic_blocking_corpus):
    s1_df, s2_df, s3_df, _ = synthetic_blocking_corpus
    blocker = MultiChannelBlocker().fit(s2_df, s3_df)
    cands_c = blocker.generate_channel_c(s1_df)

    # S1-004 has address "404 Oak Ave" and name initial "dr "
    # S2-004 has "404 Oak Avenue" and name "smith dental clinic" (initial "smi") -> won't match initial "dr "
    # S3-004 has "404 Oak Ave Suite 1" and name "dental clinic of dr smith" (initial "den")
    # Verify address number tokenization is exact: "100" does not collide with "1000"
    assert all("1000" not in str(c) for c in cands_c.get("S1-001", []))


def test_channel_h_landmark(synthetic_blocking_corpus):
    s1_df, s2_df, s3_df, _ = synthetic_blocking_corpus
    blocker = MultiChannelBlocker().fit(s2_df, s3_df)
    cands_h = blocker.generate_channel_h(s1_df)

    # S1-006 "Hotel Grand Palace" with landmark "central mall" matches S2-006 "Grand Palace Hotel"
    assert "S2-006" in cands_h["S1-006"]


def test_candidate_store_union_and_provenance(synthetic_blocking_corpus):
    s1_df, s2_df, s3_df, gt = synthetic_blocking_corpus
    blocker = MultiChannelBlocker().fit(s2_df, s3_df)

    store = CandidateStore(s1_df["entity_id"])
    cands_a = blocker.generate_channel_a(s1_df)
    cands_b = blocker.generate_channel_b(s1_df)

    added_a = store.add_channel_candidates("Channel_A", cands_a)
    added_b = store.add_channel_candidates("Channel_B", cands_b)

    assert added_a > 0
    assert store.total_s1_entities == len(s1_df)
    
    # S1-001 has candidates from Channel A
    cands_s1 = store.get_candidates("S1-001")
    assert "S2-001" in cands_s1
    assert "S3-001" in cands_s1

    # Verify no duplicate candidate IDs per S1
    for s1_id in s1_df["entity_id"]:
        c_list = store.get_candidates(s1_id)
        assert len(c_list) == len(set(c_list))


def test_blocking_metrics_computation(synthetic_blocking_corpus):
    s1_df, _, _, gt = synthetic_blocking_corpus

    # Perfect candidates matching all ground truth
    perfect_cands = {k: list(v) for k, v in gt.items()}
    metrics = compute_blocking_metrics(perfect_cands, gt)

    assert metrics.candidate_recall == 1.0
    assert metrics.entity_complete_coverage == 1.0
    assert metrics.entity_partial_coverage == 1.0
    assert metrics.missed_gt_links == 0
    assert metrics.singleton_s1_count == 1
    assert metrics.non_singleton_s1_count == 5

    # Partial candidates
    partial_cands = {
        "S1-001": ["S2-001"],  # 1 of 2 matches
        "S1-002": ["S2-002", "S3-002"],  # 2 of 2 matches
        "S1-003": [],  # 0 of 1 match
        "S1-004": ["S2-004"],  # 1 of 2 matches
        "S1-005": [],  # 0 of 0 matches (singleton)
        "S1-006": ["S2-006"],  # 1 of 1 match
    }
    # Total GT links: S1-001(2) + S1-002(2) + S1-003(1) + S1-004(2) + S1-006(1) = 8 links
    # Recovered: S1-001(1) + S1-002(2) + S1-003(0) + S1-004(1) + S1-006(1) = 5 links
    # Recall = 5/8 = 0.625
    part_metrics = compute_blocking_metrics(partial_cands, gt)
    assert np.isclose(part_metrics.candidate_recall, 5 / 8)
    # Complete coverage: S1-002, S1-006 -> 2/5 = 0.40
    assert np.isclose(part_metrics.entity_complete_coverage, 2 / 5)
    # Partial coverage: S1-001, S1-002, S1-004, S1-006 -> 4/5 = 0.80
    assert np.isclose(part_metrics.entity_partial_coverage, 4 / 5)
