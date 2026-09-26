"""Unit Tests for Phase 12: Error Analysis and Failure Taxonomy Engine."""

import pytest
from dataclasses import dataclass
from src.error_analysis import ErrorAnalyzer, ErrorTaxonomySummary


@dataclass
class MockRecord:
    name_norm: str
    addr_norm: str
    addr_numbers: list


def test_error_analyzer_categorization():
    """Verifies FP and FN categorization across franchise, common name, and missing address."""
    s1_lookup = {
        "s1_1": MockRecord(name_norm="target", addr_norm="100 main st", addr_numbers=["100"]),
        "s1_2": MockRecord(name_norm="target", addr_norm="200 oak st", addr_numbers=["200"]),
        "s1_3": MockRecord(name_norm="starbucks", addr_norm="500 pine st", addr_numbers=["500"]),
        "s1_singleton": MockRecord(name_norm="unique inc", addr_norm="999 broad st", addr_numbers=["999"]),
    }

    cand_lookup = {
        "s2_1": MockRecord(name_norm="target", addr_norm="100 main st", addr_numbers=["100"]),
        "s2_2": MockRecord(name_norm="target", addr_norm="300 elm st", addr_numbers=["300"]),
        "s2_3": MockRecord(name_norm="starbucks", addr_norm="", addr_numbers=[]),
        "s2_false": MockRecord(name_norm="unrelated corp", addr_norm="111 wall st", addr_numbers=["111"]),
    }

    gt = {
        "s1_1": {"s2_1"},
        "s1_2": {"s2_2"},
        "s1_3": {"s2_3"},
        "s1_singleton": set(),
    }

    preds = {
        "s1_1": {"s2_1"},               # TP
        "s1_2": {"s2_1"},               # FP (franchise confusion: name matches, numbers disagree)
        "s1_3": set(),                  # FN (missing address on candidate)
        "s1_singleton": {"s2_false"},   # False merge on singleton
    }

    cand_store = {
        "s1_1": ["s2_1"],
        "s1_2": ["s2_1", "s2_2"],
        "s1_3": ["s2_3"],
        "s1_singleton": ["s2_false"],
    }

    analyzer = ErrorAnalyzer()
    summary = analyzer.analyze_errors(preds, gt, cand_store, s1_lookup, cand_lookup)

    assert isinstance(summary, ErrorTaxonomySummary)
    assert summary.total_tp == 1
    assert summary.total_fp == 2
    assert summary.total_fn == 2
    assert summary.total_true_singletons == 1
    assert summary.false_merges_on_singletons == 1
    assert summary.false_merge_rate == 1.0
    assert summary.fp_franchise_branch_confusion == 1
    assert summary.fn_missing_address == 1
