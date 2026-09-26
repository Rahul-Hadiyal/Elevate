"""Unit Tests for Exact Expected-F0.5 Decision Engine and Conflict Resolution."""

import pytest
import numpy as np
from src.decision_engine import (
    f05_scalar,
    poisson_binomial_pmf,
    compute_exact_expected_f05,
    select_matches_for_entity,
    DecisionEngine,
)


def test_f05_scalar_verification():
    """Verifies official formula and problem statement worked examples."""
    # Official problem statement example: TP=2, FP=1, FN=0 -> F0.5 = 0.7143
    assert pytest.approx(f05_scalar(tp=2, fp=1, fn=0), 1e-4) == 0.7143

    # Empty true & predicted -> 1.0 (true singleton)
    assert f05_scalar(tp=0, fp=0, fn=0) == 1.0

    # False merge on singleton: TP=0, FP=1, FN=0 -> 0.0
    assert f05_scalar(tp=0, fp=1, fn=0) == 0.0

    # Missed match: TP=0, FP=0, FN=1 -> 0.0
    assert f05_scalar(tp=0, fp=0, fn=1) == 0.0


def test_poisson_binomial_pmf():
    """Verifies Poisson-Binomial DP against exact combinatorial expansion."""
    probs = [0.8, 0.4, 0.1]
    pmf = poisson_binomial_pmf(probs)
    
    assert len(pmf) == 4
    assert pytest.approx(sum(pmf), 1e-6) == 1.0

    # P(0 successes) = (1-0.8)*(1-0.4)*(1-0.1) = 0.2 * 0.6 * 0.9 = 0.108
    assert pytest.approx(pmf[0], 1e-5) == 0.108

    # P(3 successes) = 0.8 * 0.4 * 0.1 = 0.032
    assert pytest.approx(pmf[3], 1e-5) == 0.032


def test_exact_expected_f05_empty_set_selection():
    """Verifies that an entity with multiple weak candidates chooses the empty set."""
    # Weak candidates: p = [0.12, 0.10, 0.08, 0.06, 0.05]
    weak_cands = [
        ("c1", 0.12),
        ("c2", 0.10),
        ("c3", 0.08),
        ("c4", 0.06),
        ("c5", 0.05),
    ]

    selected = select_matches_for_entity(weak_cands)
    # Empty set should win to protect against false merge penalty on singletons
    assert len(selected) == 0


def test_exact_expected_f05_strong_selection():
    """Verifies that high probability candidates are correctly selected."""
    strong_cands = [
        ("c1", 0.95),
        ("c2", 0.90),
        ("c3", 0.05),
    ]

    selected = select_matches_for_entity(strong_cands)
    assert len(selected) == 2
    assert [c for c, _ in selected] == ["c1", "c2"]


def test_margin_guarded_conflict_resolution():
    """Verifies winner selection and ambiguous collision dropping under H1."""
    # Case 1: Clear winner (diff > margin_delta)
    # S1_A @ 0.95 vs S1_B @ 0.70 for candidate C100 (margin 0.25 > 0.05)
    cand_map = {
        "s1_A": [("c100", 0.95)],
        "s1_B": [("c100", 0.70), ("c200", 0.88)],
        "s1_C": [("c300", 0.92)],
        # Case 2: Ambiguous collision (diff < 0.05)
        # S1_D @ 0.81 vs S1_E @ 0.80 for candidate C400 (margin 0.01 < 0.05)
        "s1_D": [("c400", 0.81)],
        "s1_E": [("c400", 0.80)],
    }

    engine = DecisionEngine(margin_delta=0.05, enable_conflict_resolution=True)
    preds = engine.optimize_predictions(cand_map)

    # Winner S1_A retains c100; S1_B loses c100 and retains c200
    assert preds["s1_A"] == {"c100"}
    assert preds["s1_B"] == {"c200"}
    assert preds["s1_C"] == {"c300"}

    # Ambiguous collision drops c400 for both S1_D and S1_E (precision protection)
    assert preds["s1_D"] == set()
    assert preds["s1_E"] == set()
