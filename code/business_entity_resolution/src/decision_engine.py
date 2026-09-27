"""Exact Expected-F0.5 Decision Layer and Margin-Guarded Conflict Resolver.

Implements Poisson-Binomial Dynamic Programming subset selection to choose
candidates maximizing exact expected F0.5 per S1 entity, followed by
margin-guarded conflict resolution and loser re-optimisation under Hypothesis H1.
"""

from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

BETA2: float = 0.25  # beta = 0.5


def f05_scalar(tp: int, fp: int, fn: int) -> float:
    """Calculates official per-entity F0.5 score for exact counts."""
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0  # Correct empty singleton
    if tp == 0:
        return 0.0
    num = (1.0 + BETA2) * tp
    den = (1.0 + BETA2) * tp + BETA2 * fn + fp
    return float(num / den) if den > 0.0 else 0.0


def poisson_binomial_pmf(probs: List[float]) -> np.ndarray:
    """Computes Poisson-Binomial PMF via O(K^2) dynamic programming.
    
    dp[j] represents the probability of having exactly j successes
    among independent Bernoulli trials with probabilities `probs`.
    """
    if not probs:
        return np.array([1.0], dtype=np.float64)

    dp = np.zeros(len(probs) + 1, dtype=np.float64)
    dp[0] = 1.0

    for p in probs:
        p_val = float(np.clip(p, 0.0, 1.0))
        # Update in reverse to perform in-place DP
        dp[1:] = dp[1:] * (1.0 - p_val) + dp[:-1] * p_val
        dp[0] = dp[0] * (1.0 - p_val)

    return dp


def compute_exact_expected_f05(
    selected_probs: List[float],
    unselected_probs: List[float],
) -> float:
    """Computes exact E[F0.5] over independent candidate probabilities.
    
    TP ~ PoissonBinomial(selected_probs)
    FP = k - TP (where k = len(selected_probs))
    FN ~ PoissonBinomial(unselected_probs)
    """
    k = len(selected_probs)
    if k == 0:
        # For empty subset, score is 1.0 iff all unselected are 0 (true singleton), else 0.0
        # P(FN=0) = product(1 - p_j)
        p_empty_correct = 1.0
        for p in unselected_probs:
            p_empty_correct *= (1.0 - float(np.clip(p, 0.0, 1.0)))
        return float(p_empty_correct)

    tp_pmf = poisson_binomial_pmf(selected_probs)
    fn_pmf = poisson_binomial_pmf(unselected_probs)

    expected_f05 = 0.0
    for tp, p_tp in enumerate(tp_pmf):
        if p_tp <= 1e-12:
            continue
        fp = k - tp
        for fn, p_fn in enumerate(fn_pmf):
            if p_fn <= 1e-12:
                continue
            score = f05_scalar(tp, fp, fn)
            expected_f05 += p_tp * p_fn * score

    return float(expected_f05)


def select_matches_for_entity(
    candidate_scores: List[Tuple[str, float]],
    min_prob_filter: float = 0.01,
    max_candidates: int = 20,
) -> List[Tuple[str, float]]:
    """Selects the prefix of candidate matches maximizing exact expected F0.5.
    
    Args:
        candidate_scores: List of (cand_id, probability) tuples for a single S1 entity.
        min_prob_filter: Prune candidate probabilities below this floor.
        max_candidates: Maximum prefix depth to evaluate.
        
    Returns:
        List of selected (cand_id, probability) tuples.
    """
    if not candidate_scores:
        return []

    # Filter negligible candidates and sort descending by probability
    filtered = [
        (c_id, float(p)) for c_id, p in candidate_scores
        if p >= min_prob_filter
    ]
    if not filtered:
        return []

    filtered.sort(key=lambda x: x[1], reverse=True)
    filtered = filtered[:max_candidates]

    all_cands = [c_id for c_id, _ in filtered]
    all_probs = [p for _, p in filtered]
    total_n = len(all_probs)

    # Base case: evaluate empty set k = 0
    best_k = 0
    best_expected_val = compute_exact_expected_f05([], all_probs)

    # Evaluate prefixes k = 1..total_n
    for k in range(1, total_n + 1):
        selected_p = all_probs[:k]
        unselected_p = all_probs[k:]
        val = compute_exact_expected_f05(selected_p, unselected_p)
        if val > best_expected_val:
            best_k = k
            best_expected_val = val

    return [(all_cands[i], all_probs[i]) for i in range(best_k)]


class DecisionEngine:
    """End-to-end entity decision optimizer with margin-guarded conflict resolution."""

    def __init__(
        self,
        margin_delta: float = 0.05,
        min_prob_filter: float = 0.01,
        max_candidates_per_entity: int = 20,
        enable_conflict_resolution: bool = True,
    ):
        """Initializes decision engine settings."""
        self.margin_delta = margin_delta
        self.min_prob_filter = min_prob_filter
        self.max_candidates_per_entity = max_candidates_per_entity
        self.enable_conflict_resolution = enable_conflict_resolution

    def optimize_predictions(
        self,
        entity_candidate_map: Dict[str, List[Tuple[str, float]]],
    ) -> Dict[str, Set[str]]:
        """Selects matches per entity and resolves candidate ownership conflicts.
        
        Args:
            entity_candidate_map: Dict mapping s1_id -> [(cand_id, probability), ...]
            
        Returns:
            Dict mapping s1_id -> Set of selected matched candidate IDs.
        """
        # Step 1: Initial exact expected-F0.5 selection per S1 entity
        selected_per_s1: Dict[str, Dict[str, float]] = {}
        for s1_id, cand_pairs in entity_candidate_map.items():
            chosen = select_matches_for_entity(
                cand_pairs,
                min_prob_filter=self.min_prob_filter,
                max_candidates=self.max_candidates_per_entity,
            )
            selected_per_s1[s1_id] = {c_id: p for c_id, p in chosen}

        if not self.enable_conflict_resolution:
            return {s1_id: set(cands.keys()) for s1_id, cands in selected_per_s1.items()}

        # Step 2: Build reverse claimant index: cand_id -> [(s1_id, p), ...]
        reverse_index: Dict[str, List[Tuple[str, float]]] = {}
        for s1_id, cand_dict in selected_per_s1.items():
            for cand_id, p in cand_dict.items():
                if cand_id not in reverse_index:
                    reverse_index[cand_id] = []
                reverse_index[cand_id].append((s1_id, p))

        # Step 3: Margin-guarded conflict resolution
        conflicts_resolved = 0
        s1_entities_affected: Set[str] = set()
        # Track candidate ownership: cand_id -> winning s1_id (if clear winner)
        # ambiguous_dropped: candidates removed due to ambiguous collision (no one gets them)
        ambiguous_dropped: Set[str] = set()
        cand_winner: Dict[str, str] = {}

        for cand_id, claimants in reverse_index.items():
            if len(claimants) <= 1:
                continue

            conflicts_resolved += 1
            # Sort claimants by score descending
            claimants.sort(key=lambda x: x[1], reverse=True)
            winner_s1, winner_p = claimants[0]
            runner_up_s1, runner_up_p = claimants[1]

            score_diff = winner_p - runner_up_p

            if score_diff < self.margin_delta - 1e-9:
                # Ambiguous collision: drop edge for all claimants (precision-protective)
                ambiguous_dropped.add(cand_id)
                for s1_id, _ in claimants:
                    if cand_id in selected_per_s1[s1_id]:
                        del selected_per_s1[s1_id][cand_id]
                        s1_entities_affected.add(s1_id)
            else:
                # Clear winner: record winner, drop edge from all losers
                cand_winner[cand_id] = winner_s1
                for s1_id, _ in claimants[1:]:
                    if cand_id in selected_per_s1[s1_id]:
                        del selected_per_s1[s1_id][cand_id]
                        s1_entities_affected.add(s1_id)

        logger.info(
            f"Conflict Resolution: {conflicts_resolved} candidate collisions processed, "
            f"{len(s1_entities_affected)} S1 entities affected."
        )

        # Step 4: Loser re-optimisation for affected S1 entities
        # For each affected S1 entity, exclude:
        #   (a) Any candidate currently selected by ANOTHER entity (prevents duplicate cross-entity assignment)
        #   (b) Any candidate dropped ambiguously in Step 3
        # This guarantees:
        #   1. Zero duplicate candidate assignments across entities
        #   2. Correct retention of candidates won by the entity itself
        #   3. Clean re-evaluation of alternate candidates for losers
        all_assigned: Set[str] = {c for cands in selected_per_s1.values() for c in cands}

        for s1_id in s1_entities_affected:
            curr_assigned = set(selected_per_s1[s1_id].keys())
            # Fast exact equivalence: c_id not in ((all_assigned - curr_assigned) | ambiguous_dropped)
            remaining_cands = [
                (c_id, p) for c_id, p in entity_candidate_map.get(s1_id, [])
                if (c_id in curr_assigned or c_id not in all_assigned) and (c_id not in ambiguous_dropped)
            ]
            re_chosen = select_matches_for_entity(
                remaining_cands,
                min_prob_filter=self.min_prob_filter,
                max_candidates=self.max_candidates_per_entity,
            )
            new_cands = {c_id: p for c_id, p in re_chosen}
            selected_per_s1[s1_id] = new_cands
            all_assigned.difference_update(curr_assigned)
            all_assigned.update(new_cands.keys())

        return {s1_id: set(cands.keys()) for s1_id, cands in selected_per_s1.items()}


