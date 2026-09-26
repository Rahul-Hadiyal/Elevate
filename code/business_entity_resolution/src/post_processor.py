"""Post-Processing and Constraint Enforcement Module for Business Entity Resolution.

Applies domain constraints, precision guards, and conflict resolution over
pairwise match probability predictions:
1. 1-to-1 Source Constraint (at most 1 S2 and 1 S3 per S1 cluster).
2. Missing Address Precision Guard (elevated threshold when addresses are empty).
3. Strict Numeric and Country Mismatch Vetoes.
4. Global Reverse-Conflict Deduplication (optional 1-to-1 S1 assignment per candidate).
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np


@dataclass
class CandidatePrediction:
    """Scored candidate pair prediction."""
    s1_id: str
    cand_id: str
    prob: float
    cand_source: str  # 'S2' or 'S3'
    has_empty_addr: bool = False
    has_numeric_disagreement: bool = False
    has_country_disagreement: bool = False


class PostProcessor:
    """Configurable post-processing engine for entity resolution predictions."""

    def __init__(
        self,
        base_threshold: float = 0.60,
        empty_addr_threshold: float = 0.85,
        max_cands_per_source: int = 1,
        enforce_numeric_veto: bool = True,
        enforce_country_veto: bool = True,
        enforce_global_uniqueness: bool = False,
    ):
        self.base_threshold = base_threshold
        self.empty_addr_threshold = empty_addr_threshold
        self.max_cands_per_source = max_cands_per_source
        self.enforce_numeric_veto = enforce_numeric_veto
        self.enforce_country_veto = enforce_country_veto
        self.enforce_global_uniqueness = enforce_global_uniqueness

    def filter_and_assign(
        self,
        scored_pairs: List[CandidatePrediction],
        all_s1_ids: Optional[List[str]] = None,
    ) -> Dict[str, Set[str]]:
        """Apply post-processing rules and return final predictions map."""
        predictions: Dict[str, Set[str]] = {s1: set() for s1 in (all_s1_ids or [])}

        # Step 1: Candidate-level filtering
        valid_candidates: List[CandidatePrediction] = []

        for p in scored_pairs:
            # Country veto
            if self.enforce_country_veto and p.has_country_disagreement:
                continue

            # Numeric address mismatch veto
            if self.enforce_numeric_veto and p.has_numeric_disagreement:
                continue

            # Dynamic threshold based on missingness
            req_thresh = self.empty_addr_threshold if p.has_empty_addr else self.base_threshold
            if p.prob >= req_thresh:
                valid_candidates.append(p)

        # Step 2: Global Uniqueness (optional) or Per-S1 ranking
        if self.enforce_global_uniqueness:
            # Sort all candidate pairs globally by probability descending
            valid_candidates.sort(key=lambda x: x.prob, reverse=True)
            assigned_cands: Set[str] = set()
            s1_source_counts: Dict[Tuple[str, str], int] = {}

            for p in valid_candidates:
                if p.cand_id in assigned_cands:
                    continue

                curr_cnt = s1_source_counts.get((p.s1_id, p.cand_source), 0)
                if curr_cnt < self.max_cands_per_source:
                    if p.s1_id not in predictions:
                        predictions[p.s1_id] = set()
                    predictions[p.s1_id].add(p.cand_id)
                    assigned_cands.add(p.cand_id)
                    s1_source_counts[(p.s1_id, p.cand_source)] = curr_cnt + 1

        else:
            # Standard per-S1 top-k per source assignment
            # Group by S1
            s1_cand_map: Dict[str, List[CandidatePrediction]] = {}
            for p in valid_candidates:
                if p.s1_id not in s1_cand_map:
                    s1_cand_map[p.s1_id] = []
                s1_cand_map[p.s1_id].append(p)

            for s1_id, cands in s1_cand_map.items():
                # Separate by source
                s2_cands = [c for c in cands if c.cand_source == "S2"]
                s3_cands = [c for c in cands if c.cand_source == "S3"]

                s2_cands.sort(key=lambda x: x.prob, reverse=True)
                s3_cands.sort(key=lambda x: x.prob, reverse=True)

                selected_s2 = [c.cand_id for c in s2_cands[:self.max_cands_per_source]]
                selected_s3 = [c.cand_id for c in s3_cands[:self.max_cands_per_source]]

                if s1_id not in predictions:
                    predictions[s1_id] = set()
                predictions[s1_id].update(selected_s2 + selected_s3)

        return predictions
