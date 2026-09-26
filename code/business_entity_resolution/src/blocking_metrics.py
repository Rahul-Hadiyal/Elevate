"""Blocking and Candidate Generation Metrics Module.

Implements rigorous candidate-level and entity-level recall evaluation:
- Link Recall: True ground-truth matches recovered / Total ground-truth links.
- Entity Complete Coverage: Fraction of non-singleton S1s with 100% of their true matches in candidate pool.
- Entity Partial Coverage: Fraction of non-singleton S1s with at least one true match in candidate pool.
- Candidate Volume Statistics: Total pairs, mean, median, P90, P95, P99, max candidates per S1.
- Candidate Explosion / Zero-candidate diagnostic indicators.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np


@dataclass
class BlockingMetricsSummary:
    """Detailed summary of candidate generation quality and volume."""
    candidate_recall: float
    entity_complete_coverage: float
    entity_partial_coverage: float
    total_gt_links: int
    recovered_gt_links: int
    missed_gt_links: int
    total_s1_entities: int
    non_singleton_s1_count: int
    singleton_s1_count: int
    total_candidate_pairs: int
    avg_candidates_per_s1: float
    median_candidates_per_s1: float
    p90_candidates_per_s1: float
    p95_candidates_per_s1: float
    p99_candidates_per_s1: float
    max_candidates_per_s1: int
    zero_candidate_s1_count: int
    zero_candidate_s1_rate: float
    runtime_seconds: float = 0.0
    peak_memory_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Converts summary to dictionary with rounded floats."""
        return {
            "candidate_recall": round(self.candidate_recall, 6),
            "entity_complete_coverage": round(self.entity_complete_coverage, 6),
            "entity_partial_coverage": round(self.entity_partial_coverage, 6),
            "total_gt_links": self.total_gt_links,
            "recovered_gt_links": self.recovered_gt_links,
            "missed_gt_links": self.missed_gt_links,
            "total_s1_entities": self.total_s1_entities,
            "non_singleton_s1_count": self.non_singleton_s1_count,
            "singleton_s1_count": self.singleton_s1_count,
            "total_candidate_pairs": self.total_candidate_pairs,
            "avg_candidates_per_s1": round(self.avg_candidates_per_s1, 2),
            "median_candidates_per_s1": round(self.median_candidates_per_s1, 1),
            "p90_candidates_per_s1": round(self.p90_candidates_per_s1, 1),
            "p95_candidates_per_s1": round(self.p95_candidates_per_s1, 1),
            "p99_candidates_per_s1": round(self.p99_candidates_per_s1, 1),
            "max_candidates_per_s1": self.max_candidates_per_s1,
            "zero_candidate_s1_count": self.zero_candidate_s1_count,
            "zero_candidate_s1_rate": round(self.zero_candidate_s1_rate, 4),
            "runtime_seconds": round(self.runtime_seconds, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
        }


def compute_blocking_metrics(
    candidates: Dict[str, Union[Set[str], List[str]]],
    ground_truth: Dict[str, Union[Set[str], List[str]]],
    runtime_seconds: float = 0.0,
    peak_memory_mb: float = 0.0,
) -> BlockingMetricsSummary:
    """Computes comprehensive candidate recall and volume statistics.

    Args:
        candidates: Mapping from source1_entity_id -> set/list of candidate IDs.
        ground_truth: Mapping from source1_entity_id -> set/list of true matched IDs.
        runtime_seconds: Total execution time for generating the candidates.
        peak_memory_mb: Peak memory usage in MB.

    Returns:
        BlockingMetricsSummary with recall and candidate distribution metrics.
    """
    total_s1_entities = len(ground_truth)
    if total_s1_entities == 0:
        return BlockingMetricsSummary(
            candidate_recall=0.0,
            entity_complete_coverage=0.0,
            entity_partial_coverage=0.0,
            total_gt_links=0,
            recovered_gt_links=0,
            missed_gt_links=0,
            total_s1_entities=0,
            non_singleton_s1_count=0,
            singleton_s1_count=0,
            total_candidate_pairs=0,
            avg_candidates_per_s1=0.0,
            median_candidates_per_s1=0.0,
            p90_candidates_per_s1=0.0,
            p95_candidates_per_s1=0.0,
            p99_candidates_per_s1=0.0,
            max_candidates_per_s1=0,
            zero_candidate_s1_count=0,
            zero_candidate_s1_rate=0.0,
            runtime_seconds=runtime_seconds,
            peak_memory_mb=peak_memory_mb,
        )

    total_gt_links = 0
    recovered_gt_links = 0
    non_singleton_s1_count = 0
    singleton_s1_count = 0
    complete_recovered_entities = 0
    partial_recovered_entities = 0

    candidate_counts = np.empty(total_s1_entities, dtype=np.int32)
    zero_cand_count = 0

    for idx, (s1_id, true_matches) in enumerate(ground_truth.items()):
        true_set = set(true_matches) if not isinstance(true_matches, set) else true_matches
        n_true = len(true_set)
        cand_list = candidates.get(s1_id, ())
        cand_set = set(cand_list) if not isinstance(cand_list, set) else cand_list
        n_cands = len(cand_set)

        candidate_counts[idx] = n_cands
        if n_cands == 0:
            zero_cand_count += 1

        if n_true == 0:
            singleton_s1_count += 1
        else:
            non_singleton_s1_count += 1
            total_gt_links += n_true
            recovered = len(true_set & cand_set)
            recovered_gt_links += recovered

            if recovered == n_true:
                complete_recovered_entities += 1
            if recovered > 0:
                partial_recovered_entities += 1

    candidate_recall = (
        recovered_gt_links / total_gt_links if total_gt_links > 0 else 1.0
    )
    entity_complete_coverage = (
        complete_recovered_entities / non_singleton_s1_count
        if non_singleton_s1_count > 0
        else 1.0
    )
    entity_partial_coverage = (
        partial_recovered_entities / non_singleton_s1_count
        if non_singleton_s1_count > 0
        else 1.0
    )

    total_candidate_pairs = int(np.sum(candidate_counts))
    avg_cands = float(np.mean(candidate_counts))
    median_cands = float(np.median(candidate_counts))
    p90_cands = float(np.percentile(candidate_counts, 90))
    p95_cands = float(np.percentile(candidate_counts, 95))
    p99_cands = float(np.percentile(candidate_counts, 99))
    max_cands = int(np.max(candidate_counts))
    zero_cand_rate = zero_cand_count / total_s1_entities

    return BlockingMetricsSummary(
        candidate_recall=candidate_recall,
        entity_complete_coverage=entity_complete_coverage,
        entity_partial_coverage=entity_partial_coverage,
        total_gt_links=total_gt_links,
        recovered_gt_links=recovered_gt_links,
        missed_gt_links=total_gt_links - recovered_gt_links,
        total_s1_entities=total_s1_entities,
        non_singleton_s1_count=non_singleton_s1_count,
        singleton_s1_count=singleton_s1_count,
        total_candidate_pairs=total_candidate_pairs,
        avg_candidates_per_s1=avg_cands,
        median_candidates_per_s1=median_cands,
        p90_candidates_per_s1=p90_cands,
        p95_candidates_per_s1=p95_cands,
        p99_candidates_per_s1=p99_cands,
        max_candidates_per_s1=max_cands,
        zero_candidate_s1_count=zero_cand_count,
        zero_candidate_s1_rate=zero_cand_rate,
        runtime_seconds=runtime_seconds,
        peak_memory_mb=peak_memory_mb,
    )
