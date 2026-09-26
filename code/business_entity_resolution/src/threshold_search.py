"""Threshold Search and Entity-Level Evaluation Module for Business Entity Resolution.

Evaluates entity-level Macro F0.5 (and precision, recall, singleton accuracy) across a
fine-grained grid of probability decision thresholds.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd

from src.metrics import compute_macro_f05, EvaluationSummary


@dataclass
class ThresholdEvaluationRow:
    """Evaluation summary for a single decision threshold."""
    threshold: float
    macro_f05: float
    macro_precision: float
    macro_recall: float
    correct_singletons: int
    false_merges: int
    missed_non_singletons: int
    total_entities: int


class ThresholdOptimizer:
    """Performs threshold grid search and entity resolution evaluation."""

    def __init__(
        self,
        candidate_probs: Dict[str, List[Tuple[str, float]]],
        ground_truth: Dict[str, Union[Set[str], List[str]]],
        all_s1_ids: Optional[List[str]] = None,
    ):
        """Initialize ThresholdOptimizer.
        
        Args:
            candidate_probs: Map from s1_entity_id -> list of (cand_id, probability).
            ground_truth: Map from s1_entity_id -> set of true matched entity IDs.
            all_s1_ids: List of all evaluation S1 IDs (including singletons with 0 candidates).
        """
        self.candidate_probs = candidate_probs
        self.ground_truth = ground_truth
        
        if all_s1_ids is not None:
            self.all_s1_ids = list(all_s1_ids)
            self.ground_truth = {eid: ground_truth.get(eid, set()) for eid in self.all_s1_ids}
        else:
            self.all_s1_ids = sorted(list(set(candidate_probs.keys()) | set(ground_truth.keys())))
            self.ground_truth = ground_truth

    def evaluate_threshold(
        self,
        threshold: float,
        max_cands_per_source: Optional[int] = None,
    ) -> EvaluationSummary:
        """Evaluate macro F0.5 at a specific probability threshold.
        
        Args:
            threshold: Probability decision cutoff [0.0, 1.0].
            max_cands_per_source: Optional max entities to predict per source (e.g. at most 1 S2 and 1 S3).
            
        Returns:
            EvaluationSummary object with macro F0.5, precision, recall, etc.
        """
        predictions: Dict[str, Set[str]] = {}

        for s1_id in self.all_s1_ids:
            cands_with_scores = self.candidate_probs.get(s1_id, [])
            filtered = [(cid, score) for cid, score in cands_with_scores if score >= threshold]

            if max_cands_per_source is not None and max_cands_per_source > 0:
                # Rank by score descending and take at most max_cands_per_source per source
                filtered.sort(key=lambda x: x[1], reverse=True)
                s2_cands = [cid for cid, _ in filtered if cid.startswith("S2-") or cid.startswith("S2_")][:max_cands_per_source]
                s3_cands = [cid for cid, _ in filtered if cid.startswith("S3-") or cid.startswith("S3_")][:max_cands_per_source]
                selected_set = set(s2_cands + s3_cands)
            else:
                selected_set = {cid for cid, _ in filtered}

            predictions[s1_id] = selected_set

        return compute_macro_f05(predictions, self.ground_truth, include_detailed_results=True)

    def grid_search(
        self,
        thresholds: Optional[List[float]] = None,
        max_cands_per_source: Optional[int] = None,
    ) -> Tuple[float, EvaluationSummary, pd.DataFrame]:
        """Perform grid search over candidate decision thresholds.
        
        Args:
            thresholds: List of thresholds to evaluate. Defaults to [0.10, 0.15, ..., 0.95].
            max_cands_per_source: Optional constraint on per-source matches.
            
        Returns:
            Tuple of (optimal_threshold, best_summary, results_dataframe).
        """
        if thresholds is None:
            thresholds = [round(t, 2) for t in np.arange(0.10, 0.96, 0.05)]

        rows = []
        best_f05 = -1.0
        best_thresh = thresholds[0]
        best_summary = None

        for t in thresholds:
            summary = self.evaluate_threshold(t, max_cands_per_source=max_cands_per_source)
            rows.append({
                "threshold": t,
                "macro_f05": summary.macro_f05,
                "macro_precision": summary.macro_precision,
                "macro_recall": summary.macro_recall,
                "correct_singletons": summary.correct_singletons,
                "singleton_acc": summary.singleton_accuracy,
                "false_merges": summary.false_merge_count,
                "false_merge_rate": summary.false_merge_rate,
                "missed_non_singletons": summary.missed_non_singletons,
                "exact_matches": summary.exact_match_entities,
            })

            if summary.macro_f05 > best_f05:
                best_f05 = summary.macro_f05
                best_thresh = t
                best_summary = summary

        df_results = pd.DataFrame(rows)
        return best_thresh, best_summary, df_results
