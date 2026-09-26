"""Error Analysis & Failure Mode Categorization Engine (Phase 12).

Systematically categorizes False Positives (FPs), False Negatives (FNs),
and Singleton Errors on validation predictions to guide targeted refinements.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class ErrorTaxonomySummary:
    """Statistical breakdown of prediction failure modes."""
    total_evaluated_entities: int
    total_true_links: int
    total_predicted_links: int
    total_tp: int
    total_fp: int
    total_fn: int
    
    # FP Breakdown
    fp_common_name_collision: int
    fp_address_collision: int
    fp_franchise_branch_confusion: int
    fp_other: int
    
    # FN Breakdown
    fn_dba_trade_name: int
    fn_severe_abbreviation: int
    fn_missing_address: int
    fn_blocking_miss: int
    fn_other: int
    
    # Singleton Diagnostics
    total_true_singletons: int
    correct_singletons: int
    false_merges_on_singletons: int
    false_merge_rate: float
    singleton_accuracy: float


class ErrorAnalyzer:
    """Categorizes entity resolution matching errors across domain categories."""

    def analyze_errors(
        self,
        predictions: Dict[str, Set[str]],
        ground_truth: Dict[str, Set[str]],
        candidate_store_dict: Dict[str, List[str]],
        s1_lookup: Dict[str, Any],
        cand_lookup: Dict[str, Any],
    ) -> ErrorTaxonomySummary:
        """Categorizes all prediction errors across evaluated entities.
        
        Args:
            predictions: Dict of s1_id -> predicted matched cand IDs.
            ground_truth: Dict of s1_id -> true matched cand IDs.
            candidate_store_dict: Dict of s1_id -> generated candidates.
            s1_lookup: Dict of s1_id -> EntityRecord.
            cand_lookup: Dict of cand_id -> EntityRecord.
            
        Returns:
            ErrorTaxonomySummary with comprehensive error diagnostics.
        """
        total_entities = len(ground_truth)
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_true_links = 0
        total_pred_links = 0

        fp_common_name = 0
        fp_addr_collision = 0
        fp_franchise = 0
        fp_other = 0

        fn_dba = 0
        fn_abbrev = 0
        fn_missing_addr = 0
        fn_blocking_miss = 0
        fn_other = 0

        true_singletons = 0
        correct_singletons = 0
        false_merges = 0

        for s1_id, true_set in ground_truth.items():
            pred_set = predictions.get(s1_id, set())
            cands_generated = set(candidate_store_dict.get(s1_id, []))

            total_true_links += len(true_set)
            total_pred_links += len(pred_set)

            tp_set = pred_set & true_set
            fp_set = pred_set - true_set
            fn_set = true_set - pred_set

            total_tp += len(tp_set)
            total_fp += len(fp_set)
            total_fn += len(fn_set)

            # Singleton diagnostics
            if len(true_set) == 0:
                true_singletons += 1
                if len(pred_set) == 0:
                    correct_singletons += 1
                else:
                    false_merges += 1

            s1_rec = s1_lookup.get(s1_id)

            # Categorize False Positives
            for fp_cand in fp_set:
                cand_rec = cand_lookup.get(fp_cand)
                if not s1_rec or not cand_rec:
                    fp_other += 1
                    continue

                # Check franchise confusion (same name, differing address numbers)
                s1_nums = getattr(s1_rec, "addr_numbers", [])
                cand_nums = getattr(cand_rec, "addr_numbers", [])
                has_num_diff = bool(s1_nums and cand_nums and set(s1_nums) != set(cand_nums))

                s1_name = getattr(s1_rec, "name_norm", "")
                cand_name = getattr(cand_rec, "name_norm", "")
                s1_addr = getattr(s1_rec, "addr_norm", "")
                cand_addr = getattr(cand_rec, "addr_norm", "")

                if s1_name == cand_name and has_num_diff:
                    fp_franchise += 1
                elif s1_name == cand_name and s1_addr != cand_addr:
                    fp_common_name += 1
                elif s1_addr and s1_addr == cand_addr and s1_name != cand_name:
                    fp_addr_collision += 1
                else:
                    fp_other += 1

            # Categorize False Negatives
            for fn_cand in fn_set:
                if fn_cand not in cands_generated:
                    fn_blocking_miss += 1
                    continue

                cand_rec = cand_lookup.get(fn_cand)
                if not s1_rec or not cand_rec:
                    fn_other += 1
                    continue

                s1_name = getattr(s1_rec, "name_norm", "")
                cand_name = getattr(cand_rec, "name_norm", "")
                s1_addr = getattr(s1_rec, "addr_norm", "")
                cand_addr = getattr(cand_rec, "addr_norm", "")

                if not s1_addr or not cand_addr:
                    fn_missing_addr += 1
                elif len(s1_name) <= 4 or len(cand_name) <= 4:
                    fn_abbrev += 1
                elif s1_addr and cand_addr and s1_name != cand_name:
                    fn_dba += 1
                else:
                    fn_other += 1

        singleton_acc = float(correct_singletons / max(true_singletons, 1))
        false_merge_rate = float(false_merges / max(true_singletons, 1))

        return ErrorTaxonomySummary(
            total_evaluated_entities=total_entities,
            total_true_links=total_true_links,
            total_predicted_links=total_pred_links,
            total_tp=total_tp,
            total_fp=total_fp,
            total_fn=total_fn,
            fp_common_name_collision=fp_common_name,
            fp_address_collision=fp_addr_collision,
            fp_franchise_branch_confusion=fp_franchise,
            fp_other=fp_other,
            fn_dba_trade_name=fn_dba,
            fn_severe_abbreviation=fn_abbrev,
            fn_missing_address=fn_missing_addr,
            fn_blocking_miss=fn_blocking_miss,
            fn_other=fn_other,
            total_true_singletons=true_singletons,
            correct_singletons=correct_singletons,
            false_merges_on_singletons=false_merges,
            false_merge_rate=false_merge_rate,
            singleton_accuracy=singleton_acc,
        )
