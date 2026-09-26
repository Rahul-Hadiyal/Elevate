"""Evaluation Metrics Module for Business Entity Resolution.

Implements the official macro F0.5 evaluation metric (beta = 0.5, beta^2 = 0.25)
per the Amazon ML Challenge problem statement.

Single shared source of truth for:
- Per-entity F0.5 calculation
- Precision and Recall calculation
- Singleton handling (true singleton + empty prediction = 1.0; false merge = 0.0)
- Macro-averaged evaluation across S1 entities
- Detailed diagnostic evaluation metrics
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np


BETA: float = 0.5
BETA2: float = 0.25  # beta^2 = 0.5^2 = 0.25
BETA2_PLUS_1: float = 1.25  # 1 + beta^2 = 1.25


@dataclass(frozen=True)
class EntityMetricResult:
    """Evaluation result for a single S1 entity."""
    entity_id: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f05: float
    is_true_singleton: bool
    is_pred_singleton: bool


@dataclass(frozen=True)
class EvaluationSummary:
    """Macro-level evaluation summary across all evaluation S1 entities."""
    macro_f05: float
    macro_precision: float
    macro_recall: float
    total_entities: int
    true_singletons: int
    pred_singletons: int
    correct_singletons: int
    singleton_accuracy: float
    false_merge_count: int
    false_merge_rate: float
    missed_non_singletons: int
    exact_match_entities: int
    exact_match_rate: float
    partial_match_entities: int
    zero_score_entities: int
    per_entity_results: Optional[List[EntityMetricResult]] = None


def f05_per_entity_reference(
    predicted: Union[Set[str], Iterable[str]],
    ground_truth: Union[Set[str], Iterable[str]],
) -> float:
    """Reference implementation of per-entity F0.5 score.
    
    Args:
        predicted: Predicted matched entity IDs (Source 2 / Source 3).
        ground_truth: True matched entity IDs.
        
    Returns:
        float: F0.5 score in range [0.0, 1.0].
    """
    pred_set = set(predicted) if not isinstance(predicted, set) else predicted
    true_set = set(ground_truth) if not isinstance(ground_truth, set) else ground_truth

    # Singleton Cases
    if not true_set:
        # True singleton: correct empty prediction gets 1.0, false merge gets 0.0
        return 1.0 if not pred_set else 0.0

    # Non-singleton cases
    if not pred_set:
        # Missed all matches for a true non-singleton
        return 0.0

    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0

    fp = len(pred_set) - tp
    fn = len(true_set) - tp

    numerator = BETA2_PLUS_1 * tp
    denominator = BETA2_PLUS_1 * tp + BETA2 * fn + fp

    return numerator / denominator if denominator > 0.0 else 0.0


def evaluate_entity_detailed(
    entity_id: str,
    predicted: Union[Set[str], Iterable[str]],
    ground_truth: Union[Set[str], Iterable[str]],
) -> EntityMetricResult:
    """Calculates detailed metrics (TP, FP, FN, Precision, Recall, F0.5) for one entity."""
    pred_set = set(predicted) if not isinstance(predicted, set) else predicted
    true_set = set(ground_truth) if not isinstance(ground_truth, set) else ground_truth

    is_true_singleton = len(true_set) == 0
    is_pred_singleton = len(pred_set) == 0

    if is_true_singleton:
        if is_pred_singleton:
            return EntityMetricResult(
                entity_id=entity_id,
                tp=0,
                fp=0,
                fn=0,
                precision=1.0,
                recall=1.0,
                f05=1.0,
                is_true_singleton=True,
                is_pred_singleton=True,
            )
        else:
            return EntityMetricResult(
                entity_id=entity_id,
                tp=0,
                fp=len(pred_set),
                fn=0,
                precision=0.0,
                recall=0.0,
                f05=0.0,
                is_true_singleton=True,
                is_pred_singleton=False,
            )

    if is_pred_singleton:
        return EntityMetricResult(
            entity_id=entity_id,
            tp=0,
            fp=0,
            fn=len(true_set),
            precision=0.0,
            recall=0.0,
            f05=0.0,
            is_true_singleton=False,
            is_pred_singleton=True,
        )

    tp = len(pred_set & true_set)
    fp = len(pred_set) - tp
    fn = len(true_set) - tp

    prec = tp / len(pred_set) if len(pred_set) > 0 else 0.0
    rec = tp / len(true_set) if len(true_set) > 0 else 0.0

    if tp == 0:
        f05 = 0.0
    else:
        num = BETA2_PLUS_1 * tp
        den = BETA2_PLUS_1 * tp + BETA2 * fn + fp
        f05 = num / den if den > 0.0 else 0.0

    return EntityMetricResult(
        entity_id=entity_id,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=prec,
        recall=rec,
        f05=f05,
        is_true_singleton=False,
        is_pred_singleton=False,
    )


def compute_macro_f05(
    predictions: Dict[str, Union[Set[str], List[str]]],
    ground_truth: Dict[str, Union[Set[str], List[str]]],
    include_detailed_results: bool = False,
) -> EvaluationSummary:
    """Computes Macro F0.5 and comprehensive diagnostics across all S1 entities in ground truth.
    
    Args:
        predictions: Mapping from source1_entity_id to predicted set/list of matched IDs.
        ground_truth: Mapping from source1_entity_id to true set/list of matched IDs.
        include_detailed_results: Whether to store per-entity results in the summary.
        
    Returns:
        EvaluationSummary with macro metrics and diagnostics.
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        return EvaluationSummary(
            macro_f05=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            total_entities=0,
            true_singletons=0,
            pred_singletons=0,
            correct_singletons=0,
            singleton_accuracy=0.0,
            false_merge_count=0,
            false_merge_rate=0.0,
            missed_non_singletons=0,
            exact_match_entities=0,
            exact_match_rate=0.0,
            partial_match_entities=0,
            zero_score_entities=0,
            per_entity_results=[] if include_detailed_results else None,
        )

    f05_scores: List[float] = []
    prec_scores: List[float] = []
    rec_scores: List[float] = []
    detailed_results: List[EntityMetricResult] = []

    true_singletons = 0
    pred_singletons = 0
    correct_singletons = 0
    false_merges = 0
    missed_non_singletons = 0
    exact_matches = 0
    partial_matches = 0
    zero_scores = 0

    for s1_id, true_ids in ground_truth.items():
        pred_ids = predictions.get(s1_id, set())
        result = evaluate_entity_detailed(s1_id, pred_ids, true_ids)

        f05_scores.append(result.f05)
        prec_scores.append(result.precision)
        rec_scores.append(result.recall)

        if include_detailed_results:
            detailed_results.append(result)

        if result.is_true_singleton:
            true_singletons += 1
            if result.is_pred_singleton:
                correct_singletons += 1
            else:
                false_merges += 1
        else:
            if result.is_pred_singleton:
                missed_non_singletons += 1

        if result.is_pred_singleton:
            pred_singletons += 1

        if result.f05 == 1.0:
            exact_matches += 1
        elif result.f05 == 0.0:
            zero_scores += 1
        else:
            partial_matches += 1

    macro_f05 = float(np.mean(f05_scores))
    macro_prec = float(np.mean(prec_scores))
    macro_rec = float(np.mean(rec_scores))

    singleton_acc = (
        correct_singletons / true_singletons if true_singletons > 0 else 1.0
    )
    false_merge_rate = (
        false_merges / true_singletons if true_singletons > 0 else 0.0
    )
    exact_match_rate = exact_matches / total_entities

    return EvaluationSummary(
        macro_f05=macro_f05,
        macro_precision=macro_prec,
        macro_recall=macro_rec,
        total_entities=total_entities,
        true_singletons=true_singletons,
        pred_singletons=pred_singletons,
        correct_singletons=correct_singletons,
        singleton_accuracy=singleton_acc,
        false_merge_count=false_merges,
        false_merge_rate=false_merge_rate,
        missed_non_singletons=missed_non_singletons,
        exact_match_entities=exact_matches,
        exact_match_rate=exact_match_rate,
        partial_match_entities=partial_matches,
        zero_score_entities=zero_scores,
        per_entity_results=detailed_results if include_detailed_results else None,
    )
