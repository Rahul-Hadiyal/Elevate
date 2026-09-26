"""Hypothesis H1 Gate Module.

Empirically tests Hypothesis H1:
"Each S2 / S3 record belongs to at most one S1 entity."

Evaluates ground truth to detect any multi-claimant records (records claimed by >= 2 S1 entities),
reporting separate statistics for Source 2 and Source 3, and gating downstream conflict resolution.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging
import pandas as pd

from src.data_loader import extract_reverse_mapping


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceH1Stats:
    """H1 statistics for a single source (S2 or S3)."""
    source_tag: str
    total_distinct_matched_ids: int
    single_claimant_count: int
    multiple_claimants_count: int
    violation_rate: float
    violating_examples: Dict[str, List[str]]


@dataclass(frozen=True)
class H1GateResult:
    """Comprehensive result of the Phase 2.5 Hypothesis H1 Gate."""
    is_h1_confirmed: bool
    total_distinct_matched_ids: int
    total_single_claimant_ids: int
    total_multi_claimant_ids: int
    overall_violation_rate: float
    s2_stats: SourceH1Stats
    s3_stats: SourceH1Stats
    conflict_resolution_enabled: bool
    invariant_8_enforced: bool
    all_violating_ids: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, output_path: Union[str, Path]) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"H1 Gate report saved to {path}")


def evaluate_h1_gate(
    ground_truth_df: pd.DataFrame,
    output_json_path: Optional[Union[str, Path]] = None,
) -> H1GateResult:
    """Evaluates Hypothesis H1 against the ground truth DataFrame.
    
    Args:
        ground_truth_df: DataFrame with 'source1_entity_id' and 'matched_entity_ids'.
        output_json_path: Optional path to persist logs/h1_gate.json.
        
    Returns:
        H1GateResult with complete findings and component activation flags.
    """
    reverse_map = extract_reverse_mapping(ground_truth_df)

    s2_single = 0
    s2_multi = 0
    s2_violating: Dict[str, List[str]] = {}

    s3_single = 0
    s3_multi = 0
    s3_violating: Dict[str, List[str]] = {}

    all_violating: Dict[str, List[str]] = {}

    for mid, s1_set in reverse_map.items():
        s1_list = sorted(list(s1_set))
        if mid.startswith("S2-"):
            if len(s1_set) == 1:
                s2_single += 1
            else:
                s2_multi += 1
                s2_violating[mid] = s1_list
                all_violating[mid] = s1_list
        elif mid.startswith("S3-"):
            if len(s1_set) == 1:
                s3_single += 1
            else:
                s3_multi += 1
                s3_violating[mid] = s1_list
                all_violating[mid] = s1_list
        else:
            # Non-standard ID prefix
            if len(s1_set) > 1:
                all_violating[mid] = s1_list

    total_s2 = s2_single + s2_multi
    total_s3 = s3_single + s3_multi
    total_matched = total_s2 + total_s3

    s2_rate = s2_multi / total_s2 if total_s2 > 0 else 0.0
    s3_rate = s3_multi / total_s3 if total_s3 > 0 else 0.0
    overall_rate = len(all_violating) / total_matched if total_matched > 0 else 0.0

    is_h1_confirmed = (len(all_violating) == 0)

    s2_stats = SourceH1Stats(
        source_tag="S2",
        total_distinct_matched_ids=total_s2,
        single_claimant_count=s2_single,
        multiple_claimants_count=s2_multi,
        violation_rate=s2_rate,
        violating_examples=dict(list(s2_violating.items())[:10]),
    )

    s3_stats = SourceH1Stats(
        source_tag="S3",
        total_distinct_matched_ids=total_s3,
        single_claimant_count=s3_single,
        multiple_claimants_count=s3_multi,
        violation_rate=s3_rate,
        violating_examples=dict(list(s3_violating.items())[:10]),
    )

    result = H1GateResult(
        is_h1_confirmed=is_h1_confirmed,
        total_distinct_matched_ids=total_matched,
        total_single_claimant_ids=s2_single + s3_single,
        total_multi_claimant_ids=len(all_violating),
        overall_violation_rate=overall_rate,
        s2_stats=s2_stats,
        s3_stats=s3_stats,
        conflict_resolution_enabled=is_h1_confirmed,
        invariant_8_enforced=is_h1_confirmed,
        all_violating_ids=all_violating,
    )

    if output_json_path:
        result.save(output_json_path)

    return result
