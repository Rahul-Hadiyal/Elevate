"""Official Validation Evaluator Module for Business Entity Resolution.

Computes the competition objective: Macro F0.5 per S1 entity, alongside
comprehensive diagnostic subgroups:
- By Country (US, India, France, etc.)
- By Cardinality (Singletons/0-match, 1-match, 2-5 matches, 6+ matches)
- By Match Pattern (S2-only, S3-only, Both S2+S3, None)
- Error distribution (Exact match rate, Zero score rate, False merges, Missed links)
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import json
import logging
import numpy as np
import pandas as pd

try:
    from src.metrics import (
        EvaluationSummary,
        EntityMetricResult,
        compute_macro_f05,
        evaluate_entity_detailed,
    )
    from src.split import get_cardinality_bucket, get_match_pattern
except ImportError:
    from metrics import (
        EvaluationSummary,
        EntityMetricResult,
        compute_macro_f05,
        evaluate_entity_detailed,
    )
    from split import get_cardinality_bucket, get_match_pattern



logger = logging.getLogger(__name__)


@dataclass
class SubgroupMetrics:
    """Metrics aggregated for a specific subgroup of S1 entities."""
    group_name: str
    group_value: str
    entity_count: int
    macro_f05: float
    macro_precision: float
    macro_recall: float
    exact_match_rate: float
    zero_score_rate: float
    singleton_accuracy: Optional[float] = None
    false_merge_rate: Optional[float] = None


@dataclass
class FullEvaluationReport:
    """Complete evaluation report with macro metrics, subgroup metrics, and percentiles."""
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
    exact_match_entities: int
    exact_match_rate: float
    zero_score_entities: int
    zero_score_rate: float
    percentiles: Dict[str, float]
    subgroups: Dict[str, List[SubgroupMetrics]]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


    def to_dict(self) -> Dict[str, Any]:
        """Converts report to dictionary."""
        return {
            "macro_f05": round(self.macro_f05, 6),
            "macro_precision": round(self.macro_precision, 6),
            "macro_recall": round(self.macro_recall, 6),
            "total_entities": self.total_entities,
            "true_singletons": self.true_singletons,
            "pred_singletons": self.pred_singletons,
            "correct_singletons": self.correct_singletons,
            "singleton_accuracy": round(self.singleton_accuracy, 6),
            "false_merge_count": self.false_merge_count,
            "false_merge_rate": round(self.false_merge_rate, 6),
            "exact_match_entities": self.exact_match_entities,
            "exact_match_rate": round(self.exact_match_rate, 6),
            "zero_score_entities": self.zero_score_entities,
            "zero_score_rate": round(self.zero_score_rate, 6),
            "percentiles": {k: round(v, 6) for k, v in self.percentiles.items()},
            "subgroups": {
                k: [asdict(m) for m in v] for k, v in self.subgroups.items()
            },
            "created_at": self.created_at,
        }

    def to_markdown(self, title: str = "Official F0.5 Validation Report") -> str:
        """Generates a clean markdown table summarizing all metrics and subgroups."""
        lines = [
            f"# {title}",
            f"*Generated at: {self.created_at} UTC*",
            "",
            "## 1. Primary Macro Metrics",
            "",
            "| Metric | Value | Description |",
            "|:---|:---|:---|",
            f"| **Macro F0.5** | **{self.macro_f05:.6f}** | **Official Competition Metric** |",
            f"| Macro Precision | {self.macro_precision:.6f} | Mean per-entity precision |",
            f"| Macro Recall | {self.macro_recall:.6f} | Mean per-entity recall |",
            f"| Total S1 Entities | {self.total_entities:,} | Evaluated entities count |",
            f"| Exact Match Rate | {self.exact_match_rate * 100:.2f}% ({self.exact_match_entities:,}) | F0.5 = 1.0 (exact match) |",
            f"| Zero Score Rate | {self.zero_score_rate * 100:.2f}% ({self.zero_score_entities:,}) | F0.5 = 0.0 (complete failure) |",
            "",
            "## 2. Singleton & False Merge Diagnostics",
            "",
            "| Diagnostic | Value | Description |",
            "|:---|:---|:---|",
            f"| True Singletons | {self.true_singletons:,} ({self.true_singletons / max(1, self.total_entities) * 100:.2f}%) | Ground-truth 0-match entities |",
            f"| Predicted Singletons | {self.pred_singletons:,} ({self.pred_singletons / max(1, self.total_entities) * 100:.2f}%) | Predicted 0-match entities |",
            f"| Singleton Accuracy | {self.singleton_accuracy * 100:.2f}% | True singletons correctly predicted empty |",
            f"| False Merge Count | {self.false_merge_count:,} ({self.false_merge_rate * 100:.2f}%) | True singletons incorrectly given matches |",
            "",
            "## 3. Score Percentile Distribution",
            "",
            "| Percentile | F0.5 Score |",
            "|:---|:---|",
            f"| Min (P0) | {self.percentiles.get('p00', 0.0):.6f} |",
            f"| P25 | {self.percentiles.get('p25', 0.0):.6f} |",
            f"| Median (P50) | {self.percentiles.get('p50', 0.0):.6f} |",
            f"| P75 | {self.percentiles.get('p75', 0.0):.6f} |",
            f"| P90 | {self.percentiles.get('p90', 0.0):.6f} |",
            f"| Max (P100) | {self.percentiles.get('p100', 0.0):.6f} |",
            f"| Mean | {self.percentiles.get('mean', 0.0):.6f} |",
            f"| Std Dev | {self.percentiles.get('std', 0.0):.6f} |",
            "",
            "## 4. Subgroup Breakdowns",
            "",
        ]

        for group_name, group_metrics in self.subgroups.items():
            lines.append(f"### Subgroup: {group_name.replace('_', ' ').title()}")
            lines.append("")
            lines.append("| Category | Entities | Share (%) | Macro F0.5 | Precision | Recall | Exact Match | Zero Score |")
            lines.append("|:---|---:|---:|---:|---:|---:|---:|---:|")
            for m in group_metrics:
                share = (m.entity_count / max(1, self.total_entities)) * 100.0
                lines.append(
                    f"| `{m.group_value}` | {m.entity_count:,} | {share:.1f}% | "
                    f"**{m.macro_f05:.4f}** | {m.macro_precision:.4f} | {m.macro_recall:.4f} | "
                    f"{m.exact_match_rate * 100:.1f}% | {m.zero_score_rate * 100:.1f}% |"
                )
            lines.append("")

        return "\n".join(lines)

    def save(self, output_path: Union[str, Path]) -> None:
        """Saves report to JSON or Markdown depending on file extension."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
        elif path.suffix == ".md":
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.to_markdown())
        else:
            raise ValueError(f"Unsupported output format: {path.suffix}")


class ValidationEvaluator:
    """Evaluator for evaluating entity resolution models against ground truth."""

    def __init__(self, ground_truth: Union[pd.DataFrame, Dict[str, Union[Set[str], List[str]]]]):
        """Initializes evaluator with ground truth mapping.

        Args:
            ground_truth: Ground truth mapping (Dict or DataFrame).
        """
        if isinstance(ground_truth, pd.DataFrame):
            self.ground_truth: Dict[str, Set[str]] = {}
            for _, row in ground_truth.iterrows():
                s1_id = str(row["source1_entity_id"]).strip()
                raw_matched = row.get("matched_entity_ids")
                if pd.isna(raw_matched) or raw_matched is None or str(raw_matched).strip() == "":
                    self.ground_truth[s1_id] = set()
                elif isinstance(raw_matched, str):
                    self.ground_truth[s1_id] = {m.strip() for m in raw_matched.split(",") if m.strip()}
                elif isinstance(raw_matched, (list, set, tuple)):
                    self.ground_truth[s1_id] = {str(m).strip() for m in raw_matched if str(m).strip()}
                else:
                    self.ground_truth[s1_id] = set()
        elif isinstance(ground_truth, dict):
            self.ground_truth = {k: set(v) for k, v in ground_truth.items()}
        else:
            raise TypeError(f"Unsupported ground truth type: {type(ground_truth)}")

    def evaluate(
        self,
        predictions: Union[pd.DataFrame, Dict[str, Union[Set[str], List[str]]]],
        s1_metadata: Optional[pd.DataFrame] = None,
    ) -> FullEvaluationReport:
        """Evaluates predictions and computes macro metrics and subgroup breakdowns.

        Args:
            predictions: Predicted matches (Dict or DataFrame).
            s1_metadata: Optional DataFrame with `entity_id`, `country` for metadata slicing.

        Returns:
            FullEvaluationReport with metrics and subgroups.
        """
        # Parse predictions into dict
        pred_map: Dict[str, Set[str]] = {}
        if isinstance(predictions, pd.DataFrame):
            for _, row in predictions.iterrows():
                s1_id = str(row["source1_entity_id"]).strip()
                raw_matched = row.get("matched_entity_ids")
                if pd.isna(raw_matched) or raw_matched is None or str(raw_matched).strip() == "":
                    pred_map[s1_id] = set()
                elif isinstance(raw_matched, str):
                    pred_map[s1_id] = {m.strip() for m in raw_matched.split(",") if m.strip()}
                elif isinstance(raw_matched, (list, set, tuple)):
                    pred_map[s1_id] = {str(m).strip() for m in raw_matched if str(m).strip()}
                else:
                    pred_map[s1_id] = set()
        elif isinstance(predictions, dict):
            pred_map = {k: set(v) for k, v in predictions.items()}
        else:
            raise TypeError(f"Unsupported predictions type: {type(predictions)}")

        # Build metadata lookup
        country_lookup: Dict[str, str] = {}
        if s1_metadata is not None and "entity_id" in s1_metadata.columns:
            for _, row in s1_metadata.iterrows():
                eid = str(row["entity_id"]).strip()
                country = str(row.get("country", "UNKNOWN")).strip().upper()
                c_norm = "US" if country in ("US", "USA", "UNITED STATES") else (
                    "IN" if country in ("IN", "IND", "INDIA") else (
                        "FR" if country in ("FR", "FRA", "FRANCE") else "OTHER"
                    )
                )
                country_lookup[eid] = c_norm

        # Compute per-entity metrics
        summary = compute_macro_f05(
            predictions=pred_map,
            ground_truth=self.ground_truth,
            include_detailed_results=True,
        )

        detailed_results = summary.per_entity_results or []
        f05_scores = np.array([r.f05 for r in detailed_results], dtype=float) if detailed_results else np.array([0.0])

        # Compute score percentiles
        percentiles = {
            "p00": float(np.min(f05_scores)),
            "p25": float(np.percentile(f05_scores, 25)),
            "p50": float(np.median(f05_scores)),
            "p75": float(np.percentile(f05_scores, 75)),
            "p90": float(np.percentile(f05_scores, 90)),
            "p100": float(np.max(f05_scores)),
            "mean": float(np.mean(f05_scores)),
            "std": float(np.std(f05_scores)),
        }

        # Subgroups: Country, Cardinality, Match Pattern
        subgroups_dict: Dict[str, List[SubgroupMetrics]] = {}

        # 1. Cardinality subgroup
        card_groups: Dict[str, List[EntityMetricResult]] = {}
        pattern_groups: Dict[str, List[EntityMetricResult]] = {}
        country_groups: Dict[str, List[EntityMetricResult]] = {}

        for r in detailed_results:
            true_ids = self.ground_truth.get(r.entity_id, set())
            card_bucket = get_cardinality_bucket(len(true_ids))
            pattern = get_match_pattern(true_ids)
            country = country_lookup.get(r.entity_id, "UNKNOWN")

            card_groups.setdefault(card_bucket, []).append(r)
            pattern_groups.setdefault(pattern, []).append(r)
            country_groups.setdefault(country, []).append(r)

        def _aggregate_subgroup(name: str, group_dict: Dict[str, List[EntityMetricResult]]) -> List[SubgroupMetrics]:
            results = []
            for val in sorted(group_dict.keys()):
                group_res = group_dict[val]
                cnt = len(group_res)
                if cnt == 0:
                    continue
                f05_list = [x.f05 for x in group_res]
                prec_list = [x.precision for x in group_res]
                rec_list = [x.recall for x in group_res]
                exact_cnt = sum(1 for x in group_res if x.f05 == 1.0)
                zero_cnt = sum(1 for x in group_res if x.f05 == 0.0)

                # Singleton stats if applicable
                true_singletons = sum(1 for x in group_res if x.is_true_singleton)
                correct_singletons = sum(1 for x in group_res if x.is_true_singleton and x.is_pred_singleton)
                false_merges = sum(1 for x in group_res if x.is_true_singleton and not x.is_pred_singleton)

                sing_acc = (correct_singletons / true_singletons) if true_singletons > 0 else None
                fm_rate = (false_merges / true_singletons) if true_singletons > 0 else None

                results.append(
                    SubgroupMetrics(
                        group_name=name,
                        group_value=val,
                        entity_count=cnt,
                        macro_f05=float(np.mean(f05_list)),
                        macro_precision=float(np.mean(prec_list)),
                        macro_recall=float(np.mean(rec_list)),
                        exact_match_rate=exact_cnt / cnt,
                        zero_score_rate=zero_cnt / cnt,
                        singleton_accuracy=sing_acc,
                        false_merge_rate=fm_rate,
                    )
                )
            return results

        subgroups_dict["cardinality"] = _aggregate_subgroup("cardinality", card_groups)
        subgroups_dict["match_pattern"] = _aggregate_subgroup("match_pattern", pattern_groups)
        if country_lookup:
            subgroups_dict["country"] = _aggregate_subgroup("country", country_groups)

        zero_score_rate = summary.zero_score_entities / summary.total_entities if summary.total_entities > 0 else 0.0

        return FullEvaluationReport(
            macro_f05=summary.macro_f05,
            macro_precision=summary.macro_precision,
            macro_recall=summary.macro_recall,
            total_entities=summary.total_entities,
            true_singletons=summary.true_singletons,
            pred_singletons=summary.pred_singletons,
            correct_singletons=summary.correct_singletons,
            singleton_accuracy=summary.singleton_accuracy,
            false_merge_count=summary.false_merge_count,
            false_merge_rate=summary.false_merge_rate,
            exact_match_entities=summary.exact_match_entities,
            exact_match_rate=summary.exact_match_rate,
            zero_score_entities=summary.zero_score_entities,
            zero_score_rate=zero_score_rate,
            percentiles=percentiles,
            subgroups=subgroups_dict,
        )
