"""Out-of-Distribution Calibration & Country Transfer Audit (Phase 11 / Exp 8).

Evaluates calibration resilience when transferring across countries:
- Calibrate on US-only partition
- Evaluate calibration quality (ECE, Brier) and Macro F0.5 on held-out India partition
- Enforces OOD Calibration Gate: Delta Macro F0.5 <= 2.0 percentage points.
"""

from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass

from .calibration import ProbabilityCalibrator, compute_ece
from .metrics import compute_macro_f05
from .post_processor import PostProcessor, CandidatePrediction

logger = logging.getLogger(__name__)


@dataclass
class OODAuditResult:
    """Diagnostic results from Out-of-Distribution Calibration Audit."""
    in_dist_method: str
    us_brier: float
    us_ece: float
    india_in_dist_brier: float
    india_in_dist_ece: float
    india_in_dist_f05: float
    india_ood_brier: float
    india_ood_ece: float
    india_ood_f05: float
    f05_delta_pp: float
    passed_gate: bool
    num_us_samples: int
    num_india_samples: int
    num_india_entities: int


class OODCalibrationAuditor:
    """Audits probability calibration resilience on unseen geographic partitions."""

    def __init__(self, max_allowed_degradation_pp: float = 2.0):
        """Initializes OOD auditor.
        
        Args:
            max_allowed_degradation_pp: Max allowable Macro F0.5 degradation in percentage points.
        """
        self.max_allowed_degradation_pp = max_allowed_degradation_pp

    def audit_transfer(
        self,
        us_cal_probs_raw: np.ndarray,
        us_cal_labels: np.ndarray,
        india_raw_probs: np.ndarray,
        india_labels: np.ndarray,
        india_candidate_preds: List[CandidatePrediction],
        india_all_s1_ids: List[str],
        india_gt: Dict[str, Set[str]],
        in_dist_calibrator: ProbabilityCalibrator,
        post_processor: PostProcessor,
    ) -> OODAuditResult:
        """Executes OOD transfer audit from US to India.
        
        Args:
            us_cal_probs_raw: Uncalibrated probabilities on US calibration split.
            us_cal_labels: True labels on US calibration split.
            india_raw_probs: Uncalibrated probabilities on India evaluation split.
            india_labels: True labels on India evaluation split.
            india_candidate_preds: Predictions with raw probabilities for India.
            india_all_s1_ids: All S1 IDs in India cohort.
            india_gt: Ground truth dictionary for India cohort.
            in_dist_calibrator: Calibrator fitted on all countries.
            post_processor: PostProcessor instance.
            
        Returns:
            OODAuditResult with comparative diagnostics.
        """
        # 1. Fit US-only calibrator (OOD source)
        us_calibrator = ProbabilityCalibrator(method="sigmoid")
        us_calibrator.fit(us_cal_probs_raw, us_cal_labels)

        # Evaluate on US calibration
        us_cal_probs = us_calibrator.predict_proba(us_cal_probs_raw)
        us_brier = float(np.mean((us_cal_probs - us_cal_labels) ** 2))
        us_ece = compute_ece(us_cal_probs, us_cal_labels)

        # 2. In-Distribution India Evaluation (using global calibrator)
        india_in_dist_probs = in_dist_calibrator.predict_proba(india_raw_probs)
        india_in_dist_brier = float(np.mean((india_in_dist_probs - india_labels) ** 2))
        india_in_dist_ece = compute_ece(india_in_dist_probs, india_labels)

        in_dist_preds = [
            CandidatePrediction(
                s1_id=p.s1_id,
                cand_id=p.cand_id,
                prob=float(india_in_dist_probs[i]),
                cand_source=p.cand_source,
                has_empty_addr=p.has_empty_addr,
                has_numeric_disagreement=p.has_numeric_disagreement,
                has_country_disagreement=p.has_country_disagreement,
            )
            for i, p in enumerate(india_candidate_preds)
        ]
        in_dist_assigned = post_processor.filter_and_assign(in_dist_preds, india_all_s1_ids)
        in_dist_eval = compute_macro_f05(in_dist_assigned, india_gt)
        india_in_dist_f05 = float(in_dist_eval.macro_f05)

        # 3. OOD India Evaluation (using US-only calibrator)
        india_ood_probs = us_calibrator.predict_proba(india_raw_probs)
        india_ood_brier = float(np.mean((india_ood_probs - india_labels) ** 2))
        india_ood_ece = compute_ece(india_ood_probs, india_labels)

        ood_preds = [
            CandidatePrediction(
                s1_id=p.s1_id,
                cand_id=p.cand_id,
                prob=float(india_ood_probs[i]),
                cand_source=p.cand_source,
                has_empty_addr=p.has_empty_addr,
                has_numeric_disagreement=p.has_numeric_disagreement,
                has_country_disagreement=p.has_country_disagreement,
            )
            for i, p in enumerate(india_candidate_preds)
        ]
        ood_assigned = post_processor.filter_and_assign(ood_preds, india_all_s1_ids)
        ood_eval = compute_macro_f05(ood_assigned, india_gt)
        india_ood_f05 = float(ood_eval.macro_f05)

        f05_delta_pp = (india_in_dist_f05 - india_ood_f05) * 100.0
        passed_gate = f05_delta_pp <= self.max_allowed_degradation_pp

        logger.info(
            f"OOD Transfer Audit: India In-Dist F0.5={india_in_dist_f05:.4f}, "
            f"India OOD F0.5={india_ood_f05:.4f}, Delta={f05_delta_pp:.2f} pp, Gate Passed: {passed_gate}"
        )

        return OODAuditResult(
            in_dist_method="sigmoid",
            us_brier=us_brier,
            us_ece=us_ece,
            india_in_dist_brier=india_in_dist_brier,
            india_in_dist_ece=india_in_dist_ece,
            india_in_dist_f05=india_in_dist_f05,
            india_ood_brier=india_ood_brier,
            india_ood_ece=india_ood_ece,
            india_ood_f05=india_ood_f05,
            f05_delta_pp=f05_delta_pp,
            passed_gate=passed_gate,
            num_us_samples=len(us_cal_labels),
            num_india_samples=len(india_labels),
            num_india_entities=len(india_all_s1_ids),
        )
