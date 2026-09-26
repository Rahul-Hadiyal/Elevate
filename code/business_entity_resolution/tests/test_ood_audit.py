"""Unit Tests for Phase 11: Out-of-Distribution Calibration Audit."""

import pytest
import numpy as np
from src.ood_audit import OODCalibrationAuditor, OODAuditResult
from src.calibration import ProbabilityCalibrator
from src.post_processor import PostProcessor, CandidatePrediction


def test_ood_audit_transfer_logic():
    """Verifies that OOD transfer audit correctly evaluates calibration delta."""
    np.random.seed(42)
    n_us = 200
    n_india = 150

    us_raw = np.random.uniform(0.1, 0.9, n_us)
    us_labels = (us_raw > 0.5).astype(np.int32)

    india_raw = np.random.uniform(0.1, 0.9, n_india)
    india_labels = (india_raw > 0.5).astype(np.int32)

    india_preds = [
        CandidatePrediction(
            s1_id=f"s1_{i%30}",
            cand_id=f"s2_{i}",
            prob=0.0,
            cand_source="S2",
        )
        for i in range(n_india)
    ]
    india_all_s1 = [f"s1_{i}" for i in range(30)]
    india_gt = {f"s1_{i}": {f"s2_{i}"} for i in range(30)}

    in_dist_calibrator = ProbabilityCalibrator(method="sigmoid")
    in_dist_calibrator.fit(us_raw, us_labels)

    post_processor = PostProcessor(base_threshold=0.50)
    auditor = OODCalibrationAuditor(max_allowed_degradation_pp=5.0)

    result = auditor.audit_transfer(
        us_cal_probs_raw=us_raw,
        us_cal_labels=us_labels,
        india_raw_probs=india_raw,
        india_labels=india_labels,
        india_candidate_preds=india_preds,
        india_all_s1_ids=india_all_s1,
        india_gt=india_gt,
        in_dist_calibrator=in_dist_calibrator,
        post_processor=post_processor,
    )

    assert isinstance(result, OODAuditResult)
    assert result.f05_delta_pp <= 5.0
    assert result.passed_gate is True
    assert result.india_ood_ece >= 0.0
