"""Unit tests for inference and submission pipeline."""

import pytest
import numpy as np
import pandas as pd
from src.inference import run_batch_inference
from src.model import PairwiseScorer
from src.calibration import ProbabilityCalibrator
from src.post_processor import PostProcessor
from src.pair_features import FEATURE_NAMES


def test_batch_inference_pipeline():
    s1_df = pd.DataFrame([
        {"entity_id": "S1-1", "business_name": "Google LLC", "business_address": "1600 Amphitheatre Pkwy", "country": "US"},
        {"entity_id": "S1-2", "business_name": "Apple Inc", "business_address": "1 Apple Park Way", "country": "US"},
        {"entity_id": "S1-3", "business_name": "Unknown Entity X", "business_address": "", "country": "US"},
    ])
    s2_df = pd.DataFrame([
        {"entity_id": "S2-1", "business_name": "Google", "business_address": "1600 Amphitheatre Parkway", "country": "US"},
    ])
    s3_df = pd.DataFrame([
        {"entity_id": "S3-1", "business_name": "Apple Corp", "business_address": "1 Apple Park", "country": "US"},
    ])

    # Simple mock model that always returns high probability
    class DummyScorer:
        def predict_proba(self, X):
            return np.ones(len(X), dtype=np.float32) * 0.95

    class DummyCalibrator:
        def predict_proba(self, X):
            return X

    pp = PostProcessor(base_threshold=0.60)
    sub_df = run_batch_inference(
        s1_df, s2_df, s3_df,
        scorer=DummyScorer(),
        calibrator=DummyCalibrator(),
        post_processor=pp,
        chunk_size=2,
    )

    assert len(sub_df) == 3
    assert list(sub_df.columns) == ["source1_entity_id", "matched_entity_ids"]
    assert sub_df.loc[sub_df["source1_entity_id"] == "S1-1", "matched_entity_ids"].iloc[0] == "S2-1"
    assert sub_df.loc[sub_df["source1_entity_id"] == "S1-2", "matched_entity_ids"].iloc[0] == "S3-1"
    assert sub_df.loc[sub_df["source1_entity_id"] == "S1-3", "matched_entity_ids"].iloc[0] == ""
