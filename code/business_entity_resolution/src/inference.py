"""Inference and Submission Generation Pipeline for Business Entity Resolution.

Provides memory-efficient chunked batch prediction, probability scoring,
post-processing, and formatting of submission files meeting all challenge requirements.
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Iterator
import time
import logging
import numpy as np
import pandas as pd

from src.normalizer import EntityNormalizer
from src.index_builder import BlockingIndex
from src.blocker import MultiChannelBlocker
from src.candidate_store import CandidateStore
from src.feature_store import FeatureExtractor
from src.feature_engineer import build_entity_lookup
from src.model import PairwiseScorer
from src.calibration import ProbabilityCalibrator
from src.post_processor import PostProcessor, CandidatePrediction
from src.decision_engine import DecisionEngine
from typing import Union

logger = logging.getLogger(__name__)


def generate_candidates_for_batch(
    s1_norm_chunk: pd.DataFrame,
    blocker: MultiChannelBlocker,
    cap: int = 50,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], int]]:
    """Generate candidates for a chunk of normalized S1 records with multi-channel ranking."""
    cands_a = blocker.generate_channel_a(s1_norm_chunk)
    cands_b = blocker.generate_channel_b(s1_norm_chunk)
    cands_c = blocker.generate_channel_c(s1_norm_chunk)
    cands_d = blocker.generate_channel_d(s1_norm_chunk)
    cands_e = blocker.generate_channel_e(s1_norm_chunk)
    cands_g = blocker.generate_channel_g(s1_norm_chunk)
    cands_h = blocker.generate_channel_h(s1_norm_chunk)
    cands_i = blocker.generate_channel_i(s1_norm_chunk)
    cands_j = blocker.generate_channel_j(s1_norm_chunk)
    cands_k = blocker.generate_channel_k(s1_norm_chunk)

    store = CandidateStore(s1_norm_chunk["entity_id"])
    for ch_name, ch_cands in [
        ("Channel_A", cands_a), ("Channel_B", cands_b), ("Channel_C", cands_c),
        ("Channel_D", cands_d), ("Channel_E", cands_e), ("Channel_G", cands_g),
        ("Channel_H", cands_h), ("Channel_I", cands_i), ("Channel_J", cands_j),
        ("Channel_K", cands_k)
    ]:
        store.add_channel_candidates(ch_name, ch_cands)

    return store.get_candidate_dict(cap=cap), store.get_channel_counts()


def run_batch_inference(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    scorer: PairwiseScorer,
    calibrator: ProbabilityCalibrator,
    post_processor: Optional[Union[PostProcessor, DecisionEngine]] = None,
    decision_engine: Optional[DecisionEngine] = None,
    chunk_size: int = 100000,
    cap_cands_per_s1: int = 50,
) -> pd.DataFrame:
    """Run full entity resolution inference across test datasets."""
    normalizer = EntityNormalizer()

    logger.info("Normalizing Source 2 and Source 3 candidate datasets...")
    s2_norm = normalizer.normalize_dataframe(s2_df)
    s3_norm = normalizer.normalize_dataframe(s3_df)

    logger.info("Building Candidate Blocking Indexes on S2 and S3...")
    index = BlockingIndex(min_token_len=3, max_token_df=5000)
    index.build_indexes(s2_norm, s3_norm)
    blocker = MultiChannelBlocker(index, max_cands_per_key=100)

    logger.info("Building candidate entity attribute lookups...")
    cand_lookup = build_entity_lookup(s2_norm)
    cand_lookup.update(build_entity_lookup(s3_norm))

    total_s1 = len(s1_df)
    num_chunks = int(np.ceil(total_s1 / chunk_size))
    logger.info(f"Processing {total_s1:,d} Source 1 entities in {num_chunks} chunks of {chunk_size:,d}...")

    all_predictions: Dict[str, Set[str]] = {}

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_s1)
        s1_chunk = s1_df.iloc[start_idx:end_idx].copy()

        logger.info(f"  Chunk {chunk_idx + 1}/{num_chunks}: Normalizing {len(s1_chunk):,d} S1 records...")
        s1_norm_chunk = normalizer.normalize_dataframe(s1_chunk)
        s1_lookup = build_entity_lookup(s1_norm_chunk)

        # Generate candidates
        logger.info(f"  Chunk {chunk_idx + 1}/{num_chunks}: Generating candidates via multi-channel blocker...")
        cand_dict, ch_counts = generate_candidates_for_batch(s1_norm_chunk, blocker, cap=cap_cands_per_s1)

        pair_list = [(s1_id, cid) for s1_id, cands in cand_dict.items() for cid in cands]
        logger.info(f"  Chunk {chunk_idx + 1}/{num_chunks}: Extracted {len(pair_list):,d} candidate pairs.")

        if not pair_list:
            for s1_id in s1_norm_chunk["entity_id"]:
                all_predictions[s1_id] = set()
            continue

        # Extract features
        extractor = FeatureExtractor(s1_lookup, cand_lookup)
        batch = extractor.extract_pair_batch(pair_list, channel_counts=ch_counts)

        # Score with model
        raw_probs = scorer.predict_proba(batch.features)
        cal_probs = calibrator.predict_proba(raw_probs)

        active_engine = decision_engine if decision_engine is not None else (post_processor if isinstance(post_processor, DecisionEngine) else None)
        if active_engine is not None:
            entity_cand_map: Dict[str, List[Tuple[str, float]]] = {}
            for i, (s1_id, cid) in enumerate(batch.pair_ids):
                entity_cand_map.setdefault(s1_id, []).append((cid, float(cal_probs[i])))

            # Every S1 entity in the chunk must be represented, including
            # entities with zero candidates -> they yield an empty prediction.
            for s1_id in s1_norm_chunk["entity_id"]:
                entity_cand_map.setdefault(s1_id, [])

            chunk_preds = active_engine.optimize_predictions(entity_cand_map)
            all_predictions.update(chunk_preds)
        else:
            # Extract feature indicators for post-processing guards
            feat_names = batch.feature_names
            idx_s1_empty = feat_names.index("feat_s1_addr_is_empty") if "feat_s1_addr_is_empty" in feat_names else -1
            idx_cand_empty = feat_names.index("feat_cand_addr_is_empty") if "feat_cand_addr_is_empty" in feat_names else -1
            idx_num_disagree = feat_names.index("feat_addr_num_disagreement") if "feat_addr_num_disagreement" in feat_names else -1
            idx_country_match = feat_names.index("feat_country_exact_match") if "feat_country_exact_match" in feat_names else -1
            idx_country_s1_m = feat_names.index("feat_country_s1_missing") if "feat_country_s1_missing" in feat_names else -1
            idx_country_cand_m = feat_names.index("feat_country_cand_missing") if "feat_country_cand_missing" in feat_names else -1

            cand_preds: List[CandidatePrediction] = []
            for i, (s1_id, cid) in enumerate(batch.pair_ids):
                p = float(cal_probs[i])
                src = "S2" if (cid.startswith("S2-") or cid.startswith("S2_")) else "S3"
                
                s1_empty = bool(batch.features[i, idx_s1_empty] > 0.5) if idx_s1_empty >= 0 else False
                c_empty = bool(batch.features[i, idx_cand_empty] > 0.5) if idx_cand_empty >= 0 else False
                num_disagree = bool(batch.features[i, idx_num_disagree] > 0.5) if idx_num_disagree >= 0 else False
                
                c_match = bool(batch.features[i, idx_country_match] > 0.5) if idx_country_match >= 0 else True
                cs1_m = bool(batch.features[i, idx_country_s1_m] > 0.5) if idx_country_s1_m >= 0 else False
                cc_m = bool(batch.features[i, idx_country_cand_m] > 0.5) if idx_country_cand_m >= 0 else False
                country_disagree = (not c_match) and (not cs1_m) and (not cc_m)

                cand_preds.append(CandidatePrediction(
                    s1_id=s1_id,
                    cand_id=cid,
                    prob=p,
                    cand_source=src,
                    has_empty_addr=s1_empty or c_empty,
                    has_numeric_disagreement=num_disagree,
                    has_country_disagreement=country_disagree,
                ))

            # Apply post-processor
            proc = post_processor if isinstance(post_processor, PostProcessor) else PostProcessor()
            chunk_preds = proc.filter_and_assign(cand_preds, all_s1_ids=s1_norm_chunk["entity_id"].tolist())
            all_predictions.update(chunk_preds)

    # Format into DataFrame
    rows = []
    for s1_id in s1_df["entity_id"]:
        matched = all_predictions.get(s1_id, set())
        # Sort matched IDs for canonical determinism
        matched_str = ",".join(sorted(list(matched))) if matched else ""
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": matched_str,
        })

    return pd.DataFrame(rows)
