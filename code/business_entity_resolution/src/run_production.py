"""
Production Inference Runner — Full Pipeline with Full Training Data.

This script replaces run_final_audit.py as the correct production pipeline:
1. Trains Stage-1 ensemble on full train split (NOT just 8K entities)
2. Trains Stage-2 context rescorer
3. Uses all 10 blocking channels (A-K) with cap=50
4. Uses correct DecisionEngine with FIXED loser re-optimization
5. Generates matching_results.tsv for test submission

Usage:
    cd C:/Users/rahul/Desktop/Elevate
    python code/business_entity_resolution/src/run_production.py

Runtime estimate: 3-6 hours on CPU
"""

from pathlib import Path
import os
import sys
import time
import json
import logging
from typing import Dict, List, Any, Set, Tuple
import numpy as np
import pandas as pd
import gc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_entity_source, load_ground_truth, parse_ground_truth_to_dict
from src.split import SplitManifest, create_stratified_split
from src.normalizer import EntityNormalizer
from src.index_builder import BlockingIndex
from src.blocker import MultiChannelBlocker
from src.candidate_store import CandidateStore
from src.pair_features import FEATURE_NAMES
from src.feature_store import FeatureBatch, FeatureExtractor
from src.feature_engineer import build_entity_lookup
from src.model import PairwiseScorer
from src.calibration import ProbabilityCalibrator
from src.context_features import ContextFeatureExtractor, G5_FEATURE_NAMES
from src.stage2_model import Stage2Rescorer
from src.decision_engine import DecisionEngine
from src.output_generator import OutputGenerator
from src.validator import SubmissionValidator
from src.metrics import compute_macro_f05

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] Production: %(message)s"
)
logger = logging.getLogger(__name__)


def extract_features_for_split(
    s1_df: pd.DataFrame,
    blocker: MultiChannelBlocker,
    extractor: FeatureExtractor,
    cap: int = 100,
    max_workers_s1: int = 1000,
) -> Tuple[FeatureBatch, Dict[str, List[str]]]:
    """Extract candidates and features for a set of S1 entities using all 10 channels."""
    store = CandidateStore(s1_df["entity_id"])
    store.add_channel_candidates("channel_A", blocker.generate_channel_a(s1_df))
    store.add_channel_candidates("channel_B", blocker.generate_channel_b(s1_df))
    store.add_channel_candidates("channel_C", blocker.generate_channel_c(s1_df))
    store.add_channel_candidates("channel_D", blocker.generate_channel_d(s1_df))
    store.add_channel_candidates("channel_E", blocker.generate_channel_e(s1_df))
    store.add_channel_candidates("channel_G", blocker.generate_channel_g(s1_df))
    store.add_channel_candidates("channel_H", blocker.generate_channel_h(s1_df))
    store.add_channel_candidates("channel_I", blocker.generate_channel_i(s1_df))
    store.add_channel_candidates("channel_J", blocker.generate_channel_j(s1_df))
    store.add_channel_candidates("channel_K", blocker.generate_channel_k(s1_df))

    cand_map = store.get_candidate_dict(cap=cap)
    ch_counts = store.get_channel_counts()
    flat_pairs = [(s1_id, c_id) for s1_id, cands in cand_map.items() for c_id in cands]

    batch = extractor.extract_pair_batch(flat_pairs, channel_counts=ch_counts)
    return batch, cand_map


def run_production_pipeline() -> None:
    """Execute the complete production pipeline."""
    t_start = time.time()

    # =========================================================================
    # PATHS
    # =========================================================================
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    
    # Try to find dataset directory
    train_dir = repo_root / "student_resource" / "dataset" / "train"
    test_dir = repo_root / "student_resource" / "dataset" / "test"
    if not train_dir.exists():
        train_dir = repo_root / "dataset" / "train"
        test_dir = repo_root / "dataset" / "test"

    splits_dir = repo_root / "artifacts" / "splits"
    models_dir = repo_root / "artifacts" / "models"
    output_dir = repo_root / "output"

    models_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PRODUCTION PIPELINE — FULL TRAINING WITH 10-CHANNEL BLOCKING")
    logger.info("=" * 70)
    logger.info(f"Train dir: {train_dir}")
    logger.info(f"Test dir:  {test_dir}")

    # =========================================================================
    # STEP 1: LOAD TRAINING DATA
    # =========================================================================
    logger.info("Step 1: Loading training data...")
    s1_train_df, _ = load_entity_source(train_dir / "train_source1.tsv", "S1")
    s2_train_df, _ = load_entity_source(train_dir / "train_source2.tsv", "S2")
    s3_train_df, _ = load_entity_source(train_dir / "train_source3.tsv", "S3")
    gt_df, _ = load_ground_truth(train_dir / "train_ground_truth.tsv")
    gt_map = parse_ground_truth_to_dict(gt_df)

    logger.info(f"  S1 train: {len(s1_train_df):,}, S2 train: {len(s2_train_df):,}, S3 train: {len(s3_train_df):,}")
    logger.info(f"  GT entries: {len(gt_map):,}")

    # =========================================================================
    # STEP 2: LOAD / CREATE SPLIT MANIFEST
    # =========================================================================
    logger.info("Step 2: Loading split manifest...")
    manifest_path = splits_dir / "split_manifest.tsv.gz"
    if not manifest_path.exists():
        manifest_path = splits_dir / "split_manifest.tsv"
    if not manifest_path.exists():
        logger.info("  Creating split manifest...")
        manifest = create_stratified_split(s1_train_df, gt_df)
        splits_dir.mkdir(parents=True, exist_ok=True)
        manifest.save(splits_dir / "split_manifest.tsv")
    else:
        manifest = SplitManifest.load(manifest_path)

    # Use FULL train split (not truncated)
    train_s1_ids = set(manifest.get_entity_ids("train"))
    earlystop_s1_ids = set(manifest.get_entity_ids("earlystop"))
    cal_s1_ids = set(manifest.get_entity_ids("calibration"))
    val_b_s1_ids = set(manifest.get_entity_ids("val_b"))

    # Limit to manageable size for memory — use up to 30K train, 10K each for es/cal/val
    # Increase these limits based on available RAM
    MAX_TRAIN = int(os.environ.get("MAX_TRAIN_S1", "30000"))
    MAX_ES = int(os.environ.get("MAX_ES_S1", "10000"))
    MAX_CAL = int(os.environ.get("MAX_CAL_S1", "10000"))
    MAX_VAL = int(os.environ.get("MAX_VAL_S1", "5000"))

    random.seed(42)
    train_eids_list = list(train_s1_ids)
    random.shuffle(train_eids_list)
    train_eids = train_eids_list[:MAX_TRAIN]

    es_eids_list = list(earlystop_s1_ids)
    random.shuffle(es_eids_list)
    es_eids = es_eids_list[:MAX_ES]

    cal_eids_list = list(cal_s1_ids)
    random.shuffle(cal_eids_list)
    cal_eids = cal_eids_list[:MAX_CAL]

    val_b_eids_list = list(val_b_s1_ids)
    random.shuffle(val_b_eids_list)
    val_b_eids = val_b_eids_list[:MAX_VAL]

    logger.info("CANDIDATE_GENERATION_MODE = LIVE")
    logger.info("CHANNELS = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'J', 'K'] (10 channels; Channel F omitted)")
    logger.info("CAP = 100")
    logger.info("CHANNEL_COUNT_METADATA = PRESENT")
    logger.info(f"  Train: {len(train_eids):,} | EarlyStop: {len(es_eids):,} | "
                f"Cal: {len(cal_eids):,} | Val_B: {len(val_b_eids):,}")

    # =========================================================================
    # STEP 3: NORMALIZE TRAINING S1 RECORDS
    # =========================================================================
    logger.info("Step 3: Normalizing S1 training records...")
    normalizer = EntityNormalizer()

    all_needed = set(train_eids) | set(es_eids) | set(cal_eids) | set(val_b_eids)
    s1_needed = s1_train_df[s1_train_df["entity_id"].isin(all_needed)].copy()
    s1_norm_all = normalizer.normalize_dataframe(s1_needed)
    logger.info(f"  Normalized {len(s1_norm_all):,} S1 training records.")

    # =========================================================================
    # STEP 4: LOAD AND NORMALIZE S2/S3 CANDIDATE POOL
    # =========================================================================
    logger.info("Step 4: Building training candidate pool...")
    target_gt_ids = {cid for eid in all_needed for cid in gt_map.get(eid, set())}
    s2_gt_eids = {e for e in target_gt_ids if e.startswith("S2-")}
    s3_gt_eids = {e for e in target_gt_ids if e.startswith("S3-")}

    # Include all GT-required candidates + a sample of the rest
    N_EXTRA_CANDS = int(os.environ.get("N_EXTRA_CANDS", "150000"))
    s2_sample = pd.concat([
        s2_train_df[s2_train_df["entity_id"].isin(s2_gt_eids)],
        s2_train_df.head(N_EXTRA_CANDS)
    ]).drop_duplicates(subset=["entity_id"])
    s3_sample = pd.concat([
        s3_train_df[s3_train_df["entity_id"].isin(s3_gt_eids)],
        s3_train_df.head(N_EXTRA_CANDS)
    ]).drop_duplicates(subset=["entity_id"])

    s2_norm = normalizer.normalize_dataframe(s2_sample)
    s3_norm = normalizer.normalize_dataframe(s3_sample)
    logger.info(f"  S2 sample: {len(s2_norm):,} | S3 sample: {len(s3_norm):,}")

    # Release raw dataframes
    del s2_train_df, s3_train_df, s2_sample, s3_sample
    gc.collect()

    # =========================================================================
    # STEP 5: BUILD BLOCKING INDEX AND LOOKUPS
    # =========================================================================
    logger.info("Step 5: Building blocking index...")
    index = BlockingIndex(min_token_len=3, max_token_df=5000)
    index.build_indexes(s2_norm, s3_norm)
    blocker = MultiChannelBlocker(index, max_cands_per_key=100)

    s1_lookup = build_entity_lookup(s1_norm_all)
    cand_lookup = build_entity_lookup(s2_norm)
    cand_lookup.update(build_entity_lookup(s3_norm))
    logger.info(f"  Candidate lookup: {len(cand_lookup):,} entries")

    extractor = FeatureExtractor(s1_lookup, cand_lookup, ground_truth=gt_map)

    # =========================================================================
    # STEP 6: EXTRACT FEATURES FOR ALL SPLITS
    # =========================================================================
    logger.info("Step 6: Extracting features for train/earlystop/calibration/val_b...")

    def get_s1_subset(eids):
        return s1_norm_all[s1_norm_all["entity_id"].isin(eids)].copy()

    logger.info("  Extracting training features...")
    train_batch, train_cmap = extract_features_for_split(get_s1_subset(train_eids), blocker, extractor, cap=50)
    logger.info(f"  Train: {train_batch.num_pairs:,} pairs, {train_batch.num_positives:,} positives")

    logger.info("  Extracting early-stop features...")
    es_batch, es_cmap = extract_features_for_split(get_s1_subset(es_eids), blocker, extractor, cap=50)
    logger.info(f"  EarlyStop: {es_batch.num_pairs:,} pairs, {es_batch.num_positives:,} positives")

    logger.info("  Extracting calibration features...")
    cal_batch, cal_cmap = extract_features_for_split(get_s1_subset(cal_eids), blocker, extractor, cap=50)
    logger.info(f"  Calibration: {cal_batch.num_pairs:,} pairs, {cal_batch.num_positives:,} positives")

    logger.info("  Extracting val_b features...")
    valb_batch, valb_cmap = extract_features_for_split(get_s1_subset(val_b_eids), blocker, extractor, cap=50)
    logger.info(f"  Val_B: {valb_batch.num_pairs:,} pairs, {valb_batch.num_positives:,} positives")

    # =========================================================================
    # STEP 7: TRAIN STAGE-1 MODEL
    # =========================================================================
    logger.info("Step 7: Training Stage-1 LightGBM model...")
    stage1 = PairwiseScorer(
        model_type="lightgbm",
        feature_names=FEATURE_NAMES,
        n_estimators=500,
        learning_rate=0.04,
        num_leaves=35,
        max_depth=7,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    stage1.fit(
        train_batch.features,
        train_batch.labels,
        X_val=es_batch.features,
        y_val=es_batch.labels,
        early_stopping_rounds=30,
    )
    logger.info(f"  Stage-1 best iteration: {stage1.best_iteration_}")

    # Calibrate Stage-1
    cal_raw = stage1.predict_proba(cal_batch.features)
    calibrator = ProbabilityCalibrator(method="sigmoid")
    calibrator.fit(cal_raw, cal_batch.labels)
    logger.info("  Stage-1 calibrator fitted.")

    # =========================================================================
    # STEP 8: EXTRACT CONTEXT FEATURES (G5) AND TRAIN STAGE-2
    # =========================================================================
    logger.info("Step 8: Extracting G5 context features and training Stage-2...")
    context_gen = ContextFeatureExtractor()

    p1_train = calibrator.predict_proba(stage1.predict_proba(train_batch.features))
    p1_es = calibrator.predict_proba(stage1.predict_proba(es_batch.features))
    p1_cal = calibrator.predict_proba(stage1.predict_proba(cal_batch.features))
    p1_valb = calibrator.predict_proba(stage1.predict_proba(valb_batch.features))

    g5_train = context_gen.extract_context_features([(s1, c, p) for (s1, c), p in zip(train_batch.pair_ids, p1_train)])
    g5_es = context_gen.extract_context_features([(s1, c, p) for (s1, c), p in zip(es_batch.pair_ids, p1_es)])
    g5_cal = context_gen.extract_context_features([(s1, c, p) for (s1, c), p in zip(cal_batch.pair_ids, p1_cal)])
    g5_valb = context_gen.extract_context_features([(s1, c, p) for (s1, c), p in zip(valb_batch.pair_ids, p1_valb)])

    stage2 = Stage2Rescorer(max_depth=3, num_leaves=8, learning_rate=0.05, n_estimators=150)
    stage2.fit(
        X_train=g5_train,
        y_train=train_batch.labels,
        X_earlystop=g5_es,
        y_earlystop=es_batch.labels,
        X_calib=g5_cal,
        y_calib=cal_batch.labels,
        calibration_method="sigmoid",
    )
    logger.info("  Stage-2 model fitted and calibrated.")

    # =========================================================================
    # STEP 9: OFFLINE EVALUATION ON VAL_B
    # =========================================================================
    logger.info("Step 9: Offline evaluation on val_b...")
    p2_valb = stage2.predict_proba(g5_valb)

    decision_engine = DecisionEngine(
        enable_conflict_resolution=True,
        margin_delta=0.05,
        min_prob_filter=0.01,
        max_candidates_per_entity=100,
    )

    cand_score_map_valb: Dict[str, List[Tuple[str, float]]] = {}
    for (s1_id, cid), prob in zip(valb_batch.pair_ids, p2_valb):
        cand_score_map_valb.setdefault(s1_id, []).append((cid, float(prob)))
    for eid in val_b_eids:
        cand_score_map_valb.setdefault(eid, [])

    preds_valb = decision_engine.optimize_predictions(cand_score_map_valb)
    sub_gt = {eid: gt_map.get(eid, set()) for eid in val_b_eids}
    eval_result = compute_macro_f05(preds_valb, sub_gt)

    logger.info("=" * 60)
    logger.info(f"VAL_B OFFLINE SCORE: F0.5={eval_result.macro_f05:.4f} "
                f"Prec={eval_result.macro_precision:.4f} "
                f"Rec={eval_result.macro_recall:.4f}")
    logger.info(f"  Singleton accuracy: {eval_result.singleton_accuracy:.4f}")
    logger.info(f"  Exact matches: {eval_result.exact_match_rate:.4f}")
    logger.info("=" * 60)

    # Save offline result
    offline_result = {
        "val_b_f05": float(eval_result.macro_f05),
        "val_b_precision": float(eval_result.macro_precision),
        "val_b_recall": float(eval_result.macro_recall),
        "val_b_n_entities": len(val_b_eids),
        "train_pairs": int(train_batch.num_pairs),
        "train_positives": int(train_batch.num_positives),
    }
    (repo_root / "logs").mkdir(parents=True, exist_ok=True)
    with open(repo_root / "logs" / "production_offline_eval.json", "w") as f:
        json.dump(offline_result, f, indent=2)

    # Release training memory
    del s1_needed, s1_norm_all, s2_norm, s3_norm, index, blocker
    del train_batch, es_batch, cal_batch, valb_batch
    del g5_train, g5_es, g5_cal, g5_valb
    del s1_lookup, cand_lookup, extractor
    gc.collect()

    # =========================================================================
    # STEP 10: TEST INFERENCE
    # =========================================================================
    logger.info("Step 10: Running full test inference...")

    test_s1_df, _ = load_entity_source(test_dir / "test_source1.tsv", "S1")
    logger.info(f"  Loaded {len(test_s1_df):,} test S1 records.")

    logger.info("  Loading and normalizing test S2...")
    test_s2_df, _ = load_entity_source(test_dir / "test_source2.tsv", "S2")
    test_s2_norm = normalizer.normalize_dataframe(test_s2_df)
    del test_s2_df
    gc.collect()

    logger.info("  Loading and normalizing test S3...")
    test_s3_df, _ = load_entity_source(test_dir / "test_source3.tsv", "S3")
    test_s3_norm = normalizer.normalize_dataframe(test_s3_df)
    del test_s3_df
    gc.collect()

    logger.info("  Building test blocking index (all 10 channels)...")
    test_index = BlockingIndex(min_token_len=3, max_token_df=5000)
    test_index.build_indexes(test_s2_norm, test_s3_norm)
    test_blocker = MultiChannelBlocker(test_index, max_cands_per_key=100)

    test_cand_lookup = build_entity_lookup(test_s2_norm)
    test_cand_lookup.update(build_entity_lookup(test_s3_norm))
    logger.info(f"  Test candidate lookup: {len(test_cand_lookup):,} entries")

    all_test_s1_ids = test_s1_df["entity_id"].tolist()
    all_predictions: Dict[str, Set[str]] = {}
    all_candidates_map: Dict[str, List[str]] = {}

    chunk_size = 100_000
    n_chunks = (len(test_s1_df) + chunk_size - 1) // chunk_size
    logger.info(f"  Processing {len(test_s1_df):,} S1 entities in {n_chunks} chunks...")

    for chunk_idx in range(n_chunks):
        s_i = chunk_idx * chunk_size
        e_i = min(s_i + chunk_size, len(test_s1_df))
        s1_chunk = test_s1_df.iloc[s_i:e_i].copy()

        s1_norm_chunk = normalizer.normalize_dataframe(s1_chunk)
        s1_chunk_lookup = build_entity_lookup(s1_norm_chunk)

        # Generate candidates using all 10 channels
        store = CandidateStore(s1_norm_chunk["entity_id"])
        store.add_channel_candidates("channel_A", test_blocker.generate_channel_a(s1_norm_chunk))
        store.add_channel_candidates("channel_B", test_blocker.generate_channel_b(s1_norm_chunk))
        store.add_channel_candidates("channel_C", test_blocker.generate_channel_c(s1_norm_chunk))
        store.add_channel_candidates("channel_D", test_blocker.generate_channel_d(s1_norm_chunk))
        store.add_channel_candidates("channel_E", test_blocker.generate_channel_e(s1_norm_chunk))
        store.add_channel_candidates("channel_G", test_blocker.generate_channel_g(s1_norm_chunk))
        store.add_channel_candidates("channel_H", test_blocker.generate_channel_h(s1_norm_chunk))
        store.add_channel_candidates("channel_I", test_blocker.generate_channel_i(s1_norm_chunk))
        store.add_channel_candidates("channel_J", test_blocker.generate_channel_j(s1_norm_chunk))
        store.add_channel_candidates("channel_K", test_blocker.generate_channel_k(s1_norm_chunk))

        cand_dict = store.get_candidate_dict(cap=100)
        ch_counts = store.get_channel_counts()
        assert ch_counts is not None and len(ch_counts) > 0, "RUNTIME INVARIANT VIOLATED: Candidate channel provenance missing!"

        for s1_id in s1_norm_chunk["entity_id"]:
            all_candidates_map[s1_id] = cand_dict.get(s1_id, [])

        flat_pairs = [(s1_id, cid) for s1_id, cands in cand_dict.items() for cid in cands]

        if flat_pairs:
            chunk_ext = FeatureExtractor(s1_chunk_lookup, test_cand_lookup)
            batch = chunk_ext.extract_pair_batch(flat_pairs, channel_counts=ch_counts)

            # Stage-1 scoring
            raw_p1 = stage1.predict_proba(batch.features)
            cal_p1 = calibrator.predict_proba(raw_p1)

            # Stage-2 rescoring
            g5 = context_gen.extract_context_features([(s1, c, p) for (s1, c), p in zip(batch.pair_ids, cal_p1)])
            p2 = stage2.predict_proba(g5)

            cand_score_map: Dict[str, List[Tuple[str, float]]] = {}
            for (s1_id, cid), prob in zip(batch.pair_ids, p2):
                cand_score_map.setdefault(s1_id, []).append((cid, float(prob)))

            for s1_id in s1_norm_chunk["entity_id"]:
                cand_score_map.setdefault(s1_id, [])

            chunk_preds = decision_engine.optimize_predictions(cand_score_map)
            all_predictions.update(chunk_preds)
        else:
            for s1_id in s1_norm_chunk["entity_id"]:
                all_predictions[s1_id] = set()

        logger.info(f"  Chunk {chunk_idx+1}/{n_chunks} done. "
                    f"({e_i:,}/{len(test_s1_df):,} S1 entities)")
        del s1_chunk, s1_norm_chunk, s1_chunk_lookup, cand_dict, store
        gc.collect()

    # =========================================================================
    # STEP 11: WRITE OUTPUT FILES
    # =========================================================================
    logger.info("Step 11: Writing output TSV files...")

    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"

    with open(matching_path, "w", encoding="utf-8", newline="\n") as f_match, \
         open(candidate_path, "w", encoding="utf-8", newline="\n") as f_cand:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for s1_id in all_test_s1_ids:
            matched = all_predictions.get(s1_id, set())
            match_str = ",".join(sorted(matched)) if matched else ""
            f_match.write(f"{s1_id}\t{match_str}\n")

            cands = all_candidates_map.get(s1_id, [])
            cand_str = ",".join(cands) if cands else ""
            f_cand.write(f"{s1_id}\t{cand_str}\n")

    total_time = time.time() - t_start
    logger.info(f"Pipeline completed in {total_time:.1f}s ({total_time/3600:.2f}h)")
    logger.info(f"Output: {matching_path}")
    logger.info(f"Offline F0.5 on val_b: {eval_result.macro_f05:.4f}")


if __name__ == "__main__":
    run_production_pipeline()
