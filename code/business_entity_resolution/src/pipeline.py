"""Main Reproducible Pipeline Entry Point for Business Entity Resolution (Phase 14).

Executes end-to-end data loading, normalization, candidate generation (blocking),
feature engineering, two-stage model scoring, exact expected-F0.5 DP selection,
margin-guarded conflict resolution, and TSV output generation satisfying all 8 invariants.
"""

from pathlib import Path
import os
import sys
import time
import argparse
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_entity_source, load_ground_truth, parse_ground_truth_to_dict
from src.normalizer import EntityNormalizer
from src.index_builder import BlockingIndex
from src.blocker import MultiChannelBlocker
from src.candidate_store import CandidateStore
from src.feature_store import FeatureBatch, FeatureExtractor
from src.feature_engineer import build_entity_lookup
from src.model import PairwiseScorer
from src.calibration import ProbabilityCalibrator
from src.ensemble import EnsemblePairwiseScorer, EnsembleStage2Rescorer
from src.context_features import ContextFeatureExtractor
from src.decision_engine import DecisionEngine
from src.output_generator import OutputGenerator
from src.validator import SubmissionValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] Pipeline: %(message)s"
)
logger = logging.getLogger(__name__)


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    mode: str = "test",
    max_cands_per_s1: int = 20,
    chunk_size: int = 100000,
) -> None:
    """Executes the complete entity resolution pipeline."""
    t0_start = time.time()
    logger.info("=" * 70)
    logger.info(f"STARTING BUSINESS ENTITY RESOLUTION PIPELINE (MODE: {mode.upper()})")
    logger.info("=" * 70)

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    models_dir = repo_root / "artifacts" / "models"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalizer = EntityNormalizer()
    context_gen = ContextFeatureExtractor()
    decision_engine = DecisionEngine(enable_conflict_resolution=True, margin_delta=0.05)
    output_gen = OutputGenerator(output_dir=output_dir)
    validator = SubmissionValidator(test_dir=data_dir / "test" if mode == "test" else data_dir / "train")

    if mode == "test":
        test_dir = data_dir / "test"
        logger.info(f"Loading Test Sources from {test_dir}...")
        s1_df_raw, _ = load_entity_source(test_dir / "test_source1.tsv", "S1")
        s2_df_raw, _ = load_entity_source(test_dir / "test_source2.tsv", "S2")
        s3_df_raw, _ = load_entity_source(test_dir / "test_source3.tsv", "S3")

        logger.info(f"Loaded {len(s1_df_raw):,d} S1, {len(s2_df_raw):,d} S2, {len(s3_df_raw):,d} S3 records.")
        all_test_s1_ids = s1_df_raw["entity_id"].tolist()

        logger.info("Normalizing candidate sources (S2, S3)...")
        s2_norm = normalizer.normalize_dataframe(s2_df_raw)
        s3_norm = normalizer.normalize_dataframe(s3_df_raw)

        cand_lookup = build_entity_lookup(s2_norm)
        cand_lookup.update(build_entity_lookup(s3_norm))

        logger.info("Building Test Blocking Index...")
        blocking_index = BlockingIndex(min_token_len=3, max_token_df=5000)
        blocking_index.build_indexes(s2_norm, s3_norm)
        blocker = MultiChannelBlocker(blocking_index, max_cands_per_key=100)

        # Load trained models
        stage1_path = models_dir / "ensemble_stage1_model.joblib"
        stage2_path = models_dir / "ensemble_stage2_model.joblib"

        if stage1_path.exists() and stage2_path.exists():
            logger.info("Loading trained Ensemble Stage-1 and Stage-2 models...")
            stage1_model = EnsemblePairwiseScorer.load(stage1_path)
            stage2_model = EnsembleStage2Rescorer.load(stage2_path)
        else:
            logger.info("Loading single Stage-1 and Stage-2 models...")
            from src.stage2_model import Stage2Rescorer
            stage1_model = PairwiseScorer()  # fallback
            stage2_model = Stage2Rescorer()

        # Process S1 entities in streaming chunks
        n_chunks = (len(s1_df_raw) + chunk_size - 1) // chunk_size
        logger.info(f"Processing S1 entities in {n_chunks} chunks of {chunk_size:,d}...")

        all_candidates_map: Dict[str, List[str]] = {}
        all_matches_map: Dict[str, Set[str]] = {}

        for chunk_idx in range(n_chunks):
            start_i = chunk_idx * chunk_size
            end_i = min(len(s1_df_raw), (chunk_idx + 1) * chunk_size)
            chunk_df = s1_df_raw.iloc[start_i:end_i].copy()
            logger.info(f"Processing Chunk {chunk_idx + 1}/{n_chunks} ({len(chunk_df):,d} S1 entities)...")

            chunk_norm = normalizer.normalize_dataframe(chunk_df)
            chunk_s1_lookup = build_entity_lookup(chunk_norm)
            extractor = FeatureExtractor(chunk_s1_lookup, cand_lookup)

            # Generate candidate channels
            store = CandidateStore()
            store.add_channel_candidates("channel_A", blocker.generate_channel_a(chunk_norm))
            store.add_channel_candidates("channel_B", blocker.generate_channel_b(chunk_norm))
            store.add_channel_candidates("channel_C", blocker.generate_channel_c(chunk_norm))
            store.add_channel_candidates("channel_D", blocker.generate_channel_d(chunk_norm))
            store.add_channel_candidates("channel_E", blocker.generate_channel_e(chunk_norm))
            store.add_channel_candidates("channel_G", blocker.generate_channel_g(chunk_norm))
            store.add_channel_candidates("channel_H", blocker.generate_channel_h(chunk_norm))

            chunk_cand_map = store.get_candidate_dict(cap=max_cands_per_s1)
            for s1_id in chunk_df["entity_id"]:
                if s1_id not in chunk_cand_map:
                    chunk_cand_map[s1_id] = []
                all_candidates_map[s1_id] = chunk_cand_map[s1_id]

            # Extract features and predict
            flat_pairs = [(s1_id, cid) for s1_id, cands in chunk_cand_map.items() for cid in cands]
            if flat_pairs:
                batch = extractor.extract_pair_batch(flat_pairs)
                p1_scores = stage1_model.predict_proba(batch.features)
                g5_df = context_gen.extract_context_features([(s1, c, p) for (s1, c), p in zip(batch.pair_ids, p1_scores)])
                p2_scores = stage2_model.predict_proba(g5_df)

                cand_score_map: Dict[str, List[Tuple[str, float]]] = {}
                for (s1_id, cid), prob in zip(batch.pair_ids, p2_scores):
                    if s1_id not in cand_score_map:
                        cand_score_map[s1_id] = []
                    cand_score_map[s1_id].append((cid, float(prob)))

                for s1_id in chunk_df["entity_id"]:
                    if s1_id not in cand_score_map:
                        cand_score_map[s1_id] = []

                chunk_preds = decision_engine.optimize_predictions(cand_score_map)
                all_matches_map.update(chunk_preds)
            else:
                for s1_id in chunk_df["entity_id"]:
                    all_matches_map[s1_id] = set()

        # Write output TSV files
        match_tsv, cand_tsv = output_gen.write_submission_files(
            all_s1_ids=all_test_s1_ids,
            candidate_dict=all_candidates_map,
            matching_dict=all_matches_map,
        )

        # Validate Invariants locally and via official script
        logger.info("Step 9: Validating Output Invariants...")
        validator.validate_invariants(
            matching_path=match_tsv,
            candidate_path=cand_tsv,
            expected_s1_ids=set(all_test_s1_ids),
            enforce_h1=True,
        )
        validator.run_official_validator(
            matching_path=match_tsv,
            candidate_path=cand_tsv,
        )

    logger.info(f"Pipeline finished successfully in {time.time() - t0_start:.2f} seconds.")


def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "test"], default="test")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--chunk-size", type=int, default=100000)
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
