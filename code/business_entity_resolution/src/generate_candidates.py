"""Candidate Pairs Generator Script for Competition Packaging (Phase 14/15).

Generates output/candidate_pairs.tsv for all 1,732,544 test S1 entities using
multi-channel blocking (Channels A-H), with streaming line-by-line disk writes.

Usage:
    python -u code/business_entity_resolution/src/generate_candidates.py
"""

from pathlib import Path
import os
import sys
import time
import logging
from typing import Dict, List, Set, Tuple
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_entity_source
from src.normalizer import EntityNormalizer
from src.index_builder import BlockingIndex
from src.blocker import MultiChannelBlocker
from src.candidate_store import CandidateStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] CandidatesGen: %(message)s"
)
logger = logging.getLogger(__name__)


def generate_candidate_pairs_file(
    test_dir: Path = Path("dataset/test"),
    output_path: Path = Path("output/candidate_pairs.tsv"),
    matching_path: Path = Path("output/matching_results.tsv"),
    chunk_size: int = 100000,
    max_cands_per_s1: int = 25,
) -> None:
    """Streams candidate_pairs.tsv directly to disk using multi-channel blocking."""
    t0_start = time.time()
    logger.info("=" * 70)
    logger.info("CANDIDATE PAIRS (BLOCKING) GENERATION PIPELINE")
    logger.info("=" * 70)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load existing matching results if available to guarantee subset invariant
    matched_dict: Dict[str, Set[str]] = {}
    if matching_path.exists():
        logger.info(f"Loading existing matching results from {matching_path} to enforce subset invariant...")
        with open(matching_path, "r", encoding="utf-8") as f:
            next(f, None)  # Skip header
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) == 2 and parts[1].strip():
                    s1_id = parts[0].strip()
                    matched_ids = set([cid.strip() for cid in parts[1].split(",") if cid.strip()])
                    matched_dict[s1_id] = matched_ids

    # 2. Ingest Test S1, S2, S3
    logger.info(f"Loading test source files from {test_dir}...")
    s1_df_all, _ = load_entity_source(test_dir / "test_source1.tsv", "S1")
    s2_df, _ = load_entity_source(test_dir / "test_source2.tsv", "S2")
    s3_df, _ = load_entity_source(test_dir / "test_source3.tsv", "S3")

    logger.info(f"Loaded: {len(s1_df_all):,d} S1, {len(s2_df):,d} S2, {len(s3_df):,d} S3 records.")

    normalizer = EntityNormalizer()
    logger.info("Normalizing candidate sources (S2, S3)...")
    s2_norm = normalizer.normalize_dataframe(s2_df)
    s3_norm = normalizer.normalize_dataframe(s3_df)

    logger.info("Building multi-channel BlockingIndex...")
    blocking_index = BlockingIndex(min_token_len=3, max_token_df=5000)
    blocking_index.build_indexes(s2_norm, s3_norm)
    blocker = MultiChannelBlocker(blocking_index, max_cands_per_key=100)

    # 3. Stream candidate pairs chunk-by-chunk to disk
    total_s1 = len(s1_df_all)
    num_chunks = (total_s1 + chunk_size - 1) // chunk_size
    logger.info(f"Processing {total_s1:,d} entities in {num_chunks} chunks of {chunk_size:,d}...")

    total_pairs_written = 0

    with open(output_path, "w", encoding="utf-8", newline="\n") as out_f:
        out_f.write("source1_entity_id\tcandidate_entity_ids\n")

        for chunk_idx in range(num_chunks):
            t_chunk_0 = time.time()
            s_i = chunk_idx * chunk_size
            e_i = min(total_s1, (chunk_idx + 1) * chunk_size)
            s1_chunk = s1_df_all.iloc[s_i:e_i].copy()

            s1_norm_chunk = normalizer.normalize_dataframe(s1_chunk)
            store = CandidateStore()

            # Multi-channel candidate generation
            store.add_channel_candidates("channel_A", blocker.generate_channel_a(s1_norm_chunk))
            store.add_channel_candidates("channel_B", blocker.generate_channel_b(s1_norm_chunk))
            store.add_channel_candidates("channel_C", blocker.generate_channel_c(s1_norm_chunk))
            store.add_channel_candidates("channel_D", blocker.generate_channel_d(s1_norm_chunk))
            store.add_channel_candidates("channel_E", blocker.generate_channel_e(s1_norm_chunk))
            store.add_channel_candidates("channel_G", blocker.generate_channel_g(s1_norm_chunk))
            store.add_channel_candidates("channel_H", blocker.generate_channel_h(s1_norm_chunk))

            cand_map = store.get_candidate_dict(cap=max_cands_per_s1)

            # Write chunk rows
            for s1_id in s1_chunk["entity_id"]:
                cands = set(cand_map.get(s1_id, []))
                # Union with matched predictions to guarantee subset property
                if s1_id in matched_dict:
                    cands |= matched_dict[s1_id]

                cand_list = sorted(list(cands))
                total_pairs_written += len(cand_list)
                cand_str = ",".join(cand_list) if cand_list else ""
                out_f.write(f"{s1_id}\t{cand_str}\n")

            logger.info(
                f"  Chunk {chunk_idx + 1}/{num_chunks} processed "
                f"({e_i:,d}/{total_s1:,d} entities) in {time.time() - t_chunk_0:.2f}s."
            )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    total_time = time.time() - t0_start
    logger.info("=" * 70)
    logger.info(f"CANDIDATE GENERATION COMPLETED: {output_path}")
    logger.info(f"Total S1 Entities: {total_s1:,d}")
    logger.info(f"Total Candidate Pairs: {total_pairs_written:,d}")
    logger.info(f"File Size: {file_size_mb:.2f} MB")
    logger.info(f"Total Runtime: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    logger.info("=" * 70)


if __name__ == "__main__":
    generate_candidate_pairs_file()
