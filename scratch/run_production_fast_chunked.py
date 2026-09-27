#!/usr/bin/env python3
"""Fast Indexed In-Memory Production Inference Runner.

Generates full-corpus predictions for 1,732,544 S1 entities across 41.5M candidate pairs
by keeping raw candidate data indexed in-memory and building feature-precomputed representations
on-demand per 100k chunk. This prevents OS page thrashing and sustains 25,000+ pairs/sec.
"""

import sys, os, time, math, psutil, gc, json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd
import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'code' / 'business_entity_resolution'))

from src.data_loader import load_entity_source
from src.normalizer import EntityNormalizer
from src.feature_engineer import build_entity_lookup
from src.decision_engine import DecisionEngine
from scratch.test_precomputed_63_features import (
    PrecomputedEntityV2, compute_all_63_pair_features
)

TOTAL_ENTITIES = 1732544
CHUNK_SIZE = 100000

def get_checkpoint_path() -> Path:
    return REPO_ROOT / 'artifacts' / 'inference_chunks' / 'inference_checkpoint.json'

def load_checkpoint() -> Set[int]:
    ckpt_path = get_checkpoint_path()
    if ckpt_path.exists():
        try:
            with open(ckpt_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data.get('completed_chunks', []))
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}")
    return set()

def save_checkpoint(completed: Set[int]):
    ckpt_path = get_checkpoint_path()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ckpt_path, 'w', encoding='utf-8') as f:
        json.dump({'completed_chunks': sorted(list(completed))}, f, indent=2)

def run_production_inference():
    proc = psutil.Process()
    test_dir = REPO_ROOT / 'student_resource' / 'dataset' / 'test'
    output_dir = REPO_ROOT / 'output'
    models_dir = REPO_ROOT / 'artifacts' / 'models'
    chunks_dir = REPO_ROOT / 'artifacts' / 'inference_chunks'
    subs_dir = REPO_ROOT / 'artifacts' / 'submissions'
    chunks_dir.mkdir(parents=True, exist_ok=True)
    subs_dir.mkdir(parents=True, exist_ok=True)

    cand_pairs_path = output_dir / 'candidate_pairs.tsv'

    print("=" * 80)
    print("FAST INDEXED IN-MEMORY FULL PRODUCTION INFERENCE")
    print(f"Total S1 Entities:     {TOTAL_ENTITIES:,}")
    print(f"Chunk Size:            {CHUNK_SIZE:,}")
    total_chunks = math.ceil(TOTAL_ENTITIES / CHUNK_SIZE)
    print(f"Total Chunks:          {total_chunks}")
    print("=" * 80, flush=True)

    # 1. Load models upfront
    print("[1/4] Loading models and calibrator...", flush=True)
    t0 = time.time()
    model = joblib.load(models_dir / 'production_scorer_v2.joblib')
    calib = joblib.load(models_dir / 'production_calibrator_v2.joblib')
    print(f"      Models loaded in {time.time()-t0:.2f}s", flush=True)

    # 2. Load Raw Candidate Index into memory (one-time)
    print("[2/4] Loading Raw Candidate Index (S2 + S3)...", flush=True)
    t0 = time.time()
    s2_df, _ = load_entity_source(test_dir / 'test_source2.tsv', 'S2')
    s3_df, _ = load_entity_source(test_dir / 'test_source3.tsv', 'S3')
    cand_raw_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    cand_raw_df.set_index('entity_id', inplace=True)
    gc.collect()
    print(f"      Raw Candidate Index Ready: {len(cand_raw_df):,} rows in {time.time()-t0:.2f}s (RAM: {proc.memory_info().rss/1e6:.1f} MB)", flush=True)

    # 3. Load Candidate Pairs Map (one-time)
    print("[3/4] Loading Candidate Pairs Map...", flush=True)
    t0 = time.time()
    cand_map: Dict[str, List[str]] = {}
    with open(cand_pairs_path, 'r', encoding='utf-8') as f:
        next(f) # Skip header
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) == 2 and parts[1]:
                cand_map[parts[0]] = [cid.strip() for cid in parts[1].split(',') if cid.strip()][:150]
    print(f"      Candidate Pairs Loaded: {len(cand_map):,} entities in {time.time()-t0:.2f}s (RAM: {proc.memory_info().rss/1e6:.1f} MB)", flush=True)

    # 4. Load S1 Full DataFrame (one-time)
    print("[4/4] Loading S1 Full DataFrame...", flush=True)
    t0 = time.time()
    s1_full_df, _ = load_entity_source(test_dir / 'test_source1.tsv', 'S1')
    print(f"      S1 Loaded: {len(s1_full_df):,} rows in {time.time()-t0:.2f}s (RAM: {proc.memory_info().rss/1e6:.1f} MB)", flush=True)

    completed_chunks = load_checkpoint()
    print(f"\nExisting Completed Chunks: {sorted(list(completed_chunks))}", flush=True)

    normalizer = EntityNormalizer()
    engine = DecisionEngine(
        enable_conflict_resolution=True,
        margin_delta=0.05,
        min_prob_filter=0.01,
        max_candidates_per_entity=50,
    )

    t_pipeline_start = time.time()
    total_processed_pairs = 0

    for chunk_idx in range(total_chunks):
        chunk_file = chunks_dir / f"chunk_{chunk_idx:03d}.tsv"
        if chunk_idx in completed_chunks and chunk_file.exists():
            print(f"\n--> Chunk {chunk_idx + 1}/{total_chunks} [ALREADY COMPLETED - SKIPPING]", flush=True)
            continue

        s_i = chunk_idx * CHUNK_SIZE
        e_i = min(s_i + CHUNK_SIZE, TOTAL_ENTITIES)
        chunk_n_s1 = e_i - s_i

        print(f"\n================================================================================", flush=True)
        print(f"--> Starting Chunk {chunk_idx + 1}/{total_chunks}: S1 rows [{s_i:,} .. {e_i:,}] ({chunk_n_s1:,} entities)", flush=True)
        print(f"================================================================================", flush=True)
        t_chunk_start = time.time()

        # Step A: Slice chunk S1 and candidate pairs
        s1_chunk_df = s1_full_df.iloc[s_i:e_i].copy()
        s1_ids = list(s1_chunk_df['entity_id'])

        pair_list = [(s1, c) for s1 in s1_ids for c in cand_map.get(s1, [])]
        needed_cands = {c for _, c in pair_list}
        n_pairs = len(pair_list)
        total_processed_pairs += n_pairs
        print(f"  [A] Pairs extracted: {n_pairs:,} candidate pairs ({len(needed_cands):,} unique candidates)", flush=True)

        # Step B: Normalize chunk entities & candidates on demand
        t0 = time.time()
        s1_norm = normalizer.normalize_dataframe(s1_chunk_df)
        s1_lookup = build_entity_lookup(s1_norm)
        del s1_chunk_df, s1_norm

        common_cands = cand_raw_df.index.intersection(needed_cands)
        cand_sub_df = cand_raw_df.loc[common_cands].reset_index()
        cand_norm = normalizer.normalize_dataframe(cand_sub_df)
        cand_lookup = build_entity_lookup(cand_norm)
        del cand_sub_df, cand_norm
        print(f"  [B] Normalization completed in {time.time()-t0:.2f}s ({len(s1_lookup):,} S1 + {len(cand_lookup):,} Cands)", flush=True)

        # Step C: Precompute entity representations
        t0 = time.time()
        name_df = Counter()
        for row in cand_lookup.values():
            for t in set(row.get('name_norm', '').split()):
                name_df[t] += 1
        total_docs = len(cand_lookup) + len(s1_lookup)
        log_total = math.log(max(1000, total_docs))
        token_idf_dict = {t: max(0.1, log_total - math.log(df)) for t, df in name_df.items()}

        s1_pre = {eid: PrecomputedEntityV2(row, name_df) for eid, row in s1_lookup.items()}
        cand_pre = {eid: PrecomputedEntityV2(row, name_df) for eid, row in cand_lookup.items()}
        del s1_lookup, cand_lookup
        print(f"  [C] PrecomputedEntityV2 built in {time.time()-t0:.2f}s (RAM: {proc.memory_info().rss/1e6:.1f} MB)", flush=True)

        # Step D: Extract 63 features
        t0 = time.time()
        feat_mat = np.zeros((n_pairs, 63), dtype=np.float32)
        for p_i, (s1_id, cid) in enumerate(pair_list):
            s1_obj = s1_pre.get(s1_id)
            c_obj = cand_pre.get(cid)
            if s1_obj and c_obj:
                feat_mat[p_i] = compute_all_63_pair_features(
                    s1_obj, c_obj, token_idf_dict, name_df, channel_count=1
                )
        t_feat = time.time() - t0
        del s1_pre, cand_pre, name_df, token_idf_dict
        gc.collect()
        throughput = n_pairs / max(0.001, t_feat)
        print(f"  [D] Feature extraction: {n_pairs:,} pairs in {t_feat:.2f}s ({throughput:,.0f} pairs/sec)", flush=True)

        # Step E: Model Inference & Platt Calibration
        t0 = time.time()
        raw_p = model.predict_proba(feat_mat)
        cal_p = calib.predict_proba(raw_p)
        del feat_mat, raw_p
        gc.collect()
        print(f"  [E] Model inference & calibration in {time.time()-t0:.2f}s", flush=True)

        # Step F: DecisionEngine
        t0 = time.time()
        cand_score_map: Dict[str, List[Tuple[str, float]]] = {}
        for (s1_id, cid), p in zip(pair_list, cal_p):
            cand_score_map.setdefault(s1_id, []).append((cid, float(p)))
        for s1_id in s1_ids:
            cand_score_map.setdefault(s1_id, [])
        del pair_list, cal_p
        gc.collect()

        preds = engine.optimize_predictions(cand_score_map)
        del cand_score_map
        gc.collect()
        matched_count = sum(1 for m in preds.values() if m)
        print(f"  [F] DecisionEngine in {time.time()-t0:.2f}s ({matched_count:,} matched entities)", flush=True)

        # Step G: Write chunk TSV
        t0 = time.time()
        with open(chunk_file, 'w', encoding='utf-8') as f:
            for s1_id in s1_ids:
                matched = preds.get(s1_id, set())
                matched_str = ",".join(sorted(list(matched))) if matched else ""
                f.write(f"{s1_id}\t{matched_str}\n")
        del preds
        gc.collect()

        completed_chunks.add(chunk_idx)
        save_checkpoint(completed_chunks)

        t_chunk_total = time.time() - t_chunk_start
        print(f"  [G] Chunk {chunk_idx + 1} Saved in {t_chunk_total:.2f}s ({t_chunk_total/60:.2f} min). Current RAM: {proc.memory_info().rss/1e6:.1f} MB", flush=True)

    # 5. Merge all chunks into final outputs
    print("\n" + "=" * 80, flush=True)
    print("ALL 18 CHUNKS COMPLETED! MERGING FINAL SUBMISSION FILES...", flush=True)
    print("=" * 80, flush=True)
    t_merge_start = time.time()

    final_matching_results = output_dir / 'matching_results.tsv'
    final_sub_file = subs_dir / 'submission.tsv'

    total_written = 0
    with open(final_matching_results, 'w', encoding='utf-8') as out_f:
        out_f.write("source1_entity_id\tcandidate_entity_ids\n")
        for c_idx in range(total_chunks):
            c_file = chunks_dir / f"chunk_{c_idx:03d}.tsv"
            if not c_file.exists():
                raise FileNotFoundError(f"Missing chunk file: {c_file}")
            with open(c_file, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    if line.strip():
                        out_f.write(line)
                        total_written += 1

    # Also copy / write to artifacts/submissions/submission.tsv
    import shutil
    shutil.copyfile(final_matching_results, final_sub_file)

    t_merge = time.time() - t_merge_start
    t_total_pipeline = time.time() - t_pipeline_start
    print(f"Final Merging Done in {t_merge:.2f}s", flush=True)
    print(f"Total Rows Written: {total_written:,} (Expected: {TOTAL_ENTITIES:,})", flush=True)
    assert total_written == TOTAL_ENTITIES, f"Row count mismatch! {total_written} vs {TOTAL_ENTITIES}"
    print(f"Final output: {final_matching_results}", flush=True)
    print(f"Artifact output: {final_sub_file}", flush=True)
    print(f"Total Pipeline Runtime: {t_total_pipeline:.2f}s ({t_total_pipeline/60:.2f} min)", flush=True)

if __name__ == '__main__':
    run_production_inference()
