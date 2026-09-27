"""
Full-Corpus Production Pipeline Runner — Optimized High-Throughput Engine.

Executes test inference across all 1,732,544 S1 entities and ~41.5M candidate pairs
using the frozen V2 scoring architecture:
- Candidate Retention: cap=150
- Channels: A-K + Channel N (candidate_pairs.tsv intact)
- Features: 63 production features (56 RapidFuzz + 7 Name IDF)
- Models: production_scorer_v2.joblib + production_calibrator_v2.joblib
- DecisionEngine: Poisson-Binomial DP (min_prob=0.01) + O(1) Margin-Guarded Conflict Resolution
- Resumable Chunk Checkpointing: artifacts/inference_chunks/
"""

import sys, os, time, math, json, hashlib, psutil
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd
import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'code' / 'business_entity_resolution'))

from src.data_loader import load_entity_source, validate_ground_truth_table
from src.normalizer import EntityNormalizer
from src.feature_engineer import build_entity_lookup
from src.decision_engine import DecisionEngine
from scratch.test_precomputed_63_features import (
    PrecomputedEntityV2, compute_all_63_pair_features
)

def compute_sha256(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha.update(block)
    return sha.hexdigest()

def main():
    t_start_total = time.time()
    print("=" * 80)
    print("FULL-CORPUS OPTIMIZED PRODUCTION INFERENCE & VALIDATION")
    print("=" * 80)

    test_dir = REPO_ROOT / 'student_resource' / 'dataset' / 'test'
    output_dir = REPO_ROOT / 'output'
    models_dir = REPO_ROOT / 'artifacts' / 'models'
    subs_dir = REPO_ROOT / 'artifacts' / 'submissions'
    perf_dir = REPO_ROOT / 'artifacts' / 'performance'
    ckpt_dir = REPO_ROOT / 'artifacts' / 'inference_chunks'

    output_dir.mkdir(parents=True, exist_ok=True)
    subs_dir.mkdir(parents=True, exist_ok=True)
    perf_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cand_pairs_path = output_dir / 'candidate_pairs.tsv'
    submission_tsv = subs_dir / 'submission.tsv'
    matching_tsv = output_dir / 'matching_results.tsv'

    proc = psutil.Process()
    timing_stats = {}

    # 1. Load Candidate Pairs Index
    print("Step 1: Loading candidate pairs map from candidate_pairs.tsv...")
    t0 = time.time()
    cand_map: Dict[str, List[str]] = {}
    needed_candidates: Set[str] = set()

    with open(cand_pairs_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) == 2 and parts[1]:
                s1_id = parts[0]
                cands = [cid.strip() for cid in parts[1].split(',') if cid.strip()][:150]
                cand_map[s1_id] = cands
                needed_candidates.update(cands)

    timing_stats["time_load_candidate_pairs_sec"] = time.time() - t0
    total_pairs = sum(len(c) for c in cand_map.values())
    print(f"Loaded {len(cand_map):,d} S1 candidate entries ({total_pairs:,d} total candidate pairs).")
    print(f"Unique candidate entities required: {len(needed_candidates):,d} (Time: {timing_stats['time_load_candidate_pairs_sec']:.2f}s)")

    # 2. Load and Precompute Test Candidate Entities (S2 & S3)
    print("\nStep 2: Loading and precomputing candidate entity pools...")
    t0 = time.time()
    normalizer = EntityNormalizer()

    print("  Loading and normalizing test Source 2...")
    s2_df, _ = load_entity_source(test_dir / 'test_source2.tsv', 'S2')
    s2_sub_df = s2_df[s2_df['entity_id'].isin(needed_candidates)].copy()
    s2_norm = normalizer.normalize_dataframe(s2_sub_df)
    cand_lookup = build_entity_lookup(s2_norm)
    del s2_df, s2_sub_df

    print("  Loading and normalizing test Source 3...")
    s3_df, _ = load_entity_source(test_dir / 'test_source3.tsv', 'S3')
    s3_sub_df = s3_df[s3_df['entity_id'].isin(needed_candidates)].copy()
    s3_norm = normalizer.normalize_dataframe(s3_sub_df)
    cand_lookup.update(build_entity_lookup(s3_norm))
    del s3_df, s3_sub_df

    timing_stats["time_load_and_normalize_candidates_sec"] = time.time() - t0
    print(f"Candidate lookup initialized: {len(cand_lookup):,d} entities (Time: {timing_stats['time_load_and_normalize_candidates_sec']:.2f}s)")

    # 3. Load Test Source 1
    print("\nStep 3: Loading test Source 1 entities...")
    t0 = time.time()
    test_s1_df, _ = load_entity_source(test_dir / 'test_source1.tsv', 'S1')
    total_test_s1 = len(test_s1_df)
    timing_stats["time_load_s1_sec"] = time.time() - t0
    print(f"Loaded {total_test_s1:,d} S1 records (Time: {timing_stats['time_load_s1_sec']:.2f}s)")

    # 4. Build Token IDF Dictionary
    print("\nStep 4: Building token IDF map from candidate vocabulary...")
    t0 = time.time()
    name_df = Counter()
    for row in cand_lookup.values():
        for t in set(row.get('name_norm', '').split()):
            name_df[t] += 1
    total_docs = len(cand_lookup) + total_test_s1
    log_total = math.log(max(1000, total_docs))
    token_idf_dict = {t: max(0.1, log_total - math.log(df)) for t, df in name_df.items()}

    print("  Precomputing fast candidate representation objects (PrecomputedEntityV2)...")
    cand_precomputed = {
        eid: PrecomputedEntityV2(row, name_df) for eid, row in cand_lookup.items()
    }
    del cand_lookup
    timing_stats["time_precompute_candidates_sec"] = time.time() - t0
    print(f"Candidate precomputation complete ({len(cand_precomputed):,d} objects, Time: {timing_stats['time_precompute_candidates_sec']:.2f}s)")

    # 5. Load Frozen Production Models
    print("\nStep 5: Loading production_scorer_v2 and production_calibrator_v2...")
    model = joblib.load(models_dir / 'production_scorer_v2.joblib')
    calib = joblib.load(models_dir / 'production_calibrator_v2.joblib')
    engine = DecisionEngine(
        enable_conflict_resolution=True,
        margin_delta=0.05,
        min_prob_filter=0.01,
        max_candidates_per_entity=50,
    )

    # 6. Execute Resumable Chunked Inference
    chunk_size = 100000
    num_chunks = int(np.ceil(total_test_s1 / chunk_size))
    print(f"\nStep 6: Executing chunked inference across {num_chunks} chunks ({chunk_size:,d} S1/chunk)...")

    checkpoint_manifest_path = ckpt_dir / 'inference_checkpoint.json'
    completed_chunks = set()
    if checkpoint_manifest_path.exists():
        with open(checkpoint_manifest_path, 'r') as f:
            completed_chunks = set(json.load(f).get("completed_chunks", []))
        print(f"Found existing checkpoint manifest: {len(completed_chunks)}/{num_chunks} chunks already finished.")

    t_inf_start = time.time()
    chunk_runtimes = []

    for chunk_idx in range(num_chunks):
        s_i = chunk_idx * chunk_size
        e_i = min(s_i + chunk_size, total_test_s1)
        chunk_file = ckpt_dir / f"chunk_{chunk_idx:03d}.tsv"

        if chunk_idx in completed_chunks and chunk_file.exists():
            print(f"  Chunk {chunk_idx+1:02d}/{num_chunks}: Checkpoint found ({chunk_file.name}). SKIPPING.")
            continue

        t_c0 = time.time()
        s1_chunk = test_s1_df.iloc[s_i:e_i]
        s1_norm_chunk = normalizer.normalize_dataframe(s1_chunk)
        s1_chunk_lookup = build_entity_lookup(s1_norm_chunk)

        s1_chunk_precomputed = {
            eid: PrecomputedEntityV2(row, name_df) for eid, row in s1_chunk_lookup.items()
        }

        # Candidate pairs for chunk
        s1_ids = list(s1_chunk["entity_id"])
        pair_list = [(s1_id, cid) for s1_id in s1_ids for cid in cand_map.get(s1_id, [])]
        n_pairs = len(pair_list)

        if n_pairs == 0:
            chunk_preds = {s1_id: set() for s1_id in s1_ids}
        else:
            # Fast 63-feature extraction
            feat_mat = np.zeros((n_pairs, 63), dtype=np.float32)
            for p_i, (s1_id, cid) in enumerate(pair_list):
                s1_obj = s1_chunk_precomputed.get(s1_id)
                c_obj = cand_precomputed.get(cid)
                if s1_obj and c_obj:
                    feat_mat[p_i] = compute_all_63_pair_features(
                        s1_obj, c_obj, token_idf_dict, name_df, channel_count=1
                    )

            # Model prediction & calibration
            raw_p = model.predict_proba(feat_mat)
            cal_p = calib.predict_proba(raw_p)

            # DecisionEngine
            cand_score_map: Dict[str, List[Tuple[str, float]]] = {}
            for (s1_id, cid), p in zip(pair_list, cal_p):
                cand_score_map.setdefault(s1_id, []).append((cid, float(p)))
            for s1_id in s1_ids:
                cand_score_map.setdefault(s1_id, [])

            chunk_preds = engine.optimize_predictions(cand_score_map)

        # Write chunk TSV
        with open(chunk_file, 'w', encoding='utf-8') as f:
            for s1_id in s1_ids:
                matched = chunk_preds.get(s1_id, set())
                matched_str = ",".join(sorted(list(matched))) if matched else ""
                f.write(f"{s1_id}\t{matched_str}\n")

        t_chunk_elapsed = time.time() - t_c0
        chunk_runtimes.append(t_chunk_elapsed)
        completed_chunks.add(chunk_idx)

        # Update checkpoint manifest
        with open(checkpoint_manifest_path, 'w') as f:
            json.dump({"completed_chunks": sorted(list(completed_chunks))}, f, indent=2)

        throughput = n_pairs / t_chunk_elapsed if n_pairs > 0 else 0
        mem_mb = proc.memory_info().rss / (1024 * 1024)
        print(f"  Chunk {chunk_idx+1:02d}/{num_chunks}: Completed {len(s1_ids):,d} S1 ({n_pairs:,d} pairs) in {t_chunk_elapsed:.2f}s ({throughput:,.0f} pairs/sec, RAM: {mem_mb:.0f} MB)")

    timing_stats["time_inference_total_sec"] = time.time() - t_inf_start

    # 7. Merge Chunk Checkpoints into Final TSV Files
    print("\nStep 7: Merging chunk TSVs into final matching_results.tsv and submission.tsv...")
    t0 = time.time()

    with open(matching_tsv, 'w', encoding='utf-8', newline='\n') as f_mat, \
         open(submission_tsv, 'w', encoding='utf-8', newline='\n') as f_sub:
        header = "source1_entity_id\tmatched_entity_ids\n"
        f_mat.write(header)
        f_sub.write(header)

        for chunk_idx in range(num_chunks):
            chunk_file = ckpt_dir / f"chunk_{chunk_idx:03d}.tsv"
            with open(chunk_file, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    f_mat.write(line)
                    f_sub.write(line)

    timing_stats["time_merge_tsv_sec"] = time.time() - t0
    final_sha = compute_sha256(matching_tsv)
    matching_size_mb = matching_tsv.stat().st_size / (1024 * 1024)
    print(f"Final output written: {matching_tsv} ({matching_size_mb:.2f} MB, SHA-256: {final_sha})")

    # 8. Submission Constraint Validation
    print("\nStep 8: Running official validation constraints...")
    sub_df = pd.read_csv(matching_tsv, sep='\t', dtype=str).fillna("")
    val_report = validate_ground_truth_table(sub_df, file_path=matching_tsv, strict=True)
    print(f"Validation Report: valid={val_report.is_valid}, rows={val_report.row_count:,d}, errors={len(val_report.errors)}")
    assert val_report.is_valid, f"SUBMISSION VALIDATION FAILED! Errors: {val_report.errors}"

    # Compute Statistics
    matched_col = sub_df["matched_entity_ids"].fillna("").astype(str)
    unmatched_count = int((matched_col == "").sum())
    matched_count = int((matched_col != "").sum())
    match_lens = matched_col.apply(lambda x: len([i for i in x.split(",") if i.strip()])).values
    total_matches = int(sum(match_lens))

    s2_count = sum(1 for m in matched_col if any(c.startswith("S2-") for c in m.split(",") if c))
    s3_count = sum(1 for m in matched_col if any(c.startswith("S3-") for c in m.split(",") if c))

    t_total_elapsed = time.time() - t_start_total

    run_summary = {
        "status": "PASS",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "submission_file": str(matching_tsv),
        "sha256_checksum": final_sha,
        "file_size_mb": round(matching_size_mb, 2),
        "total_s1_entities": total_test_s1,
        "total_candidate_pairs": total_pairs,
        "matched_entities": matched_count,
        "unmatched_entities": unmatched_count,
        "match_percentage": round((matched_count / total_test_s1) * 100, 2),
        "total_selected_links": total_matches,
        "entities_with_s2": s2_count,
        "entities_with_s3": s3_count,
        "timing_breakdown": {
            "load_candidate_pairs_sec": round(timing_stats["time_load_candidate_pairs_sec"], 2),
            "load_and_normalize_candidates_sec": round(timing_stats["time_load_and_normalize_candidates_sec"], 2),
            "precompute_candidates_sec": round(timing_stats["time_precompute_candidates_sec"], 2),
            "inference_chunks_total_sec": round(timing_stats["time_inference_total_sec"], 2),
            "merge_tsv_sec": round(timing_stats["time_merge_tsv_sec"], 2),
            "total_elapsed_sec": round(t_total_elapsed, 2),
            "total_elapsed_minutes": round(t_total_elapsed / 60, 2),
        },
        "performance_metrics": {
            "overall_pairs_per_sec": round(total_pairs / max(1, timing_stats["time_inference_total_sec"]), 1),
            "peak_rss_mb": round(proc.memory_info().rss / (1024 * 1024), 1),
            "chunk_size": chunk_size,
            "num_chunks": num_chunks,
        },
        "validation_report": {
            "is_valid": val_report.is_valid,
            "row_count": val_report.row_count,
            "errors": val_report.errors,
            "warnings": val_report.warnings,
        }
    }

    with open(perf_dir / 'final_production_validation.json', 'w') as f:
        json.dump(run_summary, f, indent=2)

    print("\n" + "=" * 80)
    print("FULL PRODUCTION RUN SUMMARY:")
    print(f"Status:               {run_summary['status']}")
    print(f"Total S1 Entities:    {total_test_s1:,d}")
    print(f"Total Candidate Pairs:{total_pairs:,d}")
    print(f"Matched S1 Entities:  {matched_count:,d} ({run_summary['match_percentage']}%)")
    print(f"Unmatched Entities:   {unmatched_count:,d}")
    print(f"Total Output Matches: {total_matches:,d}")
    print(f"Total Runtime:        {run_summary['timing_breakdown']['total_elapsed_minutes']} minutes ({t_total_elapsed:.1f}s)")
    print(f"Inference Throughput: {run_summary['performance_metrics']['overall_pairs_per_sec']:,.0f} pairs/sec")
    print(f"Peak RAM (RSS):       {run_summary['performance_metrics']['peak_rss_mb']} MB")
    print(f"SHA-256 Checksum:     {final_sha}")
    print(f"Official Validator:   {'PASSED' if val_report.is_valid else 'FAILED'}")
    print("=" * 80)

if __name__ == '__main__':
    main()
