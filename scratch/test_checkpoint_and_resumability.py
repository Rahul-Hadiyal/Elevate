import sys, os, time, json, hashlib, math
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd
import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'code' / 'business_entity_resolution'))

from src.decision_engine import DecisionEngine
from scratch.test_precomputed_63_features import (
    PrecomputedEntityV2, compute_all_63_pair_features
)

def run_chunk_inference(
    s1_chunk_ids: List[str],
    cand_map: Dict[str, List[str]],
    s1_precomputed: Dict[str, PrecomputedEntityV2],
    cand_precomputed: Dict[str, PrecomputedEntityV2],
    model: Any,
    calib: Any,
    engine: DecisionEngine,
    token_idf_dict: Dict[str, float],
    name_df: Counter,
) -> Dict[str, List[str]]:
    pair_list = [(s1, c) for s1 in s1_chunk_ids for c in cand_map.get(s1, [])]
    n_pairs = len(pair_list)
    if n_pairs == 0:
        return {s1: [] for s1 in s1_chunk_ids}

    feat_mat = np.zeros((n_pairs, 63), dtype=np.float32)
    for i, (s1_id, cid) in enumerate(pair_list):
        s1_obj = s1_precomputed.get(s1_id)
        c_obj = cand_precomputed.get(cid)
        if s1_obj and c_obj:
            feat_mat[i] = compute_all_63_pair_features(s1_obj, c_obj, token_idf_dict, name_df, channel_count=1)

    raw_p = model.predict_proba(feat_mat)
    cal_p = calib.predict_proba(raw_p)

    cand_score_map = {}
    for (s1_id, cid), p in zip(pair_list, cal_p):
        cand_score_map.setdefault(s1_id, []).append((cid, float(p)))
    for s1_id in s1_chunk_ids:
        cand_score_map.setdefault(s1_id, [])

    preds = engine.optimize_predictions(cand_score_map)
    return {s1: sorted(list(preds.get(s1, set()))) for s1 in s1_chunk_ids}

def write_chunk_tsv(chunk_file: Path, chunk_results: Dict[str, List[str]]):
    with open(chunk_file, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, matched in chunk_results.items():
            f.write(f"{s1_id}\t{','.join(matched)}\n")

def read_chunk_tsv(chunk_file: Path) -> Dict[str, List[str]]:
    res = {}
    with open(chunk_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            s1_id = p[0]
            matched = [c.strip() for c in p[1].split(',') if c.strip()] if (len(p) == 2 and p[1]) else []
            res[s1_id] = matched
    return res

def main():
    print("=" * 80)
    print("RESUMABLE CHECKPOINTING & INTERRUPTION SIMULATION TEST")
    print("=" * 80)

    checkpoint_dir = REPO_ROOT / 'artifacts' / 'performance' / 'test_checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    models_dir = REPO_ROOT / 'artifacts' / 'models'
    output_dir = REPO_ROOT / 'output'

    model = joblib.load(models_dir / 'production_scorer_v2.joblib')
    calib = joblib.load(models_dir / 'production_calibrator_v2.joblib')
    engine = DecisionEngine(
        enable_conflict_resolution=True,
        margin_delta=0.05,
        min_prob_filter=0.01,
        max_candidates_per_entity=50,
    )

    # 1. Take 1,500 S1 entities split into 3 chunks of 500
    s1_all = []
    cand_map = {}
    needed_cands = set()

    with open(output_dir / 'candidate_pairs.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) == 2 and parts[1]:
                s1_id = parts[0]
                cands = [cid.strip() for cid in parts[1].split(',') if cid.strip()][:150]
                cand_map[s1_id] = cands
                needed_cands.update(cands)
                s1_all.append(s1_id)
                if len(s1_all) >= 1500:
                    break

    # Mock precomputed entities for fast test
    from src.data_loader import load_entity_source
    from src.normalizer import EntityNormalizer
    from src.feature_engineer import build_entity_lookup
    test_dir = REPO_ROOT / 'student_resource' / 'dataset' / 'test'
    normalizer = EntityNormalizer()
    s1_df, _ = load_entity_source(test_dir / 'test_source1.tsv', 'S1')
    s1_norm = normalizer.normalize_dataframe(s1_df[s1_df['entity_id'].isin(s1_all)])
    s1_lookup = build_entity_lookup(s1_norm)

    s2_df, _ = load_entity_source(test_dir / 'test_source2.tsv', 'S2')
    s2_norm = normalizer.normalize_dataframe(s2_df[s2_df['entity_id'].isin(needed_cands)])
    cand_lookup = build_entity_lookup(s2_norm)

    s3_df, _ = load_entity_source(test_dir / 'test_source3.tsv', 'S3')
    s3_norm = normalizer.normalize_dataframe(s3_df[s3_df['entity_id'].isin(needed_cands)])
    cand_lookup.update(build_entity_lookup(s3_norm))

    name_df = Counter()
    for s in s1_lookup.values():
        for t in set(s.get('name_norm', '').split()):
            name_df[t] += 1
    for c in cand_lookup.values():
        for t in set(c.get('name_norm', '').split()):
            name_df[t] += 1
    total_docs = len(s1_lookup) + len(cand_lookup)
    log_total = math.log(max(1000, total_docs))
    token_idf_dict = {t: max(0.1, log_total - math.log(df)) for t, df in name_df.items()}

    s1_precomputed = {eid: PrecomputedEntityV2(s1_lookup[eid], name_df) for eid in s1_all}
    cand_precomputed = {cid: PrecomputedEntityV2(cand_lookup[cid], name_df) for cid in needed_cands}

    chunks = [s1_all[0:500], s1_all[500:1000], s1_all[1000:1500]]

    # Step A: Run clean non-interrupted reference run
    print("\nStep A: Running clean reference run (no interruption)...")
    clean_results = {}
    for c_idx, chunk_ids in enumerate(chunks):
        res = run_chunk_inference(chunk_ids, cand_map, s1_precomputed, cand_precomputed, model, calib, engine, token_idf_dict, name_df)
        clean_results.update(res)

    clean_file = checkpoint_dir / 'clean_merged.tsv'
    write_chunk_tsv(clean_file, clean_results)
    clean_sha = hashlib.sha256(open(clean_file, 'rb').read()).hexdigest()
    print(f"Clean reference run completed. SHA-256: {clean_sha}")

    # Step B: Simulate Run 1 (completes Chunk 0 and Chunk 1, then crashes before Chunk 2)
    print("\nStep B: Simulating interrupted run (completes Chunk 0 and Chunk 1, crashes before Chunk 2)...")
    c0_file = checkpoint_dir / 'chunk_000.tsv'
    c1_file = checkpoint_dir / 'chunk_001.tsv'
    c2_file = checkpoint_dir / 'chunk_002.tsv'

    if c0_file.exists(): os.remove(c0_file)
    if c1_file.exists(): os.remove(c1_file)
    if c2_file.exists(): os.remove(c2_file)

    write_chunk_tsv(c0_file, run_chunk_inference(chunks[0], cand_map, s1_precomputed, cand_precomputed, model, calib, engine, token_idf_dict, name_df))
    write_chunk_tsv(c1_file, run_chunk_inference(chunks[1], cand_map, s1_precomputed, cand_precomputed, model, calib, engine, token_idf_dict, name_df))
    print("Chunk 0 saved.")
    print("Chunk 1 saved.")
    print(">>> SIMULATED CRASH / TERMINATION BEFORE CHUNK 2 <<<")

    # Step C: Restart pipeline with checkpoint detection
    print("\nStep C: Restarting pipeline with checkpoint detection...")
    resumed_results = {}
    recomputed_chunks = []
    skipped_chunks = []

    for c_idx, chunk_ids in enumerate(chunks):
        ckpt_file = checkpoint_dir / f'chunk_{c_idx:03d}.tsv'
        if ckpt_file.exists():
            skipped_chunks.append(c_idx)
            print(f"  Chunk {c_idx}: Checkpoint found at {ckpt_file.name}. SKIPPING recomputation.")
            chunk_data = read_chunk_tsv(ckpt_file)
            resumed_results.update(chunk_data)
        else:
            recomputed_chunks.append(c_idx)
            print(f"  Chunk {c_idx}: No checkpoint. Computing chunk...")
            chunk_data = run_chunk_inference(chunk_ids, cand_map, s1_precomputed, cand_precomputed, model, calib, engine, token_idf_dict, name_df)
            write_chunk_tsv(ckpt_file, chunk_data)
            resumed_results.update(chunk_data)

    assert skipped_chunks == [0, 1], f"Expected chunks [0, 1] to be skipped, got {skipped_chunks}"
    assert recomputed_chunks == [2], f"Expected chunk [2] to be computed, got {recomputed_chunks}"
    print(f"CHECKPOINT SKIPPING VERIFIED: Skipped chunks {skipped_chunks}, resumed chunk {recomputed_chunks}.")

    # Step D: Merge resumed chunks and verify byte-for-byte SHA-256 match
    resumed_file = checkpoint_dir / 'resumed_merged.tsv'
    write_chunk_tsv(resumed_file, resumed_results)
    resumed_sha = hashlib.sha256(open(resumed_file, 'rb').read()).hexdigest()

    print(f"\nComparing output hashes:")
    print(f"  Clean run SHA-256:   {clean_sha}")
    print(f"  Resumed run SHA-256: {resumed_sha}")

    assert clean_sha == resumed_sha, "BYTE-FOR-BYTE EQUIVALENCE FAILED BETWEEN CLEAN AND RESUMED RUN!"
    print("\nSUCCESS: BYTE-FOR-BYTE OUTPUT EQUIVALENCE CONFIRMED (PASS)!")

    # Clean up test checkpoints
    for f in checkpoint_dir.glob('*.tsv'):
        os.remove(f)

    report = {
        "benchmark": "Resumable Checkpointing & Interruption Simulation",
        "clean_run_sha256": clean_sha,
        "resumed_run_sha256": resumed_sha,
        "byte_for_byte_equality": (clean_sha == resumed_sha),
        "skipped_chunks_detected": skipped_chunks,
        "resumed_chunks_computed": recomputed_chunks,
        "status": "PASS"
    }
    with open(REPO_ROOT / 'artifacts' / 'performance' / 'checkpoint_validation.json', 'w') as f:
        json.dump(report, f, indent=2)

if __name__ == '__main__':
    main()
