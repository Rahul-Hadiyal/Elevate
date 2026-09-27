import sys, os, time, math, psutil
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

def test_indexed_chunk():
    proc = psutil.Process()
    test_dir = REPO_ROOT / 'student_resource' / 'dataset' / 'test'
    output_dir = REPO_ROOT / 'output'
    models_dir = REPO_ROOT / 'artifacts' / 'models'
    cand_pairs_path = output_dir / 'candidate_pairs.tsv'

    print("=" * 80)
    print("FAST INDEXED IN-MEMORY CHUNK BENCHMARK (100,000 S1)")
    print("=" * 80)

    # 1. Load candidate index into memory
    t0 = time.time()
    s2_df, _ = load_entity_source(test_dir / 'test_source2.tsv', 'S2')
    s3_df, _ = load_entity_source(test_dir / 'test_source3.tsv', 'S3')
    cand_raw_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    cand_raw_df.set_index('entity_id', inplace=True)
    print(f"[1] Raw Candidate Index Ready: {len(cand_raw_df):,} entities in {time.time()-t0:.2f}s (RAM: {proc.memory_info().rss/1e6:.1f} MB)")

    # 2. Load candidate pairs map for Chunk 1 (100,000 to 200,000)
    t0 = time.time()
    chunk_size = 100000
    chunk_idx = 1
    s_i = chunk_idx * chunk_size
    e_i = s_i + chunk_size

    s1_full_df, _ = load_entity_source(test_dir / 'test_source1.tsv', 'S1')
    s1_chunk_df = s1_full_df.iloc[s_i:e_i].copy()
    del s1_full_df

    s1_ids = list(s1_chunk_df['entity_id'])
    s1_ids_set = set(s1_ids)

    cand_map: Dict[str, List[str]] = {}
    with open(cand_pairs_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) == 2 and parts[1]:
                if parts[0] in s1_ids_set:
                    cand_map[parts[0]] = [cid.strip() for cid in parts[1].split(',') if cid.strip()][:150]

    pair_list = [(s1, c) for s1 in s1_ids for c in cand_map.get(s1, [])]
    needed_cands = {c for _, c in pair_list}
    n_pairs = len(pair_list)
    print(f"[2] Chunk 1 Pairs Loaded: {len(s1_ids):,} S1 entities, {n_pairs:,} candidate pairs, {len(needed_cands):,} unique candidates in {time.time()-t0:.2f}s")

    # 3. Fast Index Intersection & Normalization
    t0 = time.time()
    normalizer = EntityNormalizer()
    s1_norm = normalizer.normalize_dataframe(s1_chunk_df)
    s1_lookup = build_entity_lookup(s1_norm)
    del s1_chunk_df, s1_norm

    common_cands = cand_raw_df.index.intersection(needed_cands)
    cand_sub_df = cand_raw_df.loc[common_cands].reset_index()
    cand_norm = normalizer.normalize_dataframe(cand_sub_df)
    cand_lookup = build_entity_lookup(cand_norm)
    del cand_sub_df, cand_norm
    print(f"[3] Chunk Entities Normalized: {len(s1_lookup):,} S1 + {len(cand_lookup):,} Cands in {time.time()-t0:.2f}s (RAM: {proc.memory_info().rss/1e6:.1f} MB)")

    # 4. Token IDF Map & Fast Precomputation
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
    t_pre = time.time() - t0
    print(f"[4] PrecomputedEntityV2 Built in {t_pre:.2f}s (RAM: {proc.memory_info().rss/1e6:.1f} MB)")

    # 5. Fast 63-Feature Extraction
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
    throughput = n_pairs / t_feat
    print(f"[5] Feature Extraction: {n_pairs:,} pairs in {t_feat:.2f}s ({throughput:,.0f} pairs/sec) (RAM: {proc.memory_info().rss/1e6:.1f} MB)")

    # 6. Model & Calibration Inference
    t0 = time.time()
    model = joblib.load(models_dir / 'production_scorer_v2.joblib')
    calib = joblib.load(models_dir / 'production_calibrator_v2.joblib')
    raw_p = model.predict_proba(feat_mat)
    cal_p = calib.predict_proba(raw_p)
    t_inf = time.time() - t0
    print(f"[6] Model Scoring & Platt Calibration in {t_inf:.2f}s")

    # 7. DecisionEngine
    t0 = time.time()
    engine = DecisionEngine(
        enable_conflict_resolution=True,
        margin_delta=0.05,
        min_prob_filter=0.01,
        max_candidates_per_entity=50,
    )
    cand_score_map: Dict[str, List[Tuple[str, float]]] = {}
    for (s1_id, cid), p in zip(pair_list, cal_p):
        cand_score_map.setdefault(s1_id, []).append((cid, float(p)))
    for s1_id in s1_ids:
        cand_score_map.setdefault(s1_id, [])

    preds = engine.optimize_predictions(cand_score_map)
    t_de = time.time() - t0
    matched_count = sum(1 for m in preds.values() if m)
    print(f"[7] DecisionEngine in {t_de:.2f}s ({matched_count:,} matched)")

    # Write Chunk TSV
    t0 = time.time()
    chunk_file = output_dir / 'test_chunk_001.tsv'
    with open(chunk_file, 'w', encoding='utf-8') as f:
        for s1_id in s1_ids:
            matched = preds.get(s1_id, set())
            matched_str = ",".join(sorted(list(matched))) if matched else ""
            f.write(f"{s1_id}\t{matched_str}\n")
    print(f"[8] Chunk TSV Written in {time.time()-t0:.2f}s")

    t_chunk_total = t_pre + t_feat + t_inf + t_de
    print("\n" + "=" * 80)
    print(f"CHUNK 1 (100,000 S1) BENCHMARK SUMMARY:")
    print(f"Candidate Pairs:         {n_pairs:,}")
    print(f"Feature Throughput:      {throughput:,.0f} pairs/sec")
    print(f"Chunk Computation Time:  {t_chunk_total:.1f}s ({t_chunk_total/60:.2f} minutes)")
    print(f"Peak Working Set (RAM):  {proc.memory_info().rss/1e6:.1f} MB")
    print(f"Extrapolated 17 Chunks:  {t_chunk_total * 17 / 60:.1f} minutes ({t_chunk_total * 17 / 3600:.2f} hours)")
    print("=" * 80)

if __name__ == '__main__':
    test_indexed_chunk()
