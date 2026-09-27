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

def benchmark_10k_chunk():
    proc = psutil.Process()
    test_dir = REPO_ROOT / 'student_resource' / 'dataset' / 'test'
    output_dir = REPO_ROOT / 'output'
    models_dir = REPO_ROOT / 'artifacts' / 'models'
    cand_pairs_path = output_dir / 'candidate_pairs.tsv'

    print("=" * 80)
    print("10,000 S1 BENCHMARK: PROFILING ACTUAL EXECUTION PATH")
    print("=" * 80)

    # 1. Load candidate pairs for first 10,000 S1
    t0 = time.time()
    cand_map: Dict[str, List[str]] = {}
    with open(cand_pairs_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) == 2 and parts[1]:
                cand_map[parts[0]] = [cid.strip() for cid in parts[1].split(',') if cid.strip()][:150]
            if len(cand_map) >= 10000:
                break
    
    s1_ids = list(cand_map.keys())
    pair_list = [(s1, c) for s1 in s1_ids for c in cand_map[s1]]
    needed_cands = {c for _, c in pair_list}
    n_pairs = len(pair_list)
    print(f"[1] Candidate Pairs Loaded: {len(s1_ids):,} S1 entities, {n_pairs:,} pairs, {len(needed_cands):,} unique candidates in {time.time()-t0:.2f}s")

    # 2. Load S1 entities
    t0 = time.time()
    normalizer = EntityNormalizer()
    s1_df, _ = load_entity_source(test_dir / 'test_source1.tsv', 'S1')
    s1_sub_df = s1_df[s1_df['entity_id'].isin(s1_ids)].copy()
    s1_norm = normalizer.normalize_dataframe(s1_sub_df)
    s1_lookup = build_entity_lookup(s1_norm)
    del s1_df, s1_sub_df
    print(f"[2] S1 Entities Loaded & Normalized: {len(s1_lookup):,} in {time.time()-t0:.2f}s")

    # 3. Load Candidate entities (S2 & S3) ONLY for needed_cands
    t0 = time.time()
    s2_df, _ = load_entity_source(test_dir / 'test_source2.tsv', 'S2')
    s2_sub_df = s2_df[s2_df['entity_id'].isin(needed_cands)].copy()
    s2_norm = normalizer.normalize_dataframe(s2_sub_df)
    cand_lookup = build_entity_lookup(s2_norm)
    del s2_df, s2_sub_df

    s3_df, _ = load_entity_source(test_dir / 'test_source3.tsv', 'S3')
    s3_sub_df = s3_df[s3_df['entity_id'].isin(needed_cands)].copy()
    s3_norm = normalizer.normalize_dataframe(s3_sub_df)
    cand_lookup.update(build_entity_lookup(s3_norm))
    del s3_df, s3_sub_df
    print(f"[3] Needed Candidates Loaded & Normalized: {len(cand_lookup):,} in {time.time()-t0:.2f}s (RAM: {proc.memory_info().rss/(1024*1024):.1f} MB)")

    # 4. Token IDF Map
    t0 = time.time()
    name_df = Counter()
    for row in cand_lookup.values():
        for t in set(row.get('name_norm', '').split()):
            name_df[t] += 1
    total_docs = len(cand_lookup) + len(s1_lookup)
    log_total = math.log(max(1000, total_docs))
    token_idf_dict = {t: max(0.1, log_total - math.log(df)) for t, df in name_df.items()}

    # Precompute representation ONLY for needed entities
    t_pre0 = time.time()
    s1_pre = {eid: PrecomputedEntityV2(row, name_df) for eid, row in s1_lookup.items()}
    cand_pre = {eid: PrecomputedEntityV2(row, name_df) for eid, row in cand_lookup.items()}
    t_pre = time.time() - t_pre0
    print(f"[4] PrecomputedEntityV2 constructed for {len(s1_pre):,} S1 + {len(cand_pre):,} Cands in {t_pre:.2f}s (RAM: {proc.memory_info().rss/(1024*1024):.1f} MB)")

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
    print(f"[5] Feature Extraction: {n_pairs:,} pairs in {t_feat:.2f}s ({throughput:,.0f} pairs/sec) (RAM: {proc.memory_info().rss/(1024*1024):.1f} MB)")

    # 6. Model & Calibration Inference
    t0 = time.time()
    model = joblib.load(models_dir / 'production_scorer_v2.joblib')
    calib = joblib.load(models_dir / 'production_calibrator_v2.joblib')
    raw_p = model.predict_proba(feat_mat)
    cal_p = calib.predict_proba(raw_p)
    t_inf = time.time() - t0
    print(f"[6] Model Scoring & Platt Calibration: {n_pairs:,} pairs in {t_inf:.2f}s (RAM: {proc.memory_info().rss/(1024*1024):.1f} MB)")

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
    print(f"[7] DecisionEngine: {len(s1_ids):,} S1 in {t_de:.2f}s ({matched_count:,} matched) (RAM: {proc.memory_info().rss/(1024*1024):.1f} MB)")

    # Total per 10k S1:
    t_total = t_pre + t_feat + t_inf + t_de
    extrapolated_100k = t_total * 10
    extrapolated_full_1732k = t_total * 173.25
    print("\n" + "=" * 80)
    print(f"BENCHMARK SUMMARY FOR 10,000 S1:")
    print(f"Feature Extraction: {t_feat:.2f}s ({throughput:,.0f} pairs/sec)")
    print(f"Scoring + Calibration: {t_inf:.2f}s")
    print(f"DecisionEngine: {t_de:.2f}s")
    print(f"Total Computation (10k S1): {t_total:.2f}s")
    print(f"Extrapolated 100k S1 Chunk: {extrapolated_100k:.1f}s ({extrapolated_100k/60:.2f} minutes)")
    print(f"Extrapolated Full 1.73M Corpus: {extrapolated_full_1732k/60:.1f} minutes ({extrapolated_full_1732k/3600:.2f} hours)")
    print("=" * 80)

if __name__ == '__main__':
    benchmark_10k_chunk()
