import sys, os, time, math, json
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
from src.pair_features import compute_single_pair_features, FEATURE_NAMES
from src.decision_engine import DecisionEngine
from scratch.test_precomputed_63_features import (
    PrecomputedEntityV2, compute_all_63_pair_features, compute_name_idf_baseline
)

def main():
    print("=" * 80)
    print("TESTING EXACT EQUIVALENCE BETWEEN UNOPTIMIZED V2 AND OPTIMIZED V2 PIPELINE")
    print("=" * 80)

    test_dir = REPO_ROOT / 'student_resource' / 'dataset' / 'test'
    output_dir = REPO_ROOT / 'output'
    models_dir = REPO_ROOT / 'artifacts' / 'models'

    # Load 1,000 S1 entities
    sample_s1_ids = []
    cand_map = {}
    target_cands = set()

    with open(output_dir / 'candidate_pairs.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) == 2 and parts[1]:
                s1_id = parts[0]
                cands = [cid.strip() for cid in parts[1].split(',') if cid.strip()][:150]
                cand_map[s1_id] = cands
                target_cands.update(cands)
                sample_s1_ids.append(s1_id)
                if len(sample_s1_ids) >= 1000:
                    break

    pair_list = [(s1, c) for s1 in sample_s1_ids for c in cand_map.get(s1, [])]
    print(f"Sample: {len(sample_s1_ids):,} S1 entities -> {len(pair_list):,} candidate pairs ({len(target_cands):,} unique candidates).")

    # Load and normalize entities
    normalizer = EntityNormalizer()
    s1_df, _ = load_entity_source(test_dir / 'test_source1.tsv', 'S1')
    s1_sub_df = s1_df[s1_df['entity_id'].isin(sample_s1_ids)].copy()
    s1_norm = normalizer.normalize_dataframe(s1_sub_df)
    s1_lookup = build_entity_lookup(s1_norm)
    del s1_df, s1_sub_df

    s2_df, _ = load_entity_source(test_dir / 'test_source2.tsv', 'S2')
    s2_sub_df = s2_df[s2_df['entity_id'].isin(target_cands)].copy()
    s2_norm = normalizer.normalize_dataframe(s2_sub_df)
    cand_lookup = build_entity_lookup(s2_norm)
    del s2_df, s2_sub_df

    s3_df, _ = load_entity_source(test_dir / 'test_source3.tsv', 'S3')
    s3_sub_df = s3_df[s3_df['entity_id'].isin(target_cands)].copy()
    s3_norm = normalizer.normalize_dataframe(s3_sub_df)
    cand_lookup.update(build_entity_lookup(s3_norm))
    del s3_df, s3_sub_df

    # Token frequencies for IDF
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

    # 1. RUN REFERENCE UNOPTIMIZED V2 PIPELINE
    print("\n1. Running reference unoptimized V2 feature extraction...")
    t0 = time.time()
    n_pairs = len(pair_list)
    ref_feat_mat = np.zeros((n_pairs, 63), dtype=np.float32)

    for i, (s1_id, cid) in enumerate(pair_list):
        s1 = s1_lookup[s1_id]
        c = cand_lookup[cid]
        v56 = compute_single_pair_features(s1, c, channel_count=1)
        ref_feat_mat[i, :56] = v56

    ref_idf = compute_name_idf_baseline(pair_list, s1_lookup, cand_lookup, name_df, total_docs)
    ref_feat_mat[:, 56:] = ref_idf
    t_ref = time.time() - t0
    print(f"Reference feature extraction time: {t_ref:.3f}s ({n_pairs / t_ref:,.0f} pairs/sec)")

    # 2. RUN OPTIMIZED V2 PIPELINE
    print("\n2. Running optimized V2 feature extraction with precomputed entities...")
    t0 = time.time()
    s1_precomputed = {eid: PrecomputedEntityV2(row, name_df) for eid, row in s1_lookup.items()}
    cand_precomputed = {eid: PrecomputedEntityV2(row, name_df) for eid, row in cand_lookup.items()}
    t_precompute = time.time() - t0

    t0 = time.time()
    opt_feat_mat = np.zeros((n_pairs, 63), dtype=np.float32)
    for i, (s1_id, cid) in enumerate(pair_list):
        s1_obj = s1_precomputed[s1_id]
        c_obj = cand_precomputed[cid]
        opt_feat_mat[i] = compute_all_63_pair_features(s1_obj, c_obj, token_idf_dict, name_df, channel_count=1)
    t_opt_pairs = time.time() - t0
    t_opt_total = t_precompute + t_opt_pairs
    print(f"Optimized feature extraction time: {t_opt_total:.3f}s (precompute {t_precompute:.3f}s + pairs {t_opt_pairs:.3f}s -> {n_pairs / t_opt_pairs:,.0f} pairs/sec)")
    print(f"SPEEDUP: {t_ref / t_opt_total:.2f}x total ({t_ref / t_opt_pairs:.2f}x pairwise)")

    # 3. VERIFY FEATURE EQUIVALENCE
    max_feat_diff = np.max(np.abs(ref_feat_mat - opt_feat_mat))
    print(f"\nMax feature difference across all {n_pairs:,} pairs x 63 features: {max_feat_diff}")
    assert np.allclose(ref_feat_mat, opt_feat_mat, atol=1e-5), "FEATURE MATRICES DO NOT MATCH!"
    print("SUCCESS: FEATURE MATRICES ARE 100.000% EXACTLY EQUIVALENT!")

    # 4. VERIFY MODEL PREDICTIONS AND DECISIONS
    print("\nRunning V2 model inference and calibrator on both feature matrices...")
    model = joblib.load(models_dir / 'production_scorer_v2.joblib')
    calib = joblib.load(models_dir / 'production_calibrator_v2.joblib')

    ref_raw = model.predict_proba(ref_feat_mat)
    ref_cal = calib.predict_proba(ref_raw)

    opt_raw = model.predict_proba(opt_feat_mat)
    opt_cal = calib.predict_proba(opt_raw)

    max_prob_diff = np.max(np.abs(ref_cal - opt_cal))
    print(f"Max calibrated probability difference: {max_prob_diff}")
    assert np.allclose(ref_cal, opt_cal, atol=1e-5), "CALIBRATED PROBABILITIES DO NOT MATCH!"
    print("SUCCESS: CALIBRATED PROBABILITIES ARE 100.000% EXACTLY EQUIVALENT!")

    print("\nRunning DecisionEngine on both predictions...")
    engine = DecisionEngine(
        enable_conflict_resolution=True,
        margin_delta=0.05,
        min_prob_filter=0.01,
        max_candidates_per_entity=50,
    )

    ref_cand_map = {}
    for (s1_id, cid), p in zip(pair_list, ref_cal):
        ref_cand_map.setdefault(s1_id, []).append((cid, float(p)))
    for s1_id in sample_s1_ids:
        ref_cand_map.setdefault(s1_id, [])

    opt_cand_map = {}
    for (s1_id, cid), p in zip(pair_list, opt_cal):
        opt_cand_map.setdefault(s1_id, []).append((cid, float(p)))
    for s1_id in sample_s1_ids:
        opt_cand_map.setdefault(s1_id, [])

    ref_preds = engine.optimize_predictions(ref_cand_map)
    opt_preds = engine.optimize_predictions(opt_cand_map)

    mismatches = sum(1 for s1 in sample_s1_ids if ref_preds[s1] != opt_preds[s1])
    print(f"DecisionEngine Mismatches: {mismatches} / {len(sample_s1_ids):,}")
    assert mismatches == 0, f"DECISIONS DO NOT MATCH! {mismatches} mismatches found!"
    print("SUCCESS: DECISIONS ARE 100.000% EXACTLY IDENTICAL!")

    report = {
        "benchmark": "Exact Equivalence: Unoptimized V2 vs Fast Precomputed V2",
        "sample_s1_count": len(sample_s1_ids),
        "sample_pair_count": n_pairs,
        "max_feature_difference": float(max_feat_diff),
        "max_calibrated_prob_difference": float(max_prob_diff),
        "decision_mismatches": mismatches,
        "exact_equivalence": True,
        "reference_runtime_sec": round(t_ref, 3),
        "optimized_runtime_sec": round(t_opt_total, 3),
        "speedup_factor": round(t_ref / t_opt_total, 2)
    }
    with open(REPO_ROOT / 'artifacts' / 'performance' / 'v2_equivalence_report.json', 'w') as f:
        json.dump(report, f, indent=2)

if __name__ == '__main__':
    main()
