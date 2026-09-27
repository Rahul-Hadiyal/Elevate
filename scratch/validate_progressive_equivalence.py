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
from src.pair_features import compute_single_pair_features
from src.decision_engine import DecisionEngine
from scratch.test_precomputed_63_features import (
    PrecomputedEntityV2, compute_all_63_pair_features, compute_name_idf_baseline
)

def run_cohort_equivalence_test(
    cohort_name: str,
    s1_ids: List[str],
    cand_map: Dict[str, List[str]],
    s1_lookup: Dict[str, Dict[str, Any]],
    cand_lookup: Dict[str, Dict[str, Any]],
    model: Any,
    calib: Any,
    name_df: Counter,
    token_idf_dict: Dict[str, float],
    total_docs: int,
    engine: DecisionEngine,
) -> Dict[str, Any]:
    print(f"\n--- Running Equivalence Test: {cohort_name} ({len(s1_ids):,} S1 entities) ---")
    
    # 1. Build pair list
    pair_list = [(s1, c) for s1 in s1_ids for c in cand_map.get(s1, [])]
    n_pairs = len(pair_list)
    print(f"Total pairs to evaluate: {n_pairs:,}")

    # 2. Reference Pipeline Feature Extraction
    t0 = time.time()
    ref_feat_mat = np.zeros((n_pairs, 63), dtype=np.float32)
    for i, (s1_id, cid) in enumerate(pair_list):
        s1 = s1_lookup[s1_id]
        c = cand_lookup[cid]
        v56 = compute_single_pair_features(s1, c, channel_count=1)
        ref_feat_mat[i, :56] = v56

    ref_idf = compute_name_idf_baseline(pair_list, s1_lookup, cand_lookup, name_df, total_docs)
    ref_feat_mat[:, 56:] = ref_idf
    t_ref_feat = time.time() - t0

    # 3. Optimized Pipeline Feature Extraction
    t0 = time.time()
    s1_precomputed = {eid: PrecomputedEntityV2(s1_lookup[eid], name_df) for eid in s1_ids if eid in s1_lookup}
    needed_cands = {cid for _, cid in pair_list}
    cand_precomputed = {cid: PrecomputedEntityV2(cand_lookup[cid], name_df) for cid in needed_cands if cid in cand_lookup}
    t_precompute = time.time() - t0

    t0 = time.time()
    opt_feat_mat = np.zeros((n_pairs, 63), dtype=np.float32)
    for i, (s1_id, cid) in enumerate(pair_list):
        s1_obj = s1_precomputed[s1_id]
        c_obj = cand_precomputed[cid]
        opt_feat_mat[i] = compute_all_63_pair_features(s1_obj, c_obj, token_idf_dict, name_df, channel_count=1)
    t_opt_pairs = time.time() - t0
    t_opt_feat = t_precompute + t_opt_pairs

    # Compare features
    max_feat_diff = float(np.max(np.abs(ref_feat_mat - opt_feat_mat)))
    print(f"Max feature difference: {max_feat_diff:.8f}")

    # 4. Model & Calibrator Inference
    t0 = time.time()
    ref_raw = model.predict_proba(ref_feat_mat)
    ref_cal = calib.predict_proba(ref_raw)
    t_ref_inf = time.time() - t0

    t0 = time.time()
    opt_raw = model.predict_proba(opt_feat_mat)
    opt_cal = calib.predict_proba(opt_raw)
    t_opt_inf = time.time() - t0

    max_raw_diff = float(np.max(np.abs(ref_raw - opt_raw)))
    max_cal_diff = float(np.max(np.abs(ref_cal - opt_cal)))
    print(f"Max raw prob difference: {max_raw_diff:.8f}, Max cal prob difference: {max_cal_diff:.8f}")

    # 5. DecisionEngine Optimization & Conflict Resolution
    ref_cand_map = {}
    for (s1_id, cid), p in zip(pair_list, ref_cal):
        ref_cand_map.setdefault(s1_id, []).append((cid, float(p)))
    for s1_id in s1_ids:
        ref_cand_map.setdefault(s1_id, [])

    opt_cand_map = {}
    for (s1_id, cid), p in zip(pair_list, opt_cal):
        opt_cand_map.setdefault(s1_id, []).append((cid, float(p)))
    for s1_id in s1_ids:
        opt_cand_map.setdefault(s1_id, [])

    t0 = time.time()
    ref_preds = engine.optimize_predictions(ref_cand_map)
    t_ref_dec = time.time() - t0

    t0 = time.time()
    opt_preds = engine.optimize_predictions(opt_cand_map)
    t_opt_dec = time.time() - t0

    # Compare decisions
    decision_mismatches = 0
    matched_ref = 0
    matched_opt = 0
    for s1_id in s1_ids:
        r_set = ref_preds.get(s1_id, set())
        o_set = opt_preds.get(s1_id, set())
        if r_set:
            matched_ref += 1
        if o_set:
            matched_opt += 1
        if r_set != o_set:
            decision_mismatches += 1

    print(f"Decision mismatches: {decision_mismatches} / {len(s1_ids):,}")
    print(f"Matched entities: Reference = {matched_ref:,}, Optimized = {matched_opt:,}")
    print(f"Feature Speedup: {t_ref_feat / t_opt_feat:.2f}x (Total: {(t_ref_feat + t_ref_inf + t_ref_dec) / (t_opt_feat + t_opt_inf + t_opt_dec):.2f}x)")

    return {
        "cohort": cohort_name,
        "s1_count": len(s1_ids),
        "pair_count": n_pairs,
        "max_feature_diff": max_feat_diff,
        "max_raw_prob_diff": max_raw_diff,
        "max_cal_prob_diff": max_cal_diff,
        "decision_mismatches": decision_mismatches,
        "matched_ref": matched_ref,
        "matched_opt": matched_opt,
        "exact_match_percentage": round(((len(s1_ids) - decision_mismatches) / len(s1_ids)) * 100.0, 4),
        "reference_runtime_sec": round(t_ref_feat + t_ref_inf + t_ref_dec, 3),
        "optimized_runtime_sec": round(t_opt_feat + t_opt_inf + t_opt_dec, 3),
        "speedup_factor": round((t_ref_feat + t_ref_inf + t_ref_dec) / (t_opt_feat + t_opt_inf + t_opt_dec), 2)
    }

def main():
    print("=" * 80)
    print("PROGRESSIVE PRODUCTION EQUIVALENCE BENCHMARK")
    print("=" * 80)

    test_dir = REPO_ROOT / 'student_resource' / 'dataset' / 'test'
    output_dir = REPO_ROOT / 'output'
    models_dir = REPO_ROOT / 'artifacts' / 'models'

    # Load production models
    print("Loading production_scorer_v2 and production_calibrator_v2...")
    model = joblib.load(models_dir / 'production_scorer_v2.joblib')
    calib = joblib.load(models_dir / 'production_calibrator_v2.joblib')

    engine = DecisionEngine(
        enable_conflict_resolution=True,
        margin_delta=0.05,
        min_prob_filter=0.01,
        max_candidates_per_entity=50,
    )

    # Load 15,000 S1 from candidate pairs to support 1k, 10k, and diverse cohorts
    print("Scanning candidate pairs from candidate_pairs.tsv...")
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
                if len(s1_all) >= 15000:
                    break

    normalizer = EntityNormalizer()
    print("Loading test_source1 slice...")
    s1_df, _ = load_entity_source(test_dir / 'test_source1.tsv', 'S1')
    s1_sub_df = s1_df[s1_df['entity_id'].isin(s1_all)].copy()
    s1_norm = normalizer.normalize_dataframe(s1_sub_df)
    s1_lookup = build_entity_lookup(s1_norm)
    del s1_df, s1_sub_df

    print("Loading candidate records from test_source2 and test_source3...")
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

    # Token frequencies for IDF
    print("Building vocabulary token IDF map...")
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

    # Cohort 1: 1,000 S1
    res_1k = run_cohort_equivalence_test(
        "Cohort 1 (1,000 S1 Entities)",
        s1_all[:1000],
        cand_map, s1_lookup, cand_lookup,
        model, calib, name_df, token_idf_dict, total_docs, engine
    )

    # Cohort 2: 10,000 S1
    res_10k = run_cohort_equivalence_test(
        "Cohort 2 (10,000 S1 Entities)",
        s1_all[:10000],
        cand_map, s1_lookup, cand_lookup,
        model, calib, name_df, token_idf_dict, total_docs, engine
    )

    # Cohort 3: Diverse Cases Cohort (Filter diverse subsets)
    # Find Indian entities, multi-candidate (>20 cands) entities, long names/addresses
    diverse_s1 = []
    for s1_id in s1_all:
        s1_row = s1_lookup.get(s1_id, {})
        cands = cand_map.get(s1_id, [])
        is_india = (s1_row.get("country_norm") == "india")
        is_multi = (len(cands) >= 20)
        has_both_s2_s3 = any(c.startswith("S2-") for c in cands) and any(c.startswith("S3-") for c in cands)
        if is_india or (is_multi and has_both_s2_s3):
            diverse_s1.append(s1_id)
            if len(diverse_s1) >= 2000:
                break

    res_diverse = run_cohort_equivalence_test(
        "Cohort 3 (Diverse Subsets: India, High-Cardinality, S2+S3, Conflicting)",
        diverse_s1,
        cand_map, s1_lookup, cand_lookup,
        model, calib, name_df, token_idf_dict, total_docs, engine
    )

    all_results = [res_1k, res_10k, res_diverse]

    out_file = REPO_ROOT / 'artifacts' / 'performance' / 'progressive_equivalence_results.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("PROGRESSIVE EQUIVALENCE SUMMARY:")
    for r in all_results:
        print(f"[{r['cohort']}]: Exact Match: {r['exact_match_percentage']}% | Max Feat Diff: {r['max_feature_diff']:.8f} | Max Prob Diff: {r['max_cal_prob_diff']:.8f} | Speedup: {r['speedup_factor']}x")
    print("=" * 80)

if __name__ == '__main__':
    main()
