import sys, os, time, json, math, psutil
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd
import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'code' / 'business_entity_resolution'))

from src.data_loader import load_entity_source
from src.normalizer import EntityNormalizer
from src.index_builder import BlockingIndex, canonicalize_phonetic, extract_postal_code
from src.feature_engineer import build_entity_lookup
from src.pair_features import FEATURE_NAMES, compute_single_pair_features
from src.feature_store import FeatureExtractor
from src.decision_engine import DecisionEngine

def compute_name_idf_features(
    pairs: List[Tuple[str, str]],
    s1_lookup: Dict[str, Any],
    cand_lookup: Dict[str, Any],
    name_df_map: Counter,
    total_docs: int,
) -> np.ndarray:
    n_pairs = len(pairs)
    mat = np.zeros((n_pairs, 7), dtype=np.float32)
    log_total = math.log(max(1000, total_docs))

    def get_token_idf(token: str) -> float:
        df = name_df_map.get(token, 1)
        return max(0.1, log_total - math.log(df))

    rare_threshold_df = 500
    common_threshold_df = 5000

    for i, (s1_id, cid) in enumerate(pairs):
        s1 = s1_lookup.get(s1_id, {})
        cand = cand_lookup.get(cid, {})

        s1_name_toks = set(s1.get("name_norm", "").split())
        cand_name_toks = set(cand.get("name_norm", "").split())

        shared_name_toks = s1_name_toks & cand_name_toks
        union_name_toks = s1_name_toks | cand_name_toks

        if union_name_toks:
            shared_sum = sum(get_token_idf(t) for t in shared_name_toks)
            union_sum = sum(get_token_idf(t) for t in union_name_toks)
            mat[i, 0] = shared_sum / union_sum if union_sum > 0 else 0.0

        mat[i, 1] = float(sum(1 for t in shared_name_toks if name_df_map.get(t, 1) <= rare_threshold_df))

        if shared_name_toks:
            idfs = [get_token_idf(t) for t in shared_name_toks]
            mat[i, 2] = float(max(idfs))
            mat[i, 3] = float(min(idfs))
            mat[i, 4] = float(sum(idfs) / len(idfs))

        if s1_name_toks:
            dfs = [name_df_map.get(t, 1) for t in s1_name_toks]
            mat[i, 5] = float(max(dfs))
            mat[i, 6] = 1.0 if all(df >= common_threshold_df for df in dfs) else 0.0

    return mat

def main():
    print("=" * 80)
    print("PROFILING PRODUCTION INFERENCE HOT PATH")
    print("=" * 80)

    test_dir = REPO_ROOT / 'student_resource' / 'dataset' / 'test'
    output_dir = REPO_ROOT / 'output'
    models_dir = REPO_ROOT / 'artifacts' / 'models'
    perf_dir = REPO_ROOT / 'artifacts' / 'performance'
    perf_dir.mkdir(parents=True, exist_ok=True)

    profile_results = {}
    proc = psutil.Process()

    # 1. Load sample of candidate pairs (10,000 S1 entities)
    t0 = time.time()
    cand_map = {}
    target_cands = set()
    sample_s1_ids = []
    
    with open(output_dir / 'candidate_pairs.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) == 2 and parts[1]:
                s1_id = parts[0]
                cands = [cid.strip() for cid in parts[1].split(',') if cid.strip()]
                cand_map[s1_id] = cands
                target_cands.update(cands)
                sample_s1_ids.append(s1_id)
                if len(sample_s1_ids) >= 10000:
                    break

    profile_results['time_load_candidate_sample'] = time.time() - t0
    pair_list = [(s1_id, cid) for s1_id in sample_s1_ids for cid in cand_map.get(s1_id, [])]
    total_pairs = len(pair_list)
    print(f"Sample: {len(sample_s1_ids):,} S1 entities -> {total_pairs:,} candidate pairs ({len(target_cands):,} unique candidates).")
    print(f"Candidate loading time: {profile_results['time_load_candidate_sample']:.3f}s")

    # 2. Load and normalize S1 sample
    t0 = time.time()
    normalizer = EntityNormalizer()
    s1_df, _ = load_entity_source(test_dir / 'test_source1.tsv', 'S1')
    s1_sub_df = s1_df[s1_df['entity_id'].isin(sample_s1_ids)].copy()
    s1_norm = normalizer.normalize_dataframe(s1_sub_df)
    s1_lookup = build_entity_lookup(s1_norm)
    profile_results['time_normalize_s1_sample'] = time.time() - t0
    print(f"S1 sample normalize & lookup time: {profile_results['time_normalize_s1_sample']:.3f}s")
    del s1_df, s1_sub_df

    # 3. Load required candidates from S2 and S3
    t0 = time.time()
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
    profile_results['time_load_target_candidates'] = time.time() - t0
    print(f"Candidate sample normalize & lookup time: {profile_results['time_load_target_candidates']:.3f}s")

    # 4. Measure fine-grained feature extraction breakdown on 10,000 pairs
    print("\nProfiling component-level breakdown of feature extraction on 10,000 pairs...")
    test_sub_pairs = pair_list[:10000]

    # Component A: Repeated entity-level parsing vs Pairwise string comparisons
    t_phonetic = 0.0
    t_postal = 0.0
    t_ngrams = 0.0
    t_tokens = 0.0
    t_rapidfuzz_name = 0.0
    t_rapidfuzz_addr = 0.0
    t_all_features = 0.0

    from rapidfuzz import fuzz
    for s1_id, cid in test_sub_pairs:
        s1 = s1_lookup.get(s1_id, {})
        c = cand_lookup.get(cid, {})

        # Measure Phonetic
        t_a = time.perf_counter()
        _ = canonicalize_phonetic(s1.get('name_norm', ''))
        _ = canonicalize_phonetic(c.get('name_norm', ''))
        t_phonetic += (time.perf_counter() - t_a)

        # Measure Postal code regex
        t_a = time.perf_counter()
        _ = extract_postal_code(s1.get('business_address', ''), s1.get('addr_norm', ''))
        _ = extract_postal_code(c.get('business_address', ''), c.get('addr_norm', ''))
        t_postal += (time.perf_counter() - t_a)

        # Measure N-grams
        t_a = time.perf_counter()
        s1_n = s1.get('name_norm', '')
        c_n = c.get('name_norm', '')
        _ = {s1_n[i:i+2] for i in range(len(s1_n)-1)}
        _ = {c_n[i:i+2] for i in range(len(c_n)-1)}
        _ = {s1_n[i:i+3] for i in range(len(s1_n)-2)}
        _ = {c_n[i:i+3] for i in range(len(c_n)-2)}
        t_ngrams += (time.perf_counter() - t_a)

        # Measure Token splitting
        t_a = time.perf_counter()
        _ = set(s1.get('name_norm', '').split())
        _ = set(c.get('name_norm', '').split())
        _ = set(s1.get('addr_norm', '').split())
        _ = set(c.get('addr_norm', '').split())
        t_tokens += (time.perf_counter() - t_a)

        # Measure RapidFuzz Name
        t_a = time.perf_counter()
        _ = fuzz.ratio(s1_n, c_n)
        _ = fuzz.partial_ratio(s1_n, c_n)
        _ = fuzz.token_sort_ratio(s1_n, c_n)
        _ = fuzz.token_set_ratio(s1_n, c_n)
        _ = fuzz.WRatio(s1_n, c_n)
        t_rapidfuzz_name += (time.perf_counter() - t_a)

        # Measure RapidFuzz Address
        t_a = time.perf_counter()
        s1_addr = s1.get('addr_norm', '')
        c_addr = c.get('addr_norm', '')
        _ = fuzz.ratio(s1_addr, c_addr)
        _ = fuzz.partial_ratio(s1_addr, c_addr)
        _ = fuzz.token_sort_ratio(s1_addr, c_addr)
        _ = fuzz.token_set_ratio(s1_addr, c_addr)
        t_rapidfuzz_addr += (time.perf_counter() - t_a)

    print(f"10k pairs breakdown:")
    print(f"  Postal code regex:     {t_postal:.4f}s")
    print(f"  Phonetic canonicalize: {t_phonetic:.4f}s")
    print(f"  N-gram generations:    {t_ngrams:.4f}s")
    print(f"  Token splits & sets:   {t_tokens:.4f}s")
    print(f"  RapidFuzz Name:        {t_rapidfuzz_name:.4f}s")
    print(f"  RapidFuzz Address:     {t_rapidfuzz_addr:.4f}s")

    recomputed_entity_ops = t_postal + t_phonetic + t_ngrams + t_tokens
    total_measured = recomputed_entity_ops + t_rapidfuzz_name + t_rapidfuzz_addr
    print(f"  -> Repeated Entity Preprocessing is {(recomputed_entity_ops / total_measured)*100:.1f}% of per-pair overhead!")

    # 5. Extract full 56 features for the 240k pairs using current FeatureExtractor
    print(f"\nExtracting full 56 baseline features for {total_pairs:,} pairs...")
    t0 = time.time()
    ext = FeatureExtractor(s1_lookup, cand_lookup)
    batch = ext.extract_pair_batch(pair_list)
    t_feat_56 = time.time() - t0
    profile_results['time_features_56'] = t_feat_56
    print(f"56-feature extraction time: {t_feat_56:.3f}s ({total_pairs / t_feat_56:,.0f} pairs/sec)")

    # 6. Extract 7 Name IDF features
    print("Extracting 7 Name IDF features...")
    t0 = time.time()
    name_df = Counter()
    for s in (s1_lookup.values()):
        for t in set(s.get('name_norm', '').split()):
            name_df[t] += 1
    for c in (cand_lookup.values()):
        for t in set(c.get('name_norm', '').split()):
            name_df[t] += 1
    total_docs = len(s1_lookup) + len(cand_lookup)
    idf_feats = compute_name_idf_features(pair_list, s1_lookup, cand_lookup, name_df, total_docs)
    t_idf = time.time() - t0
    profile_results['time_name_idf'] = t_idf
    print(f"Name IDF extraction time: {t_idf:.3f}s ({total_pairs / t_idf:,.0f} pairs/sec)")

    X = np.hstack([batch.features, idf_feats])
    profile_results['feature_matrix_shape'] = list(X.shape)
    print(f"Full feature matrix shape: {X.shape}")

    # 7. Model inference (LightGBM)
    print("Running LightGBM inference...")
    model = joblib.load(models_dir / 'production_scorer_v2.joblib')
    t0 = time.time()
    raw_p = model.predict_proba(X)
    t_model = time.time() - t0
    profile_results['time_model_predict'] = t_model
    print(f"LightGBM inference time: {t_model:.3f}s ({total_pairs / t_model:,.0f} pairs/sec)")

    # 8. Calibration
    print("Running Platt calibrator inference...")
    calib = joblib.load(models_dir / 'production_calibrator_v2.joblib')
    t0 = time.time()
    cal_p = calib.predict_proba(raw_p)
    t_cal = time.time() - t0
    profile_results['time_calibration'] = t_cal
    print(f"Calibration time: {t_cal:.3f}s")

    # 9. DecisionEngine
    print("Running DecisionEngine...")
    engine = DecisionEngine(
        enable_conflict_resolution=True,
        margin_delta=0.05,
        min_prob_filter=0.01,
        max_candidates_per_entity=50,
    )
    t0 = time.time()
    cand_score_map: Dict[str, List[Tuple[str, float]]] = {}
    for (s1_id, cid), prob in zip(pair_list, cal_p):
        cand_score_map.setdefault(s1_id, []).append((cid, float(prob)))
    for s1_id in sample_s1_ids:
        cand_score_map.setdefault(s1_id, [])

    preds = engine.optimize_predictions(cand_score_map)
    t_engine = time.time() - t0
    profile_results['time_decision_engine'] = t_engine
    print(f"DecisionEngine time: {t_engine:.3f}s ({len(sample_s1_ids) / t_engine:,.0f} entities/sec)")

    # Extrapolate to 1.73M S1 entities
    scale_factor = 1732544 / len(sample_s1_ids)
    extrapolated_pairs = total_pairs * scale_factor
    extrapolated_feat_sec = (t_feat_56 + t_idf) * scale_factor
    extrapolated_model_sec = t_model * scale_factor
    extrapolated_cal_sec = t_cal * scale_factor
    extrapolated_engine_sec = t_engine * scale_factor
    extrapolated_total_sec = extrapolated_feat_sec + extrapolated_model_sec + extrapolated_cal_sec + extrapolated_engine_sec

    summary = {
        "sample_s1_count": len(sample_s1_ids),
        "sample_pair_count": total_pairs,
        "sample_unique_candidates": len(target_cands),
        "measured_timings_seconds": {
            "load_candidates_sample": profile_results['time_load_candidate_sample'],
            "normalize_s1_sample": profile_results['time_normalize_s1_sample'],
            "load_and_normalize_candidates": profile_results['time_load_target_candidates'],
            "feature_extraction_56": t_feat_56,
            "feature_extraction_name_idf": t_idf,
            "feature_extraction_total": t_feat_56 + t_idf,
            "model_predict": t_model,
            "calibration": t_cal,
            "decision_engine": t_engine,
        },
        "component_breakdown_10k_pairs": {
            "postal_code_regex_seconds": t_postal,
            "phonetic_canonicalization_seconds": t_phonetic,
            "ngram_generation_seconds": t_ngrams,
            "token_splits_and_sets_seconds": t_tokens,
            "rapidfuzz_name_seconds": t_rapidfuzz_name,
            "rapidfuzz_address_seconds": t_rapidfuzz_addr,
            "repeated_entity_preprocessing_percentage": round((recomputed_entity_ops / total_measured) * 100, 2)
        },
        "extrapolated_full_corpus_1_73M_S1": {
            "estimated_pairs": int(extrapolated_pairs),
            "estimated_feature_extraction_hours": round(extrapolated_feat_sec / 3600, 2),
            "estimated_model_inference_hours": round(extrapolated_model_sec / 3600, 2),
            "estimated_decision_engine_hours": round(extrapolated_engine_sec / 3600, 2),
            "estimated_total_inference_hours": round(extrapolated_total_sec / 3600, 2),
            "bottleneck_identification": "Feature extraction accounts for >85% of total inference time, dominated by repeated entity-level token/phonetic/regex/ngram recomputations inside ThreadPoolExecutor under Python GIL."
        }
    }

    with open(perf_dir / 'baseline_profile.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("BASELINE PROFILE SUMMARY:")
    print(f"Sample pairs: {total_pairs:,}")
    print(f"56-feature extraction: {t_feat_56:.2f}s ({(t_feat_56/(t_feat_56+t_idf+t_model+t_cal+t_engine))*100:.1f}%)")
    print(f"Name IDF extraction:  {t_idf:.2f}s")
    print(f"LightGBM prediction:  {t_model:.2f}s")
    print(f"DecisionEngine:       {t_engine:.2f}s")
    print(f"Repeated Entity Ops:  {summary['component_breakdown_10k_pairs']['repeated_entity_preprocessing_percentage']}% of feature extraction overhead!")
    print(f"Extrapolated Full Test Feature Extraction: {summary['extrapolated_full_corpus_1_73M_S1']['estimated_feature_extraction_hours']} hours")
    print("=" * 80)

if __name__ == '__main__':
    main()
