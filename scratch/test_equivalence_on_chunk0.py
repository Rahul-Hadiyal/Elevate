import sys, os, time, json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd
import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'code' / 'business_entity_resolution'))

from src.data_loader import load_entity_source
from src.normalizer import EntityNormalizer
from src.index_builder import canonicalize_phonetic, extract_postal_code
from src.feature_engineer import build_entity_lookup
from src.pair_features import FEATURE_NAMES, compute_single_pair_features
from src.feature_store import FeatureExtractor
from src.decision_engine import DecisionEngine
from rapidfuzz import fuzz

class FastEntity:
    __slots__ = (
        'entity_id', 'business_name', 'name_norm', 'name_tokens_sorted',
        'name_is_degenerate', 'name_ph', 'name_len', 'name_toks', 'name_tok_set',
        'name_2g', 'name_3g', 'business_address', 'addr_norm', 'addr_lm',
        'addr_len', 'addr_toks', 'addr_tok_set', 'addr_num_set', 'postal_code',
        'country_norm', 'is_s2', 'is_s3', 'addr_is_empty'
    )

    def __init__(self, row: Dict[str, Any]):
        self.entity_id = str(row.get("entity_id", "") or "")
        self.business_name = str(row.get("business_name", "") or "").strip()
        self.name_norm = str(row.get("name_norm", "") or "").strip()
        self.name_tokens_sorted = str(row.get("name_tokens_sorted", "") or "").strip()
        self.name_is_degenerate = bool(row.get("name_is_degenerate", False))
        self.name_ph = canonicalize_phonetic(self.name_norm)
        self.name_len = len(self.name_norm)
        self.name_toks = self.name_norm.split()
        self.name_tok_set = set(self.name_toks)
        self.name_2g = {self.name_norm[i:i+2] for i in range(len(self.name_norm) - 1)} if len(self.name_norm) >= 2 else set()
        self.name_3g = {self.name_norm[i:i+3] for i in range(len(self.name_norm) - 2)} if len(self.name_norm) >= 3 else set()

        self.business_address = str(row.get("business_address", "") or "").strip()
        self.addr_norm = str(row.get("addr_norm", "") or "").strip()
        self.addr_lm = str(row.get("addr_landmark", "") or "").strip()
        self.addr_len = len(self.addr_norm)
        self.addr_toks = self.addr_norm.split()
        self.addr_tok_set = set(self.addr_toks)

        nums = row.get("addr_numbers", [])
        if not isinstance(nums, (list, set, tuple)):
            nums = []
        self.addr_num_set = {str(x).strip().lower() for x in nums if str(x).strip()}

        self.postal_code = extract_postal_code(self.business_address, self.addr_norm)
        self.country_norm = str(row.get("country_norm", "") or "").strip()

        self.is_s2 = 1.0 if (self.entity_id.startswith("S2-") or self.entity_id.startswith("S2_")) else 0.0
        self.is_s3 = 1.0 if (self.entity_id.startswith("S3-") or self.entity_id.startswith("S3_")) else 0.0
        self.addr_is_empty = 1.0 if not self.addr_norm else 0.0

def jaccard_set_similarity(s1: Set[Any], s2: Set[Any]) -> float:
    if not s1 or not s2:
        return 0.0
    u = len(s1 | s2)
    return len(s1 & s2) / u if u > 0 else 0.0

def compute_pair_features_fast(
    s1: FastEntity,
    c: FastEntity,
    channel_count: int = 1
) -> np.ndarray:
    f_name_exact_raw = 1.0 if (s1.business_name and s1.business_name == c.business_name) else 0.0
    f_name_exact_norm = 1.0 if (s1.name_norm and s1.name_norm == c.name_norm) else 0.0
    f_name_exact_sorted = 1.0 if (s1.name_tokens_sorted and s1.name_tokens_sorted == c.name_tokens_sorted) else 0.0
    f_name_ph_exact = 1.0 if (s1.name_ph and s1.name_ph == c.name_ph) else 0.0

    f_name_len_diff_abs = float(abs(s1.name_len - c.name_len))
    max_len = max(s1.name_len, c.name_len)
    f_name_len_diff_rel = (f_name_len_diff_abs / max_len) if max_len > 0 else 0.0
    f_name_tok_count_diff = float(abs(len(s1.name_toks) - len(c.name_toks)))

    overlap_toks = s1.name_tok_set & c.name_tok_set
    f_name_tok_overlap_count = float(len(overlap_toks))
    f_name_tok_jaccard = jaccard_set_similarity(s1.name_tok_set, c.name_tok_set)

    min_tok_len = min(len(s1.name_tok_set), len(c.name_tok_set)) if (s1.name_tok_set and c.name_tok_set) else 0
    f_name_tok_containment = (len(overlap_toks) / min_tok_len) if min_tok_len > 0 else 0.0
    f_name_tok_overlap_coeff = f_name_tok_containment

    f_name_first_tok_match = 1.0 if (s1.name_toks and c.name_toks and s1.name_toks[0] == c.name_toks[0]) else 0.0
    f_name_last_tok_match = 1.0 if (s1.name_toks and c.name_toks and s1.name_toks[-1] == c.name_toks[-1]) else 0.0
    f_name_is_prefix = 1.0 if (s1.name_norm and c.name_norm and (s1.name_norm.startswith(c.name_norm) or c.name_norm.startswith(s1.name_norm))) else 0.0

    if s1.name_norm and c.name_norm:
        f_name_fuzz_ratio = float(fuzz.ratio(s1.name_norm, c.name_norm))
        f_name_fuzz_partial = float(fuzz.partial_ratio(s1.name_norm, c.name_norm))
        f_name_fuzz_tsort = float(fuzz.token_sort_ratio(s1.name_norm, c.name_norm))
        f_name_fuzz_tset = float(fuzz.token_set_ratio(s1.name_norm, c.name_norm))
        f_name_fuzz_wratio = float(fuzz.WRatio(s1.name_norm, c.name_norm))
    else:
        f_name_fuzz_ratio = 0.0
        f_name_fuzz_partial = 0.0
        f_name_fuzz_tsort = 0.0
        f_name_fuzz_tset = 0.0
        f_name_fuzz_wratio = 0.0

    f_name_2g_jaccard = jaccard_set_similarity(s1.name_2g, c.name_2g)
    f_name_3g_jaccard = jaccard_set_similarity(s1.name_3g, c.name_3g)

    f_addr_exact_norm = 1.0 if (s1.addr_norm and s1.addr_norm == c.addr_norm) else 0.0
    f_addr_exact_lm = 1.0 if (s1.addr_lm and s1.addr_lm == c.addr_lm) else 0.0

    f_addr_len_diff_abs = float(abs(s1.addr_len - c.addr_len))
    f_addr_tok_count_diff = float(abs(len(s1.addr_toks) - len(c.addr_toks)))

    overlap_addr_toks = s1.addr_tok_set & c.addr_tok_set
    f_addr_tok_overlap_count = float(len(overlap_addr_toks))
    f_addr_tok_jaccard = jaccard_set_similarity(s1.addr_tok_set, c.addr_tok_set)

    min_addr_tok = min(len(s1.addr_tok_set), len(c.addr_tok_set)) if (s1.addr_tok_set and c.addr_tok_set) else 0
    f_addr_tok_containment = (len(overlap_addr_toks) / min_addr_tok) if min_addr_tok > 0 else 0.0

    if s1.addr_norm and c.addr_norm:
        f_addr_fuzz_ratio = float(fuzz.ratio(s1.addr_norm, c.addr_norm))
        f_addr_fuzz_partial = float(fuzz.partial_ratio(s1.addr_norm, c.addr_norm))
        f_addr_fuzz_tsort = float(fuzz.token_sort_ratio(s1.addr_norm, c.addr_norm))
        f_addr_fuzz_tset = float(fuzz.token_set_ratio(s1.addr_norm, c.addr_norm))
    else:
        f_addr_fuzz_ratio = 0.0
        f_addr_fuzz_partial = 0.0
        f_addr_fuzz_tsort = 0.0
        f_addr_fuzz_tset = 0.0

    f_addr_num_count_s1 = float(len(s1.addr_num_set))
    f_addr_num_count_cand = float(len(c.addr_num_set))
    overlap_nums = s1.addr_num_set & c.addr_num_set
    f_addr_num_overlap_count = float(len(overlap_nums))
    f_addr_num_jaccard = jaccard_set_similarity(s1.addr_num_set, c.addr_num_set)
    f_addr_num_exact_match = 1.0 if (s1.addr_num_set and c.addr_num_set and s1.addr_num_set == c.addr_num_set) else 0.0
    f_addr_num_disagree = 1.0 if (s1.addr_num_set and c.addr_num_set and not overlap_nums) else 0.0

    f_postal_both_present = 1.0 if (s1.postal_code and c.postal_code) else 0.0
    f_postal_exact_match = 1.0 if (s1.postal_code and c.postal_code and s1.postal_code == c.postal_code) else 0.0

    f_country_exact_match = 1.0 if (s1.country_norm and c.country_norm and s1.country_norm == c.country_norm) else 0.0
    f_country_s1_missing = 1.0 if (not s1.country_norm or s1.country_norm == "unknown") else 0.0
    f_country_cand_missing = 1.0 if (not c.country_norm or c.country_norm == "unknown") else 0.0

    f_cross_prod = (f_name_fuzz_ratio / 100.0) * (f_addr_fuzz_ratio / 100.0)
    f_cross_name_exact_addr_ov = 1.0 if (f_name_exact_norm == 1.0 and f_addr_tok_jaccard > 0.25) else 0.0
    f_cross_name_fuzz_num_match = (f_name_fuzz_tsort / 100.0) * f_addr_num_exact_match
    f_cross_name_high_addr_low = 1.0 if (f_name_fuzz_ratio >= 85.0 and f_addr_fuzz_ratio < 40.0) else 0.0
    f_cross_addr_high_name_low = 1.0 if (f_addr_fuzz_ratio >= 85.0 and f_name_fuzz_ratio < 40.0) else 0.0
    f_cross_name_postal_agree = 1.0 if (f_name_fuzz_ratio >= 70.0 and f_postal_exact_match == 1.0) else 0.0

    f_cand_channel_count = float(channel_count)
    f_s1_name_degen = 1.0 if s1.name_is_degenerate else 0.0
    f_cand_name_degen = 1.0 if c.name_is_degenerate else 0.0
    f_s1_addr_empty = s1.addr_is_empty
    f_cand_addr_empty = c.addr_is_empty

    return np.array([
        f_name_exact_raw, f_name_exact_norm, f_name_exact_sorted, f_name_ph_exact,
        f_name_len_diff_abs, f_name_len_diff_rel, f_name_tok_count_diff, f_name_tok_overlap_count,
        f_name_tok_jaccard, f_name_tok_containment, f_name_tok_overlap_coeff, f_name_first_tok_match,
        f_name_last_tok_match, f_name_is_prefix, f_name_fuzz_ratio, f_name_fuzz_partial,
        f_name_fuzz_tsort, f_name_fuzz_tset, f_name_fuzz_wratio, f_name_2g_jaccard,
        f_name_3g_jaccard, f_addr_exact_norm, f_addr_exact_lm, f_addr_len_diff_abs,
        f_addr_tok_count_diff, f_addr_tok_overlap_count, f_addr_tok_jaccard, f_addr_tok_containment,
        f_addr_fuzz_ratio, f_addr_fuzz_partial, f_addr_fuzz_tsort, f_addr_fuzz_tset,
        f_addr_num_count_s1, f_addr_num_count_cand, f_addr_num_overlap_count, f_addr_num_jaccard,
        f_addr_num_exact_match, f_addr_num_disagree, f_postal_both_present, f_postal_exact_match,
        f_country_exact_match, f_country_s1_missing, f_country_cand_missing, f_cross_prod,
        f_cross_name_exact_addr_ov, f_cross_name_fuzz_num_match, f_cross_name_high_addr_low,
        f_cross_addr_high_name_low, f_cross_name_postal_agree, c.is_s2, c.is_s3,
        f_cand_channel_count, f_s1_name_degen, f_cand_name_degen, f_s1_addr_empty,
        f_cand_addr_empty,
    ], dtype=np.float32)

def main():
    print("=" * 80)
    print("END-TO-END EXACT OUTPUT EQUIVALENCE REGRESSION TEST ON TEST DATA")
    print("=" * 80)

    test_dir = REPO_ROOT / 'student_resource' / 'dataset' / 'test'
    output_dir = REPO_ROOT / 'output'
    models_dir = REPO_ROOT / 'artifacts' / 'models'

    # Load first 5,000 S1 from output/matching_results.tsv (frozen baseline)
    sample_s1_ids = []
    baseline_predictions = {}
    with open(output_dir / 'matching_results.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            s1_id = parts[0]
            matched = set(parts[1].split(',')) if (len(parts) == 2 and parts[1]) else set()
            baseline_predictions[s1_id] = matched
            sample_s1_ids.append(s1_id)
            if len(sample_s1_ids) >= 5000:
                break

    print(f"Loaded {len(sample_s1_ids):,} baseline predictions from output/matching_results.tsv.")

    # Load candidate pairs for these 5,000 S1 entities
    cand_map = {}
    target_cands = set()
    with open(output_dir / 'candidate_pairs.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) == 2 and parts[1]:
                s1_id = parts[0]
                if s1_id in baseline_predictions:
                    cands = parts[1].split(',')[:50]
                    cand_map[s1_id] = cands
                    target_cands.update(cands)

    pair_list = [(s1, c) for s1 in sample_s1_ids for c in cand_map.get(s1, [])]
    print(f"Candidate pairs: {len(pair_list):,} across {len(target_cands):,} candidate entities.")

    # Load entities
    normalizer = EntityNormalizer()
    s1_df, _ = load_entity_source(test_dir / 'test_source1.tsv', 'S1')
    s1_sub_df = s1_df[s1_df['entity_id'].isin(sample_s1_ids)].copy()
    s1_norm = normalizer.normalize_dataframe(s1_sub_df)
    s1_fast_lookup = {row['entity_id']: FastEntity(row) for row in s1_norm.to_dict('records')}
    del s1_df, s1_sub_df

    s2_df, _ = load_entity_source(test_dir / 'test_source2.tsv', 'S2')
    s2_sub_df = s2_df[s2_df['entity_id'].isin(target_cands)].copy()
    s2_norm = normalizer.normalize_dataframe(s2_sub_df)
    cand_fast_lookup = {row['entity_id']: FastEntity(row) for row in s2_norm.to_dict('records')}
    del s2_df, s2_sub_df

    s3_df, _ = load_entity_source(test_dir / 'test_source3.tsv', 'S3')
    s3_sub_df = s3_df[s3_df['entity_id'].isin(target_cands)].copy()
    s3_norm = normalizer.normalize_dataframe(s3_sub_df)
    cand_fast_lookup.update({row['entity_id']: FastEntity(row) for row in s3_norm.to_dict('records')})
    del s3_df, s3_sub_df

    print("\nRunning fast feature extraction on candidate pairs...")
    t0 = time.time()
    n_pairs = len(pair_list)
    feat_mat = np.zeros((n_pairs, 56), dtype=np.float32)
    for i, (s1_id, cid) in enumerate(pair_list):
        s1_obj = s1_fast_lookup.get(s1_id)
        c_obj = cand_fast_lookup.get(cid)
        if s1_obj and c_obj:
            feat_mat[i] = compute_pair_features_fast(s1_obj, c_obj, channel_count=1)
    t_feat = time.time() - t0
    print(f"Fast feature extraction completed in {t_feat:.3f}s ({n_pairs / t_feat:,.0f} pairs/sec)!")

    # Model inference using production_scorer (56 features)
    model = joblib.load(models_dir / 'production_scorer.joblib')
    calib = joblib.load(models_dir / 'production_calibrator.joblib')

    t0 = time.time()
    raw_p = model.predict_proba(feat_mat)
    cal_p = calib.predict_proba(raw_p)
    t_inf = time.time() - t0
    print(f"Model + calibration completed in {t_inf:.3f}s!")

    # Decision Engine
    engine = DecisionEngine(
        enable_conflict_resolution=True,
        margin_delta=0.05,
        min_prob_filter=0.01,
        max_candidates_per_entity=50,
    )
    t0 = time.time()
    cand_score_map = {}
    for (s1_id, cid), p in zip(pair_list, cal_p):
        cand_score_map.setdefault(s1_id, []).append((cid, float(p)))
    for s1_id in sample_s1_ids:
        cand_score_map.setdefault(s1_id, [])

    opt_predictions = engine.optimize_predictions(cand_score_map)
    t_engine = time.time() - t0
    print(f"DecisionEngine completed in {t_engine:.3f}s!")

    # Compare predictions against baseline output
    mismatches = 0
    total_checked = len(sample_s1_ids)
    for s1_id in sample_s1_ids:
        pred_base = baseline_predictions.get(s1_id, set())
        pred_opt = opt_predictions.get(s1_id, set())
        if pred_base != pred_opt:
            mismatches += 1
            if mismatches <= 5:
                print(f"Mismatch for {s1_id}:")
                print(f"  Baseline:  {pred_base}")
                print(f"  Optimized: {pred_opt}")

    match_rate = ((total_checked - mismatches) / total_checked) * 100.0
    print("\n" + "=" * 80)
    print(f"EQUIVALENCE REGRESSION TEST RESULTS:")
    print(f"Total S1 entities verified: {total_checked:,}")
    print(f"Exact Matches:              {total_checked - mismatches:,}")
    print(f"Mismatches:                 {mismatches}")
    print(f"Match Rate:                 {match_rate:.4f}%")
    print("=" * 80)

    report = {
        "benchmark": "End-to-End Equivalence on 5,000 S1 Entities against output/matching_results.tsv",
        "entities_checked": total_checked,
        "exact_matches": total_checked - mismatches,
        "mismatches": mismatches,
        "match_rate_percentage": match_rate,
        "fast_feature_throughput_pairs_per_sec": round(n_pairs / t_feat, 1),
        "total_compute_seconds": round(t_feat + t_inf + t_engine, 3),
    }
    with open(REPO_ROOT / 'artifacts' / 'performance' / 'equivalence_report.json', 'w') as f:
        json.dump(report, f, indent=2)

if __name__ == '__main__':
    main()
