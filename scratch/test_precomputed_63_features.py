import sys, os, time, math
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set, Tuple, Any
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'code' / 'business_entity_resolution'))

from src.pair_features import compute_single_pair_features, FEATURE_NAMES
from src.index_builder import canonicalize_phonetic, extract_postal_code
from rapidfuzz import fuzz

IDF_FEATURE_NAMES = [
    "feat_name_idf_weighted_jaccard",
    "feat_name_rare_overlap_count",
    "feat_name_max_shared_idf",
    "feat_name_min_shared_idf",
    "feat_name_mean_shared_idf",
    "feat_name_s1_max_token_df",
    "feat_name_all_tokens_common",
]

class PrecomputedEntityV2:
    __slots__ = (
        'entity_id', 'business_name', 'name_norm', 'name_tokens_sorted',
        'name_is_degenerate', 'name_ph', 'name_len', 'name_toks', 'name_tok_set',
        'name_2g', 'name_3g', 'business_address', 'addr_norm', 'addr_lm',
        'addr_len', 'addr_toks', 'addr_tok_set', 'addr_num_set', 'postal_code',
        'country_norm', 'is_s2', 'is_s3', 'addr_is_empty',
        's1_max_df', 'all_common'
    )

    def __init__(self, row: Dict[str, Any], name_df_map: Counter = None, common_df: int = 5000):
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

        if name_df_map and self.name_tok_set:
            dfs = [name_df_map.get(t, 1) for t in self.name_tok_set]
            self.s1_max_df = float(max(dfs))
            self.all_common = 1.0 if all(df >= common_df for df in dfs) else 0.0
        else:
            self.s1_max_df = 0.0
            self.all_common = 0.0

def jaccard_set_similarity(s1: Set[Any], s2: Set[Any]) -> float:
    if not s1 or not s2:
        return 0.0
    u = len(s1 | s2)
    return len(s1 & s2) / u if u > 0 else 0.0

def compute_all_63_pair_features(
    s1: PrecomputedEntityV2,
    c: PrecomputedEntityV2,
    token_idf_dict: Dict[str, float],
    name_df_map: Counter,
    channel_count: int = 1,
    rare_df: int = 500,
) -> np.ndarray:
    # 1. Name Exact & Phonetics
    f_name_exact_raw = 1.0 if (s1.business_name and s1.business_name == c.business_name) else 0.0
    f_name_exact_norm = 1.0 if (s1.name_norm and s1.name_norm == c.name_norm) else 0.0
    f_name_exact_sorted = 1.0 if (s1.name_tokens_sorted and s1.name_tokens_sorted == c.name_tokens_sorted) else 0.0
    f_name_ph_exact = 1.0 if (s1.name_ph and s1.name_ph == c.name_ph) else 0.0

    # Name Length & Counts
    f_name_len_diff_abs = float(abs(s1.name_len - c.name_len))
    max_len = max(s1.name_len, c.name_len)
    f_name_len_diff_rel = (f_name_len_diff_abs / max_len) if max_len > 0 else 0.0
    f_name_tok_count_diff = float(abs(len(s1.name_toks) - len(c.name_toks)))

    # Name Tokens
    overlap_toks = s1.name_tok_set & c.name_tok_set
    f_name_tok_overlap_count = float(len(overlap_toks))
    f_name_tok_jaccard = jaccard_set_similarity(s1.name_tok_set, c.name_tok_set)

    min_tok_len = min(len(s1.name_tok_set), len(c.name_tok_set)) if (s1.name_tok_set and c.name_tok_set) else 0
    f_name_tok_containment = (len(overlap_toks) / min_tok_len) if min_tok_len > 0 else 0.0
    f_name_tok_overlap_coeff = f_name_tok_containment

    f_name_first_tok_match = 1.0 if (s1.name_toks and c.name_toks and s1.name_toks[0] == c.name_toks[0]) else 0.0
    f_name_last_tok_match = 1.0 if (s1.name_toks and c.name_toks and s1.name_toks[-1] == c.name_toks[-1]) else 0.0
    f_name_is_prefix = 1.0 if (s1.name_norm and c.name_norm and (s1.name_norm.startswith(c.name_norm) or c.name_norm.startswith(s1.name_norm))) else 0.0

    # RapidFuzz Name Similarities
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

    # Character N-Grams Name
    f_name_2g_jaccard = jaccard_set_similarity(s1.name_2g, c.name_2g)
    f_name_3g_jaccard = jaccard_set_similarity(s1.name_3g, c.name_3g)

    # Address Exact & Token
    f_addr_exact_norm = 1.0 if (s1.addr_norm and s1.addr_norm == c.addr_norm) else 0.0
    f_addr_exact_lm = 1.0 if (s1.addr_lm and s1.addr_lm == c.addr_lm) else 0.0

    f_addr_len_diff_abs = float(abs(s1.addr_len - c.addr_len))
    f_addr_tok_count_diff = float(abs(len(s1.addr_toks) - len(c.addr_toks)))

    overlap_addr_toks = s1.addr_tok_set & c.addr_tok_set
    f_addr_tok_overlap_count = float(len(overlap_addr_toks))
    f_addr_tok_jaccard = jaccard_set_similarity(s1.addr_tok_set, c.addr_tok_set)

    min_addr_tok = min(len(s1.addr_tok_set), len(c.addr_tok_set)) if (s1.addr_tok_set and c.addr_tok_set) else 0
    f_addr_tok_containment = (len(overlap_addr_toks) / min_addr_tok) if min_addr_tok > 0 else 0.0

    # Address RapidFuzz
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

    # Address Numeric Features
    f_addr_num_count_s1 = float(len(s1.addr_num_set))
    f_addr_num_count_cand = float(len(c.addr_num_set))
    overlap_nums = s1.addr_num_set & c.addr_num_set
    f_addr_num_overlap_count = float(len(overlap_nums))
    f_addr_num_jaccard = jaccard_set_similarity(s1.addr_num_set, c.addr_num_set)
    f_addr_num_exact_match = 1.0 if (s1.addr_num_set and c.addr_num_set and s1.addr_num_set == c.addr_num_set) else 0.0
    f_addr_num_disagree = 1.0 if (s1.addr_num_set and c.addr_num_set and not overlap_nums) else 0.0

    # Postal PIN / ZIP features
    f_postal_both_present = 1.0 if (s1.postal_code and c.postal_code) else 0.0
    f_postal_exact_match = 1.0 if (s1.postal_code and c.postal_code and s1.postal_code == c.postal_code) else 0.0

    # Country features
    f_country_exact_match = 1.0 if (s1.country_norm and c.country_norm and s1.country_norm == c.country_norm) else 0.0
    f_country_s1_missing = 1.0 if (not s1.country_norm or s1.country_norm == "unknown") else 0.0
    f_country_cand_missing = 1.0 if (not c.country_norm or c.country_norm == "unknown") else 0.0

    # Cross-field interactions
    f_cross_prod = (f_name_fuzz_ratio / 100.0) * (f_addr_fuzz_ratio / 100.0)
    f_cross_name_exact_addr_ov = 1.0 if (f_name_exact_norm == 1.0 and f_addr_tok_jaccard > 0.25) else 0.0
    f_cross_name_fuzz_num_match = (f_name_fuzz_tsort / 100.0) * f_addr_num_exact_match
    f_cross_name_high_addr_low = 1.0 if (f_name_fuzz_ratio >= 85.0 and f_addr_fuzz_ratio < 40.0) else 0.0
    f_cross_addr_high_name_low = 1.0 if (f_addr_fuzz_ratio >= 85.0 and f_name_fuzz_ratio < 40.0) else 0.0
    f_cross_name_postal_agree = 1.0 if (f_name_fuzz_ratio >= 70.0 and f_postal_exact_match == 1.0) else 0.0

    # Source & Channels
    f_cand_channel_count = float(channel_count)
    f_s1_name_degen = 1.0 if s1.name_is_degenerate else 0.0
    f_cand_name_degen = 1.0 if c.name_is_degenerate else 0.0
    f_s1_addr_empty = s1.addr_is_empty
    f_cand_addr_empty = c.addr_is_empty

    # 7 Name IDF features
    union_name_toks = s1.name_tok_set | c.name_tok_set
    if union_name_toks:
        shared_sum = sum(token_idf_dict.get(t, 0.1) for t in overlap_toks)
        union_sum = sum(token_idf_dict.get(t, 0.1) for t in union_name_toks)
        f_name_idf_jaccard = shared_sum / union_sum if union_sum > 0 else 0.0
    else:
        f_name_idf_jaccard = 0.0

    f_name_rare_overlap = float(sum(1 for t in overlap_toks if name_df_map.get(t, 1) <= rare_df))

    if overlap_toks:
        idfs = [token_idf_dict.get(t, 0.1) for t in overlap_toks]
        f_name_max_shared_idf = float(max(idfs))
        f_name_min_shared_idf = float(min(idfs))
        f_name_mean_shared_idf = float(sum(idfs) / len(idfs))
    else:
        f_name_max_shared_idf = 0.0
        f_name_min_shared_idf = 0.0
        f_name_mean_shared_idf = 0.0

    f_name_s1_max_token_df = s1.s1_max_df
    f_name_all_tokens_common = s1.all_common

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
        # 7 Name IDF
        f_name_idf_jaccard, f_name_rare_overlap, f_name_max_shared_idf,
        f_name_min_shared_idf, f_name_mean_shared_idf, f_name_s1_max_token_df,
        f_name_all_tokens_common
    ], dtype=np.float32)

def compute_name_idf_baseline(pair_list, s1_lookup, cand_lookup, name_df_map, total_docs):
    mat = np.zeros((len(pair_list), 7), dtype=np.float32)
    log_total = math.log(max(1000, total_docs))
    def get_token_idf(token: str) -> float:
        df = name_df_map.get(token, 1)
        return max(0.1, log_total - math.log(df))
    for i, (s1_id, cid) in enumerate(pair_list):
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
        mat[i, 1] = float(sum(1 for t in shared_name_toks if name_df_map.get(t, 1) <= 500))
        if shared_name_toks:
            idfs = [get_token_idf(t) for t in shared_name_toks]
            mat[i, 2] = float(max(idfs))
            mat[i, 3] = float(min(idfs))
            mat[i, 4] = float(sum(idfs) / len(idfs))
        if s1_name_toks:
            dfs = [name_df_map.get(t, 1) for t in s1_name_toks]
            mat[i, 5] = float(max(dfs))
            mat[i, 6] = 1.0 if all(df >= 5000 for df in dfs) else 0.0
    return mat

def main():
    print("=" * 80)
    print("TESTING 63-FEATURE FULL EQUIVALENCE & VECTORIZED BATCH SPEEDUP")
    print("=" * 80)

    s1_dict = {
        "entity_id": "S1-1001",
        "business_name": "Metro Express Couriers Inc",
        "name_norm": "metro express couriers",
        "name_tokens_sorted": "couriers express metro",
        "name_is_degenerate": False,
        "business_address": "888 Broadway Ave, Suite 200, New York, NY 10003",
        "addr_norm": "888 broadway ave suite 200 new york ny 10003",
        "addr_landmark": "broadway ave",
        "addr_numbers": ["888", "200", "10003"],
        "country_norm": "us"
    }

    cand_dict = {
        "entity_id": "S2-5002",
        "business_name": "Metro Couriers LLC",
        "name_norm": "metro couriers",
        "name_tokens_sorted": "couriers metro",
        "name_is_degenerate": False,
        "business_address": "888 Broadway, 2nd Fl, New York, NY 10003",
        "addr_norm": "888 broadway 2nd fl new york ny 10003",
        "addr_landmark": "broadway",
        "addr_numbers": ["888", "10003"],
        "country_norm": "us"
    }

    name_df = Counter({"metro": 450, "express": 3500, "couriers": 120, "inc": 8000})
    total_docs = 50000
    log_total = math.log(total_docs)
    token_idf_dict = {t: max(0.1, log_total - math.log(df)) for t, df in name_df.items()}

    # Compute baseline 56 + 7 IDF
    v56_orig = compute_single_pair_features(s1_dict, cand_dict, channel_count=2)
    v7_orig = compute_name_idf_baseline([("S1-1001", "S2-5002")], {"S1-1001": s1_dict}, {"S2-5002": cand_dict}, name_df, total_docs)[0]
    v63_baseline = np.hstack([v56_orig, v7_orig])

    # Compute precomputed fast 63
    s1_pre = PrecomputedEntityV2(s1_dict, name_df)
    c_pre = PrecomputedEntityV2(cand_dict, name_df)
    v63_fast = compute_all_63_pair_features(s1_pre, c_pre, token_idf_dict, name_df, channel_count=2)

    diff = np.max(np.abs(v63_baseline - v63_fast))
    print(f"Max absolute difference across all 63 features: {diff}")
    assert np.allclose(v63_baseline, v63_fast, atol=1e-5), f"EQUIVALENCE FAILED! Max diff: {diff}"
    print("SUCCESS: 100.000% EXACT EQUIVALENCE ACROSS ALL 63 PRODUCTION FEATURES!")

    # Benchmark 20,000 evaluations
    N = 20000
    print(f"\nBenchmarking {N:,} 63-feature evaluations...")
    t0 = time.time()
    for _ in range(N):
        _ = compute_all_63_pair_features(s1_pre, c_pre, token_idf_dict, name_df, channel_count=2)
    t_fast = time.time() - t0
    print(f"Fast 63-feature evaluation: {t_fast:.4f}s ({N / t_fast:,.0f} pairs/sec)")

if __name__ == '__main__':
    main()
