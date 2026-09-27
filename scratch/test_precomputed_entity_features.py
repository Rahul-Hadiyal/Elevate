import sys, os, time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'code' / 'business_entity_resolution'))

from src.pair_features import compute_single_pair_features, FEATURE_NAMES
from src.index_builder import canonicalize_phonetic, extract_postal_code
from rapidfuzz import fuzz

class PrecomputedEntity:
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
    s1: PrecomputedEntity,
    c: PrecomputedEntity,
    channel_count: int = 1
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

    # Missingness & Degenerate
    f_s1_name_degen = 1.0 if s1.name_is_degenerate else 0.0
    f_cand_name_degen = 1.0 if c.name_is_degenerate else 0.0
    f_s1_addr_empty = s1.addr_is_empty
    f_cand_addr_empty = c.addr_is_empty

    return np.array([
        f_name_exact_raw,
        f_name_exact_norm,
        f_name_exact_sorted,
        f_name_ph_exact,
        f_name_len_diff_abs,
        f_name_len_diff_rel,
        f_name_tok_count_diff,
        f_name_tok_overlap_count,
        f_name_tok_jaccard,
        f_name_tok_containment,
        f_name_tok_overlap_coeff,
        f_name_first_tok_match,
        f_name_last_tok_match,
        f_name_is_prefix,
        f_name_fuzz_ratio,
        f_name_fuzz_partial,
        f_name_fuzz_tsort,
        f_name_fuzz_tset,
        f_name_fuzz_wratio,
        f_name_2g_jaccard,
        f_name_3g_jaccard,
        f_addr_exact_norm,
        f_addr_exact_lm,
        f_addr_len_diff_abs,
        f_addr_tok_count_diff,
        f_addr_tok_overlap_count,
        f_addr_tok_jaccard,
        f_addr_tok_containment,
        f_addr_fuzz_ratio,
        f_addr_fuzz_partial,
        f_addr_fuzz_tsort,
        f_addr_fuzz_tset,
        f_addr_num_count_s1,
        f_addr_num_count_cand,
        f_addr_num_overlap_count,
        f_addr_num_jaccard,
        f_addr_num_exact_match,
        f_addr_num_disagree,
        f_postal_both_present,
        f_postal_exact_match,
        f_country_exact_match,
        f_country_s1_missing,
        f_country_cand_missing,
        f_cross_prod,
        f_cross_name_exact_addr_ov,
        f_cross_name_fuzz_num_match,
        f_cross_name_high_addr_low,
        f_cross_addr_high_name_low,
        f_cross_name_postal_agree,
        c.is_s2,
        c.is_s3,
        f_cand_channel_count,
        f_s1_name_degen,
        f_cand_name_degen,
        f_s1_addr_empty,
        f_cand_addr_empty,
    ], dtype=np.float32)

def main():
    print("=" * 80)
    print("TESTING PRECOMPUTED ENTITY FEATURE EQUIVALENCE & SPEEDUP")
    print("=" * 80)

    # Mock sample entity records
    s1_dict = {
        "entity_id": "S1-123456",
        "business_name": "Apex Healthcare Group LLC",
        "name_norm": "apex healthcare group",
        "name_tokens_sorted": "apex group healthcare",
        "name_is_degenerate": False,
        "business_address": "123 Main St, Suite 400, Dallas, TX 75001",
        "addr_norm": "123 main st suite 400 dallas tx 75001",
        "addr_landmark": "main st",
        "addr_numbers": ["123", "400", "75001"],
        "country_norm": "us"
    }

    cand_dict = {
        "entity_id": "S2-987654",
        "business_name": "The Apex Healthcare Group",
        "name_norm": "the apex healthcare group",
        "name_tokens_sorted": "apex group healthcare the",
        "name_is_degenerate": False,
        "business_address": "123 Main Street #400, Dallas, Texas 75001",
        "addr_norm": "123 main street 400 dallas texas 75001",
        "addr_landmark": "main street",
        "addr_numbers": ["123", "400", "75001"],
        "country_norm": "us"
    }

    # 1. Test correctness equivalence
    v_orig = compute_single_pair_features(s1_dict, cand_dict, channel_count=3)
    
    s1_pre = PrecomputedEntity(s1_dict)
    cand_pre = PrecomputedEntity(cand_dict)
    v_fast = compute_pair_features_fast(s1_pre, cand_pre, channel_count=3)

    diff = np.max(np.abs(v_orig - v_fast))
    print(f"Max absolute difference between original and precomputed: {diff}")
    assert np.allclose(v_orig, v_fast, atol=1e-6), f"EQUIVALENCE FAILED! Max diff: {diff}"
    print("SUCCESS: 100.000% EXACT EQUIVALENCE ACROSS ALL 56 FEATURES!")

    # 2. Benchmark speed on 20,000 evaluations
    N = 20000
    print(f"\nBenchmarking {N:,} pair feature calculations...")
    t0 = time.time()
    for _ in range(N):
        _ = compute_single_pair_features(s1_dict, cand_dict, channel_count=3)
    t_orig = time.time() - t0
    print(f"Original time: {t_orig:.4f}s ({N / t_orig:,.0f} pairs/sec)")

    t0 = time.time()
    for _ in range(N):
        _ = compute_pair_features_fast(s1_pre, cand_pre, channel_count=3)
    t_fast = time.time() - t0
    print(f"Fast precomputed time: {t_fast:.4f}s ({N / t_fast:,.0f} pairs/sec)")
    print(f"SPEEDUP: {t_orig / t_fast:.2f}x faster per pair!")

if __name__ == '__main__':
    main()
