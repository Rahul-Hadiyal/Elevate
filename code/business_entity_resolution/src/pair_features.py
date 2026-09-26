"""Pairwise feature computation module for Business Entity Resolution.

Computes fine-grained, CPU-efficient similarity and agreement features between
S1 entity records and candidate S2/S3 entity records without label leakage.
"""

import re
from typing import Dict, List, Any, Tuple, Optional, Set
import numpy as np
from rapidfuzz import fuzz

from src.index_builder import canonicalize_phonetic, extract_postal_code


# Feature column names metadata
FEATURE_NAMES: List[str] = [
    # 1. Exact & Identity features
    "feat_name_exact_raw",
    "feat_name_exact_norm",
    "feat_name_exact_sorted",
    "feat_name_phonetic_exact",
    # 2. Length & Count differences
    "feat_name_len_diff_abs",
    "feat_name_len_diff_rel",
    "feat_name_tok_count_diff",
    # 3. Token-level overlap & set similarities
    "feat_name_tok_overlap_count",
    "feat_name_tok_jaccard",
    "feat_name_tok_containment",
    "feat_name_tok_overlap_coeff",
    "feat_name_first_tok_match",
    "feat_name_last_tok_match",
    "feat_name_is_prefix",
    # 4. Fuzzy character & token similarities (RapidFuzz)
    "feat_name_fuzz_ratio",
    "feat_name_fuzz_partial_ratio",
    "feat_name_fuzz_token_sort",
    "feat_name_fuzz_token_set",
    "feat_name_fuzz_wratio",
    # 5. Character N-Gram similarities
    "feat_name_char_2gram_jaccard",
    "feat_name_char_3gram_jaccard",
    # 6. Address Exact & Token similarities
    "feat_addr_exact_norm",
    "feat_addr_exact_landmark",
    "feat_addr_len_diff_abs",
    "feat_addr_tok_count_diff",
    "feat_addr_tok_overlap_count",
    "feat_addr_tok_jaccard",
    "feat_addr_tok_containment",
    # 7. Address Fuzzy similarities
    "feat_addr_fuzz_ratio",
    "feat_addr_fuzz_partial_ratio",
    "feat_addr_fuzz_token_sort",
    "feat_addr_fuzz_token_set",
    # 8. Address Numeric Anchor features
    "feat_addr_num_count_s1",
    "feat_addr_num_count_cand",
    "feat_addr_num_overlap_count",
    "feat_addr_num_jaccard",
    "feat_addr_num_exact_match",
    "feat_addr_num_disagreement",
    # 9. Postal PIN / ZIP features
    "feat_postal_both_present",
    "feat_postal_exact_match",
    # 10. Country features
    "feat_country_exact_match",
    "feat_country_s1_missing",
    "feat_country_cand_missing",
    # 11. Cross-field interaction features
    "feat_cross_name_addr_fuzz_prod",
    "feat_cross_name_exact_addr_overlap",
    "feat_cross_name_fuzz_num_match",
    "feat_cross_name_high_addr_low",
    "feat_cross_addr_high_name_low",
    "feat_cross_name_postal_agree",
    # 12. Candidate Source & Channel metadata
    "feat_cand_is_s2",
    "feat_cand_is_s3",
    "feat_cand_channel_count",
    # 13. Missingness & Degenerate indicators
    "feat_s1_name_is_degenerate",
    "feat_cand_name_is_degenerate",
    "feat_s1_addr_is_empty",
    "feat_cand_addr_is_empty",
]


def compute_char_ngrams(text: str, n: int) -> Set[str]:
    """Extract character n-grams from string."""
    if len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def jaccard_set_similarity(s1: Set[Any], s2: Set[Any]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not s1 or not s2:
        return 0.0
    u = len(s1 | s2)
    return len(s1 & s2) / u if u > 0 else 0.0


def compute_single_pair_features(
    s1_row: Dict[str, Any],
    cand_row: Dict[str, Any],
    channel_count: int = 1,
) -> np.ndarray:
    """Compute feature vector for a single (S1, Candidate) pair.

    Args:
        s1_row: Dict containing S1 record fields (raw & normalized).
        cand_row: Dict containing Candidate record fields (raw & normalized).
        channel_count: Number of blocking channels that produced this candidate pair.

    Returns:
        np.ndarray of shape (len(FEATURE_NAMES),) with float32 values.
    """
    # 1. Names
    s1_name_raw = str(s1_row.get("business_name", "") or "").strip()
    cand_name_raw = str(cand_row.get("business_name", "") or "").strip()

    s1_name_norm = str(s1_row.get("name_norm", "") or "").strip()
    cand_name_norm = str(cand_row.get("name_norm", "") or "").strip()

    s1_name_sorted = str(s1_row.get("name_tokens_sorted", "") or "").strip()
    cand_name_sorted = str(cand_row.get("name_tokens_sorted", "") or "").strip()

    s1_name_degen = bool(s1_row.get("name_is_degenerate", False))
    cand_name_degen = bool(cand_row.get("name_is_degenerate", False))

    # Phonetics
    s1_ph = canonicalize_phonetic(s1_name_norm)
    cand_ph = canonicalize_phonetic(cand_name_norm)

    # 2. Addresses
    s1_addr_raw = str(s1_row.get("business_address", "") or "").strip()
    cand_addr_raw = str(cand_row.get("business_address", "") or "").strip()

    s1_addr_norm = str(s1_row.get("addr_norm", "") or "").strip()
    cand_addr_norm = str(cand_row.get("addr_norm", "") or "").strip()

    s1_lm = str(s1_row.get("addr_landmark", "") or "").strip()
    cand_lm = str(cand_row.get("addr_landmark", "") or "").strip()

    s1_nums = s1_row.get("addr_numbers", [])
    if not isinstance(s1_nums, (list, set, tuple)):
        s1_nums = []
    s1_num_set = {str(x).strip().lower() for x in s1_nums if str(x).strip()}

    cand_nums = cand_row.get("addr_numbers", [])
    if not isinstance(cand_nums, (list, set, tuple)):
        cand_nums = []
    cand_num_set = {str(x).strip().lower() for x in cand_nums if str(x).strip()}

    # 3. Country
    s1_c = str(s1_row.get("country_norm", "") or "").strip()
    cand_c = str(cand_row.get("country_norm", "") or "").strip()

    # 4. Source ID
    cand_eid = str(cand_row.get("entity_id", "") or "")
    is_s2 = 1.0 if (cand_eid.startswith("S2-") or cand_eid.startswith("S2_")) else 0.0
    is_s3 = 1.0 if (cand_eid.startswith("S3-") or cand_eid.startswith("S3_")) else 0.0

    # ------------------ FEATURE EXTRACTION ------------------
    # Name Exact
    f_name_exact_raw = 1.0 if (s1_name_raw and s1_name_raw == cand_name_raw) else 0.0
    f_name_exact_norm = 1.0 if (s1_name_norm and s1_name_norm == cand_name_norm) else 0.0
    f_name_exact_sorted = 1.0 if (s1_name_sorted and s1_name_sorted == cand_name_sorted) else 0.0
    f_name_ph_exact = 1.0 if (s1_ph and s1_ph == cand_ph) else 0.0

    # Name Length & Counts
    s1_name_len = len(s1_name_norm)
    cand_name_len = len(cand_name_norm)
    f_name_len_diff_abs = float(abs(s1_name_len - cand_name_len))
    max_len = max(s1_name_len, cand_name_len)
    f_name_len_diff_rel = (f_name_len_diff_abs / max_len) if max_len > 0 else 0.0

    s1_toks = s1_name_norm.split()
    cand_toks = cand_name_norm.split()
    s1_tok_set = set(s1_toks)
    cand_tok_set = set(cand_toks)

    f_name_tok_count_diff = float(abs(len(s1_toks) - len(cand_toks)))

    # Name Tokens
    overlap_toks = s1_tok_set & cand_tok_set
    f_name_tok_overlap_count = float(len(overlap_toks))
    f_name_tok_jaccard = jaccard_set_similarity(s1_tok_set, cand_tok_set)

    min_tok_len = min(len(s1_tok_set), len(cand_tok_set)) if (s1_tok_set and cand_tok_set) else 0
    f_name_tok_containment = (len(overlap_toks) / min_tok_len) if min_tok_len > 0 else 0.0
    f_name_tok_overlap_coeff = f_name_tok_containment

    f_name_first_tok_match = 1.0 if (s1_toks and cand_toks and s1_toks[0] == cand_toks[0]) else 0.0
    f_name_last_tok_match = 1.0 if (s1_toks and cand_toks and s1_toks[-1] == cand_toks[-1]) else 0.0
    f_name_is_prefix = 1.0 if (s1_name_norm and cand_name_norm and (s1_name_norm.startswith(cand_name_norm) or cand_name_norm.startswith(s1_name_norm))) else 0.0

    # RapidFuzz Name Similarities
    if s1_name_norm and cand_name_norm:
        f_name_fuzz_ratio = float(fuzz.ratio(s1_name_norm, cand_name_norm))
        f_name_fuzz_partial = float(fuzz.partial_ratio(s1_name_norm, cand_name_norm))
        f_name_fuzz_tsort = float(fuzz.token_sort_ratio(s1_name_norm, cand_name_norm))
        f_name_fuzz_tset = float(fuzz.token_set_ratio(s1_name_norm, cand_name_norm))
        f_name_fuzz_wratio = float(fuzz.WRatio(s1_name_norm, cand_name_norm))
    else:
        f_name_fuzz_ratio = 0.0
        f_name_fuzz_partial = 0.0
        f_name_fuzz_tsort = 0.0
        f_name_fuzz_tset = 0.0
        f_name_fuzz_wratio = 0.0

    # Character N-Grams Name
    s1_2g = compute_char_ngrams(s1_name_norm, 2)
    cand_2g = compute_char_ngrams(cand_name_norm, 2)
    f_name_2g_jaccard = jaccard_set_similarity(s1_2g, cand_2g)

    s1_3g = compute_char_ngrams(s1_name_norm, 3)
    cand_3g = compute_char_ngrams(cand_name_norm, 3)
    f_name_3g_jaccard = jaccard_set_similarity(s1_3g, cand_3g)

    # Address Exact & Token
    f_addr_exact_norm = 1.0 if (s1_addr_norm and s1_addr_norm == cand_addr_norm) else 0.0
    f_addr_exact_lm = 1.0 if (s1_lm and s1_lm == cand_lm) else 0.0

    s1_addr_len = len(s1_addr_norm)
    cand_addr_len = len(cand_addr_norm)
    f_addr_len_diff_abs = float(abs(s1_addr_len - cand_addr_len))

    s1_addr_toks = s1_addr_norm.split()
    cand_addr_toks = cand_addr_norm.split()
    s1_addr_tok_set = set(s1_addr_toks)
    cand_addr_tok_set = set(cand_addr_toks)

    f_addr_tok_count_diff = float(abs(len(s1_addr_toks) - len(cand_addr_toks)))

    overlap_addr_toks = s1_addr_tok_set & cand_addr_tok_set
    f_addr_tok_overlap_count = float(len(overlap_addr_toks))
    f_addr_tok_jaccard = jaccard_set_similarity(s1_addr_tok_set, cand_addr_tok_set)

    min_addr_tok = min(len(s1_addr_tok_set), len(cand_addr_tok_set)) if (s1_addr_tok_set and cand_addr_tok_set) else 0
    f_addr_tok_containment = (len(overlap_addr_toks) / min_addr_tok) if min_addr_tok > 0 else 0.0

    # Address RapidFuzz
    if s1_addr_norm and cand_addr_norm:
        f_addr_fuzz_ratio = float(fuzz.ratio(s1_addr_norm, cand_addr_norm))
        f_addr_fuzz_partial = float(fuzz.partial_ratio(s1_addr_norm, cand_addr_norm))
        f_addr_fuzz_tsort = float(fuzz.token_sort_ratio(s1_addr_norm, cand_addr_norm))
        f_addr_fuzz_tset = float(fuzz.token_set_ratio(s1_addr_norm, cand_addr_norm))
    else:
        f_addr_fuzz_ratio = 0.0
        f_addr_fuzz_partial = 0.0
        f_addr_fuzz_tsort = 0.0
        f_addr_fuzz_tset = 0.0

    # Address Numeric Features
    f_addr_num_count_s1 = float(len(s1_num_set))
    f_addr_num_count_cand = float(len(cand_num_set))
    overlap_nums = s1_num_set & cand_num_set
    f_addr_num_overlap_count = float(len(overlap_nums))
    f_addr_num_jaccard = jaccard_set_similarity(s1_num_set, cand_num_set)
    f_addr_num_exact_match = 1.0 if (s1_num_set and cand_num_set and s1_num_set == cand_num_set) else 0.0
    f_addr_num_disagree = 1.0 if (s1_num_set and cand_num_set and not overlap_nums) else 0.0

    # Postal PIN / ZIP features
    s1_pin = extract_postal_code(s1_addr_raw, s1_addr_norm)
    cand_pin = extract_postal_code(cand_addr_raw, cand_addr_norm)
    f_postal_both_present = 1.0 if (s1_pin and cand_pin) else 0.0
    f_postal_exact_match = 1.0 if (s1_pin and cand_pin and s1_pin == cand_pin) else 0.0

    # Country features
    f_country_exact_match = 1.0 if (s1_c and cand_c and s1_c == cand_c) else 0.0
    f_country_s1_missing = 1.0 if (not s1_c or s1_c == "unknown") else 0.0
    f_country_cand_missing = 1.0 if (not cand_c or cand_c == "unknown") else 0.0

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
    f_s1_name_degen = 1.0 if s1_name_degen else 0.0
    f_cand_name_degen = 1.0 if cand_name_degen else 0.0
    f_s1_addr_empty = 1.0 if not s1_addr_norm else 0.0
    f_cand_addr_empty = 1.0 if not cand_addr_norm else 0.0

    feat_vector = np.array([
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
        is_s2,
        is_s3,
        f_cand_channel_count,
        f_s1_name_degen,
        f_cand_name_degen,
        f_s1_addr_empty,
        f_cand_addr_empty,
    ], dtype=np.float32)

    return feat_vector
