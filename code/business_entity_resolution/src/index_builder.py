"""Index Builder Module for Multi-Channel Candidate Blocking.

Constructs high-throughput inverted indexes, signature maps, and sparse TF-IDF
structures over Source 2 and Source 3 entity records:
- Channel A: Exact Normalized Name + Country
- Channel B: Name Token Signatures (Sorted Tokens, Core Sorted, Prefix-2 + Country)
- Channel C: Address Numeric Anchors (Street Number + Name Initial-3 + Country)
- Channel D: Address Inverted Index (Postal / Street Token + Name Initial-3 + Country)
- Channel E: Rare Name Token Inverted Index (Low document frequency tokens + Country)
- Channel H: Landmark Anchor + Name Initial-3 + Country
- Channel I: Phonetic / Transliteration Key Indexes (Phonetic Name, Phonetic Sorted, Phonetic Prefix-2 + Country)
- Channel J: Core Token Pair Inverted Index (Combinations of informative name tokens + Country)
- Channel K: Address Number + Street Token Inverted Index (Primary Number + Street Token + Country)
- Channel L: Postal PIN / ZIP Code + Name Anchor Inverted Index (PIN/ZIP + Name Token + Country)
- Channel M: Rare Character 3-Gram Inverted Index (Low-DF 3-gram pairs + Country)
"""

from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import logging
import re
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


logger = logging.getLogger(__name__)

# Generalized high-frequency phonetic & transliteration mappings
PHONETIC_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    (r"\bshree\b|\bshri\b", "sri"),
    (r"\blaxmi\b", "lakshmi"),
    (r"\bchoudhary\b|\bchoudhury\b|\bchaudhry\b", "chaudhary"),
    (r"\benterprises\b|\benterprise\b|\bent\b", "enterprise"),
    (r"\bassociates\b|\bassoc\b", "associate"),
    (r"\btraders\b|\btrader\b", "trader"),
    (r"\bservices\b|\bservice\b|\bserv\b", "service"),
    (r"\btechnologies\b|\btechnology\b|\btech\b", "tech"),
    (r"\bindustries\b|\bindustry\b|\bind\b", "industry"),
    (r"\bcentre\b|\bcenter\b", "center"),
    (r"\binternational\b|\bintl\b", "international"),
    (r"\bpharmaceuticals\b|\bpharma\b", "pharma"),
    (r"\bcorporation\b|\bcorp\b", "corp"),
    (r"\bmedical\b|\bmed\b", "medical"),
    (r"\bbrothers\b|\bbros\b", "bros"),
    (r"\bagencies\b|\bagency\b", "agency"),
)

PIN_REGEX = re.compile(r"\b(?:[1-9][0-9]{5}|[0-9]{5})\b")


def canonicalize_phonetic(name_norm: str) -> str:
    """Applies high-frequency phonetic and transliteration canonicalizations."""
    if not name_norm:
        return ""
    t = name_norm.lower()
    for pat, rep in PHONETIC_REPLACEMENTS:
        t = re.sub(pat, rep, t)
    # Collapse repeating double consonants: e.g. "millennium" -> "milenium", "aggarwal" -> "agarwal"
    t = re.sub(r"([a-z])\1+", r"\1", t)
    return t.strip()


def extract_postal_code(addr_raw: str, addr_norm: str) -> str:
    """Extracts 6-digit Indian PIN code or 5-digit US ZIP code."""
    for text in (addr_raw, addr_norm):
        if not text:
            continue
        m = PIN_REGEX.search(str(text))
        if m:
            return m.group(0)
    return ""


class BlockingIndex:
    """Unified container for multi-channel blocking indices over Source 2 and Source 3."""

    def __init__(
        self,
        max_token_df: int = 500,
        min_token_len: int = 3,
    ):
        """Initializes blocking index structures.
        
        Args:
            max_token_df: Maximum document frequency for inverted index tokens to prevent candidate explosions.
            min_token_len: Minimum token character length for indexing.
        """
        self.max_token_df = max_token_df
        self.min_token_len = min_token_len

        # Channel A: (name_norm, country_norm) -> [cand_id, ...]
        self.idx_exact_name_country: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        # Channel B: (name_tokens_sorted, country_norm) & (prefix_2, country_norm)
        self.idx_sorted_name_country: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_core_sorted_country: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_name_prefix2_country: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        # Channel C: (primary_number, name_initial_3, country_norm) -> [cand_id, ...]
        self.idx_addr_num_name3_country: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)

        # Channel D: (addr_token, name_initial_3, country_norm) -> [cand_id, ...]
        self.idx_addr_tok_name3_country: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)

        # Channel E: (rare_name_token, country_norm) -> [cand_id, ...]
        self.idx_rare_name_tok_country: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        # Channel H: (addr_landmark, country_norm) & (addr_landmark, name_initial_3, country_norm)
        self.idx_landmark_country: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_landmark_name3_country: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)

        # Channel I: Phonetic Key Indexes
        self.idx_phonetic_name_country: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_phonetic_sorted_country: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.idx_phonetic_prefix2_country: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        # Channel J: Core Token Pair Inverted Index
        self.idx_core_pair_country: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)

        # Channel K: Address Number + Street Token Inverted Index (no name initial required)
        self.idx_addr_num_street_country: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)

        # Channel L: Postal PIN / ZIP Code + Name Token Anchor
        self.idx_pin_name_tok_country: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)

        # Channel M: Rare Character 3-Gram Pair Inverted Index
        self.idx_char_3gram_pair_country: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)

        # Global document frequency trackers
        self._name_token_df: Counter = Counter()
        self._addr_token_df: Counter = Counter()
        self._char_3gram_df: Counter = Counter()

    def build_indexes(
        self,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        include_tfidf: bool = False,
    ) -> "BlockingIndex":
        """Builds all candidate blocking indices over S2 and S3 DataFrames.

        Args:
            s2_df: Normalized Source 2 DataFrame.
            s3_df: Normalized Source 3 DataFrame.
            include_tfidf: Whether to fit and transform sparse TF-IDF matrices.

        Returns:
            self
        """
        logger.info("Indexing Source 2 and Source 3 records for candidate generation...")

        # Step 1: Pre-calculate token frequencies
        for df in (s2_df, s3_df):
            if df is None or df.empty:
                continue
            name_norms = df["name_norm"].fillna("").astype(str).tolist()
            addr_norms = df["addr_norm"].fillna("").astype(str).tolist()
            for name in name_norms:
                toks = set(name.split())
                for t in toks:
                    if len(t) >= self.min_token_len:
                        self._name_token_df[t] += 1
                if len(name) >= 3:
                    for i in range(len(name) - 2):
                        g = name[i:i+3]
                        self._char_3gram_df[g] += 1

            for addr in addr_norms:
                toks = set(addr.split())
                for t in toks:
                    if len(t) >= self.min_token_len:
                        self._addr_token_df[t] += 1

        # Step 2: Populate inverted indices
        for df in (s2_df, s3_df):
            if df is None or df.empty:
                continue
            eids = df["entity_id"].astype(str).str.strip().tolist()
            name_norms = df["name_norm"].fillna("").astype(str).str.strip().tolist()
            name_sorteds = df["name_tokens_sorted"].fillna("").astype(str).str.strip().tolist()
            addr_raws = df["business_address"].fillna("").astype(str).tolist() if "business_address" in df.columns else [""] * len(eids)
            addr_norms = df["addr_norm"].fillna("").astype(str).str.strip().tolist()
            addr_landmarks = df["addr_landmark"].fillna("").astype(str).str.strip().tolist()
            addr_numbers = df["addr_numbers"].tolist()
            country_norms = df["country_norm"].fillna("unknown").astype(str).str.strip().tolist()
            name_degens = df["name_is_degenerate"].tolist()

            for i in range(len(eids)):
                eid = eids[i]
                name_norm = name_norms[i]
                name_sorted = name_sorteds[i]
                addr_raw = addr_raws[i]
                addr_norm = addr_norms[i]
                addr_lm = addr_landmarks[i]
                nums = addr_numbers[i]
                c_norm = country_norms[i]
                is_degen = name_degens[i]

                if not name_norm or is_degen:
                    continue

                name_tokens = name_norm.split()
                name_initial_3 = name_norm[:3]

                # Channel A: Exact name + country
                self.idx_exact_name_country[(name_norm, c_norm)].append(eid)

                # Channel B1: Sorted tokens + country
                if name_sorted and name_sorted != name_norm:
                    self.idx_sorted_name_country[(name_sorted, c_norm)].append(eid)

                # Channel B2: First 2 tokens + country (for multi-token names)
                if len(name_tokens) >= 2:
                    p2 = f"{name_tokens[0]} {name_tokens[1]}"
                    self.idx_name_prefix2_country[(p2, c_norm)].append(eid)

                # Channel C: Primary Address Number + Name Initial 3 + Country
                primary_num = ""
                if isinstance(nums, list) and len(nums) > 0:
                    primary_num = str(nums[0]).strip().lower()
                    if primary_num:
                        self.idx_addr_num_name3_country[(primary_num, name_initial_3, c_norm)].append(eid)

                # Channel D: Address Distinctive Token + Name Initial 3 + Country
                if addr_norm:
                    addr_tokens = addr_norm.split()
                    for atok in addr_tokens:
                        if len(atok) >= self.min_token_len and self._addr_token_df.get(atok, 0) <= self.max_token_df:
                            self.idx_addr_tok_name3_country[(atok, name_initial_3, c_norm)].append(eid)

                # Channel E: Rare Distinctive Name Token + Country
                if len(name_tokens) >= 2:
                    for ntok in name_tokens:
                        if len(ntok) >= self.min_token_len:
                            df_count = self._name_token_df.get(ntok, 0)
                            if 2 <= df_count <= self.max_token_df:
                                self.idx_rare_name_tok_country[(ntok, c_norm)].append(eid)

                # Channel H: Landmark component + Country
                if addr_lm:
                    self.idx_landmark_country[(addr_lm, c_norm)].append(eid)
                    self.idx_landmark_name3_country[(addr_lm, name_initial_3, c_norm)].append(eid)

                # Channel I: Phonetic / Transliteration Key Indexes
                ph_name = canonicalize_phonetic(name_norm)
                if ph_name:
                    self.idx_phonetic_name_country[(ph_name, c_norm)].append(eid)
                    ph_tokens = ph_name.split()
                    if len(ph_tokens) >= 2:
                        ph_sorted = " ".join(sorted(ph_tokens))
                        ph_p2 = f"{ph_tokens[0]} {ph_tokens[1]}"
                        self.idx_phonetic_sorted_country[(ph_sorted, c_norm)].append(eid)
                        self.idx_phonetic_prefix2_country[(ph_p2, c_norm)].append(eid)

                # Channel J: Core Token Pair Inverted Index
                core_name_toks = [
                    t for t in name_tokens
                    if len(t) >= self.min_token_len and self._name_token_df.get(t, 0) <= 2000
                ]
                if 2 <= len(core_name_toks) <= 6:
                    for t1, t2 in combinations(sorted(core_name_toks), 2):
                        self.idx_core_pair_country[(t1, t2, c_norm)].append(eid)

                # Channel K: Address Number + Street Token Inverted Index
                if primary_num and addr_norm:
                    for atok in addr_norm.split()[:4]:
                        if len(atok) >= self.min_token_len and atok != primary_num:
                            if self._addr_token_df.get(atok, 0) <= 2000:
                                self.idx_addr_num_street_country[(primary_num, atok, c_norm)].append(eid)

                # Channel L: Postal PIN / ZIP Code + Name Token Anchor
                pin_code = extract_postal_code(addr_raw, addr_norm)
                if pin_code and len(name_tokens) >= 1:
                    first_tok = name_tokens[0]
                    if len(first_tok) >= self.min_token_len:
                        self.idx_pin_name_tok_country[(pin_code, first_tok, c_norm)].append(eid)

                # Channel M: Rare Character 3-Gram Pair Inverted Index
                for ntok in name_tokens:
                    if len(ntok) >= 5:
                        grams = [ntok[j:j+3] for j in range(len(ntok) - 2)]
                        rare_grams = sorted(set(grams), key=lambda g: self._char_3gram_df.get(g, 0))
                        if len(rare_grams) >= 2:
                            g1, g2 = sorted([rare_grams[0], rare_grams[1]])
                            if self._char_3gram_df.get(g1, 0) <= 5000 and self._char_3gram_df.get(g2, 0) <= 5000:
                                self.idx_char_3gram_pair_country[(g1, g2, c_norm)].append(eid)


        logger.info(
            f"BlockingIndex built: "
            f"Exact: {len(self.idx_exact_name_country):,}, "
            f"Phonetic: {len(self.idx_phonetic_name_country):,}, "
            f"Core Pairs: {len(self.idx_core_pair_country):,}, "
            f"Addr Num+Street: {len(self.idx_addr_num_street_country):,}, "
            f"PIN+Name: {len(self.idx_pin_name_tok_country):,}, "
            f"3-Gram Pairs: {len(self.idx_char_3gram_pair_country):,}"
        )

        return self
