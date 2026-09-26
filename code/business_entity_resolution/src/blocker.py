"""Multi-Channel Candidate Generation and Blocking Engine.

Implements production-grade, CPU-efficient candidate blocking channels:
- Channel A: Exact Normalized Name + Country
- Channel B: Name Token Signatures (Sorted tokens, First-2 tokens + Country)
- Channel C: Address Numeric Anchors (Primary number + Name initial-3 + Country)
- Channel D: Address Inverted Index (Distinctive address tokens + Name initial-3 + Country)
- Channel E: Rare Name Token Inverted Index (Low document frequency tokens + Country)
- Channel F: Sparse Character 3-gram TF-IDF Retrieval
- Channel G: Controlled 4-gram Prefix + Address Anchors
- Channel H: Landmark Anchor + Name Anchor Cross-Field Blocking
- Channel I: Phonetic & Transliteration Canonicalization Blocking
- Channel J: Core Token Pair Inverted Index
- Channel K: Address Number + Street Token Inverted Index (No Name Initial constraint)
- Channel L: Postal PIN / ZIP Code + Name Token Anchor
- Channel M: Rare Character 3-Gram Inverted Index

All channels include candidate explosion guards (per-key caps) and open-set country support.
"""

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import logging
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

try:
    from src.index_builder import (
        BlockingIndex,
        canonicalize_phonetic,
        extract_postal_code,
    )
    from src.candidate_store import CandidateStore
except ImportError:
    from index_builder import (
        BlockingIndex,
        canonicalize_phonetic,
        extract_postal_code,
    )
    from candidate_store import CandidateStore


logger = logging.getLogger(__name__)


class MultiChannelBlocker:
    """Multi-channel candidate generation engine."""

    def __init__(
        self,
        index: Optional[BlockingIndex] = None,
        max_cands_per_key: int = 100,
    ):
        """Initializes blocker with index.
        
        Args:
            index: Pre-built BlockingIndex.
            max_cands_per_key: Hard safety cap per index key to prevent candidate explosion.
        """
        self.index = index or BlockingIndex()
        self.max_cands_per_key = max_cands_per_key

    def fit(
        self,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
        include_tfidf: bool = False,
    ) -> "MultiChannelBlocker":
        """Fits blocking index over Source 2 and Source 3 DataFrames."""
        self.index.build_indexes(s2_df, s3_df, include_tfidf=include_tfidf)
        return self

    def _extract_s1_vectors(self, s1_df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], List[list], List[str], List[bool]]:
        """Fast vectorized extraction of columns from S1 DataFrame."""
        eids = s1_df["entity_id"].astype(str).str.strip().tolist()
        name_norms = s1_df["name_norm"].fillna("").astype(str).str.strip().tolist() if "name_norm" in s1_df.columns else [""] * len(eids)
        name_sorteds = s1_df["name_tokens_sorted"].fillna("").astype(str).str.strip().tolist() if "name_tokens_sorted" in s1_df.columns else [""] * len(eids)
        addr_raws = s1_df["business_address"].fillna("").astype(str).tolist() if "business_address" in s1_df.columns else [""] * len(eids)
        addr_norms = s1_df["addr_norm"].fillna("").astype(str).str.strip().tolist() if "addr_norm" in s1_df.columns else [""] * len(eids)
        addr_landmarks = s1_df["addr_landmark"].fillna("").astype(str).str.strip().tolist() if "addr_landmark" in s1_df.columns else [""] * len(eids)
        addr_numbers = s1_df["addr_numbers"].tolist() if "addr_numbers" in s1_df.columns else [[]] * len(eids)
        country_norms = s1_df["country_norm"].fillna("unknown").astype(str).str.strip().tolist() if "country_norm" in s1_df.columns else ["unknown"] * len(eids)
        name_degens = s1_df["name_is_degenerate"].tolist() if "name_is_degenerate" in s1_df.columns else [False] * len(eids)

        return eids, name_norms, name_sorteds, addr_raws, addr_norms, addr_landmarks, addr_numbers, country_norms, name_degens

    def generate_channel_a(self, s1_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Channel A: Exact Normalized Name + Country."""
        eids, name_norms, _, _, _, _, _, country_norms, degens = self._extract_s1_vectors(s1_df)
        cands: Dict[str, List[str]] = {}

        for eid, name_norm, c_norm, degen in zip(eids, name_norms, country_norms, degens):
            if not name_norm or degen:
                cands[eid] = []
                continue
            key = (name_norm, c_norm)
            matched = self.index.idx_exact_name_country.get(key, [])
            if len(matched) > self.max_cands_per_key:
                matched = matched[:self.max_cands_per_key]
            cands[eid] = list(matched)

        return cands

    def generate_channel_b(self, s1_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Channel B: Name Token Signatures (Sorted tokens & Prefix-2 + Country)."""
        eids, name_norms, name_sorteds, _, _, _, _, country_norms, degens = self._extract_s1_vectors(s1_df)
        cands: Dict[str, List[str]] = {}

        for eid, name_norm, name_sorted, c_norm, degen in zip(eids, name_norms, name_sorteds, country_norms, degens):
            if not name_norm or degen:
                cands[eid] = []
                continue

            found: Set[str] = set()
            # B1: Sorted tokens
            if name_sorted:
                key_s = (name_sorted, c_norm)
                for mid in self.index.idx_sorted_name_country.get(key_s, [])[:self.max_cands_per_key]:
                    found.add(mid)

            # B2: Prefix 2 tokens (for multi-token names)
            toks = name_norm.split()
            if len(toks) >= 2:
                p2 = f"{toks[0]} {toks[1]}"
                key_p = (p2, c_norm)
                for mid in self.index.idx_name_prefix2_country.get(key_p, [])[:self.max_cands_per_key]:
                    found.add(mid)

            cands[eid] = list(found)

        return cands

    def generate_channel_c(self, s1_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Channel C: Address Numeric Anchors (Primary number + Name initial-3 + Country)."""
        eids, name_norms, _, _, _, _, addr_numbers, country_norms, degens = self._extract_s1_vectors(s1_df)
        cands: Dict[str, List[str]] = {}

        for eid, name_norm, nums, c_norm, degen in zip(eids, name_norms, addr_numbers, country_norms, degens):
            if not name_norm or degen or not isinstance(nums, list) or len(nums) == 0:
                cands[eid] = []
                continue

            primary_num = str(nums[0]).strip().lower()
            if not primary_num:
                cands[eid] = []
                continue

            name_initial_3 = name_norm[:3]
            key = (primary_num, name_initial_3, c_norm)
            matched = self.index.idx_addr_num_name3_country.get(key, [])
            if len(matched) > self.max_cands_per_key:
                matched = matched[:self.max_cands_per_key]
            cands[eid] = list(matched)

        return cands

    def generate_channel_d(self, s1_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Channel D: Address Inverted Index (Distinctive address tokens + Name initial-3 + Country)."""
        eids, name_norms, _, _, addr_norms, _, _, country_norms, degens = self._extract_s1_vectors(s1_df)
        cands: Dict[str, List[str]] = {}

        for eid, name_norm, addr_norm, c_norm, degen in zip(eids, name_norms, addr_norms, country_norms, degens):
            if not name_norm or degen or not addr_norm:
                cands[eid] = []
                continue

            name_initial_3 = name_norm[:3]
            found: Set[str] = set()
            addr_toks = addr_norm.split()

            for atok in addr_toks:
                if len(atok) >= self.index.min_token_len:
                    df_cnt = self.index._addr_token_df.get(atok, 0)
                    if 2 <= df_cnt <= self.index.max_token_df:
                        key = (atok, name_initial_3, c_norm)
                        for mid in self.index.idx_addr_tok_name3_country.get(key, [])[:30]:
                            found.add(mid)
                            if len(found) >= self.max_cands_per_key:
                                break
                if len(found) >= self.max_cands_per_key:
                    break

            cands[eid] = list(found)

        return cands

    def generate_channel_e(self, s1_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Channel E: Rare Distinctive Name Tokens + Country."""
        eids, name_norms, _, _, _, _, _, country_norms, degens = self._extract_s1_vectors(s1_df)
        cands: Dict[str, List[str]] = {}

        for eid, name_norm, c_norm, degen in zip(eids, name_norms, country_norms, degens):
            if not name_norm or degen:
                cands[eid] = []
                continue

            toks = name_norm.split()
            if len(toks) < 2:
                cands[eid] = []
                continue

            found: Set[str] = set()
            for ntok in toks:
                if len(ntok) >= self.index.min_token_len:
                    df_cnt = self.index._name_token_df.get(ntok, 0)
                    if 2 <= df_cnt <= self.index.max_token_df:
                        key = (ntok, c_norm)
                        for mid in self.index.idx_rare_name_tok_country.get(key, [])[:30]:
                            found.add(mid)
                            if len(found) >= self.max_cands_per_key:
                                break
                if len(found) >= self.max_cands_per_key:
                    break

            cands[eid] = list(found)

        return cands

    def generate_channel_g(self, s1_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Channel G: Controlled 4-gram Prefix + Address Anchor Blocking."""
        eids, name_norms, _, _, addr_norms, _, addr_numbers, country_norms, degens = self._extract_s1_vectors(s1_df)
        cands: Dict[str, List[str]] = {}

        for eid, name_norm, addr_norm, nums, c_norm, degen in zip(eids, name_norms, addr_norms, addr_numbers, country_norms, degens):
            if not name_norm or len(name_norm) < 4 or degen:
                cands[eid] = []
                continue

            found: Set[str] = set()
            # If address has numbers, check number + 3-char prefix
            if isinstance(nums, list) and len(nums) > 0:
                p_num = str(nums[0]).strip().lower()
                key = (p_num, name_norm[:3], c_norm)
                for mid in self.index.idx_addr_num_name3_country.get(key, [])[:30]:
                    found.add(mid)

            # Check address distinctive tokens with 3-char prefix
            if addr_norm:
                for atok in addr_norm.split()[:3]:
                    if len(atok) >= self.index.min_token_len:
                        key_a = (atok, name_norm[:3], c_norm)
                        for mid in self.index.idx_addr_tok_name3_country.get(key_a, [])[:20]:
                            found.add(mid)
                            if len(found) >= self.max_cands_per_key:
                                break

            cands[eid] = list(found)

        return cands

    def generate_channel_h(self, s1_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Channel H: Landmark Component + Name Anchor Cross-Field Blocking."""
        eids, name_norms, _, _, _, addr_landmarks, _, country_norms, degens = self._extract_s1_vectors(s1_df)
        cands: Dict[str, List[str]] = {}

        for eid, name_norm, addr_lm, c_norm, degen in zip(eids, name_norms, addr_landmarks, country_norms, degens):
            if not name_norm or degen or not addr_lm:
                cands[eid] = []
                continue

            found: Set[str] = set()
            name_initial_3 = name_norm[:3]
            key_name3 = (addr_lm, name_initial_3, c_norm)
            for mid in self.index.idx_landmark_name3_country.get(key_name3, [])[:self.max_cands_per_key]:
                found.add(mid)

            key_lm = (addr_lm, c_norm)
            for mid in self.index.idx_landmark_country.get(key_lm, [])[:30]:
                found.add(mid)
                if len(found) >= self.max_cands_per_key:
                    break

            cands[eid] = list(found)

        return cands

    def generate_channel_i(self, s1_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Channel I: Phonetic / Transliteration Canonicalization Blocking."""
        eids, name_norms, _, _, _, _, _, country_norms, degens = self._extract_s1_vectors(s1_df)
        cands: Dict[str, List[str]] = {}

        for eid, name_norm, c_norm, degen in zip(eids, name_norms, country_norms, degens):
            if not name_norm or degen:
                cands[eid] = []
                continue

            ph_name = canonicalize_phonetic(name_norm)
            if not ph_name:
                cands[eid] = []
                continue

            found: Set[str] = set()
            key_ph = (ph_name, c_norm)
            for mid in self.index.idx_phonetic_name_country.get(key_ph, [])[:self.max_cands_per_key]:
                found.add(mid)

            ph_tokens = ph_name.split()
            if len(ph_tokens) >= 2:
                ph_sorted = " ".join(sorted(ph_tokens))
                key_phs = (ph_sorted, c_norm)
                for mid in self.index.idx_phonetic_sorted_country.get(key_phs, [])[:self.max_cands_per_key]:
                    found.add(mid)

                ph_p2 = f"{ph_tokens[0]} {ph_tokens[1]}"
                key_php2 = (ph_p2, c_norm)
                for mid in self.index.idx_phonetic_prefix2_country.get(key_php2, [])[:self.max_cands_per_key]:
                    found.add(mid)

            cands[eid] = list(found)

        return cands

    def generate_channel_j(self, s1_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Channel J: Core Token Pair Inverted Index."""
        eids, name_norms, _, _, _, _, _, country_norms, degens = self._extract_s1_vectors(s1_df)
        cands: Dict[str, List[str]] = {}

        for eid, name_norm, c_norm, degen in zip(eids, name_norms, country_norms, degens):
            if not name_norm or degen:
                cands[eid] = []
                continue

            toks = [
                t for t in name_norm.split()
                if len(t) >= self.index.min_token_len and self.index._name_token_df.get(t, 0) <= 2000
            ]
            if len(toks) < 2 or len(toks) > 6:
                cands[eid] = []
                continue

            found: Set[str] = set()
            for t1, t2 in combinations(sorted(toks), 2):
                key = (t1, t2, c_norm)
                for mid in self.index.idx_core_pair_country.get(key, [])[:25]:
                    found.add(mid)
                    if len(found) >= self.max_cands_per_key:
                        break
                if len(found) >= self.max_cands_per_key:
                    break

            cands[eid] = list(found)

        return cands

    def generate_channel_k(self, s1_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Channel K: Address Number + Street Token Inverted Index."""
        eids, _, _, _, addr_norms, _, addr_numbers, country_norms, _ = self._extract_s1_vectors(s1_df)
        cands: Dict[str, List[str]] = {}

        for eid, addr_norm, nums, c_norm in zip(eids, addr_norms, addr_numbers, country_norms):
            if not addr_norm or not isinstance(nums, list) or len(nums) == 0:
                cands[eid] = []
                continue

            primary_num = str(nums[0]).strip().lower()
            if not primary_num:
                cands[eid] = []
                continue

            found: Set[str] = set()
            for atok in addr_norm.split()[:4]:
                if len(atok) >= self.index.min_token_len and atok != primary_num:
                    if self.index._addr_token_df.get(atok, 0) <= 2000:
                        key = (primary_num, atok, c_norm)
                        for mid in self.index.idx_addr_num_street_country.get(key, [])[:25]:
                            found.add(mid)
                            if len(found) >= self.max_cands_per_key:
                                break

                if len(found) >= self.max_cands_per_key:
                    break

            cands[eid] = list(found)

        return cands

    def generate_channel_l(self, s1_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Channel L: Postal PIN / ZIP Code + Name Token Anchor."""
        eids, name_norms, _, addr_raws, addr_norms, _, _, country_norms, degens = self._extract_s1_vectors(s1_df)
        cands: Dict[str, List[str]] = {}

        for eid, name_norm, addr_raw, addr_norm, c_norm, degen in zip(eids, name_norms, addr_raws, addr_norms, country_norms, degens):
            if not name_norm or degen:
                cands[eid] = []
                continue

            pin_code = extract_postal_code(addr_raw, addr_norm)
            toks = name_norm.split()
            if not pin_code or len(toks) == 0:
                cands[eid] = []
                continue

            found: Set[str] = set()
            first_tok = toks[0]
            if len(first_tok) >= self.index.min_token_len:
                key = (pin_code, first_tok, c_norm)
                for mid in self.index.idx_pin_name_tok_country.get(key, [])[:30]:
                    found.add(mid)

            cands[eid] = list(found)

        return cands

    def generate_channel_m(self, s1_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Channel M: Rare Character 3-Gram Inverted Index."""
        eids, name_norms, _, _, _, _, _, country_norms, degens = self._extract_s1_vectors(s1_df)
        cands: Dict[str, List[str]] = {}

        for eid, name_norm, c_norm, degen in zip(eids, name_norms, country_norms, degens):
            if not name_norm or degen:
                cands[eid] = []
                continue

            found: Set[str] = set()
            for ntok in name_norm.split():
                if len(ntok) >= 5:
                    grams = [ntok[j:j+3] for j in range(len(ntok) - 2)]
                    rare_grams = sorted(set(grams), key=lambda g: self.index._char_3gram_df.get(g, 0))
                    if len(rare_grams) >= 2:
                        g1, g2 = sorted([rare_grams[0], rare_grams[1]])
                        key = (g1, g2, c_norm)
                        for mid in self.index.idx_char_3gram_pair_country.get(key, [])[:30]:
                            found.add(mid)
                            if len(found) >= self.max_cands_per_key:
                                break
                if len(found) >= self.max_cands_per_key:
                    break

            cands[eid] = list(found)

        return cands

