"""Corpus Profiling and EDA Module.

Provides comprehensive, reproducible descriptive statistics and profiling for:
- Source 1, Source 2, and Source 3 entity tables
- Ground truth match cardinality, singleton distribution, and source patterns
- Cross-source lexical overlap and vocabulary statistics
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.h1_gate import H1GateResult, evaluate_h1_gate


def detect_script(text: str) -> str:
    """Detects the primary script of a string."""
    if not text or not text.strip():
        return "EMPTY"
    
    latin_count = 0
    devanagari_count = 0
    digit_count = 0
    other_count = 0

    for ch in text:
        if ch.isspace() or unicodedata.category(ch).startswith("P"):
            continue
        if ch.isdigit():
            digit_count += 1
            continue
        try:
            name = unicodedata.name(ch)
            if "LATIN" in name:
                latin_count += 1
            elif "DEVANAGARI" in name:
                devanagari_count += 1
            else:
                other_count += 1
        except ValueError:
            other_count += 1

    total = latin_count + devanagari_count + other_count
    if total == 0:
        return "NUMERIC_ONLY" if digit_count > 0 else "PUNCTUATION_ONLY"
    
    if latin_count / total > 0.8:
        return "LATIN"
    elif devanagari_count / total > 0.8:
        return "DEVANAGARI"
    elif latin_count > 0 and devanagari_count > 0:
        return "MIXED_LATIN_DEVANAGARI"
    else:
        return "OTHER"


def compute_distribution_stats(values: List[Union[int, float]]) -> Dict[str, float]:
    """Computes summary quantiles and moments for a numeric sequence."""
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


@dataclass
class SourceProfile:
    """Detailed profile of an individual entity table (S1, S2, or S3)."""
    source_tag: str
    total_rows: int
    unique_ids: int
    duplicate_id_count: int
    null_name_count: int
    null_address_count: int
    null_country_count: int
    country_distribution: Dict[str, int]
    name_char_length_stats: Dict[str, float]
    name_token_count_stats: Dict[str, float]
    addr_char_length_stats: Dict[str, float]
    addr_token_count_stats: Dict[str, float]
    name_scripts: Dict[str, int]
    addr_scripts: Dict[str, int]
    exact_duplicate_names_count: int
    exact_duplicate_addresses_count: int


@dataclass
class GroundTruthProfile:
    """Detailed profile of training ground truth."""
    total_s1_entities: int
    total_matched_links: int
    singleton_count: int
    singleton_rate: float
    one_match_count: int
    multi_match_count: int
    matches_per_s1_stats: Dict[str, float]
    s2_matches_count: int
    s3_matches_count: int
    both_s2_s3_count: int
    s2_only_count: int
    s3_only_count: int
    neither_count: int
    distinct_s2_matched: int
    distinct_s3_matched: int
    h1_gate_result: Dict[str, Any]


@dataclass
class CorpusProfileReport:
    """Full end-to-end dataset profile."""
    dataset_name: str
    sources: Dict[str, SourceProfile]
    ground_truth: Optional[GroundTruthProfile] = None
    cross_source_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "sources": {k: asdict(v) for k, v in self.sources.items()},
            "ground_truth": asdict(self.ground_truth) if self.ground_truth else None,
            "cross_source_stats": self.cross_source_stats,
        }

    def save(self, output_path: Union[str, Path]) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def profile_entity_dataframe(df: pd.DataFrame, source_tag: str) -> SourceProfile:
    """Profiles an entity table."""
    total_rows = len(df)
    if total_rows == 0:
        return SourceProfile(
            source_tag=source_tag,
            total_rows=0,
            unique_ids=0,
            duplicate_id_count=0,
            null_name_count=0,
            null_address_count=0,
            null_country_count=0,
            country_distribution={},
            name_char_length_stats=compute_distribution_stats([]),
            name_token_count_stats=compute_distribution_stats([]),
            addr_char_length_stats=compute_distribution_stats([]),
            addr_token_count_stats=compute_distribution_stats([]),
            name_scripts={},
            addr_scripts={},
            exact_duplicate_names_count=0,
            exact_duplicate_addresses_count=0,
        )

    unique_ids = df["entity_id"].nunique()
    duplicate_ids = total_rows - unique_ids

    names = df["business_name"].fillna("").astype(str)
    addresses = df["business_address"].fillna("").astype(str)
    countries = df["country"].fillna("UNKNOWN").astype(str)

    null_names = int((names.str.strip() == "").sum())
    null_addresses = int((addresses.str.strip() == "").sum())
    null_countries = int((countries.str.strip() == "").sum())

    country_dist = countries.value_counts().to_dict()

    # Fast length and token statistics
    name_char_lens = names.str.len().tolist()
    name_token_counts = [len(s.split()) for s in names]
    addr_char_lens = addresses.str.len().tolist()
    addr_token_counts = [len(s.split()) for s in addresses]

    # Fast script detection (sample up to 50,000 records if large)
    sample_size = min(50000, total_rows)
    if total_rows > sample_size:
        sample_indices = np.random.RandomState(42).choice(total_rows, size=sample_size, replace=False)
        sample_names = names.iloc[sample_indices]
        sample_addrs = addresses.iloc[sample_indices]
    else:
        sample_names = names
        sample_addrs = addresses

    name_scripts: Dict[str, int] = {}
    for s in sample_names:
        script = detect_script(s)
        name_scripts[script] = name_scripts.get(script, 0) + 1

    addr_scripts: Dict[str, int] = {}
    for s in sample_addrs:
        script = detect_script(s)
        addr_scripts[script] = addr_scripts.get(script, 0) + 1

    exact_dup_names = int((names.duplicated()).sum())
    exact_dup_addrs = int((addresses.duplicated()).sum())

    return SourceProfile(
        source_tag=source_tag,
        total_rows=total_rows,
        unique_ids=unique_ids,
        duplicate_id_count=duplicate_ids,
        null_name_count=null_names,
        null_address_count=null_addresses,
        null_country_count=null_countries,
        country_distribution=country_dist,
        name_char_length_stats=compute_distribution_stats(name_char_lens),
        name_token_count_stats=compute_distribution_stats(name_token_counts),
        addr_char_length_stats=compute_distribution_stats(addr_char_lens),
        addr_token_count_stats=compute_distribution_stats(addr_token_counts),
        name_scripts=name_scripts,
        addr_scripts=addr_scripts,
        exact_duplicate_names_count=exact_dup_names,
        exact_duplicate_addresses_count=exact_dup_addrs,
    )


def profile_ground_truth(df: pd.DataFrame) -> GroundTruthProfile:
    """Profiles the ground truth matching table and executes the H1 Gate."""
    total_s1 = len(df)
    if total_s1 == 0:
        return GroundTruthProfile(
            total_s1_entities=0,
            total_matched_links=0,
            singleton_count=0,
            singleton_rate=0.0,
            one_match_count=0,
            multi_match_count=0,
            matches_per_s1_stats=compute_distribution_stats([]),
            s2_matches_count=0,
            s3_matches_count=0,
            both_s2_s3_count=0,
            s2_only_count=0,
            s3_only_count=0,
            neither_count=0,
            distinct_s2_matched=0,
            distinct_s3_matched=0,
            h1_gate_result=asdict(evaluate_h1_gate(df)),
        )

    match_counts: List[int] = []
    singleton_count = 0
    one_match_count = 0
    multi_match_count = 0

    s2_only_count = 0
    s3_only_count = 0
    both_count = 0
    neither_count = 0

    distinct_s2: Set[str] = set()
    distinct_s3: Set[str] = set()
    total_matched_links = 0

    match_col = df["matched_entity_ids"].fillna("").astype(str)
    for match_str in match_col:
        cleaned = match_str.strip()
        if not cleaned:
            match_counts.append(0)
            singleton_count += 1
            neither_count += 1
            continue

        raw_ids = [item.strip() for item in cleaned.split(",") if item.strip()]
        count = len(raw_ids)
        match_counts.append(count)
        total_matched_links += count

        if count == 0:
            singleton_count += 1
            neither_count += 1
        elif count == 1:
            one_match_count += 1
        else:
            multi_match_count += 1

        has_s2 = any(mid.startswith("S2-") for mid in raw_ids)
        has_s3 = any(mid.startswith("S3-") for mid in raw_ids)

        if has_s2 and has_s3:
            both_count += 1
        elif has_s2:
            s2_only_count += 1
        elif has_s3:
            s3_only_count += 1

        for mid in raw_ids:
            if mid.startswith("S2-"):
                distinct_s2.add(mid)
            elif mid.startswith("S3-"):
                distinct_s3.add(mid)

    h1_result = evaluate_h1_gate(df)

    return GroundTruthProfile(
        total_s1_entities=total_s1,
        total_matched_links=total_matched_links,
        singleton_count=singleton_count,
        singleton_rate=singleton_count / total_s1 if total_s1 > 0 else 0.0,
        one_match_count=one_match_count,
        multi_match_count=multi_match_count,
        matches_per_s1_stats=compute_distribution_stats(match_counts),
        s2_matches_count=len(distinct_s2),
        s3_matches_count=len(distinct_s3),
        both_s2_s3_count=both_count,
        s2_only_count=s2_only_count,
        s3_only_count=s3_only_count,
        neither_count=neither_count,
        distinct_s2_matched=len(distinct_s2),
        distinct_s3_matched=len(distinct_s3),
        h1_gate_result=h1_result.to_dict(),
    )


def build_corpus_profile(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    gt_df: Optional[pd.DataFrame] = None,
    dataset_name: str = "Corpus",
) -> CorpusProfileReport:
    """Builds an integrated corpus profile across all sources."""
    sources: Dict[str, SourceProfile] = {
        "S1": profile_entity_dataframe(s1_df, "S1"),
        "S2": profile_entity_dataframe(s2_df, "S2"),
        "S3": profile_entity_dataframe(s3_df, "S3"),
    }

    gt_profile = profile_ground_truth(gt_df) if gt_df is not None else None

    # Cross-source lexical overlap
    s1_names = set(s1_df["business_name"].fillna("").str.lower().str.strip())
    s2_names = set(s2_df["business_name"].fillna("").str.lower().str.strip())
    s3_names = set(s3_df["business_name"].fillna("").str.lower().str.strip())

    s1_names.discard("")
    s2_names.discard("")
    s3_names.discard("")

    cross_stats = {
        "s1_unique_lower_names": len(s1_names),
        "s2_unique_lower_names": len(s2_names),
        "s3_unique_lower_names": len(s3_names),
        "exact_name_overlap_s1_s2": len(s1_names & s2_names),
        "exact_name_overlap_s1_s3": len(s1_names & s3_names),
        "exact_name_overlap_s2_s3": len(s2_names & s3_names),
        "exact_name_overlap_all_three": len(s1_names & s2_names & s3_names),
    }

    return CorpusProfileReport(
        dataset_name=dataset_name,
        sources=sources,
        ground_truth=gt_profile,
        cross_source_stats=cross_stats,
    )
