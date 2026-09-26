"""Diagnostic Script for Analyzing Missed Ground-Truth Links in Phase 5 Blocking.

Loads missed S1 -> S2/S3 true match pairs and categorizes root failure causes:
1. Minor typo / spelling variation in name (Levenshtein similarity >= 0.8)
2. Token permutation / subset in name
3. Transliteration / phonetic variation (e.g., Shree vs Sri, Laxmi vs Lakshmi)
4. Acronym / Initialism / Prefix difference (e.g., Dr. Smith vs Smith)
5. Address number mismatch / missing number
6. Address token overlap with different name phrasing
7. Heavy abbreviation in name or address
"""

from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Set, Tuple, Optional, Union
import json
import logging
import re

import pandas as pd
import numpy as np
from rapidfuzz import fuzz, distance

from src.data_loader import load_entity_source, load_ground_truth, parse_ground_truth_to_dict
from src.normalizer import EntityNormalizer
from src.split import SplitManifest
from src.index_builder import BlockingIndex
from src.blocker import MultiChannelBlocker
from src.candidate_store import CandidateStore


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Missed_Diagnoser")


# Common phonetic / transliteration normalization rules for Indian & International names
PHONETIC_REPLACEMENTS = [
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
    (r"\bintl\b|\binternational\b", "international"),
    (r"\bpharmaceuticals\b|\bpharma\b", "pharma"),
    (r"\bcorporation\b|\bcorp\b", "corp"),
    (r"\bmedical\b|\bmed\b", "medical"),
]


def apply_phonetic_canonicalization(text: str) -> str:
    """Applies high-frequency phonetic and transliteration canonicalizations."""
    if not text:
        return ""
    t = text.lower()
    for pat, rep in PHONETIC_REPLACEMENTS:
        t = re.sub(pat, rep, t)
    # Collapse double consonants: e.g. "millennium" -> "milenium", "aggarwal" -> "agarwal"
    t = re.sub(r"([a-z])\1+", r"\1", t)
    return t.strip()


def run_diagnostics(sample_size: int = 20000) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    train_dir = repo_root / "dataset" / "train"
    splits_dir = repo_root / "artifacts" / "splits"

    logger.info("Loading split manifest and data...")
    manifest = SplitManifest.load(splits_dir / "split_manifest.tsv.gz")
    
    s1_df_all, _ = load_entity_source(train_dir / "train_source1.tsv", "S1")
    gt_df, _ = load_ground_truth(train_dir / "train_ground_truth.tsv")
    gt_dict_all = parse_ground_truth_to_dict(gt_df)

    val_a_s1 = manifest.filter_s1_dataframe(s1_df_all, "val_a")
    val_a_gt = manifest.filter_ground_truth(gt_dict_all, "val_a")

    s2_df, _ = load_entity_source(train_dir / "train_source2.tsv", "S2")
    s3_df, _ = load_entity_source(train_dir / "train_source3.tsv", "S3")

    normalizer = EntityNormalizer.from_config_dir(repo_root / "code" / "business_entity_resolution" / "configs")
    val_a_s1_norm = normalizer.normalize_dataframe(val_a_s1)
    s2_norm = normalizer.normalize_dataframe(s2_df)
    s3_norm = normalizer.normalize_dataframe(s3_df)

    logger.info("Building baseline blocker...")
    index = BlockingIndex(max_token_df=500, min_token_len=3)
    blocker = MultiChannelBlocker(index=index, max_cands_per_key=100)
    blocker.fit(s2_norm, s3_norm)

    # Build baseline candidate store (A+B+C+D+E+G+H)
    store = CandidateStore(val_a_s1_norm["entity_id"])
    store.add_channel_candidates("A", blocker.generate_channel_a(val_a_s1_norm))
    store.add_channel_candidates("B", blocker.generate_channel_b(val_a_s1_norm))
    store.add_channel_candidates("C", blocker.generate_channel_c(val_a_s1_norm))
    store.add_channel_candidates("D", blocker.generate_channel_d(val_a_s1_norm))
    store.add_channel_candidates("E", blocker.generate_channel_e(val_a_s1_norm))
    store.add_channel_candidates("G", blocker.generate_channel_g(val_a_s1_norm))
    store.add_channel_candidates("H", blocker.generate_channel_h(val_a_s1_norm))

    all_cands = store.get_candidate_dict()

    # Fast indexed lookup for S2 and S3 records
    logger.info("Indexing target records for lookup...")
    target_df = pd.concat([s2_norm, s3_norm], ignore_index=True)
    target_lookup = target_df.set_index("entity_id").to_dict("index")
    s1_lookup = val_a_s1_norm.set_index("entity_id").to_dict("index")

    # Extract missed GT pairs
    logger.info("Extracting and categorizing missed GT links...")
    missed_pairs = []
    for s1_id, true_set in val_a_gt.items():
        cand_set = set(all_cands.get(s1_id, []))
        for mid in true_set:
            if mid not in cand_set:
                missed_pairs.append((s1_id, mid))

    logger.info(f"Total missed GT pairs: {len(missed_pairs):,} / {sum(len(v) for v in val_a_gt.values()):,}")

    # Subsample if large for fast categorization
    sample_pairs = missed_pairs[:sample_size]

    categories: Counter = Counter()
    india_categories: Counter = Counter()
    us_categories: Counter = Counter()

    diagnosed_examples = []

    for s1_id, target_id in sample_pairs:
        s1_rec = s1_lookup.get(s1_id)
        t_rec = target_lookup.get(target_id)
        if not s1_rec or not t_rec:
            categories["missing_data"] += 1
            continue

        s1_name_norm = str(s1_rec.get("name_norm", ""))
        t_name_norm = str(t_rec.get("name_norm", ""))
        s1_addr_norm = str(s1_rec.get("addr_norm", ""))
        t_addr_norm = str(t_rec.get("addr_norm", ""))
        s1_c = str(s1_rec.get("country_norm", ""))
        t_c = str(t_rec.get("country_norm", ""))

        s1_name_toks = set(s1_name_norm.split())
        t_name_toks = set(t_name_norm.split())
        shared_name_toks = s1_name_toks & t_name_toks

        s1_addr_toks = set(s1_addr_norm.split())
        t_addr_toks = set(t_addr_norm.split())
        shared_addr_toks = s1_addr_toks & t_addr_toks

        name_fuzz = fuzz.ratio(s1_name_norm, t_name_norm)
        token_sort_fuzz = fuzz.token_sort_ratio(s1_name_norm, t_name_norm)
        token_set_fuzz = fuzz.token_set_ratio(s1_name_norm, t_name_norm)

        # Phonetic canonicalization
        s1_ph = apply_phonetic_canonicalization(s1_name_norm)
        t_ph = apply_phonetic_canonicalization(t_name_norm)
        ph_match = (s1_ph == t_ph) or (fuzz.ratio(s1_ph, t_ph) >= 85)

        cat = "other_discrepancy"

        if s1_c != t_c and s1_c != "unknown" and t_c != "unknown":
            cat = "country_mismatch"
        elif ph_match and s1_ph != "":
            cat = "phonetic_transliteration_variation"
        elif token_set_fuzz >= 90 and len(shared_name_toks) >= 1:
            cat = "name_token_subset_or_order"
        elif name_fuzz >= 75:
            cat = "minor_name_typo_or_edit_distance"
        elif len(shared_name_toks) >= 1:
            cat = "shared_single_name_token_different_phrasing"
        elif len(shared_addr_toks) >= 3 or (len(shared_addr_toks) >= 2 and fuzz.token_set_ratio(s1_addr_norm, t_addr_norm) >= 80):
            cat = "strong_address_match_dissimilar_name"
        elif len(shared_addr_toks) >= 1:
            cat = "partial_address_match_dissimilar_name"
        else:
            cat = "severe_name_and_address_discrepancy"

        categories[cat] += 1
        if s1_c == "india":
            india_categories[cat] += 1
        else:
            us_categories[cat] += 1

        if len(diagnosed_examples) < 25:
            diagnosed_examples.append({
                "s1_id": s1_id,
                "target_id": target_id,
                "s1_name_raw": s1_rec.get("business_name"),
                "t_name_raw": t_rec.get("business_name"),
                "s1_name_norm": s1_name_norm,
                "t_name_norm": t_name_norm,
                "s1_addr_raw": s1_rec.get("business_address"),
                "t_addr_raw": t_rec.get("business_address"),
                "country": s1_c,
                "category": cat,
                "name_fuzz": name_fuzz,
                "token_set_fuzz": token_set_fuzz,
                "shared_name_tokens": list(shared_name_toks),
                "shared_addr_tokens": list(shared_addr_toks),
            })

    total_diag = len(sample_pairs)
    summary = {
        "total_missed_pairs_sample": total_diag,
        "overall_categories": {k: {"count": v, "pct": round(v / total_diag * 100, 2)} for k, v in categories.most_common()},
        "india_categories": {k: {"count": v, "pct": round(v / sum(india_categories.values()) * 100, 2)} for k, v in india_categories.most_common()},
        "us_categories": {k: {"count": v, "pct": round(v / sum(us_categories.values()) * 100, 2)} for k, v in us_categories.most_common()},
        "diagnosed_examples": diagnosed_examples,
    }

    logger.info("Diagnostic Analysis Completed:")
    for cat, stat in summary["overall_categories"].items():
        logger.info(f"  {cat:45s}: {stat['count']:,} ({stat['pct']}%)")

    return summary


if __name__ == "__main__":
    res = run_diagnostics(sample_size=30000)
    with open("logs/missed_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
