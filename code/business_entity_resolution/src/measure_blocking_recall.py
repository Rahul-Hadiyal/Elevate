"""Measures blocking recall on a TRAIN sample. This is the hard ceiling.

Phase 1 Step 1.3 / 1.3b implementation.
"""

import collections
import random
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.normalizer import EntityNormalizer
from src.index_builder import BlockingIndex
from src.blocker import MultiChannelBlocker
from src.candidate_store import CandidateStore

SAMPLE_SIZE = 25000       # Fast point estimate sample (Step 1.3c)
CAP_VALUES  = [15, 25, 50, 100]
SEED        = 42

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()

    print("=" * 70)
    print("PHASE 1: MEASURE BLOCKING RECALL & MISS CLASSIFICATION")
    print("=" * 70)

    # 1. Load Ground Truth
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    train_dir = repo_root / "dataset" / "train"
    if not train_dir.exists():
        train_dir = Path("dataset/train")

    gt_file = train_dir / "train_ground_truth.tsv"
    print(f"Loading Ground Truth from {gt_file}...")
    gt = {}
    with open(gt_file, "r", encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 2 and parts[1].strip():
                ids = [x.strip() for x in parts[1].split(",") if x.strip()]
                gt[parts[0].strip()] = set(ids)
            elif len(parts) == 1:
                gt[parts[0].strip()] = set()

    # 2. Ingest Train Sources
    print("Loading Source 1, 2, 3...")
    s1 = pd.read_csv(train_dir / "train_source1.tsv", sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(train_dir / "train_source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(train_dir / "train_source3.tsv", sep="\t", dtype=str).fillna("")

    s1_sample = s1.sample(n=min(SAMPLE_SIZE, len(s1)), random_state=SEED).copy()

    print(f"Normalizing S1 sample ({len(s1_sample):,d} records) and S2 ({len(s2):,d}), S3 ({len(s3):,d})...")
    normalizer = EntityNormalizer()
    s1n = normalizer.normalize_dataframe(s1_sample)
    s2n = normalizer.normalize_dataframe(s2)
    s3n = normalizer.normalize_dataframe(s3)

    print("Building Multi-Channel BlockingIndex...")
    index = BlockingIndex(min_token_len=2, max_token_df=5000)
    index.build_indexes(s2n, s3n)
    blocker = MultiChannelBlocker(index, max_cands_per_key=200)

    print("Generating candidate channels...")
    channels = {}
    for ch in "ABCDEGHIJKLM":
        fn = getattr(blocker, f"generate_channel_{ch.lower()}", None)
        if fn is not None:
            t_ch = time.time()
            cands = fn(s1n)
            channels[f"Channel_{ch}"] = cands
            print(f"  Channel {ch}: {len(cands):,d} pairs in {time.time() - t_ch:.2f}s")

    print("\n--- BLOCKING RECALL SWEEP ACROSS CAPS ---")
    best_cand_dict = None
    for cap in CAP_VALUES:
        store = CandidateStore(s1n["entity_id"])
        for name, cands in channels.items():
            store.add_channel_candidates(name, cands)
        cand_dict = store.get_candidate_dict(cap=cap)
        if cap == 50:
            best_cand_dict = cand_dict

        total_true = 0
        got = 0
        n_cands = []
        for s1_id in s1n["entity_id"]:
            truth = gt.get(s1_id, set())
            cands = set(cand_dict.get(s1_id, []))
            total_true += len(truth)
            got += len(truth & cands)
            n_cands.append(len(cands))

        recall = got / total_true if total_true else float("nan")
        print(f"cap={cap:<4} BLOCKING RECALL = {recall:.4f}  "
              f"mean_cands={np.mean(n_cands):.1f} max_cands={max(n_cands) if n_cands else 0}")

    # Per-channel marginal recall at cap=50
    print("\n--- PER-CHANNEL MARGINAL RECALL (cap=50) ---")
    for drop in [None] + list(channels.keys()):
        store = CandidateStore(s1n["entity_id"])
        for name, cands in channels.items():
            if name == drop:
                continue
            store.add_channel_candidates(name, cands)
        cd = store.get_candidate_dict(cap=50)
        tt = 0
        g = 0
        for s1_id in s1n["entity_id"]:
            truth = gt.get(s1_id, set())
            tt += len(truth)
            g += len(truth & set(cd.get(s1_id, [])))
        label = "ALL CHANNELS" if drop is None else f"without {drop}"
        print(f"  {label:<28} recall = {g/tt:.4f}")

    # Step 1.3b: Miss Classification
    print("\n--- MISS CLASSIFICATION (cap=50) ---")
    cand_lookup = {}
    for df_n in (s2n, s3n):
        for row in df_n.itertuples(index=False):
            cand_lookup[row.entity_id] = (
                set(str(getattr(row, "name_tokens", "")).split()),
                set(str(getattr(row, "addr_tokens", "")).split()),
                set(str(getattr(row, "addr_numbers", "")).split()),
            )

    stats = collections.Counter()
    for row in s1n.itertuples(index=False):
        s1_id = row.entity_id
        truth = gt.get(s1_id, set())
        found = set(best_cand_dict.get(s1_id, []))
        missed = truth - found
        if not missed:
            continue
        s1_name = set(str(getattr(row, "name_tokens", "")).split())
        s1_addr = set(str(getattr(row, "addr_tokens", "")).split())
        s1_num  = set(str(getattr(row, "addr_numbers", "")).split())
        for m in missed:
            if m not in cand_lookup:
                stats["NOT_IN_SOURCE_FILES"] += 1
                continue
            c_name, c_addr, c_num = cand_lookup[m]
            shares_name = bool(s1_name & c_name)
            shares_addr = bool(s1_addr & c_addr)
            shares_num  = bool(s1_num  & c_num)
            if shares_name or shares_addr or shares_num:
                stats["RETRIEVABLE_but_dropped"] += 1
                if shares_name: stats["  ...shares name token"] += 1
                if shares_addr: stats["  ...shares addr token"] += 1
                if shares_num:  stats["  ...shares numeric"]    += 1
            else:
                stats["UNRETRIEVABLE_no_shared_token"] += 1

    total_missed = (stats["RETRIEVABLE_but_dropped"]
                    + stats["UNRETRIEVABLE_no_shared_token"]
                    + stats["NOT_IN_SOURCE_FILES"])
    for k, v in stats.most_common():
        pct = v / total_missed if total_missed else 0
        print(f"  {k:<34} {v:>8,}  ({pct:6.2%})")

    print(f"\nCompleted in {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()
