"""Sweeps base_threshold and per-source cap against true macro-F0.5.

Phase 3 Step 3.3 implementation.
"""

import collections
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BETA2 = 0.25

def f05(tp: int, fp: int, fn: int) -> float:
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    if tp == 0:
        return 0.0
    den = (1 + BETA2) * tp + BETA2 * fn + fp
    return ((1 + BETA2) * tp) / den if den > 0 else 0.0

def source_of(cid: str) -> str:
    return "S2" if (cid.startswith("S2-") or cid.startswith("S2_")) else "S3"

def main():
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
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 2 and p[1].strip():
                gt[p[0].strip()] = set(x.strip() for x in p[1].split(",") if x.strip())
            elif len(p) == 1:
                gt[p[0].strip()] = set()

    scored_file = repo_root / "artifacts" / "val_a_scored.tsv"
    if not scored_file.exists():
        scored_file = Path("artifacts/val_a_scored.tsv")

    print(f"Loading scored pairs from {scored_file}...")
    df = pd.read_csv(scored_file, sep="\t", dtype=str)
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce").fillna(0.0)

    scored = collections.defaultdict(list)
    for r in df.itertuples(index=False):
        scored[r.s1_id].append((r.cand_id, float(r.prob)))
        
    entities = [e for e in scored if e in gt]
    print(f"Evaluating on {len(entities):,d} entities with ground truth...")

    print(f"\n{'thr':>6} {'cap=1':>9} {'cap=2':>9} {'cap=3':>9} {'cap=999':>9} {'meanpred':>9}")
    best = (-1.0, None, None)
    
    for thr in np.arange(0.20, 0.96, 0.025):
        row = []
        for cap in (1, 2, 3, 999):
            tot = 0.0
            npred = 0
            for e in entities:
                keep = [(c, p) for c, p in scored[e] if p >= thr]
                sel = set()
                for src in ("S2", "S3"):
                    sub = sorted([x for x in keep if source_of(x[0]) == src],
                                 key=lambda x: -x[1])[:cap]
                    sel |= {c for c, _ in sub}
                t = gt[e]
                tp = len(sel & t)
                tot += f05(tp, len(sel) - tp, len(t) - tp)
                npred += len(sel)
            score = tot / len(entities)
            row.append(score)
            if score > best[0]:
                best = (score, float(thr), cap)
        mp = npred / len(entities)
        print(f"{thr:6.3f} {row[0]:9.4f} {row[1]:9.4f} {row[2]:9.4f} {row[3]:9.4f} {mp:9.2f}")

    print(f"\n======================================================================")
    print(f"BEST CONFIG: Macro-F0.5 = {best[0]:.4f} at Threshold = {best[1]:.3f}, Cap = {best[2]}")
    print(f"======================================================================")

if __name__ == "__main__":
    main()
