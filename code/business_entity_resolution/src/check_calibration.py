"""Calibration and ECE verification script.

Phase 4 Step 4.1 implementation.
"""

import collections
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration import compute_ece

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
    df = df[df.s1_id.isin(gt)]

    probs = df["prob"].to_numpy()
    y = np.array([1 if c in gt.get(s, set()) else 0 for s, c in zip(df.s1_id, df.cand_id)])

    ece = compute_ece(probs, y, n_bins=10)
    print("=" * 70)
    print(f"CALIBRATION METRICS: ECE = {ece:.4f}")
    print("=" * 70)
    print(f"\n{'bin':>12} {'n':>10} {'mean_p':>8} {'actual':>8} {'gap':>8}")
    for lo in np.arange(0, 1.0, 0.1):
        m = (probs >= lo) & (probs < lo + 0.1)
        if m.sum() == 0:
            continue
        mp, ac = probs[m].mean(), y[m].mean()
        flag = "  <-- OFF" if abs(mp - ac) > 0.10 else ""
        print(f"  [{lo:.1f},{lo+0.1:.1f}) {m.sum():10d} {mp:8.3f} {ac:8.3f} {mp-ac:+8.3f}{flag}")

    if ece <= 0.05:
        print("\nGATE 4 VERDICT: PASSED (ECE <= 0.05). DecisionEngine is safe to use.")
    else:
        print("\nGATE 4 VERDICT: ECE > 0.05. Isotonic recalibration required before DecisionEngine.")

if __name__ == "__main__":
    main()
