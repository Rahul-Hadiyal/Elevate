"""Family-level validation split. Splitting unit is the S1 entity + all its matches.

Phase 3 Step 3.1 implementation.
"""

import random
import json
from pathlib import Path

SEED = 42
SPLIT = {
    "train": 0.45,
    "earlystop": 0.15,
    "calibration": 0.10,
    "val_a": 0.20,
    "val_b": 0.10
}   # sums to 1.00

def main():
    random.seed(SEED)
    
    # Path resolution
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    train_dir = repo_root / "dataset" / "train"
    if not train_dir.exists():
        train_dir = Path("dataset/train")
        
    gt_file = train_dir / "train_ground_truth.tsv"
    print(f"Reading S1 entity IDs from {gt_file}...")
    
    ids = []
    with open(gt_file, "r", encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip("\r\n").split("\t")
            if parts and parts[0].strip():
                ids.append(parts[0].strip())
                
    random.shuffle(ids)

    n = len(ids)
    out = {}
    start = 0
    for name, frac in SPLIT.items():
        end = start + int(n * frac)
        out[name] = ids[start:end]
        start = end
    out["val_b"].extend(ids[start:])  # remainder into the last fold

    assert sum(len(v) for v in out.values()) == n
    
    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_file = artifacts_dir / "validation_split.json"
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        
    print(f"Validation split created at {out_file}:")
    for k, v in out.items():
        print(f"  {k:<12} {len(v):>9,}  ({len(v)/n:.4f})")

if __name__ == "__main__":
    main()
