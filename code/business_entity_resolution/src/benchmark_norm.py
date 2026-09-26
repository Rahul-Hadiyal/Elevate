"""Benchmark normalization engine on real dataset samples."""

import sys
from pathlib import Path
import time
import psutil
import pandas as pd

src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.normalizer import EntityNormalizer


def main():
    print("--- RUNNING NORMALIZATION BENCHMARK ON REAL DATA ---", flush=True)
    normalizer = EntityNormalizer.from_config_dir("code/business_entity_resolution/configs")

    # Load 100,000 records from train and test (stratified across US, India, France)
    df_train = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", nrows=50000)
    df_test = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", nrows=50000)
    df_sample = pd.concat([df_train, df_test], ignore_index=True)

    country_counts = df_sample["country"].value_counts().to_dict()
    print(f"Sample size: {len(df_sample):,} records across countries: {country_counts}", flush=True)

    # Benchmark speed and memory
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024**2)

    t0 = time.time()
    norm_df = normalizer.normalize_dataframe(df_sample)
    elapsed = time.time() - t0

    mem_after = process.memory_info().rss / (1024**2)
    throughput = len(df_sample) / elapsed

    print(f"\nThroughput: {throughput:,.1f} rows/second ({elapsed:.2f}s for {len(df_sample):,} rows)", flush=True)
    print(f"Memory delta: {mem_after - mem_before:.2f} MB", flush=True)

    raw_unique_names = df_sample["business_name"].nunique()
    norm_unique_names = norm_df["name_norm"].nunique()
    name_reduction = (raw_unique_names - norm_unique_names) / raw_unique_names

    raw_unique_addrs = df_sample["business_address"].nunique()
    norm_unique_addrs = norm_df["addr_norm"].nunique()
    addr_reduction = (raw_unique_addrs - norm_unique_addrs) / raw_unique_addrs

    print("\nCollision Analysis:", flush=True)
    print(f"  Distinct Raw Names: {raw_unique_names:,} -> Distinct Normalized: {norm_unique_names:,} (Superficial noise removed: {name_reduction:.2%})", flush=True)
    print(f"  Distinct Raw Addrs: {raw_unique_addrs:,} -> Distinct Normalized: {norm_unique_addrs:,} (Superficial noise removed: {addr_reduction:.2%})", flush=True)

    deg_names = int(norm_df["name_is_degenerate"].sum())
    deg_addrs = int(norm_df["addr_is_degenerate"].sum())
    print("\nDegenerate Records:", flush=True)
    print(f"  Degenerate Names: {deg_names:,} ({deg_names/len(df_sample):.4%})", flush=True)
    print(f"  Degenerate Addresses: {deg_addrs:,} ({deg_addrs/len(df_sample):.4%})", flush=True)

    print("\n=== REAL EXAMPLES BEFORE / AFTER ===", flush=True)
    for country in ["US", "India", "France"]:
        subset = norm_df[norm_df["country"].str.lower() == country.lower()]
        if len(subset) > 0:
            row = subset.iloc[0]
            print(f"\n[{country.upper()}] Entity ID: {row['entity_id']}", flush=True)
            print(f"  RAW NAME:       {row['business_name']}", flush=True)
            print(f"  NORM NAME:      {row['name_norm']}", flush=True)
            print(f"  RAW ADDR:       {row['business_address']}", flush=True)
            print(f"  NORM ADDR:      {row['addr_norm']}", flush=True)
            print(f"  ADDR CORE:      {row['addr_core']}", flush=True)
            print(f"  ADDR LANDMARK:  {row['addr_landmark']}", flush=True)
            print(f"  ADDR NUMBERS:   {row['addr_numbers']}", flush=True)


if __name__ == "__main__":
    main()
