"""Executable runner for Phase 2: Full EDA, Corpus Profiling & H1 Gate on Real Competition Data."""

import sys
from pathlib import Path
import time
import json
import numpy as np
import pandas as pd

# Add source directory
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.data_loader import load_entity_source, load_ground_truth
from src.h1_gate import evaluate_h1_gate
from src.profiler import profile_entity_dataframe, profile_ground_truth, build_corpus_profile, detect_script


def log_print(msg: str) -> None:
    print(msg, flush=True)


def main():
    log_print("================================================================================")
    log_print("  PHASE 2: REAL DATA INGESTION, CORPUS PROFILING & HYPOTHESIS H1 EVALUATION")
    log_print("================================================================================")
    start_total = time.time()

    data_dir = Path("dataset")
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. INGESTION & CONTRACT VALIDATION
    # -------------------------------------------------------------------------
    log_print("\n>>> [STAGE 1/4] Ingesting and Validating TSV Files...")

    t0 = time.time()
    train_s1_df, s1_report = load_entity_source(train_dir / "train_source1.tsv", "S1", strict=True)
    log_print(f"  [TRAIN S1] {len(train_s1_df):,} rows in {time.time()-t0:.2f}s | Unique IDs: {s1_report.stats['unique_ids']:,}")

    t0 = time.time()
    train_s2_df, s2_report = load_entity_source(train_dir / "train_source2.tsv", "S2", strict=True)
    log_print(f"  [TRAIN S2] {len(train_s2_df):,} rows in {time.time()-t0:.2f}s | Unique IDs: {s2_report.stats['unique_ids']:,}")

    t0 = time.time()
    train_s3_df, s3_report = load_entity_source(train_dir / "train_source3.tsv", "S3", strict=True)
    log_print(f"  [TRAIN S3] {len(train_s3_df):,} rows in {time.time()-t0:.2f}s | Unique IDs: {s3_report.stats['unique_ids']:,}")

    t0 = time.time()
    gt_df, gt_report = load_ground_truth(train_dir / "train_ground_truth.tsv", valid_s1_ids=set(train_s1_df["entity_id"]), strict=True)
    log_print(f"  [TRAIN GT] {len(gt_df):,} rows in {time.time()-t0:.2f}s | Matched Links: {gt_report.stats['total_matched_links']:,}")

    t0 = time.time()
    test_s1_df, test_s1_rep = load_entity_source(test_dir / "test_source1.tsv", "S1", strict=True)
    log_print(f"  [TEST S1]  {len(test_s1_df):,} rows in {time.time()-t0:.2f}s | Unique IDs: {test_s1_rep.stats['unique_ids']:,}")

    t0 = time.time()
    test_s2_df, test_s2_rep = load_entity_source(test_dir / "test_source2.tsv", "S2", strict=True)
    log_print(f"  [TEST S2]  {len(test_s2_df):,} rows in {time.time()-t0:.2f}s | Unique IDs: {test_s2_rep.stats['unique_ids']:,}")

    t0 = time.time()
    test_s3_df, test_s3_rep = load_entity_source(test_dir / "test_source3.tsv", "S3", strict=True)
    log_print(f"  [TEST S3]  {len(test_s3_df):,} rows in {time.time()-t0:.2f}s | Unique IDs: {test_s3_rep.stats['unique_ids']:,}")

    # -------------------------------------------------------------------------
    # 2. HYPOTHESIS H1 GATE ON REAL GROUND TRUTH
    # -------------------------------------------------------------------------
    log_print("\n>>> [STAGE 2/4] Evaluating Hypothesis H1 Gate on Ground Truth...")
    t0 = time.time()
    h1_result = evaluate_h1_gate(gt_df, output_json_path=logs_dir / "h1_gate.json")
    log_print(f"  H1 Evaluation completed in {time.time()-t0:.2f}s")
    log_print(f"  * Hypothesis H1 Confirmed: {h1_result.is_h1_confirmed}")
    log_print(f"  * Total Distinct Matched IDs in GT: {h1_result.total_distinct_matched_ids:,}")
    log_print(f"  * Single Claimant IDs: {h1_result.total_single_claimant_ids:,} ({h1_result.total_single_claimant_ids / h1_result.total_distinct_matched_ids:.4%})")
    log_print(f"  * Multi-Claimant (Violating) IDs: {h1_result.total_multi_claimant_ids:,} ({h1_result.overall_violation_rate:.4%})")
    log_print(f"  * S2 Violations: {h1_result.s2_stats.multiple_claimants_count:,} / {h1_result.s2_stats.total_distinct_matched_ids:,} ({h1_result.s2_stats.violation_rate:.4%})")
    log_print(f"  * S3 Violations: {h1_result.s3_stats.multiple_claimants_count:,} / {h1_result.s3_stats.total_distinct_matched_ids:,} ({h1_result.s3_stats.violation_rate:.4%})")
    log_print(f"  * Conflict Resolution Enabled: {h1_result.conflict_resolution_enabled}")
    log_print(f"  * Invariant 8 Enforced: {h1_result.invariant_8_enforced}")

    if h1_result.all_violating_ids:
        log_print("\n  Sample H1 Multi-Claimant Records (First 10):")
        for mid, claimants in list(h1_result.all_violating_ids.items())[:10]:
            log_print(f"    {mid} -> claimed by {claimants}")

    # -------------------------------------------------------------------------
    # 3. COMPREHENSIVE CORPUS PROFILING
    # -------------------------------------------------------------------------
    log_print("\n>>> [STAGE 3/4] Profiling Train and Test Corpora...")
    t0 = time.time()
    train_profile = build_corpus_profile(train_s1_df, train_s2_df, train_s3_df, gt_df, dataset_name="TrainCorpus")
    train_profile.save(logs_dir / "train_corpus_profile.json")
    log_print(f"  Train corpus profile saved in {time.time()-t0:.2f}s")

    t0 = time.time()
    test_profile = build_corpus_profile(test_s1_df, test_s2_df, test_s3_df, None, dataset_name="TestCorpus")
    test_profile.save(logs_dir / "test_corpus_profile.json")
    log_print(f"  Test corpus profile saved in {time.time()-t0:.2f}s")

    # -------------------------------------------------------------------------
    # 4. SCALE & SUMMARY REPORT GENERATION
    # -------------------------------------------------------------------------
    log_print("\n>>> [STAGE 4/4] Compiling Scale and Distribution Reports...")
    scale_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train": {
            "s1_rows": len(train_s1_df),
            "s2_rows": len(train_s2_df),
            "s3_rows": len(train_s3_df),
            "gt_rows": len(gt_df),
            "gt_matched_links": train_profile.ground_truth.total_matched_links if train_profile.ground_truth else 0,
            "gt_singletons": train_profile.ground_truth.singleton_count if train_profile.ground_truth else 0,
            "gt_singleton_rate": train_profile.ground_truth.singleton_rate if train_profile.ground_truth else 0.0,
            "s1_countries": train_profile.sources["S1"].country_distribution,
            "s2_countries": train_profile.sources["S2"].country_distribution,
            "s3_countries": train_profile.sources["S3"].country_distribution,
        },
        "test": {
            "s1_rows": len(test_s1_df),
            "s2_rows": len(test_s2_df),
            "s3_rows": len(test_s3_df),
            "s1_countries": test_profile.sources["S1"].country_distribution,
            "s2_countries": test_profile.sources["S2"].country_distribution,
            "s3_countries": test_profile.sources["S3"].country_distribution,
        },
        "h1_gate": h1_result.to_dict(),
    }

    with open(logs_dir / "scale_report.json", "w", encoding="utf-8") as f:
        json.dump(scale_report, f, indent=2)

    log_print(f"\nAll profiling reports generated in {time.time()-start_total:.2f}s.")
    log_print("================================================================================")


if __name__ == "__main__":
    main()
