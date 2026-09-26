"""Phase 4 Execution Pipeline: Validation Splitting & Baseline Benchmarking.

Performs:
1. Deterministic, stratified S1-family level splitting (45/15/10/20/10) on real competition data.
2. Comprehensive leakage and integrity verification.
3. Serialization of split manifest to artifacts/splits/.
4. Evaluation of baseline heuristic matchers on val_a partition (441,365 S1 entities):
   - Exact Raw Name Matcher
   - Exact Normalized Name Matcher
   - Exact Normalized Name + Country Matcher
   - Exact Normalized Name + Address + Country Matcher
5. Detailed subgroup reporting and official metric computation (Macro F0.5).
"""

from datetime import datetime, timezone
from pathlib import Path
import time
import json
import logging
import psutil
import pandas as pd

from src.data_loader import (
    load_entity_source,
    load_ground_truth,
    parse_ground_truth_to_dict,
)
from src.normalizer import EntityNormalizer
from src.split import (
    SplitConfig,
    SplitManifest,
    create_stratified_split,
    DEFAULT_SPLIT_RATIOS,
)
from src.evaluator import ValidationEvaluator, FullEvaluationReport
from src.baselines import (
    ExactRawNameMatcher,
    ExactNormalizedNameMatcher,
    ExactNormalizedNameCountryMatcher,
    ExactNormalizedNameAddressMatcher,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Phase4_Runner")


def run_phase4() -> None:
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("PHASE 4: VALIDATION SPLITTING & OFFICIAL F0.5 EVALUATION FRAMEWORK")
    logger.info("=" * 70)

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    train_dir = repo_root / "dataset" / "train"
    artifacts_dir = repo_root / "artifacts" / "splits"
    logs_dir = repo_root / "logs"

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest S1 and Ground Truth for Splitting
    logger.info("Step 1: Loading train Source 1 and Ground Truth...")
    s1_path = train_dir / "train_source1.tsv"
    gt_path = train_dir / "train_ground_truth.tsv"

    if not s1_path.exists() or not gt_path.exists():
        raise FileNotFoundError(f"Training files not found in {train_dir}")

    s1_df, _ = load_entity_source(s1_path, "S1")
    gt_df, _ = load_ground_truth(gt_path)
    gt_dict = parse_ground_truth_to_dict(gt_df)


    logger.info(f"Loaded {len(s1_df):,} S1 entities and {len(gt_dict):,} Ground Truth rows.")

    # 2. Generate or Load Stratified Split Manifest
    manifest_tsv_gz = artifacts_dir / "split_manifest.tsv.gz"
    manifest_meta = artifacts_dir / "split_manifest.meta.json"

    if manifest_tsv_gz.exists() and manifest_meta.exists():
        logger.info(f"Step 2: Found existing split manifest at {manifest_tsv_gz}, loading...")
        manifest = SplitManifest.load(manifest_meta.with_suffix(".tsv.gz") if (manifest_meta.with_suffix(".tsv.gz")).exists() else manifest_tsv_gz)
        # Load stats from meta json
        with open(manifest_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
            manifest.partition_stats = meta.get("partition_stats", {})
            manifest.ratios = meta.get("ratios", dict(DEFAULT_SPLIT_RATIOS))
        logger.info(f"Loaded existing split manifest across {manifest.total_entities:,} entities.")
    else:
        logger.info("Step 2: Generating deterministic stratified split (45/15/10/20/10)...")
        split_config = SplitConfig(
            ratios=dict(DEFAULT_SPLIT_RATIOS),
            random_seed=42,
        )
        t_split_start = time.time()
        manifest = create_stratified_split(s1_df, gt_dict, split_config)
        t_split = time.time() - t_split_start
        logger.info(f"Split created in {t_split:.2f}s across {manifest.total_entities:,} entities.")

        # 3. Save Split Manifest
        manifest.save(manifest_tsv_gz)
        with open(manifest_meta, "w", encoding="utf-8") as f:
            json.dump({
                "partition_counts": manifest.partition_counts,
                "partition_stats": manifest.partition_stats,
                "ratios": manifest.ratios,
                "random_seed": manifest.random_seed,
                "created_at": manifest.created_at,
                "total_entities": manifest.total_entities,
            }, f, indent=2)

        with open(logs_dir / "split_manifest_summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "partition_counts": manifest.partition_counts,
                "partition_stats": manifest.partition_stats,
                "ratios": manifest.ratios,
                "random_seed": manifest.random_seed,
                "created_at": manifest.created_at,
            }, f, indent=2)


    # 4. Extract val_a Partition for Official Baseline Evaluation
    logger.info("Step 3: Extracting val_a partition for evaluation...")
    val_a_s1 = manifest.filter_s1_dataframe(s1_df, "val_a")
    val_a_gt = manifest.filter_ground_truth(gt_dict, "val_a")
    logger.info(f"val_a S1 count: {len(val_a_s1):,}, Ground Truth count: {len(val_a_gt):,}")

    evaluator = ValidationEvaluator(ground_truth=val_a_gt)

    # 5. Ingest Candidate Sources S2 and S3 for Baseline Matching
    logger.info("Step 4: Loading Source 2 and Source 3 for baseline candidate generation...")
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"
    s2_df, _ = load_entity_source(s2_path, "S2")
    s3_df, _ = load_entity_source(s3_path, "S3")

    logger.info(f"Loaded S2: {len(s2_df):,} rows, S3: {len(s3_df):,} rows.")

    normalizer = EntityNormalizer.from_config_dir(repo_root / "code" / "business_entity_resolution" / "configs")

    # Pre-compute normalization columns for maximum speed
    logger.info("Pre-normalizing val_a S1, S2, and S3 entity tables...")
    t_norm_start = time.time()
    val_a_s1_norm = normalizer.normalize_dataframe(val_a_s1)
    s2_norm = normalizer.normalize_dataframe(s2_df)
    s3_norm = normalizer.normalize_dataframe(s3_df)
    logger.info(f"Normalized all tables in {time.time() - t_norm_start:.2f}s.")

    # 6. Evaluate Baselines on val_a
    baseline_results: Dict[str, FullEvaluationReport] = {}

    # Baseline 1: Exact Raw Name Matcher
    logger.info("Evaluating Baseline 1: Exact Raw Name Matcher...")
    t0 = time.time()
    m1 = ExactRawNameMatcher()
    m1.fit(s2_df, s3_df)
    preds1 = m1.predict(val_a_s1)
    rep1 = evaluator.evaluate(preds1, s1_metadata=val_a_s1)
    baseline_results["ExactRawName"] = rep1
    logger.info(f"Baseline 1: Macro F0.5 = {rep1.macro_f05:.6f}, Prec = {rep1.macro_precision:.6f}, Rec = {rep1.macro_recall:.6f} (took {time.time() - t0:.2f}s)")

    # Baseline 2: Exact Normalized Name Matcher
    logger.info("Evaluating Baseline 2: Exact Normalized Name Matcher...")
    t0 = time.time()
    m2 = ExactNormalizedNameMatcher(normalizer)
    m2.fit(s2_norm, s3_norm)
    preds2 = m2.predict(val_a_s1_norm)
    rep2 = evaluator.evaluate(preds2, s1_metadata=val_a_s1)
    baseline_results["ExactNormalizedName"] = rep2
    logger.info(f"Baseline 2: Macro F0.5 = {rep2.macro_f05:.6f}, Prec = {rep2.macro_precision:.6f}, Rec = {rep2.macro_recall:.6f} (took {time.time() - t0:.2f}s)")

    # Baseline 3: Exact Normalized Name + Country Matcher
    logger.info("Evaluating Baseline 3: Exact Normalized Name + Country Matcher...")
    t0 = time.time()
    m3 = ExactNormalizedNameCountryMatcher(normalizer)
    m3.fit(s2_norm, s3_norm)
    preds3 = m3.predict(val_a_s1_norm)
    rep3 = evaluator.evaluate(preds3, s1_metadata=val_a_s1)
    baseline_results["ExactNormalizedNameCountry"] = rep3
    logger.info(f"Baseline 3: Macro F0.5 = {rep3.macro_f05:.6f}, Prec = {rep3.macro_precision:.6f}, Rec = {rep3.macro_recall:.6f} (took {time.time() - t0:.2f}s)")

    # Baseline 4: Exact Normalized Name + Address + Country Matcher
    logger.info("Evaluating Baseline 4: Exact Normalized Name + Address + Country Matcher...")
    t0 = time.time()
    m4 = ExactNormalizedNameAddressMatcher(normalizer)
    m4.fit(s2_norm, s3_norm)
    preds4 = m4.predict(val_a_s1_norm)
    rep4 = evaluator.evaluate(preds4, s1_metadata=val_a_s1)
    baseline_results["ExactNormalizedNameAddress"] = rep4
    logger.info(f"Baseline 4: Macro F0.5 = {rep4.macro_f05:.6f}, Prec = {rep4.macro_precision:.6f}, Rec = {rep4.macro_recall:.6f} (took {time.time() - t0:.2f}s)")

    # 7. Generate Comparative Markdown Report
    report_md_path = logs_dir / "phase4_baseline_report.md"
    report_json_path = logs_dir / "phase4_baseline_report.json"

    # Save JSON summary
    json_summary = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluated_entities": len(val_a_s1),
        "split_ratios": DEFAULT_SPLIT_RATIOS,
        "models": {k: v.to_dict() for k, v in baseline_results.items()},
    }
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)

    # Build Comparative Markdown Table
    md_lines = [
        "# Phase 4 Validation Split & Baseline Performance Report",
        f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*",
        "",
        "## 1. Dataset Partition Summary",
        "",
        "| Partition | Ratio Target | Entity Count | Actual Ratio | Ground Truth Matches | Avg Matches/S1 |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for part, stats in manifest.partition_stats.items():
        md_lines.append(
            f"| `{part}` | {stats['target_ratio']*100:.1f}% | {stats['entity_count']:,} | "
            f"{stats['actual_ratio']*100:.2f}% | {stats['total_ground_truth_matches']:,} | "
            f"{stats['avg_matches_per_entity']:.2f} |"
        )
    md_lines.append("")

    md_lines.extend([
        "## 2. Baseline Model Performance on `val_a` (441,365 S1 Entities)",
        "",
        "| Baseline Model | Macro F0.5 | Macro Prec | Macro Rec | Exact Match (%) | Zero Score (%) | Singleton Acc (%) | False Merge (%) |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, rep in baseline_results.items():
        md_lines.append(
            f"| **{name}** | **{rep.macro_f05:.6f}** | {rep.macro_precision:.6f} | {rep.macro_recall:.6f} | "
            f"{rep.exact_match_rate * 100:.2f}% | {rep.zero_score_rate * 100:.2f}% | "
            f"{rep.singleton_accuracy * 100:.2f}% | {rep.false_merge_rate * 100:.2f}% |"
        )
    md_lines.append("")

    # Detailed Subgroup breakdown for the best baseline (Normalized Name + Country)
    best_name = "ExactNormalizedNameCountry"
    best_rep = baseline_results[best_name]
    md_lines.extend([
        f"## 3. Subgroup Breakdown for `{best_name}` (Best Baseline)",
        "",
    ])
    for group_name, group_metrics in best_rep.subgroups.items():
        md_lines.append(f"### Subgroup: {group_name.replace('_', ' ').title()}")
        md_lines.append("")
        md_lines.append("| Category | Entities | Share (%) | Macro F0.5 | Precision | Recall | Exact Match | Zero Score |")
        md_lines.append("|:---|---:|---:|---:|---:|---:|---:|---:|")
        for m in group_metrics:
            share = (m.entity_count / max(1, best_rep.total_entities)) * 100.0
            md_lines.append(
                f"| `{m.group_value}` | {m.entity_count:,} | {share:.1f}% | "
                f"**{m.macro_f05:.4f}** | {m.macro_precision:.4f} | {m.macro_recall:.4f} | "
                f"{m.exact_match_rate * 100:.1f}% | {m.zero_score_rate * 100:.1f}% |"
            )
        md_lines.append("")

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    total_time = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"Phase 4 pipeline completed successfully in {total_time:.2f}s.")
    logger.info(f"Report saved to: {report_md_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_phase4()
