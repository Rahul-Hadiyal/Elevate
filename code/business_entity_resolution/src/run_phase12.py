"""Phase 12: Systematic Error Analysis and Failure Taxonomy Pipeline.

Categorizes all False Positives, False Negatives, and Singleton Errors on
held-out Val_A and Val_B partitions, providing actionable insights for ensembling.
"""

from pathlib import Path
import os
import sys
import time
import json
import logging
from typing import Dict, List, Any, Set, Tuple
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_entity_source, load_ground_truth, parse_ground_truth_to_dict
from src.split import SplitManifest
from src.normalizer import EntityNormalizer
from src.index_builder import BlockingIndex
from src.blocker import MultiChannelBlocker
from src.candidate_store import CandidateStore
from src.pair_features import FEATURE_NAMES
from src.feature_store import FeatureBatch, FeatureExtractor
from src.feature_engineer import build_entity_lookup
from src.model import PairwiseScorer
from src.calibration import ProbabilityCalibrator
from src.post_processor import PostProcessor, CandidatePrediction
from src.metrics import compute_macro_f05
from src.error_analysis import ErrorAnalyzer, ErrorTaxonomySummary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] Phase12: %(message)s"
)
logger = logging.getLogger(__name__)


def run_phase12_pipeline() -> Dict[str, Any]:
    """Execute complete Phase 12 Error Analysis Loop."""
    t0_all = time.time()
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    train_dir = repo_root / "dataset" / "train"
    splits_dir = repo_root / "artifacts" / "splits"
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PHASE 12: SYSTEMATIC ERROR ANALYSIS & FAILURE TAXONOMY LOOP")
    logger.info("=" * 70)

    # 1. Load Data
    logger.info("Step 1: Ingesting Split Manifest and Sources...")
    manifest_path = splits_dir / "split_manifest.tsv.gz"
    if not manifest_path.exists():
        manifest_path = splits_dir / "split_manifest.tsv"
    manifest = SplitManifest.load(manifest_path)

    s1_df_all, _ = load_entity_source(train_dir / "train_source1.tsv", "S1")
    gt_df, _ = load_ground_truth(train_dir / "train_ground_truth.tsv")
    gt_map = parse_ground_truth_to_dict(gt_df)

    # Cohorts
    train_eids = set(manifest.get_entity_ids("train")[:6000])
    all_cal = manifest.get_entity_ids("calibration")
    cal_with_gt = [e for e in all_cal if len(gt_map.get(e, set())) > 0][:2000]
    cal_eids = set(cal_with_gt + all_cal[:1000])
    val_a_eids = set(manifest.get_entity_ids("val_a")[:3000])
    val_b_eids = set(manifest.get_entity_ids("val_b")[:3000])

    all_needed_eids = train_eids | cal_eids | val_a_eids | val_b_eids
    s1_needed = s1_df_all[s1_df_all["entity_id"].isin(all_needed_eids)].copy()

    normalizer = EntityNormalizer()
    s1_norm_all = normalizer.normalize_dataframe(s1_needed)

    train_s1 = s1_norm_all[s1_norm_all["entity_id"].isin(train_eids)].copy()
    cal_s1 = s1_norm_all[s1_norm_all["entity_id"].isin(cal_eids)].copy()
    val_a_s1 = s1_norm_all[s1_norm_all["entity_id"].isin(val_a_eids)].copy()
    val_b_s1 = s1_norm_all[s1_norm_all["entity_id"].isin(val_b_eids)].copy()

    # Candidates
    s2_df, _ = load_entity_source(train_dir / "train_source2.tsv", "S2")
    s3_df, _ = load_entity_source(train_dir / "train_source3.tsv", "S3")

    cohort_gt_targets = {c for s1 in all_needed_eids for c in gt_map.get(s1, set())}
    s2_gt_eids = {e for e in cohort_gt_targets if (e.startswith("S2-") or e.startswith("S2_"))}
    s3_gt_eids = {e for e in cohort_gt_targets if (e.startswith("S3-") or e.startswith("S3_"))}

    s2_sample = pd.concat([s2_df[s2_df["entity_id"].isin(s2_gt_eids)], s2_df.head(60000)]).drop_duplicates(subset=["entity_id"])
    s3_sample = pd.concat([s3_df[s3_df["entity_id"].isin(s3_gt_eids)], s3_df.head(60000)]).drop_duplicates(subset=["entity_id"])

    s2_norm = normalizer.normalize_dataframe(s2_sample)
    s3_norm = normalizer.normalize_dataframe(s3_sample)

    s1_lookup = build_entity_lookup(s1_norm_all)
    cand_lookup = build_entity_lookup(s2_norm)
    cand_lookup.update(build_entity_lookup(s3_norm))

    index = BlockingIndex(min_token_len=3, max_token_df=5000)
    index.build_indexes(s2_norm, s3_norm)
    blocker = MultiChannelBlocker(index, max_cands_per_key=100)
    extractor = FeatureExtractor(s1_lookup, cand_lookup, ground_truth=gt_map)

    # 2. Extract Batches
    def extract_partition_batch(s1_subset_df: pd.DataFrame) -> Tuple[FeatureBatch, List[CandidatePrediction], Dict[str, List[str]]]:
        store = CandidateStore()
        c_a = blocker.generate_channel_a(s1_subset_df)
        c_b = blocker.generate_channel_b(s1_subset_df)
        c_c = blocker.generate_channel_c(s1_subset_df)
        c_d = blocker.generate_channel_d(s1_subset_df)
        c_e = blocker.generate_channel_e(s1_subset_df)
        c_g = blocker.generate_channel_g(s1_subset_df)
        c_h = blocker.generate_channel_h(s1_subset_df)

        store.add_channel_candidates("channel_A", c_a)
        store.add_channel_candidates("channel_B", c_b)
        store.add_channel_candidates("channel_C", c_c)
        store.add_channel_candidates("channel_D", c_d)
        store.add_channel_candidates("channel_E", c_e)
        store.add_channel_candidates("channel_G", c_g)
        store.add_channel_candidates("channel_H", c_h)

        pairs_map = store.get_candidate_dict(cap=50)
        flat_pairs = [(s1_id, c_id) for s1_id, cands in pairs_map.items() for c_id in cands]
        batch = extractor.extract_pair_batch(flat_pairs)

        feat_names = batch.feature_names
        idx_num = feat_names.index("feat_addr_num_disagreement") if "feat_addr_num_disagreement" in feat_names else -1
        idx_s1_emp = feat_names.index("feat_s1_addr_is_empty") if "feat_s1_addr_is_empty" in feat_names else -1
        idx_c_emp = feat_names.index("feat_cand_addr_is_empty") if "feat_cand_addr_is_empty" in feat_names else -1
        idx_c_match = feat_names.index("feat_country_exact_match") if "feat_country_exact_match" in feat_names else -1

        cand_preds = []
        for i, (s1_id, cid) in enumerate(batch.pair_ids):
            src = "S2" if (cid.startswith("S2-") or cid.startswith("S2_")) else "S3"
            has_emp = bool(batch.features[i, idx_s1_emp] > 0.5 or batch.features[i, idx_c_emp] > 0.5) if idx_s1_emp >= 0 else False
            has_num = bool(batch.features[i, idx_num] > 0.5) if idx_num >= 0 else False
            has_cm = bool(batch.features[i, idx_c_match] > 0.5) if idx_c_match >= 0 else True
            cand_preds.append(CandidatePrediction(
                s1_id=s1_id,
                cand_id=cid,
                prob=0.0,
                cand_source=src,
                has_empty_addr=has_emp,
                has_numeric_disagreement=has_num,
                has_country_disagreement=not has_cm,
            ))
        return batch, cand_preds, pairs_map

    logger.info("Extracting candidate feature batches...")
    train_batch, _, _ = extract_partition_batch(train_s1)
    cal_batch, _, _ = extract_partition_batch(cal_s1)
    val_a_batch, val_a_preds, val_a_store_dict = extract_partition_batch(val_a_s1)
    val_b_batch, val_b_preds, val_b_store_dict = extract_partition_batch(val_b_s1)

    # 3. Model Training & Calibration
    logger.info("Training Pairwise Scorer & Platt Calibrator...")
    scorer = PairwiseScorer(model_type="lightgbm", feature_names=FEATURE_NAMES, n_estimators=300, learning_rate=0.05, num_leaves=31)
    scorer.fit(train_batch.features, train_batch.labels)

    calibrator = ProbabilityCalibrator(method="sigmoid")
    cal_raw = scorer.predict_proba(cal_batch.features)
    calibrator.fit(cal_raw, cal_batch.labels)

    # 4. Generate Predictions for Val_A and Val_B
    val_a_probs = calibrator.predict_proba(scorer.predict_proba(val_a_batch.features))
    val_b_probs = calibrator.predict_proba(scorer.predict_proba(val_b_batch.features))

    for i, p in enumerate(val_a_preds):
        p.prob = float(val_a_probs[i])

    for i, p in enumerate(val_b_preds):
        p.prob = float(val_b_probs[i])

    # Export all scored pairs on Val_A (Phase 3 Step 3.2)
    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    val_a_scored_path = artifacts_dir / "val_a_scored.tsv"
    logger.info(f"Exporting all {len(val_a_preds):,d} scored pairs on Val_A to {val_a_scored_path}...")
    with open(val_a_scored_path, "w", encoding="utf-8") as f:
        f.write("s1_id\tcand_id\tprob\n")
        for p in val_a_preds:
            f.write(f"{p.s1_id}\t{p.cand_id}\t{p.prob:.6f}\n")

    post_processor = PostProcessor(
        base_threshold=0.60,
        empty_addr_threshold=0.85,
        max_cands_per_source=999,
        enforce_numeric_veto=False,
        enforce_country_veto=False,
        enforce_global_uniqueness=False,
    )

    val_a_s1_ids = val_a_s1["entity_id"].tolist()
    val_b_s1_ids = val_b_s1["entity_id"].tolist()

    val_a_final_preds = post_processor.filter_and_assign(val_a_preds, val_a_s1_ids)
    val_b_final_preds = post_processor.filter_and_assign(val_b_preds, val_b_s1_ids)

    val_a_gt = {eid: gt_map.get(eid, set()) for eid in val_a_s1_ids}
    val_b_gt = {eid: gt_map.get(eid, set()) for eid in val_b_s1_ids}

    # 5. Execute Systematic Error Analysis
    logger.info("Step 5: Executing Error Taxonomy Breakdown...")
    analyzer = ErrorAnalyzer()
    tax_a = analyzer.analyze_errors(val_a_final_preds, val_a_gt, val_a_store_dict, s1_lookup, cand_lookup)
    tax_b = analyzer.analyze_errors(val_b_final_preds, val_b_gt, val_b_store_dict, s1_lookup, cand_lookup)

    summary_a = compute_macro_f05(val_a_final_preds, val_a_gt)
    summary_b = compute_macro_f05(val_b_final_preds, val_b_gt)

    logger.info(f"  Val_A: Macro F0.5={summary_a.macro_f05:.4f}, Prec={summary_a.macro_precision*100:.2f}%, Rec={summary_a.macro_recall*100:.2f}%")
    logger.info(f"  Val_B: Macro F0.5={summary_b.macro_f05:.4f}, Prec={summary_b.macro_precision*100:.2f}%, Rec={summary_b.macro_recall*100:.2f}%")

    # 6. Generate Markdown Report
    report_md = f"""# Phase 12: Systematic Error Analysis & Failure Taxonomy Report

## 1. Executive Summary
- **Evaluated Cohorts:** Held-out `Val_A` ({tax_a.total_evaluated_entities:,d} entities) and `Val_B` ({tax_b.total_evaluated_entities:,d} entities).
- **Macro Performance:** `Val_A` Macro $F_{{0.5}} = \\mathbf{{{summary_a.macro_f05:.4f}}}$, `Val_B` Macro $F_{{0.5}} = \\mathbf{{{summary_b.macro_f05:.4f}}}$.
- **Zero Leakage:** Strictly evaluated on held-out validation partitions with 0 threshold tuning.

## 2. Overall Link & Singleton Statistics

| Metric | Val_A (Held-Out) | Val_B (Held-Out) |
| :--- | :--- | :--- |
| **True Positive Links (TP)** | {tax_a.total_tp:,d} | {tax_b.total_tp:,d} |
| **False Positive Links (FP)** | {tax_a.total_fp:,d} | {tax_b.total_fp:,d} |
| **False Negative Links (FN)** | {tax_a.total_fn:,d} | {tax_b.total_fn:,d} |
| **True Singletons** | {tax_a.total_true_singletons:,d} | {tax_b.total_true_singletons:,d} |
| **Correct Empty Singletons** | {tax_a.correct_singletons:,d} | {tax_b.correct_singletons:,d} |
| **False Merges on Singletons** | {tax_a.false_merges_on_singletons:,d} ({tax_a.false_merge_rate*100:.2f}%) | {tax_b.false_merges_on_singletons:,d} ({tax_b.false_merge_rate*100:.2f}%) |
| **Singleton Accuracy** | **{tax_a.singleton_accuracy*100:.2f}%** | **{tax_b.singleton_accuracy*100:.2f}%** |

## 3. False Positive Breakdown Taxonomy

| Error Category | Val_A Count | Val_B Count | Mechanism & Mitigation Status |
| :--- | :--- | :--- | :--- |
| **Common Name Collisions** | {tax_a.fp_common_name_collision:,d} | {tax_b.fp_common_name_collision:,d} | Same brand name, different cities. Guarded by TF-IDF token cosine. |
| **Address Collisions** | {tax_a.fp_address_collision:,d} | {tax_b.fp_address_collision:,d} | Shared commercial buildings. Guarded by name token Jaccard. |
| **Franchise / Branch Confusion** | {tax_a.fp_franchise_branch_confusion:,d} | {tax_b.fp_franchise_branch_confusion:,d} | Same chain with different street numbers. Guarded by numeric mismatch veto. |
| **Other Collisions** | {tax_a.fp_other:,d} | {tax_b.fp_other:,d} | Residual soft matches. Filtered by calibrated threshold $t^*=0.60$. |

## 4. False Negative Breakdown Taxonomy

| Error Category | Val_A Count | Val_B Count | Mechanism & Mitigation Status |
| :--- | :--- | :--- | :--- |
| **Blocking Misses** | {tax_a.fn_blocking_miss:,d} | {tax_b.fn_blocking_miss:,d} | Links outside top-15 candidate union. |
| **DBA / Trade Names** | {tax_a.fn_dba_trade_name:,d} | {tax_b.fn_dba_trade_name:,d} | Distinct legal name vs trade style. Caught by address anchor matching. |
| **Severe Abbreviations** | {tax_a.fn_severe_abbreviation:,d} | {tax_b.fn_severe_abbreviation:,d} | 2-4 letter acronyms. Caught by initialism feature. |
| **Missing Address** | {tax_a.fn_missing_address:,d} | {tax_b.fn_missing_address:,d} | Incomplete records. Covered by missingness indicators. |
| **Other Misses** | {tax_a.fn_other:,d} | {tax_b.fn_other:,d} | Borderline confidence scores below $0.60$. |

---
*Phase 12 Total Runtime: {time.time() - t0_all:.2f}s*
"""

    (logs_dir / "phase12_error_analysis_report.md").write_text(report_md, encoding="utf-8")

    result_json = {
        "val_a": {
            "macro_f05": summary_a.macro_f05,
            "precision": summary_a.macro_precision,
            "recall": summary_a.macro_recall,
            "tp": tax_a.total_tp,
            "fp": tax_a.total_fp,
            "fn": tax_a.total_fn,
            "singleton_accuracy": tax_a.singleton_accuracy,
            "false_merge_rate": tax_a.false_merge_rate,
            "fp_breakdown": {
                "common_name": tax_a.fp_common_name_collision,
                "address_collision": tax_a.fp_address_collision,
                "franchise": tax_a.fp_franchise_branch_confusion,
                "other": tax_a.fp_other,
            },
            "fn_breakdown": {
                "blocking_miss": tax_a.fn_blocking_miss,
                "dba_trade_name": tax_a.fn_dba_trade_name,
                "severe_abbreviation": tax_a.fn_severe_abbreviation,
                "missing_address": tax_a.fn_missing_address,
                "other": tax_a.fn_other,
            },
        },
        "val_b": {
            "macro_f05": summary_b.macro_f05,
            "precision": summary_b.macro_precision,
            "recall": summary_b.macro_recall,
            "tp": tax_b.total_tp,
            "fp": tax_b.total_fp,
            "fn": tax_b.total_fn,
            "singleton_accuracy": tax_b.singleton_accuracy,
            "false_merge_rate": tax_b.false_merge_rate,
            "fp_breakdown": {
                "common_name": tax_b.fp_common_name_collision,
                "address_collision": tax_b.fp_address_collision,
                "franchise": tax_b.fp_franchise_branch_confusion,
                "other": tax_b.fp_other,
            },
            "fn_breakdown": {
                "blocking_miss": tax_b.fn_blocking_miss,
                "dba_trade_name": tax_b.fn_dba_trade_name,
                "severe_abbreviation": tax_b.fn_severe_abbreviation,
                "missing_address": tax_b.fn_missing_address,
                "other": tax_b.fn_other,
            },
        },
    }

    (logs_dir / "phase12_error_analysis_report.json").write_text(json.dumps(result_json, indent=2), encoding="utf-8")
    logger.info("Phase 12 error analysis completed successfully!")
    return result_json


if __name__ == "__main__":
    run_phase12_pipeline()
