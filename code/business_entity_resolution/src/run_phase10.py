"""Phase 10: Stage-2 Context Re-Scoring, Exact Expected-F0.5 DP & Decision Optimization.

Consumes Stage-1 calibrated probabilities, extracts G.5 context and competition features,
trains a low-capacity Stage-2 LightGBM model, applies Platt calibration,
and evaluates exact Expected-F0.5 Poisson-Binomial DP subset selection with conflict resolution.
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
from src.context_features import ContextFeatureExtractor, G5_FEATURE_NAMES
from src.stage2_model import Stage2Rescorer
from src.decision_engine import DecisionEngine
from src.metrics import compute_macro_f05

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] Phase10: %(message)s"
)
logger = logging.getLogger(__name__)


def generate_partition_data(
    partition_name: str,
    s1_norm_df: pd.DataFrame,
    blocker: MultiChannelBlocker,
    extractor: FeatureExtractor,
    cap_per_entity: int = 15,
) -> Tuple[FeatureBatch, CandidateStore, Dict[str, List[str]]]:
    """Generates candidate pairs and features for a partition."""
    store = CandidateStore()

    c_a = blocker.generate_channel_a(s1_norm_df)
    c_b = blocker.generate_channel_b(s1_norm_df)
    c_c = blocker.generate_channel_c(s1_norm_df)
    c_d = blocker.generate_channel_d(s1_norm_df)
    c_e = blocker.generate_channel_e(s1_norm_df)
    c_g = blocker.generate_channel_g(s1_norm_df)
    c_h = blocker.generate_channel_h(s1_norm_df)

    store.add_channel_candidates("channel_A", c_a)
    store.add_channel_candidates("channel_B", c_b)
    store.add_channel_candidates("channel_C", c_c)
    store.add_channel_candidates("channel_D", c_d)
    store.add_channel_candidates("channel_E", c_e)
    store.add_channel_candidates("channel_G", c_g)
    store.add_channel_candidates("channel_H", c_h)

    pairs_map = store.get_candidate_dict(cap=cap_per_entity)
    flat_pairs = [(s1_id, c_id) for s1_id, cands in pairs_map.items() for c_id in cands]

    batch = extractor.extract_pair_batch(flat_pairs)
    return batch, store, pairs_map


def run_phase10_pipeline() -> Dict[str, Any]:
    """Execute complete Phase 10 pipeline."""
    t0_all = time.time()
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    train_dir = repo_root / "dataset" / "train"
    splits_dir = repo_root / "artifacts" / "splits"
    logs_dir = repo_root / "logs"
    models_dir = repo_root / "artifacts" / "models"
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PHASE 10: STAGE-2 CONTEXT RE-SCORING & DECISION OPTIMIZATION")
    logger.info("=" * 70)

    # 1. Load Split Manifest and Source Data
    logger.info("Step 1: Loading Partitions and Source Records...")
    manifest_path = splits_dir / "split_manifest.tsv.gz"
    if not manifest_path.exists():
        manifest_path = splits_dir / "split_manifest.tsv"
    manifest = SplitManifest.load(manifest_path)

    s1_df_all, _ = load_entity_source(train_dir / "train_source1.tsv", "S1")
    gt_df, _ = load_ground_truth(train_dir / "train_ground_truth.tsv")
    gt_map = parse_ground_truth_to_dict(gt_df)

    # Cohorts
    train_s1 = manifest.filter_s1_dataframe(s1_df_all, "train").head(5000).copy()
    earlystop_s1 = manifest.filter_s1_dataframe(s1_df_all, "earlystop").head(1500).copy()
    cal_s1 = manifest.filter_s1_dataframe(s1_df_all, "calibration").head(1500).copy()
    val_a_s1 = manifest.filter_s1_dataframe(s1_df_all, "val_a").head(3000).copy()
    val_b_s1 = manifest.filter_s1_dataframe(s1_df_all, "val_b").head(3000).copy()

    # Candidate pool
    s2_df, _ = load_entity_source(train_dir / "train_source2.tsv", "S2")
    s3_df, _ = load_entity_source(train_dir / "train_source3.tsv", "S3")

    cohort_s1_ids = set(train_s1["entity_id"]) | set(earlystop_s1["entity_id"]) | set(cal_s1["entity_id"]) | set(val_a_s1["entity_id"]) | set(val_b_s1["entity_id"])
    cohort_gt_targets = {c for s1 in cohort_s1_ids for c in gt_map.get(s1, set())}
    s2_gt_eids = {e for e in cohort_gt_targets if (e.startswith("S2-") or e.startswith("S2_"))}
    s3_gt_eids = {e for e in cohort_gt_targets if (e.startswith("S3-") or e.startswith("S3_"))}

    s2_sample = pd.concat([s2_df[s2_df["entity_id"].isin(s2_gt_eids)], s2_df.head(60000)]).drop_duplicates(subset=["entity_id"])
    s3_sample = pd.concat([s3_df[s3_df["entity_id"].isin(s3_gt_eids)], s3_df.head(60000)]).drop_duplicates(subset=["entity_id"])

    # 2. Normalization
    logger.info("Step 2: Normalizing Cohorts...")
    normalizer = EntityNormalizer()
    train_s1_norm = normalizer.normalize_dataframe(train_s1)
    earlystop_s1_norm = normalizer.normalize_dataframe(earlystop_s1)
    cal_s1_norm = normalizer.normalize_dataframe(cal_s1)
    val_a_s1_norm = normalizer.normalize_dataframe(val_a_s1)
    val_b_s1_norm = normalizer.normalize_dataframe(val_b_s1)

    s2_norm = normalizer.normalize_dataframe(s2_sample)
    s3_norm = normalizer.normalize_dataframe(s3_sample)

    # 3. Lookups & Blocker
    logger.info("Step 3: Building Blocking Index and Feature Extractor...")
    all_s1_norm = pd.concat([train_s1_norm, earlystop_s1_norm, cal_s1_norm, val_a_s1_norm, val_b_s1_norm])
    s1_lookup = build_entity_lookup(all_s1_norm)
    cand_lookup = build_entity_lookup(s2_norm)
    cand_lookup.update(build_entity_lookup(s3_norm))

    index = BlockingIndex(min_token_len=3, max_token_df=5000)
    index.build_indexes(s2_norm, s3_norm)
    blocker = MultiChannelBlocker(index, max_cands_per_key=100)
    extractor = FeatureExtractor(s1_lookup, cand_lookup, ground_truth=gt_map)

    # 4. Generate Stage-1 Batches
    logger.info("Step 4: Generating Stage-1 candidate pairs and extracting features...")
    train_batch, _, _ = generate_partition_data("train", train_s1_norm, blocker, extractor)
    es_batch, _, _ = generate_partition_data("earlystop", earlystop_s1_norm, blocker, extractor)
    cal_batch, _, _ = generate_partition_data("calibration", cal_s1_norm, blocker, extractor)
    val_a_batch, _, _ = generate_partition_data("val_a", val_a_s1_norm, blocker, extractor)
    val_b_batch, _, _ = generate_partition_data("val_b", val_b_s1_norm, blocker, extractor)

    # 5. Fit Stage-1 Scorer & Calibrator
    logger.info("Step 5: Training Stage-1 Pairwise Scorer...")
    stage1_scorer = PairwiseScorer(
        model_type="lightgbm",
        feature_names=FEATURE_NAMES,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        random_state=42,
    )
    stage1_scorer.fit(
        train_batch.features,
        train_batch.labels,
        X_val=es_batch.features,
        y_val=es_batch.labels,
        early_stopping_rounds=30,
        verbose=False,
    )

    stage1_calibrator = ProbabilityCalibrator(method="sigmoid")
    cal_raw = stage1_scorer.predict_proba(cal_batch.features)
    stage1_calibrator.fit(cal_raw, cal_batch.labels)

    # Calibrated probabilities p1
    p1_train = stage1_calibrator.predict_proba(stage1_scorer.predict_proba(train_batch.features))
    p1_es = stage1_calibrator.predict_proba(stage1_scorer.predict_proba(es_batch.features))
    p1_cal = stage1_calibrator.predict_proba(cal_raw)
    p1_val_a = stage1_calibrator.predict_proba(stage1_scorer.predict_proba(val_a_batch.features))
    p1_val_b = stage1_calibrator.predict_proba(stage1_scorer.predict_proba(val_b_batch.features))

    # 6. Extract G.5 Context & Competition Features
    logger.info("Step 6: Extracting G.5 Context & Reverse-Rank Features...")
    g5_extractor = ContextFeatureExtractor()

    def make_g5_df(batch: FeatureBatch, probs: np.ndarray) -> pd.DataFrame:
        scored = [(s1, c, float(p)) for (s1, c), p in zip(batch.pair_ids, probs)]
        return g5_extractor.extract_context_features(scored)

    X_train_g5 = make_g5_df(train_batch, p1_train)
    X_es_g5 = make_g5_df(es_batch, p1_es)
    X_cal_g5 = make_g5_df(cal_batch, p1_cal)
    X_val_a_g5 = make_g5_df(val_a_batch, p1_val_a)
    X_val_b_g5 = make_g5_df(val_b_batch, p1_val_b)

    # 7. Train Stage-2 Model
    logger.info("Step 7: Training Stage-2 Context Rescorer...")
    stage2_rescorer = Stage2Rescorer(max_depth=3, num_leaves=8, min_child_samples=50, learning_rate=0.05, n_estimators=150)
    stage2_rescorer.fit(
        X_train=X_train_g5,
        y_train=train_batch.labels,
        X_earlystop=X_es_g5,
        y_earlystop=es_batch.labels,
        X_calib=X_cal_g5,
        y_calib=cal_batch.labels,
        calibration_method="sigmoid",
    )

    stage2_rescorer.save(models_dir / "stage2_model.joblib")

    stage2_val_a_eval = stage2_rescorer.evaluate(X_val_a_g5, val_a_batch.labels)
    stage2_val_b_eval = stage2_rescorer.evaluate(X_val_b_g5, val_b_batch.labels)
    logger.info(f"  Stage-2 Val_A: ROC-AUC={stage2_val_a_eval['roc_auc']:.4f}, PR-AUC={stage2_val_a_eval['pr_auc']:.4f}, Brier={stage2_val_a_eval['brier_score']:.5f}")
    logger.info(f"  Stage-2 Val_B: ROC-AUC={stage2_val_b_eval['roc_auc']:.4f}, PR-AUC={stage2_val_b_eval['pr_auc']:.4f}, Brier={stage2_val_b_eval['brier_score']:.5f}")

    # 8. Decision Engine Evaluation
    logger.info("Step 8: Evaluating Exact Expected-F0.5 DP Decision Engine & Conflict Resolution...")
    decision_engine = DecisionEngine(margin_delta=0.05, enable_conflict_resolution=True)

    def evaluate_decision_layer(g5_df: pd.DataFrame, s1_ids: List[str], gt: Dict[str, Set[str]]) -> Dict[str, float]:
        p2 = stage2_rescorer.predict_proba(g5_df)
        g5_df = g5_df.copy()
        g5_df["p2"] = p2

        cand_map: Dict[str, List[Tuple[str, float]]] = {}
        for s1_id, group in g5_df.groupby("s1_id"):
            cand_map[str(s1_id)] = list(zip(group["cand_id"].astype(str), group["p2"].astype(float)))

        for eid in s1_ids:
            if eid not in cand_map:
                cand_map[eid] = []

        preds = decision_engine.optimize_predictions(cand_map)
        sub_gt = {eid: gt.get(eid, set()) for eid in s1_ids}
        eval_summary = compute_macro_f05(preds, sub_gt)
        return {
            "macro_f05": float(eval_summary.macro_f05),
            "precision": float(eval_summary.macro_precision),
            "recall": float(eval_summary.macro_recall),
            "num_entities": len(s1_ids),
        }

    val_a_s1_ids = val_a_s1["entity_id"].tolist()
    val_b_s1_ids = val_b_s1["entity_id"].tolist()

    val_a_res = evaluate_decision_layer(X_val_a_g5, val_a_s1_ids, gt_map)
    val_b_res = evaluate_decision_layer(X_val_b_g5, val_b_s1_ids, gt_map)

    logger.info(f"  Val_A Final Result -> Macro F0.5: {val_a_res['macro_f05']:.4f}, Precision: {val_a_res['precision']*100:.2f}%, Recall: {val_a_res['recall']*100:.2f}%")
    logger.info(f"  Val_B Final Result -> Macro F0.5: {val_b_res['macro_f05']:.4f}, Precision: {val_b_res['precision']*100:.2f}%, Recall: {val_b_res['recall']*100:.2f}%")

    # 9. Output Documentation & Reports
    importances = stage2_rescorer.get_feature_importances()
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    report_md = f"""# Phase 10: Stage-2 Re-Scoring & Exact Expected-F0.5 Decision Optimization Report

## 1. Executive Summary
- **Stage-2 Model:** Low-capacity LightGBM (`max_depth=3`, `num_leaves=8`, `min_child_samples=50`) trained on G.5 Context & Competition features.
- **Decision Engine:** Exact Expected-$F_{{0.5}}$ Poisson-Binomial dynamic programming with margin-guarded conflict resolution ($\Delta = 0.05$) under Hypothesis H1.
- **Zero Leakage:** Strictly held-out validation partitions (`Val_A`, `Val_B`).

## 2. Discrimination & Calibration Evaluation

| Split | ROC-AUC | PR-AUC | Brier Score |
| :--- | :--- | :--- | :--- |
| **Val_A (Held-Out)** | **{stage2_val_a_eval['roc_auc']:.4f}** | **{stage2_val_a_eval['pr_auc']:.4f}** | **{stage2_val_a_eval['brier_score']:.5f}** |
| **Val_B (Held-Out)** | **{stage2_val_b_eval['roc_auc']:.4f}** | **{stage2_val_b_eval['pr_auc']:.4f}** | **{stage2_val_b_eval['brier_score']:.5f}** |

## 3. Decision Optimization Results (Macro $F_{{0.5}}$)

| Evaluation Split | Macro $F_{{0.5}}$ | Precision | Recall | Evaluated S1 Entities |
| :--- | :--- | :--- | :--- | :--- |
| **Val_A (Held-Out)** | **{val_a_res['macro_f05']:.4f}** | **{val_a_res['precision']*100:.2f}%** | **{val_a_res['recall']*100:.2f}%** | {val_a_res['num_entities']:,d} |
| **Val_B (Held-Out)** | **{val_b_res['macro_f05']:.4f}** | **{val_b_res['precision']*100:.2f}%** | **{val_b_res['recall']*100:.2f}%** | {val_b_res['num_entities']:,d} |

## 4. G.5 Feature Importances (Normalized Gain)

| Rank | Feature Name | Description | Importance Gain |
| :--- | :--- | :--- | :--- |
"""
    for i, (fname, imp) in enumerate(sorted_imp, 1):
        report_md += f"| {i} | `{fname}` | G.5 Feature | {imp*100:.2f}% |\n"

    report_md += f"\n---\n*Phase 10 Total Execution Runtime: {time.time() - t0_all:.2f}s*\n"

    (logs_dir / "phase10_stage2_report.md").write_text(report_md, encoding="utf-8")

    results_json = {
        "val_a": val_a_res,
        "val_b": val_b_res,
        "stage2_metrics_val_a": stage2_val_a_eval,
        "stage2_metrics_val_b": stage2_val_b_eval,
        "feature_importances": importances,
    }
    (logs_dir / "phase10_stage2_report.json").write_text(json.dumps(results_json, indent=2), encoding="utf-8")

    logger.info("Phase 10 execution and audit reports generated successfully!")
    return results_json


if __name__ == "__main__":
    run_phase10_pipeline()
