"""Phase 5 Execution Pipeline: Multi-Channel Candidate Generation & Blocking Evaluation.

Executes:
1. Ingestion of canonical Phase 4 split (val_a partition: 441,362 S1 entities, 1,528,532 GT links).
2. Multi-representation normalization across val_a, Source 2 (5.03M), and Source 3 (5.28M).
3. Multi-channel indexing over S2 and S3 records.
4. Independent evaluation of Channels A through H.
5. Progressive marginal ablation (A -> A+B -> A+B+C -> ... -> Final Union).
6. Candidate cap sensitivity experiments (No Cap, 100, 50, 30, 20).
7. Comprehensive error analysis (Missed matches, Country, S2 vs S3, Cardinality).
8. Export of detailed reports to logs/phase5_blocking_report.json and logs/phase5_blocking_report.md.
"""

from datetime import datetime, timezone
from pathlib import Path
import time
import json
import logging
import psutil
import pandas as pd
import numpy as np

from src.data_loader import (
    load_entity_source,
    load_ground_truth,
    parse_ground_truth_to_dict,
)
from src.normalizer import EntityNormalizer
from src.split import SplitManifest
from src.index_builder import BlockingIndex
from src.candidate_store import CandidateStore
from src.blocker import MultiChannelBlocker
from src.blocking_metrics import compute_blocking_metrics, BlockingMetricsSummary


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Phase5_Runner")


def get_process_memory_mb() -> float:
    """Returns current process RSS memory in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def run_phase5() -> None:
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("PHASE 5: MULTI-CHANNEL CANDIDATE GENERATION & BLOCKING ENGINE")
    logger.info("=" * 70)

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    train_dir = repo_root / "dataset" / "train"
    artifacts_dir = repo_root / "artifacts" / "blocking"
    splits_dir = repo_root / "artifacts" / "splits"
    logs_dir = repo_root / "logs"

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest Canonical Split and Ground Truth
    logger.info("Step 1: Loading canonical Phase 4 split manifest...")
    manifest_path = splits_dir / "split_manifest.tsv.gz"
    if not manifest_path.exists():
        manifest_path = splits_dir / "split_manifest.tsv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found in {splits_dir}. Run Phase 4 first.")

    manifest = SplitManifest.load(manifest_path)
    logger.info(f"Loaded split manifest with {manifest.total_entities:,} total S1 entities.")

    logger.info("Loading Source 1 and Ground Truth...")
    s1_df_all, _ = load_entity_source(train_dir / "train_source1.tsv", "S1")
    gt_df, _ = load_ground_truth(train_dir / "train_ground_truth.tsv")
    gt_dict_all = parse_ground_truth_to_dict(gt_df)

    val_a_s1 = manifest.filter_s1_dataframe(s1_df_all, "val_a")
    val_a_gt = manifest.filter_ground_truth(gt_dict_all, "val_a")
    logger.info(f"val_a Partition: {len(val_a_s1):,} S1 entities, {sum(len(v) for v in val_a_gt.values()):,} Ground Truth links.")

    # 2. Ingest Candidate Sources S2 and S3
    logger.info("Step 2: Loading Source 2 and Source 3...")
    s2_df, _ = load_entity_source(train_dir / "train_source2.tsv", "S2")
    s3_df, _ = load_entity_source(train_dir / "train_source3.tsv", "S3")
    logger.info(f"Loaded S2: {len(s2_df):,} rows, S3: {len(s3_df):,} rows.")

    # 3. Normalization
    logger.info("Step 3: Normalizing entity records...")
    normalizer = EntityNormalizer.from_config_dir(repo_root / "code" / "business_entity_resolution" / "configs")
    t0_norm = time.time()
    val_a_s1_norm = normalizer.normalize_dataframe(val_a_s1)
    s2_norm = normalizer.normalize_dataframe(s2_df)
    s3_norm = normalizer.normalize_dataframe(s3_df)
    logger.info(f"Normalized all records in {time.time() - t0_norm:.2f}s (RAM: {get_process_memory_mb():.1f} MB).")

    # 4. Build Multi-Channel Blocking Index
    logger.info("Step 4: Building BlockingIndex over S2 and S3...")
    t0_idx = time.time()
    index = BlockingIndex(max_token_df=500, min_token_len=3)
    blocker = MultiChannelBlocker(index=index, max_cands_per_key=100)
    blocker.fit(s2_norm, s3_norm)
    logger.info(f"BlockingIndex constructed in {time.time() - t0_idx:.2f}s (RAM: {get_process_memory_mb():.1f} MB).")

    # 5. Benchmark Individual Channels
    logger.info("Step 5: Benchmarking Individual Blocking Channels...")
    channels = [
        ("Channel_A", "Exact Normalized Name + Country", lambda: blocker.generate_channel_a(val_a_s1_norm)),
        ("Channel_B", "Name Token Signatures (Sorted / Prefix-2 + Country)", lambda: blocker.generate_channel_b(val_a_s1_norm)),
        ("Channel_C", "Address Numeric Anchors (Num + Name3 + Country)", lambda: blocker.generate_channel_c(val_a_s1_norm)),
        ("Channel_D", "Address Distinctive Tokens (Tok + Name3 + Country)", lambda: blocker.generate_channel_d(val_a_s1_norm)),
        ("Channel_E", "Rare Distinctive Name Tokens (DF <= 500 + Country)", lambda: blocker.generate_channel_e(val_a_s1_norm)),
        ("Channel_F", "Sparse Character 3-gram TF-IDF (Top-5 per Country)", lambda: blocker.generate_channel_f_sparse_tfidf(val_a_s1_norm, s2_norm, s3_norm, top_k=5)),
        ("Channel_G", "Controlled 4-gram Prefix + Address Anchors", lambda: blocker.generate_channel_g(val_a_s1_norm)),
        ("Channel_H", "Landmark Anchor + Name Anchor Cross-Field", lambda: blocker.generate_channel_h(val_a_s1_norm)),
    ]

    channel_candidates: Dict[str, Dict[str, List[str]]] = {}
    channel_metrics: Dict[str, BlockingMetricsSummary] = {}

    for ch_id, ch_desc, gen_func in channels:
        logger.info(f"Evaluating {ch_id}: {ch_desc}...")
        t0 = time.time()
        cands = gen_func()
        t_elapsed = time.time() - t0
        mem_mb = get_process_memory_mb()
        metrics = compute_blocking_metrics(cands, val_a_gt, runtime_seconds=t_elapsed, peak_memory_mb=mem_mb)
        channel_candidates[ch_id] = cands
        channel_metrics[ch_id] = metrics
        logger.info(
            f"  {ch_id} Result: Recall = {metrics.candidate_recall*100:.2f}%, "
            f"Coverage = {metrics.entity_complete_coverage*100:.2f}%, "
            f"Avg Cands = {metrics.avg_candidates_per_s1:.2f}, "
            f"Pairs = {metrics.total_candidate_pairs:,} (took {t_elapsed:.2f}s)"
        )

    # 6. Progressive Marginal Ablation
    logger.info("Step 6: Running Progressive Marginal Ablation across Channels...")
    store = CandidateStore(val_a_s1_norm["entity_id"])
    ablation_steps: List[Dict[str, Any]] = []

    prev_recall = 0.0
    prev_pairs = 0
    t0_abl_start = time.time()

    for ch_id, ch_desc, _ in channels:
        t0_step = time.time()
        new_pairs = store.add_channel_candidates(ch_id, channel_candidates[ch_id])
        t_step = time.time() - t0_step
        current_dict = store.get_candidate_dict()
        current_metrics = compute_blocking_metrics(
            current_dict, val_a_gt, runtime_seconds=time.time() - t0_abl_start, peak_memory_mb=get_process_memory_mb()
        )

        marginal_recall = current_metrics.candidate_recall - prev_recall
        marginal_pairs = current_metrics.total_candidate_pairs - prev_pairs
        efficiency = (marginal_recall * current_metrics.total_gt_links) / max(1, marginal_pairs)

        ablation_record = {
            "channel_added": ch_id,
            "description": ch_desc,
            "total_recall": round(current_metrics.candidate_recall, 6),
            "marginal_recall": round(marginal_recall, 6),
            "entity_complete_coverage": round(current_metrics.entity_complete_coverage, 6),
            "entity_partial_coverage": round(current_metrics.entity_partial_coverage, 6),
            "total_candidate_pairs": current_metrics.total_candidate_pairs,
            "marginal_candidate_pairs": marginal_pairs,
            "efficiency_ratio": round(efficiency, 6),
            "avg_candidates_per_s1": round(current_metrics.avg_candidates_per_s1, 2),
            "p95_candidates_per_s1": round(current_metrics.p95_candidates_per_s1, 1),
            "p99_candidates_per_s1": round(current_metrics.p99_candidates_per_s1, 1),
            "max_candidates_per_s1": current_metrics.max_candidates_per_s1,
            "cumulative_runtime_sec": round(time.time() - t0_abl_start, 2),
        }
        ablation_steps.append(ablation_record)
        prev_recall = current_metrics.candidate_recall
        prev_pairs = current_metrics.total_candidate_pairs

        logger.info(
            f"  + {ch_id}: Recall -> {current_metrics.candidate_recall*100:.2f}% "
            f"(+{marginal_recall*100:.2f}%), Total Pairs: {current_metrics.total_candidate_pairs:,} "
            f"(+{marginal_pairs:,}), Avg: {current_metrics.avg_candidates_per_s1:.1f}/S1"
        )

    final_union_metrics = current_metrics

    # 7. Candidate Cap Experimentation
    logger.info("Step 7: Evaluating Candidate Cap Tradeoffs...")
    caps = [None, 150, 100, 75, 50, 30, 20]
    cap_results: List[Dict[str, Any]] = []

    for cap_val in caps:
        capped_dict = store.get_candidate_dict(cap=cap_val)
        cap_m = compute_blocking_metrics(capped_dict, val_a_gt)
        cap_label = str(cap_val) if cap_val is not None else "No Cap"
        cap_rec = {
            "cap": cap_label,
            "candidate_recall": round(cap_m.candidate_recall, 6),
            "entity_complete_coverage": round(cap_m.entity_complete_coverage, 6),
            "total_candidate_pairs": cap_m.total_candidate_pairs,
            "avg_candidates_per_s1": round(cap_m.avg_candidates_per_s1, 2),
            "p95_candidates_per_s1": round(cap_m.p95_candidates_per_s1, 1),
            "p99_candidates_per_s1": round(cap_m.p99_candidates_per_s1, 1),
            "max_candidates_per_s1": cap_m.max_candidates_per_s1,
        }
        cap_results.append(cap_rec)
        logger.info(
            f"  Cap = {cap_label:6s}: Recall = {cap_m.candidate_recall*100:.2f}%, "
            f"Coverage = {cap_m.entity_complete_coverage*100:.2f}%, "
            f"Total Pairs = {cap_m.total_candidate_pairs:,}, Avg = {cap_m.avg_candidates_per_s1:.2f}"
        )

    # 8. Detailed Subgroup and Error Analysis
    logger.info("Step 8: Performing Detailed Subgroup and Error Diagnostics...")
    all_final_candidates = store.get_candidate_dict()

    # Subgroups: Country, Cardinality, Match Pattern
    country_subgroup_stats: Dict[str, Dict[str, Any]] = {}
    s1_countries = val_a_s1_norm["country_norm"].tolist()
    s1_eids = val_a_s1_norm["entity_id"].tolist()

    for c_norm in ["us", "india", "france", "other", "unknown"]:
        matching_eids = [eid for eid, c in zip(s1_eids, s1_countries) if c == c_norm]
        if not matching_eids:
            continue
        sub_cands = {eid: all_final_candidates.get(eid, []) for eid in matching_eids}
        sub_gt = {eid: val_a_gt.get(eid, set()) for eid in matching_eids}
        sub_m = compute_blocking_metrics(sub_cands, sub_gt)
        country_subgroup_stats[c_norm] = {
            "entity_count": len(matching_eids),
            "candidate_recall": round(sub_m.candidate_recall, 6),
            "entity_complete_coverage": round(sub_m.entity_complete_coverage, 6),
            "avg_candidates_per_s1": round(sub_m.avg_candidates_per_s1, 2),
            "total_candidate_pairs": sub_m.total_candidate_pairs,
        }

    # S2 vs S3 Link Recovery Analysis
    total_s2_gt = 0
    recovered_s2_gt = 0
    total_s3_gt = 0
    recovered_s3_gt = 0

    for s1_id, true_set in val_a_gt.items():
        cand_set = set(all_final_candidates.get(s1_id, []))
        for mid in true_set:
            if mid.startswith("S2-"):
                total_s2_gt += 1
                if mid in cand_set:
                    recovered_s2_gt += 1
            elif mid.startswith("S3-"):
                total_s3_gt += 1
                if mid in cand_set:
                    recovered_s3_gt += 1

    source_link_recovery = {
        "Source_2": {
            "total_links": total_s2_gt,
            "recovered_links": recovered_s2_gt,
            "recall": round(recovered_s2_gt / max(1, total_s2_gt), 6),
        },
        "Source_3": {
            "total_links": total_s3_gt,
            "recovered_links": recovered_s3_gt,
            "recall": round(recovered_s3_gt / max(1, total_s3_gt), 6),
        },
    }

    # Missed matches sampling & categorization
    missed_examples = []
    for s1_id, true_set in val_a_gt.items():
        if len(missed_examples) >= 20:
            break
        cand_set = set(all_final_candidates.get(s1_id, []))
        missed = true_set - cand_set
        if missed:
            s1_row = val_a_s1_norm[val_a_s1_norm["entity_id"] == s1_id].iloc[0]
            missed_examples.append({
                "s1_id": s1_id,
                "s1_name": s1_row["business_name"],
                "s1_address": s1_row["business_address"],
                "s1_country": s1_row["country"],
                "missed_ids": list(missed),
                "candidates_generated": len(cand_set),
            })

    # 9. Generate Reports
    logger.info("Step 9: Generating Phase 5 Markdown and JSON Reports...")
    report_json_path = logs_dir / "phase5_blocking_report.json"
    report_md_path = logs_dir / "phase5_blocking_report.md"

    json_report = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluated_dataset": {
            "s1_count": len(s1_df_all),
            "val_a_count": len(val_a_s1),
            "s2_count": len(s2_df),
            "s3_count": len(s3_df),
            "val_a_gt_links": sum(len(v) for v in val_a_gt.values()),
        },
        "per_channel_metrics": {k: v.to_dict() for k, v in channel_metrics.items()},
        "progressive_ablation": ablation_steps,
        "candidate_caps": cap_results,
        "final_union": final_union_metrics.to_dict(),
        "country_subgroups": country_subgroup_stats,
        "source_link_recovery": source_link_recovery,
        "missed_examples_sample": missed_examples,
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)

    # Build Markdown Report
    md_lines = [
        "# Phase 5 Candidate Generation & Multi-Channel Blocking Report",
        f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*",
        "",
        "## 1. Dataset & Validation Scope",
        "",
        f"- **Evaluated Partition**: `val_a` ({len(val_a_s1):,} S1 entities / 20.0% canonical split)",
        f"- **Ground Truth Links**: {sum(len(v) for v in val_a_gt.values()):,} true links in `val_a`",
        f"- **Candidate Search Pool**: Source 2 ({len(s2_df):,} rows) + Source 3 ({len(s3_df):,} rows) = **{len(s2_df) + len(s3_df):,} Total Candidate Records**",
        "",
        "## 2. Individual Channel Performance",
        "",
        "| Channel ID | Blocking Mechanism | Candidate Recall | Entity Complete Coverage | Total Pairs | Avg Cands/S1 | P95 | P99 | Max | Runtime |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for ch_id, ch_desc, _ in channels:
        m = channel_metrics[ch_id]
        md_lines.append(
            f"| `{ch_id}` | {ch_desc} | **{m.candidate_recall*100:.2f}%** | {m.entity_complete_coverage*100:.2f}% | "
            f"{m.total_candidate_pairs:,} | {m.avg_candidates_per_s1:.2f} | {m.p95_candidates_per_s1:.0f} | "
            f"{m.p99_candidates_per_s1:.0f} | {m.max_candidates_per_s1:,} | {m.runtime_seconds:.2f}s |"
        )
    md_lines.append("")

    md_lines.extend([
        "## 3. Progressive Marginal Ablation",
        "",
        "| Step | Channel Added | Total Recall | Marginal Recall | Complete Coverage | Total Pairs | Marginal Pairs | Avg/S1 | Efficiency Ratio |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for step in ablation_steps:
        md_lines.append(
            f"| {step['channel_added']} | {step['description']} | **{step['total_recall']*100:.2f}%** | "
            f"+{step['marginal_recall']*100:.2f}% | {step['entity_complete_coverage']*100:.2f}% | "
            f"{step['total_candidate_pairs']:,} | +{step['marginal_candidate_pairs']:,} | "
            f"{step['avg_candidates_per_s1']:.2f} | {step['efficiency_ratio']:.4f} |"
        )
    md_lines.append("")

    md_lines.extend([
        "## 4. Final Candidate Union Summary",
        "",
        "| Metric | Value | Description |",
        "|:---|:---|:---|",
        f"| **Final Candidate Recall** | **{final_union_metrics.candidate_recall*100:.2f}%** | True GT links recovered ({final_union_metrics.recovered_gt_links:,} / {final_union_metrics.total_gt_links:,}) |",
        f"| **Entity Complete Coverage** | **{final_union_metrics.entity_complete_coverage*100:.2f}%** | S1s with 100% true matches in candidate set |",
        f"| **Entity Partial Coverage** | **{final_union_metrics.entity_partial_coverage*100:.2f}%** | S1s with >= 1 true match in candidate set |",

        f"| **Total Candidate Pairs** | **{final_union_metrics.total_candidate_pairs:,}** | Total pairs passed to Stage-1 Pairwise Scorer |",
        f"| **Average Candidates / S1** | **{final_union_metrics.avg_candidates_per_s1:.2f}** | Mean candidate fan-out per S1 entity |",
        f"| **Median Candidates / S1** | **{final_union_metrics.median_candidates_per_s1:.1f}** | Median candidate fan-out |",
        f"| **P95 / P99 Candidates** | **{final_union_metrics.p95_candidates_per_s1:.0f} / {final_union_metrics.p99_candidates_per_s1:.0f}** | Upper percentile distribution |",
        f"| **Maximum Candidates / S1** | **{final_union_metrics.max_candidates_per_s1:,}** | Peak single-entity candidate count |",
        "",
        "## 5. Candidate Cap Sensitivity Analysis",
        "",
        "| Candidate Cap | Candidate Recall | Complete Coverage | Total Pairs | Avg Cands/S1 | P95 | P99 | Max |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for c in cap_results:
        md_lines.append(
            f"| `{c['cap']}` | **{c['candidate_recall']*100:.2f}%** | {c['entity_complete_coverage']*100:.2f}% | "
            f"{c['total_candidate_pairs']:,} | {c['avg_candidates_per_s1']:.2f} | {c['p95_candidates_per_s1']:.0f} | "
            f"{c['p99_candidates_per_s1']:.0f} | {c['max_candidates_per_s1']:,} |"
        )
    md_lines.append("")

    md_lines.extend([
        "## 6. Subgroup & Error Analysis",
        "",
        "### 6.1 Breakdown by Country",
        "",
        "| Country | S1 Entities | Candidate Recall | Complete Coverage | Avg Cands/S1 | Total Pairs |",
        "|:---|---:|---:|---:|---:|---:|",
    ])

    for c_norm, c_stats in country_subgroup_stats.items():
        md_lines.append(
            f"| `{c_norm.upper()}` | {c_stats['entity_count']:,} | **{c_stats['candidate_recall']*100:.2f}%** | "
            f"{c_stats['entity_complete_coverage']*100:.2f}% | {c_stats['avg_candidates_per_s1']:.2f} | "
            f"{c_stats['total_candidate_pairs']:,} |"
        )
    md_lines.append("")

    md_lines.extend([
        "### 6.2 Breakdown by Candidate Source",
        "",
        "| Candidate Source | Total GT Links | Recovered Links | Recall (%) |",
        "|:---|---:|---:|---:|",
    ])

    for src_name, src_data in source_link_recovery.items():
        md_lines.append(
            f"| `{src_name}` | {src_data['total_links']:,} | {src_data['recovered_links']:,} | **{src_data['recall']*100:.2f}%** |"
        )
    md_lines.append("")

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    total_time = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"Phase 5 pipeline completed successfully in {total_time:.2f}s.")
    logger.info(f"Report saved to: {report_md_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_phase5()
