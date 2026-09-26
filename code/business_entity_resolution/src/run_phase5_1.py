"""Phase 5.1 Execution Pipeline: Multi-Channel Blocking Recall Recovery.

Performs:
1. Exact Verification of Baseline (A+B+C+D+E+G+H -> 78.02% link recall on val_a).
2. Independent evaluation of new recovery channels:
   - Channel I: Phonetic / Transliteration Canonicalization
   - Channel J: Core Token Pair Inverted Index
   - Channel K: Address Number + Street Token Inverted Index
   - Channel L: Postal PIN / ZIP Code + Name Token Anchor
   - Channel M: Rare Character 3-Gram Inverted Index
3. Progressive Ablation across Configurations 0 through 5.
4. Pareto analysis: Candidate Recall vs Candidate Volume vs Memory.
5. Candidate cap sensitivity on the recovered union.
6. Subgroup breakdown (India vs US, S2 vs S3) and remaining missed link analysis.
7. Export to logs/phase5_1_recall_recovery_report.json and logs/phase5_1_recall_recovery_report.md.
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
logger = logging.getLogger("Phase5_1_Runner")


def get_process_memory_mb() -> float:
    """Returns current process RSS memory in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def run_phase5_1() -> None:
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("PHASE 5.1: BLOCKING RECALL RECOVERY & PARETO FRONTIER OPTIMIZATION")
    logger.info("=" * 70)

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    train_dir = repo_root / "dataset" / "train"
    splits_dir = repo_root / "artifacts" / "splits"
    logs_dir = repo_root / "logs"

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
    total_val_a_links = sum(len(v) for v in val_a_gt.values())
    logger.info(f"val_a Partition: {len(val_a_s1):,} S1 entities, {total_val_a_links:,} Ground Truth links.")

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
    logger.info("Step 4: Building BlockingIndex (All Channels A through M) over S2 and S3...")
    t0_idx = time.time()
    index = BlockingIndex(max_token_df=500, min_token_len=3)
    blocker = MultiChannelBlocker(index=index, max_cands_per_key=100)
    blocker.fit(s2_norm, s3_norm)
    logger.info(f"BlockingIndex constructed in {time.time() - t0_idx:.2f}s (RAM: {get_process_memory_mb():.1f} MB).")

    # 5. Baseline Channels (Phase 5: A, B, C, D, E, G, H)
    logger.info("Step 5: Generating candidates for all channels...")
    channel_defs = [
        # Original Phase 5 Baseline Channels
        ("Channel_A", "Exact Normalized Name + Country", lambda: blocker.generate_channel_a(val_a_s1_norm)),
        ("Channel_B", "Name Token Signatures (Sorted / Prefix-2 + Country)", lambda: blocker.generate_channel_b(val_a_s1_norm)),
        ("Channel_C", "Address Numeric Anchors (Num + Name3 + Country)", lambda: blocker.generate_channel_c(val_a_s1_norm)),
        ("Channel_D", "Address Distinctive Tokens (Tok + Name3 + Country)", lambda: blocker.generate_channel_d(val_a_s1_norm)),
        ("Channel_E", "Rare Distinctive Name Tokens (DF <= 500 + Country)", lambda: blocker.generate_channel_e(val_a_s1_norm)),
        ("Channel_G", "Controlled 4-gram Prefix + Address Anchors", lambda: blocker.generate_channel_g(val_a_s1_norm)),
        ("Channel_H", "Landmark Anchor + Name Anchor Cross-Field", lambda: blocker.generate_channel_h(val_a_s1_norm)),
        # New Phase 5.1 Recovery Channels
        ("Channel_I", "Phonetic / Transliteration Canonicalization", lambda: blocker.generate_channel_i(val_a_s1_norm)),
        ("Channel_J", "Core Token Pair Inverted Index", lambda: blocker.generate_channel_j(val_a_s1_norm)),
        ("Channel_K", "Address Number + Street Token Inverted Index", lambda: blocker.generate_channel_k(val_a_s1_norm)),
        ("Channel_L", "Postal PIN / ZIP Code + Name Token Anchor", lambda: blocker.generate_channel_l(val_a_s1_norm)),
        ("Channel_M", "Rare Character 3-Gram Inverted Index", lambda: blocker.generate_channel_m(val_a_s1_norm)),
    ]

    channel_candidates: Dict[str, Dict[str, List[str]]] = {}
    channel_metrics: Dict[str, BlockingMetricsSummary] = {}

    for ch_id, ch_desc, gen_func in channel_defs:
        t0 = time.time()
        cands = gen_func()
        t_elapsed = time.time() - t0
        mem_mb = get_process_memory_mb()
        metrics = compute_blocking_metrics(cands, val_a_gt, runtime_seconds=t_elapsed, peak_memory_mb=mem_mb)
        channel_candidates[ch_id] = cands
        channel_metrics[ch_id] = metrics
        logger.info(
            f"  {ch_id:10s} ({ch_desc[:45]:45s}): Recall = {metrics.candidate_recall*100:6.2f}%, "
            f"Coverage = {metrics.entity_complete_coverage*100:6.2f}%, "
            f"Avg = {metrics.avg_candidates_per_s1:6.2f}, Pairs = {metrics.total_candidate_pairs:10,d} ({t_elapsed:5.2f}s)"
        )

    # 6. Verify Baseline Configuration 0 (A+B+C+D+E+G+H)
    logger.info("Step 6: Verifying Baseline Configuration 0 (A+B+C+D+E+G+H)...")
    store_base = CandidateStore(val_a_s1_norm["entity_id"])
    for ch_id in ["Channel_A", "Channel_B", "Channel_C", "Channel_D", "Channel_E", "Channel_G", "Channel_H"]:
        store_base.add_channel_candidates(ch_id, channel_candidates[ch_id])

    base_cands = store_base.get_candidate_dict()
    base_metrics = compute_blocking_metrics(base_cands, val_a_gt)
    logger.info(
        f"  Baseline Configuration 0: Recall = {base_metrics.candidate_recall*100:.2f}% "
        f"({base_metrics.recovered_gt_links:,} / {base_metrics.total_gt_links:,} links), "
        f"Pairs = {base_metrics.total_candidate_pairs:,}, Avg = {base_metrics.avg_candidates_per_s1:.2f}"
    )

    # 7. Progressive Evaluation Across Configurations 0 through 5
    logger.info("Step 7: Evaluating Progressive Configurations 0 through 5...")

    configurations = [
        ("Config_0_Baseline", "Current Baseline (A+B+C+D+E+G+H)", ["Channel_A", "Channel_B", "Channel_C", "Channel_D", "Channel_E", "Channel_G", "Channel_H"]),
        ("Config_1_Phonetic", "Baseline + Phonetic / Transliteration (A..H + I)", ["Channel_A", "Channel_B", "Channel_C", "Channel_D", "Channel_E", "Channel_G", "Channel_H", "Channel_I"]),
        ("Config_2_CorePairs", "Config 1 + Core Token Pairs (+ J)", ["Channel_A", "Channel_B", "Channel_C", "Channel_D", "Channel_E", "Channel_G", "Channel_H", "Channel_I", "Channel_J"]),
        ("Config_3_AddrStreet", "Config 2 + Addr Number & Street Token (+ K)", ["Channel_A", "Channel_B", "Channel_C", "Channel_D", "Channel_E", "Channel_G", "Channel_H", "Channel_I", "Channel_J", "Channel_K"]),
        ("Config_4_PostalPIN", "Config 3 + Postal PIN / ZIP Anchor (+ L)", ["Channel_A", "Channel_B", "Channel_C", "Channel_D", "Channel_E", "Channel_G", "Channel_H", "Channel_I", "Channel_J", "Channel_K", "Channel_L"]),
        ("Config_5_FullUnion", "Config 4 + Rare 3-Gram Inverted Index (+ M)", ["Channel_A", "Channel_B", "Channel_C", "Channel_D", "Channel_E", "Channel_G", "Channel_H", "Channel_I", "Channel_J", "Channel_K", "Channel_L", "Channel_M"]),
    ]

    config_results: List[Dict[str, Any]] = []
    active_stores: Dict[str, CandidateStore] = {}

    prev_c_rec = 0.0
    prev_c_pairs = 0

    for cfg_id, cfg_desc, ch_list in configurations:
        t0_cfg = time.time()
        c_store = CandidateStore(val_a_s1_norm["entity_id"])
        for ch_id in ch_list:
            c_store.add_channel_candidates(ch_id, channel_candidates[ch_id])
        t_cfg = time.time() - t0_cfg

        c_dict = c_store.get_candidate_dict()
        m_cfg = compute_blocking_metrics(c_dict, val_a_gt, runtime_seconds=t_cfg, peak_memory_mb=get_process_memory_mb())
        active_stores[cfg_id] = c_store

        marginal_rec = m_cfg.candidate_recall - prev_c_rec if prev_c_rec > 0 else 0.0
        marginal_pairs = m_cfg.total_candidate_pairs - prev_c_pairs if prev_c_pairs > 0 else 0

        config_results.append({
            "config_id": cfg_id,
            "description": cfg_desc,
            "channels_included": ch_list,
            "candidate_recall": round(m_cfg.candidate_recall, 6),
            "marginal_recall": round(marginal_rec, 6),
            "entity_complete_coverage": round(m_cfg.entity_complete_coverage, 6),
            "entity_partial_coverage": round(m_cfg.entity_partial_coverage, 6),
            "total_candidate_pairs": m_cfg.total_candidate_pairs,
            "marginal_pairs": marginal_pairs,
            "avg_candidates_per_s1": round(m_cfg.avg_candidates_per_s1, 2),
            "median_candidates_per_s1": round(m_cfg.median_candidates_per_s1, 1),
            "p95_candidates_per_s1": round(m_cfg.p95_candidates_per_s1, 1),
            "p99_candidates_per_s1": round(m_cfg.p99_candidates_per_s1, 1),
            "max_candidates_per_s1": m_cfg.max_candidates_per_s1,
            "runtime_seconds": round(t_cfg, 2),
            "peak_memory_mb": round(m_cfg.peak_memory_mb, 1),
        })

        logger.info(
            f"  {cfg_id:20s}: Recall = {m_cfg.candidate_recall*100:6.2f}% (+{marginal_rec*100:5.2f}%), "
            f"Coverage = {m_cfg.entity_complete_coverage*100:5.2f}%, Pairs = {m_cfg.total_candidate_pairs:10,d}, "
            f"Avg = {m_cfg.avg_candidates_per_s1:5.1f}/S1"
        )
        prev_c_rec = m_cfg.candidate_recall
        prev_c_pairs = m_cfg.total_candidate_pairs

    best_cfg_id = "Config_5_FullUnion"
    best_store = active_stores[best_cfg_id]
    best_cands = best_store.get_candidate_dict()
    best_metrics = compute_blocking_metrics(best_cands, val_a_gt)

    # 8. Candidate Cap Sensitivity on Best Recovered Configuration
    logger.info("Step 8: Evaluating Candidate Cap Tradeoffs on Best Configuration...")
    cap_values = [None, 200, 150, 100, 75, 50, 30]
    cap_results: List[Dict[str, Any]] = []

    for cap_val in cap_values:
        capped_dict = best_store.get_candidate_dict(cap=cap_val)
        cap_m = compute_blocking_metrics(capped_dict, val_a_gt)
        cap_label = str(cap_val) if cap_val is not None else "No Cap"
        cap_results.append({
            "cap": cap_label,
            "candidate_recall": round(cap_m.candidate_recall, 6),
            "entity_complete_coverage": round(cap_m.entity_complete_coverage, 6),
            "total_candidate_pairs": cap_m.total_candidate_pairs,
            "avg_candidates_per_s1": round(cap_m.avg_candidates_per_s1, 2),
            "p95_candidates_per_s1": round(cap_m.p95_candidates_per_s1, 1),
            "p99_candidates_per_s1": round(cap_m.p99_candidates_per_s1, 1),
            "max_candidates_per_s1": cap_m.max_candidates_per_s1,
        })
        logger.info(
            f"  Cap = {cap_label:6s}: Recall = {cap_m.candidate_recall*100:6.2f}%, "
            f"Coverage = {cap_m.entity_complete_coverage*100:5.2f}%, Pairs = {cap_m.total_candidate_pairs:10,d}, Avg = {cap_m.avg_candidates_per_s1:5.2f}"
        )

    # 9. Subgroup Breakdown: India vs US, S2 vs S3
    logger.info("Step 9: Analyzing Subgroups (India vs US, S2 vs S3)...")
    s1_countries = val_a_s1_norm["country_norm"].tolist()
    s1_eids = val_a_s1_norm["entity_id"].tolist()

    subgroup_analysis: Dict[str, Dict[str, Any]] = {}
    for c_norm in ["us", "india", "france", "other", "unknown"]:
        matching_eids = [eid for eid, c in zip(s1_eids, s1_countries) if c == c_norm]
        if not matching_eids:
            continue
        sub_cands = {eid: best_cands.get(eid, []) for eid in matching_eids}
        sub_gt = {eid: val_a_gt.get(eid, set()) for eid in matching_eids}
        sub_m = compute_blocking_metrics(sub_cands, sub_gt)

        # Base comparison for country
        base_sub_cands = {eid: base_cands.get(eid, []) for eid in matching_eids}
        base_sub_m = compute_blocking_metrics(base_sub_cands, sub_gt)

        subgroup_analysis[c_norm] = {
            "entity_count": len(matching_eids),
            "base_recall": round(base_sub_m.candidate_recall, 6),
            "recovered_recall": round(sub_m.candidate_recall, 6),
            "recall_gain": round(sub_m.candidate_recall - base_sub_m.candidate_recall, 6),
            "complete_coverage": round(sub_m.entity_complete_coverage, 6),
            "avg_candidates_per_s1": round(sub_m.avg_candidates_per_s1, 2),
            "total_candidate_pairs": sub_m.total_candidate_pairs,
        }

    # S2 vs S3 Recovery
    s2_total = 0
    s2_base_rec = 0
    s2_new_rec = 0
    s3_total = 0
    s3_base_rec = 0
    s3_new_rec = 0

    for s1_id, true_set in val_a_gt.items():
        c_base = set(base_cands.get(s1_id, []))
        c_new = set(best_cands.get(s1_id, []))
        for mid in true_set:
            if mid.startswith("S2-"):
                s2_total += 1
                if mid in c_base:
                    s2_base_rec += 1
                if mid in c_new:
                    s2_new_rec += 1
            elif mid.startswith("S3-"):
                s3_total += 1
                if mid in c_base:
                    s3_base_rec += 1
                if mid in c_new:
                    s3_new_rec += 1

    source_recovery = {
        "Source_2": {
            "total_links": s2_total,
            "base_recovered": s2_base_rec,
            "base_recall": round(s2_base_rec / max(1, s2_total), 6),
            "new_recovered": s2_new_rec,
            "new_recall": round(s2_new_rec / max(1, s2_total), 6),
            "gain": round((s2_new_rec - s2_base_rec) / max(1, s2_total), 6),
        },
        "Source_3": {
            "total_links": s3_total,
            "base_recovered": s3_base_rec,
            "base_recall": round(s3_base_rec / max(1, s3_total), 6),
            "new_recovered": s3_new_rec,
            "new_recall": round(s3_new_rec / max(1, s3_total), 6),
            "gain": round((s3_new_rec - s3_base_rec) / max(1, s3_total), 6),
        },
    }

    # 10. Generate Reports
    logger.info("Step 10: Generating Phase 5.1 JSON and Markdown Reports...")
    report_json_path = logs_dir / "phase5_1_recall_recovery_report.json"
    report_md_path = logs_dir / "phase5_1_recall_recovery_report.md"

    json_report = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_verification": {
            "val_a_s1_count": len(val_a_s1),
            "total_gt_links": total_val_a_links,
            "baseline_recovered_links": base_metrics.recovered_gt_links,
            "baseline_missed_links": base_metrics.missed_gt_links,
            "baseline_link_recall": round(base_metrics.candidate_recall, 6),
            "baseline_complete_coverage": round(base_metrics.entity_complete_coverage, 6),
            "baseline_total_pairs": base_metrics.total_candidate_pairs,
            "baseline_avg_cands_per_s1": round(base_metrics.avg_candidates_per_s1, 2),
        },
        "all_channel_metrics": {k: v.to_dict() for k, v in channel_metrics.items()},
        "configuration_progression": config_results,
        "best_configuration_summary": best_metrics.to_dict(),
        "candidate_caps": cap_results,
        "country_subgroups": subgroup_analysis,
        "source_recovery": source_recovery,
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)

    # Markdown Report
    md_lines = [
        "# Phase 5.1 Blocking Recall Recovery & Pareto Frontier Report",
        f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*",
        "",
        "## 1. Baseline Verification (Phase 5 Immutable Reference)",
        "",
        "| Metric | Value | Reference Status |",
        "|:---|---:|:---|",
        f"| `val_a` S1 Entities | {len(val_a_s1):,} | Canonical Phase 4 Partition (20.0%) |",
        f"| Total Ground Truth Links | {total_val_a_links:,} | Exact Ground Truth Matches in `val_a` |",
        f"| Baseline Recovered Links | {base_metrics.recovered_gt_links:,} | True links in baseline pool |",
        f"| Baseline Missed Links | {base_metrics.missed_gt_links:,} | True links missed by baseline |",
        f"| **Baseline Candidate Link Recall** | **{base_metrics.candidate_recall*100:.2f}%** | Verified Baseline Floor |",
        f"| **Baseline Entity Complete Coverage** | **{base_metrics.entity_complete_coverage*100:.2f}%** | S1s with 100% matches recovered |",
        f"| **Baseline Candidate Pairs** | **{base_metrics.total_candidate_pairs:,}** | 65.43 cands/S1 mean |",
        "",
        "## 2. Individual Channel Performance (Original vs New Recovery Channels)",
        "",
        "| Channel ID | Mechanism | Recall | Complete Coverage | Total Pairs | Avg/S1 | P95 | P99 | Max | Runtime |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for ch_id, ch_desc, _ in channel_defs:
        m = channel_metrics[ch_id]
        md_lines.append(
            f"| `{ch_id}` | {ch_desc} | **{m.candidate_recall*100:.2f}%** | {m.entity_complete_coverage*100:.2f}% | "
            f"{m.total_candidate_pairs:,} | {m.avg_candidates_per_s1:.2f} | {m.p95_candidates_per_s1:.0f} | "
            f"{m.p99_candidates_per_s1:.0f} | {m.max_candidates_per_s1:,} | {m.runtime_seconds:.2f}s |"
        )
    md_lines.append("")

    md_lines.extend([
        "## 3. Progressive Configuration Ablation (Config 0 -> Config 5)",
        "",
        "| Configuration | Description | Total Recall | Marginal Gain | Complete Coverage | Total Pairs | Marginal Pairs | Avg/S1 |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|",
    ])

    for c in config_results:
        md_lines.append(
            f"| `{c['config_id']}` | {c['description']} | **{c['candidate_recall']*100:.2f}%** | "
            f"+{c['marginal_recall']*100:.2f}% | {c['entity_complete_coverage']*100:.2f}% | "
            f"{c['total_candidate_pairs']:,} | +{c['marginal_pairs']:,} | {c['avg_candidates_per_s1']:.2f} |"
        )
    md_lines.append("")

    md_lines.extend([
        "## 4. Recovered Best Union vs Baseline Comparison",
        "",
        "| Metric | Baseline (Config 0) | Recovered (Config 5) | Absolute Improvement | Relative Delta |",
        "|:---|---:|---:|---:|---:|",
        f"| **Candidate Link Recall** | {base_metrics.candidate_recall*100:.2f}% | **{best_metrics.candidate_recall*100:.2f}%** | **+{best_metrics.candidate_recall*100 - base_metrics.candidate_recall*100:.2f}%** | **+{(best_metrics.candidate_recall - base_metrics.candidate_recall)/base_metrics.candidate_recall*100:.2f}%** |",
        f"| **Entity Complete Coverage** | {base_metrics.entity_complete_coverage*100:.2f}% | **{best_metrics.entity_complete_coverage*100:.2f}%** | **+{best_metrics.entity_complete_coverage*100 - base_metrics.entity_complete_coverage*100:.2f}%** | **+{(best_metrics.entity_complete_coverage - base_metrics.entity_complete_coverage)/base_metrics.entity_complete_coverage*100:.2f}%** |",
        f"| **Entity Partial Coverage** | {base_metrics.entity_partial_coverage*100:.2f}% | **{best_metrics.entity_partial_coverage*100:.2f}%** | **+{best_metrics.entity_partial_coverage*100 - base_metrics.entity_partial_coverage*100:.2f}%** | **+{(best_metrics.entity_partial_coverage - base_metrics.entity_partial_coverage)/base_metrics.entity_partial_coverage*100:.2f}%** |",
        f"| **Recovered GT Links** | {base_metrics.recovered_gt_links:,} | **{best_metrics.recovered_gt_links:,}** | **+{best_metrics.recovered_gt_links - base_metrics.recovered_gt_links:,}** | — |",
        f"| **Missed GT Links** | {base_metrics.missed_gt_links:,} | **{best_metrics.missed_gt_links:,}** | **-{base_metrics.missed_gt_links - best_metrics.missed_gt_links:,}** | **-{(base_metrics.missed_gt_links - best_metrics.missed_gt_links)/base_metrics.missed_gt_links*100:.2f}%** |",
        f"| **Candidate Pairs** | {base_metrics.total_candidate_pairs:,} | **{best_metrics.total_candidate_pairs:,}** | +{best_metrics.total_candidate_pairs - base_metrics.total_candidate_pairs:,} | +{(best_metrics.total_candidate_pairs - base_metrics.total_candidate_pairs)/base_metrics.total_candidate_pairs*100:.1f}% |",
        f"| **Average Candidates / S1** | {base_metrics.avg_candidates_per_s1:.2f} | **{best_metrics.avg_candidates_per_s1:.2f}** | +{best_metrics.avg_candidates_per_s1 - base_metrics.avg_candidates_per_s1:.2f} | — |",
        f"| **Median Candidates / S1** | {base_metrics.median_candidates_per_s1:.1f} | **{best_metrics.median_candidates_per_s1:.1f}** | +{best_metrics.median_candidates_per_s1 - base_metrics.median_candidates_per_s1:.1f} | — |",
        "",
        "## 5. Candidate Cap Sensitivity on Recovered Configuration",
        "",
        "| Candidate Cap | Link Recall | Complete Coverage | Total Pairs | Avg Cands / S1 | P95 | P99 | Max |",
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
        "## 6. Subgroup Recovery Analysis",
        "",
        "### 6.1 Geographic Breakdown",
        "",
        "| Country | S1 Entities | Baseline Recall | Recovered Recall | Absolute Gain | Complete Coverage | Avg Cands / S1 |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ])

    for c_norm, c_stats in subgroup_analysis.items():
        md_lines.append(
            f"| `{c_norm.upper()}` | {c_stats['entity_count']:,} | {c_stats['base_recall']*100:.2f}% | "
            f"**{c_stats['recovered_recall']*100:.2f}%** | **+{c_stats['recall_gain']*100:.2f}%** | "
            f"{c_stats['complete_coverage']*100:.2f}% | {c_stats['avg_candidates_per_s1']:.2f} |"
        )
    md_lines.append("")

    md_lines.extend([
        "### 6.2 Candidate Source Breakdown",
        "",
        "| Candidate Source | Total Links | Baseline Recall | Recovered Recall | Absolute Gain | Recovered Links |",
        "|:---|---:|---:|---:|---:|---:|",
    ])

    for src_name, src_d in source_recovery.items():
        md_lines.append(
            f"| `{src_name}` | {src_d['total_links']:,} | {src_d['base_recall']*100:.2f}% | "
            f"**{src_d['new_recall']*100:.2f}%** | **+{src_d['gain']*100:.2f}%** | {src_d['new_recovered']:,} / {src_d['total_links']:,} |"
        )
    md_lines.append("")

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    total_time = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"Phase 5.1 pipeline completed successfully in {total_time:.2f}s.")
    logger.info(f"Report saved to: {report_md_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_phase5_1()
