# 100K-S1 Baseline Benchmark & Profiling Report

**Pipeline Execution:** Unoptimized Validated Production Inference Baseline  
**Cohort:** First 100,000 S1 Entities (`test_source1.tsv`)  
**Candidate Pairs:** Exactly 2,395,399 candidate pairs from `output/candidate_pairs.tsv`  
**Date:** 2026-09-27 12:29:58  
**Git Commit:** `ddf5f32935cacdeb02821a6a277e89cc530ba086`  
**Submission SHA-256:** `74918daf652ebfe403bd574e5b6c0dc1e3d51ba24edcebf87095e424431adf86`  

---

## 1. Executive Performance Summary

- **Total Wall-Clock Time:** `680.45s` (`11.34 minutes`)
- **Extrapolated Full Test Time (1.732M S1, 41.5M pairs):** `3.27 hours`
- **Peak RSS Working Set:** `3353.2 MB`
- **Candidate Pairs Scored:** `2,395,399`
- **Feature Extraction Throughput:** `7922.5 pairs/sec`
- **DecisionEngine Throughput:** `446.6 entities/sec`
- **End-to-End Throughput:** `147.0 S1 entities/sec`

---

## 2. Complete Timing Breakdown Across Pipeline Stages

| Stage | Description | Wall Time (s) | % of Total Time | Throughput / Rate |
|---|---|---|---|---|
| `t_feature_extraction` | 56 RapidFuzz String & Token Pair Features | `302.35s` | `44.4%` | `7922.5 pairs/s` |
| `t_norm_s2` | Normalize Candidate S2 Records | `34.45s` | `5.1%` | `27603.1 records/s` |
| `t_norm_s3` | Normalize Candidate S3 Records | `27.90s` | `4.1%` | `21681.9 records/s` |
| `t_load_s2` | Ingest Candidate S2 Records from Disk | `17.72s` | `2.6%` | - |
| `t_load_s3` | Ingest Candidate S3 Records from Disk | `15.97s` | `2.3%` | - |
| `t_normalize_s1` | Normalize 100,000 S1 Entities | `3.55s` | `0.5%` | `28188.6 entities/s` |
| `t_decision_engine` | Poisson-Binomial DP & Conflict Resolution | `223.90s` | `32.9%` | `446.6 entities/s` |
| `t_model_raw_predict` | LightGBM Raw Score Prediction | `5.76s` | `0.8%` | `415988.6 pairs/s` |
| `t_build_cand_lookup` | Construct Dictionary of Normalized Candidates | `12.70s` | `1.9%` | - |
| `t_build_cand_prob_map` | Group Scored Pairs by S1 Entity | `8.33s` | `1.2%` | - |
| `t_write_tsv` | Format and Write Submission TSV | `0.10s` | `0.0%` | - |
| `t_model_cal_predict` | Platt Sigmoid Probability Calibration | `0.34s` | `0.0%` | - |
| `t_load_candidate_pairs` | Ingest Candidate Pairs from TSV | `1.05s` | `0.2%` | - |
| `t_build_s1_lookup` | Construct S1 Entity Lookup Dictionary | `0.20s` | `0.0%` | - |
| `t_flatten_pairs` | Flatten S1-Cand Dict to Pair Tuples | `5.03s` | `0.7%` | - |
| **Total** | **End-to-End Pipeline Execution** | **`680.45s`** | **100.0%** | **`147.0 entities/s`** |

---

## 3. Top Three Bottlenecks Identified

### Bottleneck 1: Pairwise Feature Extraction (`t_feature_extraction`: 302.35s, 44.4%)
- **Mechanism:** In `src/pair_features.py:compute_single_pair_features()`, every candidate pair computes 56 pairwise string distance metrics (RapidFuzz ratio, partial_ratio, token_sort, token_set, WRatio, char n-gram Jaccard, address number comparisons).
- **Redundancy:** Each S1 entity is compared against an average of 24 candidate records. Tokenization, n-gram set construction, and digit extraction are recomputed repeatedly in every pair evaluation.
- **Optimization Strategy (Phases 3 & 4):** Precompute entity token sets, n-gram sets, and address digit lists once per entity during ingestion/normalization, avoiding repeated string operations inside the pairwise loop.

### Bottleneck 2: DecisionEngine Loser Re-Optimization Loop (`t_decision_engine`: 223.90s, 32.9%)
- **Mechanism:** In `src/decision_engine.py` (lines 232-235), the Step 4 loser re-optimization iterates over all 100,000 entities in `selected_per_s1` for *every* affected entity to build `other_assigned`:
  ```python
  other_assigned = {c for other_s1, cands in selected_per_s1.items() if other_s1 != s1_id for c in cands}
  ```
- **Redundancy:** This creates an $O(N_{\text{affected}} \times N_{\text{total}})$ quadratic loop across hundreds of millions of Python iterations.
- **Optimization Strategy (Phase 6):** Precompute `all_assigned = {c for cands in selected_per_s1.values() for c in cands}` once outside the loop, reducing `other_assigned` to an $O(1)$ set difference `all_assigned - set(selected_per_s1[s1_id].keys())`. This will reduce DecisionEngine runtime from 224s to <2s.

### Bottleneck 3: Candidate Normalization & Ingestion (`t_total_candidates_prep`: 108.74s, 16.0%)
- **Mechanism:** Ingestion and regex passes in `EntityNormalizer.normalize_name()` and `normalize_address()` across 1.55M unique candidate records (`t_norm_s2`: 34.5s, `t_norm_s3`: 27.9s, disk reads: 33.7s).
- **Redundancy:** Normalization is repeated on every run instead of caching immutable normalized representations.
- **Optimization Strategy (Phase 3):** Precompute and store persistent binary arrow/feather/parquet normalized caches.

---

## 4. Submission Output Invariant Audit

- **Total S1 Output Rows:** `100,000` (exactly 100,000)
- **Zero Matches (Singletons):** `6,048` (`6.05%`)
- **Single Matches:** `14,570` (`14.57%`)
- **2 to 5 Matches:** `69,167` (`69.17%`)
- **6+ Matches:** `10,215` (`10.21%`)
- **Maximum Matches for an Entity:** `16`
- **Mean Matches per S1:** `3.111`
- **Baseline Submission SHA-256:** `74918daf652ebfe403bd574e5b6c0dc1e3d51ba24edcebf87095e424431adf86`
