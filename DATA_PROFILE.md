# Data Profile & Corpus Analysis (Real Competition Data)

**Date of Execution:** 2026-09-25  
**Data Sources:** Real Amazon ML Challenge Dataset  
**Status:** Ingested, Validated & Profiled across 24,229,173 records  

---

## 1. Scale & Manifest Overview

| File Path | Description | Rows | Unique IDs | Missing Fields | File Size |
|---|---|---|---|---|---|
| `dataset/train/train_source1.tsv` | Reference S1 training entities | **2,206,821** | 2,206,821 | 0 | 210.1 MB |
| `dataset/train/train_source2.tsv` | S2 training records | **5,034,616** | 5,034,616 | 0 | 489.3 MB |
| `dataset/train/train_source3.tsv` | S3 training records | **5,285,603** | 5,285,603 | 0 | 503.7 MB |
| `dataset/train/train_ground_truth.tsv` | Ground truth matching links | **2,206,821** | 2,206,821 | 0 | 127.0 MB |
| `dataset/test/test_source1.tsv` | Reference S1 test entities | **1,732,544** | 1,732,544 | 0 | 175.0 MB |
| `dataset/test/test_source2.tsv` | S2 test records | **4,887,273** | 4,887,273 | 0 | 509.5 MB |
| `dataset/test/test_source3.tsv` | S3 test records | **5,082,316** | 5,082,316 | 0 | 506.0 MB |
| **Total Across Corpus** | — | **26,435,994** | **26,435,994** | **0** | **2.52 GB** |

---

## 2. Hypothesis H1 Gate Evaluation

> **Hypothesis H1:** Each S2 / S3 record belongs to at most one S1 entity.

### Empirical Results on Real Ground Truth (`7,638,365` Links)
- **H1 Result:** **CONFIRMED (100.0000%)**
- **Total Distinct Matched S2/S3 IDs:** `7,638,365`
- **Single Claimant IDs:** `7,638,365` (**100.00%**)
- **Multi-Claimant Violations:** `0` (**0.00%**)
  - **Source 2 Violations:** `0` out of `3,693,619` matched S2 IDs (0.00%)
  - **Source 3 Violations:** `0` out of `3,944,746` matched S3 IDs (0.00%)
- **Architectural Consequence:**
  - `conflict_resolution_enabled` $\to$ **TRUE**
  - `invariant_8_enforced` $\to$ **TRUE** (enforcing mutual exclusivity in post-processing preserves precision without deleting true positives).

---

## 3. Match Cardinality & Singleton Distribution (Training Ground Truth)

| Metric | Count | Percentage |
|---|---|---|
| **Total S1 Reference Entities** | 2,206,821 | 100.00% |
| **Total Ground Truth Matched Links** | 7,638,365 | — |
| **Singletons (0 matches)** | 123,247 | **5.58%** |
| **1 Match Entities** | 119,157 | **5.40%** |
| **Multi-Match Entities (2+ matches)** | 1,964,417 | **89.02%** |
| **Matches S1 $\to$ Both S2 and S3** | 1,776,047 | **80.48%** |
| **Matches S1 $\to$ S2 Only** | 143,029 | **6.48%** |
| **Matches S1 $\to$ S3 Only** | 164,498 | **7.45%** |
| **Matches per S1 Entity (Quantiles)** | Mean: 3.46, Median: 3.0, P75: 5.0, P95: 6.0, Max: 11.0 |

---

## 4. Geographic Distribution & Distribution Shift (France)

### Country Distribution Breakdown

| Dataset Partition | India | United States (US) | France |
|---|---|---|---|
| **Train S1** | 883,188 (40.0%) | 1,323,633 (60.0%) | 0 (0.0%) |
| **Train S2** | 2,017,799 (40.1%) | 3,016,817 (59.9%) | 0 (0.0%) |
| **Train S3** | 2,115,547 (40.0%) | 3,170,056 (60.0%) | 0 (0.0%) |
| **Test S1** | 809,986 (**46.7%**) | 663,106 (**38.3%**) | 259,452 (**15.0%**) |
| **Test S2** | 2,312,565 (**47.3%**) | 1,871,330 (**38.3%**) | 703,378 (**14.4%**) |
| **Test S3** | 2,405,000 (**47.3%**) | 1,945,701 (**38.3%**) | 731,615 (**14.4%**) |

### Critical Finding on Distribution Shift
1. **France Presence in Test:** France constitutes **15.0% of Test S1** (259,452 entities) and ~1.43M total records across test S2/S3.
2. **Shift in Country Proportions:** While training is 60% US / 40% India, testing is **47% India / 38% US / 15% France**.
3. **Implication:** The normalization engine must derive legal suffixes and address tokens dynamically from corpus frequency, and the decision calibrator must be robust out-of-distribution.

---

## 5. Field Complexity & Script Profiling

### Length & Token Moments (Train S1)
- **Business Name Char Length:** Mean: 24.0, Median: 24.0, P95: 37.0, Max: 105.0.
- **Business Name Word Tokens:** Mean: 3.55, Median: 4.0, P95: 5.0, Max: 16.0.
- **Business Address Char Length:** Mean: 52.1, Median: 41.0, P95: 103.0, Max: 256.0.
- **Business Address Word Tokens:** Mean: 8.03, Median: 7.0, P95: 15.0, Max: 43.0.
- **Scripts:** 99.998% Latin script (including transliterated Indian entity names and accented French characters).
- **Exact Duplicate Names in S1:** 667,592 (shared common business names, e.g. "Subway", "State Bank of India", "Cafe Coffee Day").
- **Exact Duplicate Addresses in S1:** 76,215 (multi-tenant malls, commercial complexes).
