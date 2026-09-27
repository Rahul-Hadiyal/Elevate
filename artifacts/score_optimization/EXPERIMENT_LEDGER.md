# Entity Resolution Score Optimization Ledger

**Target Score:** `0.984019`  
**Evaluation Cohort:** 5,000 Held-Out S1 Entities (`val_b`, Seed 42, 17,138 Ground Truth Links)  
**Metric:** Macro F0.5 (beta = 0.5)  

---

| Exp ID | Date | Description | Feature Set | Candidate Recall | Precision | Recall | Macro F0.5 | FP Count | FN Count | Promoted? |
|---|---|---|---|---|---|---|---|---|---|---|
| **EXP-000** | 2026-09-27 | Frozen Production Baseline | 56 Features | `87.21%` | `0.9648` | `0.8629` | **`0.9288`** | `162` | `2,526` | **BASELINE** |
| **EXP-001** | 2026-09-27 | Name IDF Ablation (A1) | 63 Features | `87.21%` | `0.9654` | `0.8573` | **`0.9270`** | `113` | `2,632` | NO (Ceiling Bound) |
| **EXP-002** | 2026-09-27 | Address IDF Ablation (A2) | 58 Features | `87.21%` | `0.9647` | `0.8541` | **`0.9258`** | `135` | `2,688` | NO |
| **EXP-003** | 2026-09-27 | All IDF Ablation (A3) | 65 Features | `87.21%` | `0.9648` | `0.8570` | **`0.9264`** | `120` | `2,631` | NO |
| **EXP-004** | 2026-09-27 | IDF + Interaction (A4) | 66 Features | `87.21%` | `0.9653` | `0.8576` | **`0.9269`** | `116` | `2,632` | NO |
| **EXP-005** | 2026-09-27 | Cap=100 + Channel N + 63 Feats | 63 Features | `95.47%` | `0.9815` | `0.9291` | **`0.9641`** | `164` | `1,261` | **YES** |
| **EXP-006** | 2026-09-27 | Cap=150 + Channel N + 63 Feats | 63 Features | `97.45%` | `0.9852` | `0.9450` | **`0.9722`** | `157` | `973` | **YES (CURRENT BEST)** |

---

## Experiment Details

### EXP-000: Frozen Production Baseline
- **Model:** LightGBM (379 trees, num_leaves=35, max_depth=7, lr=0.04) + Platt Sigmoid Calibrator
- **Feature Set:** Baseline 56 pairwise RapidFuzz/token features
- **Decision Engine:** Poisson-Binomial DP (cap=50, min_prob=0.01) + Margin-Guarded Conflict Resolution (delta=0.05)
- **Macro F0.5:** `0.9288`
- **Candidate Retrieval Ceiling:** `0.9438` (Oracle)
- **Link Recall:** `87.21%` (14,946/17,138)

### EXP-001 to EXP-004: IDF Feature Ablations (Phases 10 & 11)
- **Insight:** Name IDF directly drops false positives from common business names by 26 pairs (139 -> 113) without precision loss. However, overall Macro F0.5 was strictly bounded by the candidate retrieval ceiling (87.21% link recall at cap=50).

| Exp ID | Features | Precision | Recall | Macro F0.5 | FP Count | FN Count | Gain vs A0 | Promoted? |
|---|---|---|---|---|---|---|---|---|
| **A0_Baseline_56** | 56 | `0.9645` | `0.8527` | **`0.9251`** | `139` | `2715` | `Baseline` | **NO** |
| **A1_Name_IDF** | 63 | `0.9654` | `0.8573` | **`0.9270`** | `113` | `2632` | `+0.0019` | **NO** |
| **A2_Addr_IDF** | 58 | `0.9647` | `0.8541` | **`0.9258`** | `135` | `2688` | `+0.0007` | **NO** |
| **A3_All_IDF** | 65 | `0.9648` | `0.8570` | **`0.9264`** | `120` | `2631` | `+0.0013` | **NO** |
| **A4_IDF_Interaction** | 66 | `0.9653` | `0.8576` | **`0.9269`** | `116` | `2632` | `+0.0018` | **NO** |

### EXP-005: Channel N (Address Token Blocking) + Candidate Cap 100
- **Diagnosis:** Missing-link forensics (Phase 14 & 15) proved that 949 true links were truncated by cap=50, and 522 true Indian links were cross-script name mismatches with identical address landmarks. Channel N indexes distinctive address token pairs + rare tokens.
- **Model:** Retrained LightGBM (63 features: 56 baseline + 7 Name IDF) + Platt Calibrator
- **Candidate Recall:** `95.47%` (16,362/17,138)
- **Macro Precision:** `0.9815`
- **Macro Recall:** `0.9291`
- **Macro F0.5:** `0.9641` (+0.0353 over baseline!)
- **Decision:** Promoted.

### EXP-006: Candidate Cap 150 + Channel N + 63 Features
- **Candidate Recall:** `97.45%` (16,701/17,138, Oracle Macro F0.5 = `0.9900`)
- **Macro Precision:** `0.9852`
- **Macro Recall:** `0.9450`
- **Macro F0.5:** **`0.9722`** (+0.0434 over baseline!)
- **False Positives:** 157
- **False Negatives:** 973 (down from 2,526)
- **Decision:** **PROMOTED AS NEW PRODUCTION SCORING BASELINE**.
