# Implementation Plan: Business Entity Resolution Pipeline

**Architecture Specification:** Architecture Document v3  
**Target Metric:** Macro $F_{0.5}$ over $S1$ entities (Precision-weighted $\beta=0.5$)  

---

## 1. Project Directory Structure

```
Hackathon/
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── config.py                 # Central config schema and YAML loader
│       │   ├── data_loader.py            # TSV data ingestion & validation contracts
│       │   ├── h1_gate.py                # Hypothesis H1 reverse-map validation
│       │   ├── vocab_builder.py          # Data-driven legal suffix, abbreviation & landmark vocabulary builder
│       │   ├── normalizer.py             # Multi-representation name, address, country normalization
│       │   ├── index_builder.py          # Transductive TF-IDF vectorizers (train + test corpus)
│       │   ├── blocker.py                # Multi-channel candidate generation (Channels A-H) & marginal ablation
│       │   ├── clusterer.py              # S2/S3 near-duplicate clustering for correlation handling
│       │   ├── feature_engineer.py       # Stage-1 pairwise feature extraction (G.1–G.4)
│       │   ├── model.py                  # Stage-1 LightGBM pairwise classifier
│       │   ├── calibration.py            # Dedicated fold Isotonic regression & ECE reliability audit
│       │   ├── context_features.py       # Stage-2 context & reverse-rank feature generator (G.5)
│       │   ├── stage2_model.py           # Stage-2 re-scoring model
│       │   ├── metrics.py                # Official F0.5 per-entity & macro evaluation (single source of truth)
│       │   ├── decision_engine.py        # Exact Expected-F0.5 DP subset selection + conditional conflict resolution
│       │   ├── output_generator.py       # Formats matching_results.tsv and candidate_pairs.tsv
│       │   ├── validator.py              # Submission invariant checker & official validator wrapper
│       │   └── pipeline.py               # End-to-end reproducible CLI pipeline
│       ├── configs/
│       │   ├── config.yaml
│       │   ├── legal_suffix_map.yaml
│       │   ├── abbreviation_map.yaml
│       │   ├── country_map.yaml
│       │   └── landmark_markers.yaml
│       ├── tests/
│       │   ├── test_metrics.py
│       │   ├── test_decision_engine.py
│       │   ├── test_normalization.py
│       │   ├── test_blocking.py
│       │   ├── test_invariants.py
│       │   └── test_data_loader.py
│       ├── README.md
│       └── requirements.txt
├── dataset/
│   ├── train/
│   └── test/
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── utils/
│   └── validate_submission.py
├── Documentation_template.md
├── ENVIRONMENT_REPORT.md
├── DATA_PROFILE.md
└── implementation_plan.md
```

---

## 2. Phase Execution Sequence

| Phase | Description | Key Deliverables & Gates | Status |
|---|---|---|---|
| **Phase 0** | **Environment & Problem Audit** | Hardware report, Python environment audit, Problem Statement extraction | **COMPLETE** |
| **Phase 1** | **Data Ingestion & Contracts** | `data_loader.py`, `metrics.py`, 15 passing unit tests (`pytest`), strict TSV validation, H1 data structures | **COMPLETE** |
| **Phase 2** | **EDA, Corpus Profiling & H1 Gate** | Real data profiled (26.4M records across 7 files); H1 Confirmed (0 violations in 7.6M links); `logs/scale_report.json` | **COMPLETE** |
| **Phase 3** | **Normalization Engine** | `normalizer.py`, `vocab_builder.py`, 31 passing unit tests, 35k rows/sec throughput, anti-over-normalization verified | **COMPLETE** |
| **Phase 4** | **Validation Split & Official $F_{0.5}$ Evaluator** | Canonical 45/15/10/20/10 S1-family split; official evaluator; 46 unit tests; empirical baseline floors ($F_{0.5}=0.3622$) | **COMPLETE** |
| **Phase 5** | **Candidate Generation / Multi-Channel Blocking** | `blocker.py` (Channels A–H), `index_builder.py`, `candidate_store.py`, 52 unit tests, multi-channel recall (78.02% link recall, 95.58% entity partial coverage, 28.8M pairs on 441k S1s) | **COMPLETE** |
| **Phase 6** | **Pairwise Feature Engineering (Stage-1)** | `feature_engineer.py` implementing G.1 (Name), G.2 (Address), G.3 (Country), G.4 (Cross-field) | **READY** |
| **Phase 7** | **Baseline Models (Exp 0–4)** | Logistic Regression baseline with fixed $p \ge 0.5$ rule; verify end-to-end plumbing | Planned |
| **Phase 8** | **Primary Stage-1 Model (Exp 5)** | LightGBM classifier with early stopping on `earlystop` fold; no class weight distortion | Planned |
| **Phase 9** | **Calibration & Exact Expected-$F_{0.5}$ Decision Layer (Exp 6)** | `calibration.py` (Isotonic regression on `calibration` fold), `decision_engine.py` (Poisson-binomial DP) | Planned |
| **Phase 10** | **Stage-2 Re-scoring & Conflict Resolution (Exp 7)** | `context_features.py` (G.5 features), `stage2_model.py`, margin-guarded conflict resolution | Planned |
| **Phase 11** | **Out-of-Distribution Calibration Audit (Exp 8)** | Held-out India transfer test as proxy for France; check calibration resilience | Planned |
| **Phase 12** | **Error Analysis Loop** | Systematic FP, FN, and singleton error categorization and targeted refinement | Planned |
| **Phase 13** | **Ensembling & Honest Estimation (Exp 10)** | Multi-seed LightGBM averaging + final honest evaluation on `val-B` (single pass) | Planned |
| **Phase 14** | **Test Inference & Output Generation** | Generate `matching_results.tsv` and `candidate_pairs.tsv` satisfying all invariants | Planned |
| **Phase 15** | **Official Submission Validation** | `utils/validate_submission.py` execution and automated PASS verification | Planned |
| **Phase 16** | **Reproducibility & Packaging** | Pinned `requirements.txt`, clean `README.md`, automated ZIP generator | Planned |
| **Phase 17** | **Methodology Documentation & Final Audit** | Complete `Documentation_template.md` with measured metrics and design rationale | Planned |

---

## 3. Phase Verification Summary

- **Phase 1 Tests:** 15/15 Passed (`data_loader.py`, `metrics.py`).
- **Phase 2 Gate:** 22/22 Passed (`h1_gate.py`, `profiler.py`). Real dataset verified across 26.4M records; H1 100% Confirmed.
- **Phase 3 Tests:** 31/31 Passed (`normalizer.py`, `vocab_builder.py`). Normalization throughput: 35,227 rows/sec; adversarial anti-over-normalization verified.
- **Phase 4 Tests & Benchmarks:** 46/46 Passed (`split.py`, `evaluator.py`, `baselines.py`). 45/15/10/20/10 split generated across 2,206,821 S1 entities with zero leakage; exact normalized name baseline established at Macro $F_{0.5} = 0.362225$ on 441,362 entities.
- **Phase 5 Tests & Benchmarks:** 52/52 Passed (`blocker.py`, `index_builder.py`, `candidate_store.py`, `blocking_metrics.py`). Multi-channel candidate engine evaluated across full 10.3M candidate corpus ($N = 441,362$ `val_a` entities, $1,528,532$ true links): **78.02% link recall**, **95.58% entity partial coverage**, $28,877,145$ pairs ($65.43$ cands/S1 mean, $47.0$ median).



