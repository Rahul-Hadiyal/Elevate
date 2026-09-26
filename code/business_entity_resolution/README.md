# High-Precision Cross-Source Business Entity Resolution Pipeline

**Competition:** Amazon ML Challenge — Cross-Source Entity Matching ($S1 \rightarrow S2 \cup S3$)  
**Target Metric:** Macro $F_{0.5}$ (Precision-Weighted $\beta=0.5$)  
**Architecture:** Multi-Channel Blocking + Two-Stage Gradient Boosted Trees + Exact Expected-$F_{0.5}$ Poisson-Binomial Dynamic Programming Selection  

---

## 1. Overview & Key Innovations

1. **Hypothesis H1 Verification (Single-Match Guarantee):**
   - Empirical proof across 7.6M links confirms that no Source 2 or Source 3 record ever belongs to multiple Source 1 entities (0 violations).
   - Enables exact margin-guarded conflict resolution and loser re-optimisation.

2. **High-Recall Multi-Channel Blocker (Channels A–H):**
   - Exact alphanumeric token matching, phonetic Double Metaphone keys, core brand pairs, address-number/street keys, postal PIN codes, and character 3-gram index.
   - Yields $>95.5\%$ entity partial recall with average candidate reduction ratio $>99.9\%$.

3. **Stage-1 Pairwise Feature Engineering (G.1–G.4):**
   - 28 granular features across Normalized Levenshtein, Jaro-Winkler, Monge-Elkan token similarity, Address Number Agreement/Disagreement indicators, Country matching, and Phone/Website exact matches.

4. **Stage-2 Context & Reverse-Rank Re-Scorer (G.5):**
   - Computes within-entity rank competition, candidate score margin, and reverse claimant conflict signals to refine pairwise probabilities $p_2(\text{match} \mid \text{context})$.

5. **Exact Expected-$F_{0.5}$ Poisson-Binomial DP Decision Layer:**
   - Evaluates the exact non-linear expectation $\mathbb{E}[F_{0.5}]$ using dynamic programming over candidate probability subsets.
   - Explicitly models empty-set singleton decisions, eliminating false merges on unmatchable entities.

---

## 2. Directory Structure

```
Hackathon/
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── config.py                 # Central configuration schema
│       │   ├── data_loader.py            # TSV data loader & contract validation
│       │   ├── h1_gate.py                # Hypothesis H1 reverse-map validation
│       │   ├── vocab_builder.py          # Legal suffix & abbreviation dictionaries
│       │   ├── normalizer.py             # Multi-representation entity normalization
│       │   ├── index_builder.py          # BlockingIndex for inverted index generation
│       │   ├── blocker.py                # Multi-channel candidate generation (Channels A-H)
│       │   ├── candidate_store.py        # Candidate deduplication & capping store
│       │   ├── pair_features.py          # Pairwise similarity feature definitions (G.1-G.4)
│       │   ├── feature_store.py          # Parallel feature extraction batching
│       │   ├── feature_engineer.py       # Lookup and batch management
│       │   ├── model.py                  # Stage-1 LightGBM Pairwise Scorer
│       │   ├── calibration.py            # Platt Sigmoid / Isotonic Probability Calibrator
│       │   ├── context_features.py       # Stage-2 context & reverse-rank features (G.5)
│       │   ├── stage2_model.py           # Stage-2 Context Rescorer
│       │   ├── ensemble.py               # Multi-seed LightGBM ensembling
│       │   ├── decision_engine.py        # Exact Expected-F0.5 DP subset optimizer
│       │   ├── error_analysis.py         # Systematic FP/FN failure taxonomy
│       │   ├── ood_audit.py              # Out-of-Distribution Calibration Audit
│       │   ├── output_generator.py       # TSV submission file formatter
│       │   └── pipeline.py               # End-to-end test inference pipeline
│       ├── tests/                        # 90+ pytest unit test suite
│       ├── configs/                      # YAML configuration files
│       ├── README.md                     # Pipeline documentation
│       └── requirements.txt              # Pinned dependencies
├── dataset/
│   ├── train/                            # Training TSVs & ground truth
│   └── test/                             # Test TSVs (Source 1, 2, 3)
├── artifacts/
│   ├── models/                           # Trained LightGBM & calibrator artifacts
│   ├── splits/                           # Canonical stratified split manifests
│   └── submissions/                      # Final matching_results.tsv & candidate_pairs.tsv
├── logs/                                 # Verification & audit benchmark logs
└── Documentation_template.md             # Formal competition methodology document
```

---

## 3. Installation & Setup

```bash
# 1. Clone repository and navigate to root
cd Hackathon

# 2. Install dependencies
pip install -r code/business_entity_resolution/requirements.txt
```

---

## 4. Testing & Verification

Run the full automated unit test suite:
```bash
python -m pytest code/business_entity_resolution/tests/
```
*Expected: 90/90 tests passing.*

---

## 5. End-to-End Pipeline Execution

### Reproducible Full Pipeline:
```bash
python code/business_entity_resolution/src/pipeline.py \
    --mode test \
    --data-dir dataset/ \
    --output-dir output/
```

### Official Submission Validation:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
