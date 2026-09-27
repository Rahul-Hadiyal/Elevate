# Production Pipeline Execution Graph

**System:** Amazon ML Challenge Business Entity Resolution  
**Target:** High-Performance Inference Optimization (Phase 1 Discovery)  
**Production Script:** `code/business_entity_resolution/src/run_production.py`  
**Timestamp:** 2026-09-27  

---

## 1. End-to-End Production Call Graph

```text
run_production.py:run_production_pipeline()
 │
 ├── [Stage 1: Training Data Loading]
 │     ├── src.data_loader.load_entity_source(train_source1.tsv, 'S1') -> DataFrame (2.2M rows)
 │     ├── src.data_loader.load_entity_source(train_source2.tsv, 'S2') -> DataFrame (5.0M rows)
 │     ├── src.data_loader.load_entity_source(train_source3.tsv, 'S3') -> DataFrame (5.3M rows)
 │     ├── src.data_loader.load_ground_truth(train_ground_truth.tsv) -> DataFrame (2.2M rows)
 │     └── src.data_loader.parse_ground_truth_to_dict() -> Dict[s1_id, Set[cand_id]]
 │
 ├── [Stage 2: Partition Sampling]
 │     ├── src.split.SplitManifest.load(split_manifest.tsv) -> SplitManifest
 │     ├── Stratified sampling: train (16k/30k), earlystop (2k/10k), calibration (2k/10k), val_b (5k)
 │     └── Random shuffle with seed 42 to ensure representative non-singleton strata
 │
 ├── [Stage 3: Training Entity Preprocessing & Normalization]
 │     ├── src.normalizer.EntityNormalizer.normalize_dataframe()
 │     │     ├── normalize_name(): accent strip, lower, punctuation, legal suffix canonicalization
 │     │     └── normalize_address(): street/unit abbreviations, landmark separation, digit extraction
 │     └── src.feature_engineer.build_entity_lookup() -> Dict[entity_id, Dict[attr, value]]
 │
 ├── [Stage 4: Candidate Pool S2/S3 Ingestion & Indexing]
 │     ├── Filter required ground-truth targets + sample distractors
 │     ├── src.normalizer.EntityNormalizer.normalize_dataframe() on S2 and S3 subsets
 │     ├── src.feature_engineer.build_entity_lookup() for candidate records
 │     ├── src.index_builder.BlockingIndex.build_indexes(s2_norm, s3_norm)
 │     │     ├── Inverted index: exact_name_country
 │     │     ├── Inverted index: sorted_name_country
 │     │     ├── Inverted index: name_prefix2_country
 │     │     ├── Inverted index: addr_number_name3_country
 │     │     ├── Inverted index: addr_token_name3_country
 │     │     ├── Inverted index: rare_name_token_country
 │     │     ├── Inverted index: name_4gram_country
 │     │     ├── Inverted index: landmark_name3_country
 │     │     ├── Inverted index: phonetic_name_country
 │     │     ├── Inverted index: core_token_pairs
 │     │     └── Inverted index: addr_num_street_token
 │     └── src.blocker.MultiChannelBlocker(index, max_cands_per_key=100)
 │
 ├── [Stage 5: Training Feature Extraction]
 │     └── extract_features_for_split(s1_subset, blocker, extractor, cap=100)
 │           ├── MultiChannelBlocker: generate_channel_a through generate_channel_k (10 channels)
 │           ├── CandidateStore: add_channel_candidates(), get_candidate_dict(cap=100), get_channel_counts()
 │           └── FeatureExtractor.extract_pair_batch(pair_list, channel_counts)
 │                 └── Pair-wise loop calling compute_single_pair_features() (56 features)
 │
 ├── [Stage 6: Model Training & Probability Calibration]
 │     ├── src.model.PairwiseScorer.fit(X_train, y_train, X_val, y_val, early_stopping_rounds=30)
 │     │     └── LightGBM GBDT (n_estimators=500, lr=0.04, num_leaves=35, max_depth=7)
 │     └── src.calibration.ProbabilityCalibrator.fit(raw_scores, labels)
 │           └── Platt Sigmoid Logistic Regression
 │
 ├── [Stage 7: Context Feature Extraction & Stage-2 Rescorer]
 │     ├── src.context_features.ContextFeatureExtractor.extract_context_features()
 │     │     └── G5 within-entity features: ranks, margins, z-scores, claimant count
 │     └── src.stage2_model.Stage2Rescorer.fit() (LightGBM GBDT on G5 features)
 │
 ├── [Stage 8: Offline Validation on Held-Out val_b]
 │     ├── Pairwise inference -> Context features -> Stage-2 rescore
 │     ├── src.decision_engine.DecisionEngine.optimize_predictions()
 │     │     ├── Step 1: select_matches_for_entity() (Poisson-Binomial DP, cap=100)
 │     │     ├── Step 2: Build reverse index of claimants
 │     │     ├── Step 3: Margin-guarded conflict resolution (delta=0.05)
 │     │     └── Step 4: Loser re-optimization (per-entity exclusion: other_assigned | ambiguous_dropped)
 │     └── src.metrics.compute_macro_f05() -> Val_B Macro F0.5 evaluation
 │
 └── [Stage 9: Test Inference (Dominant 2.5-3.5h Bottleneck)]
       ├── Ingest test_source1.tsv (1,732,544 rows)
       ├── Ingest & normalize test_source2.tsv (4,887,273 rows) -> test_s2_norm
       ├── Ingest & normalize test_source3.tsv (5,082,316 rows) -> test_s3_norm
       ├── Build test_index (BlockingIndex across test S2 and S3)
       ├── Build test_cand_lookup (9.97M records)
       │
       └── Chunked Loop: 18 chunks of 100,000 S1 entities
             ├── S1 chunk normalization: EntityNormalizer.normalize_dataframe()
             ├── S1 chunk lookup: build_entity_lookup()
             ├── MultiChannelBlocker on chunk (10 channels) -> store
             ├── Candidate extraction: store.get_candidate_dict(cap=100)
             ├── Channel counts: store.get_channel_counts() (assert not None)
             ├── Feature extraction: FeatureExtractor.extract_pair_batch() (56 features per pair)
             │     └── Calls compute_single_pair_features() on ~2.3M pairs per chunk
             ├── Stage-1 model prediction: stage1.predict_proba() + calibrator.predict_proba()
             ├── Stage-2 feature extraction: ContextFeatureExtractor.extract_context_features()
             ├── Stage-2 model prediction: stage2.predict_proba()
             ├── DecisionEngine.optimize_predictions() (DP prefix selection + conflict resolution)
             └── Collect predictions dictionary & candidate map
       │
       └── [Stage 10: Final Output TSV Writing]
             ├── Write output/matching_results.tsv (1,732,544 rows)
             └── Write output/candidate_pairs.tsv (1,732,544 rows)
```

---

## 2. Module & Function Summary on the Active Production Path

| Stage | Module | Function / Class | Input | Output | Configuration / Caps |
|---|---|---|---|---|---|
| **Data Ingestion** | `src.data_loader` | `load_entity_source` | Raw TSV paths | `pd.DataFrame` | Schema & ID prefix check |
| **Splitting** | `src.split` | `SplitManifest.load` | `split_manifest.tsv` | `SplitManifest` | Stratified partition lookups |
| **Normalization** | `src.normalizer` | `EntityNormalizer.normalize_dataframe` | Raw entity DataFrame | Normalized DataFrame | Legal suffix & address mappings |
| **Entity Lookup** | `src.feature_engineer` | `build_entity_lookup` | Normalized DataFrame | `Dict[str, Dict]` | Attribute dictionary per entity ID |
| **Blocking Index** | `src.index_builder` | `BlockingIndex.build_indexes` | S2/S3 normalized DFs | Inverted token indexes | `min_token_len=3, max_token_df=5000` |
| **Blocking Channels** | `src.blocker` | `MultiChannelBlocker` | S1 chunk normalized DF | Candidate tuples | 10 channels (A, B, C, D, E, G, H, I, J, K), `max_cands_per_key=100` |
| **Candidate Store** | `src.candidate_store` | `CandidateStore` | Channel outputs | Candidate dict & channel counts | Bitwise union overlap ranking, `cap=100` |
| **Feature Extraction** | `src.feature_store` | `FeatureExtractor.extract_pair_batch` | Pair tuples, lookups, channel counts | `FeatureBatch` (features, labels) | 56 RapidFuzz string & token metrics |
| **Feature Computation** | `src.pair_features` | `compute_single_pair_features` | Single S1 and Cand dicts | `List[float]` of length 56 | RapidFuzz ratios, Jaro-Winkler, token overlaps |
| **Model Scoring** | `src.model` | `PairwiseScorer.predict_proba` | Feature matrix (NumPy array) | Probabilities array | LightGBM 500 trees, 35 leaves |
| **Calibration** | `src.calibration` | `ProbabilityCalibrator.predict_proba` | Raw model log-odds | Calibrated probabilities | Platt Sigmoid scaling |
| **Context Features** | `src.context_features` | `ContextFeatureExtractor.extract_context_features` | `(s1, cand, p1)` tuples | NumPy array | 16 G5 within-entity features |
| **Stage-2 Rescorer** | `src.stage2_model` | `Stage2Rescorer.predict_proba` | G5 feature matrix | Calibrated stage-2 probabilities | LightGBM 150 trees, 8 leaves |
| **Decision Layer** | `src.decision_engine` | `DecisionEngine.optimize_predictions` | Candidate score map | `Dict[s1_id, Set[cand_id]]` | Poisson-Binomial DP, `margin_delta=0.05, cap=100` |
| **Output Writing** | `run_production.py` | Built-in file streaming | Predictions dict, candidate map | Two TSV files | Streaming write, newline `\n` |

---

## 3. Worker, Thread, and Memory Configuration

- **Current Parallelism:** Single-process Python runtime; LightGBM utilizes default OpenMP threads (`n_jobs=-1`).
- **Feature Extraction:** Executed sequentially per pair in a Python `for` loop inside `FeatureExtractor.extract_pair_batch()`.
- **Chunk Size:** $100,000$ Source-1 entities per chunk ($18$ chunks total).
- **Peak RAM:** $\approx 9.5 \text{ GB}$ with explicit `gc.collect()` after each chunk.
