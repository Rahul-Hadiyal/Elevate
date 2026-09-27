import sys, os, time, json, math
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, r'c:\Users\rahul\Desktop\Elevate\code\business_entity_resolution')

from src.data_loader import load_entity_source, load_ground_truth, parse_ground_truth_to_dict
from src.split import SplitManifest
from src.normalizer import EntityNormalizer
from src.index_builder import BlockingIndex
from src.blocker import MultiChannelBlocker
from src.candidate_store import CandidateStore
from src.pair_features import FEATURE_NAMES
from src.feature_store import FeatureExtractor
from src.feature_engineer import build_entity_lookup
from src.model import PairwiseScorer
from src.calibration import ProbabilityCalibrator
from src.decision_engine import DecisionEngine
from src.metrics import compute_macro_f05

COMMON_ADDR_STOPWORDS = {
    'floor', 'road', 'street', 'house', 'near', 'number', 'plot', 'building', 'block',
    'first', 'second', 'third', 'ground', 'main', 'opp', 'opposite', 'behind', 'beside',
    'delhi', 'india', 'state', 'city', 'lane', 'avenue', 'nagar', 'bazaar', 'market',
    'room', 'shop', 'office', 'unit', 'phase', 'sector', 'dist', 'district', 'area',
    'west', 'east', 'north', 'south', 'central', 'colony', 'complex', 'towers', 'plaza'
}

def get_distinctive_addr_tokens(addr_norm, addr_df_counter, max_df=1500):
    if not addr_norm:
        return []
    toks = [
        t for t in addr_norm.split()
        if len(t) >= 4 and t not in COMMON_ADDR_STOPWORDS and not t.isdigit()
    ]
    if addr_df_counter:
        toks = [t for t in toks if 2 <= addr_df_counter.get(t, 0) <= max_df]
    return sorted(set(toks))

def extract_channel_n_candidates(s1_df, cand_addr_pair_idx, cand_addr_rare_idx, addr_df):
    cands_dict = {}
    for eid, addr_norm, c_norm in zip(s1_df['entity_id'], s1_df['addr_norm'].fillna(''), s1_df['country_norm'].fillna('unknown')):
        toks = get_distinctive_addr_tokens(addr_norm, addr_df, max_df=1500)
        found = set()
        for t in toks:
            if addr_df.get(t, 0) <= 50:
                found.update(cand_addr_rare_idx.get((t, c_norm), [])[:25])
        if 2 <= len(toks) <= 8:
            for t1, t2 in combinations(toks[:6], 2):
                found.update(cand_addr_pair_idx.get((t1, t2, c_norm), [])[:25])
                if len(found) >= 50:
                    break
        cands_dict[eid] = list(found)
    return cands_dict

def compute_name_idf_features(
    pairs: List[Tuple[str, str]],
    s1_lookup: Dict[str, Any],
    cand_lookup: Dict[str, Any],
    name_df_map: Counter,
    total_docs: int,
) -> Tuple[np.ndarray, List[str]]:
    feature_names = [
        "feat_name_idf_weighted_jaccard",
        "feat_name_rare_overlap_count",
        "feat_name_max_shared_idf",
        "feat_name_min_shared_idf",
        "feat_name_mean_shared_idf",
        "feat_name_s1_max_token_df",
        "feat_name_all_tokens_common",
    ]
    n_pairs = len(pairs)
    mat = np.zeros((n_pairs, len(feature_names)), dtype=np.float32)
    log_total = math.log(max(1000, total_docs))

    def get_token_idf(token: str) -> float:
        df = name_df_map.get(token, 1)
        return max(0.1, log_total - math.log(df))

    rare_threshold_df = 500
    common_threshold_df = 5000

    for i, (s1_id, cid) in enumerate(pairs):
        s1 = s1_lookup.get(s1_id, {})
        cand = cand_lookup.get(cid, {})

        s1_name_toks = set(s1.get("name_norm", "").split())
        cand_name_toks = set(cand.get("name_norm", "").split())

        shared_name_toks = s1_name_toks & cand_name_toks
        union_name_toks = s1_name_toks | cand_name_toks

        if union_name_toks:
            shared_sum = sum(get_token_idf(t) for t in shared_name_toks)
            union_sum = sum(get_token_idf(t) for t in union_name_toks)
            mat[i, 0] = shared_sum / union_sum if union_sum > 0 else 0.0

        mat[i, 1] = float(sum(1 for t in shared_name_toks if name_df_map.get(t, 1) <= rare_threshold_df))

        if shared_name_toks:
            idfs = [get_token_idf(t) for t in shared_name_toks]
            mat[i, 2] = float(max(idfs))
            mat[i, 3] = float(min(idfs))
            mat[i, 4] = float(sum(idfs) / len(idfs))

        if s1_name_toks:
            dfs = [name_df_map.get(t, 1) for t in s1_name_toks]
            mat[i, 5] = float(max(dfs))
            mat[i, 6] = 1.0 if all(df >= common_threshold_df for df in dfs) else 0.0

    return mat, feature_names

def main():
    print("=" * 80)
    print("PHASE 19: END-TO-END VALIDATION (CHANNELS A-K+N, CAP=100, 63 FEATURES)")
    print("=" * 80)

    repo_root = Path(r'c:\Users\rahul\Desktop\Elevate')
    train_dir = repo_root / 'student_resource' / 'dataset' / 'train'
    splits_dir = repo_root / 'artifacts' / 'splits'
    score_opt_dir = repo_root / 'artifacts' / 'score_optimization'

    # Load GT and S1
    gt_df, _ = load_ground_truth(train_dir / 'train_ground_truth.tsv')
    gt_map = parse_ground_truth_to_dict(gt_df)
    s1_train_df, _ = load_entity_source(train_dir / 'train_source1.tsv', 'S1')

    manifest_path = splits_dir / 'split_manifest.tsv'
    if not manifest_path.exists():
        manifest_path = splits_dir / 'split_manifest.tsv.gz'
    manifest = SplitManifest.load(manifest_path)

    with open(score_opt_dir / 'baseline_manifest.json', 'r') as f:
        val_manifest = json.load(f)
    val_5k_eids = val_manifest["sample_entity_ids"]
    val_5k_gt = {eid: gt_map.get(eid, set()) for eid in val_5k_eids}
    total_true_links = sum(len(v) for v in val_5k_gt.values())

    import random
    random.seed(42)
    train_all = list(manifest.get_entity_ids('train'))
    random.shuffle(train_all)
    train_eids = train_all[:12000]

    es_all = list(manifest.get_entity_ids('earlystop'))
    random.shuffle(es_all)
    es_eids = es_all[:2000]

    cal_all = list(manifest.get_entity_ids('calibration'))
    random.shuffle(cal_all)
    cal_eids = cal_all[:2000]
    all_needed = set(train_eids) | set(es_eids) | set(cal_eids) | set(val_5k_eids)

    all_true_targets = set()
    for e in all_needed:
        all_true_targets.update(gt_map.get(e, set()))

    s2_val_true = {c for c in all_true_targets if c.startswith('S2-')}
    s3_val_true = {c for c in all_true_targets if c.startswith('S3-')}

    s2_train_df, _ = load_entity_source(train_dir / 'train_source2.tsv', 'S2')
    s3_train_df, _ = load_entity_source(train_dir / 'train_source3.tsv', 'S3')

    s2_cand_df = pd.concat([
        s2_train_df[s2_train_df['entity_id'].isin(s2_val_true)],
        s2_train_df.head(150000)
    ]).drop_duplicates(subset=['entity_id'])

    s3_cand_df = pd.concat([
        s3_train_df[s3_train_df['entity_id'].isin(s3_val_true)],
        s3_train_df.head(150000)
    ]).drop_duplicates(subset=['entity_id'])
    del s2_train_df, s3_train_df

    normalizer = EntityNormalizer()
    s2_cand_norm = normalizer.normalize_dataframe(s2_cand_df)
    s3_cand_norm = normalizer.normalize_dataframe(s3_cand_df)
    cand_lookup = build_entity_lookup(s2_cand_norm)
    cand_lookup.update(build_entity_lookup(s3_cand_norm))

    idx = BlockingIndex(min_token_len=3, max_token_df=5000)
    idx.build_indexes(s2_cand_norm, s3_cand_norm)
    blocker = MultiChannelBlocker(idx, max_cands_per_key=100)

    # Address Channel N indexes
    addr_df = Counter()
    for df in (s2_cand_norm, s3_cand_norm):
        for addr in df['addr_norm'].fillna(''):
            for t in set(addr.split()):
                if len(t) >= 4 and t not in COMMON_ADDR_STOPWORDS:
                    addr_df[t] += 1

    cand_addr_pair_idx = defaultdict(list)
    cand_addr_rare_idx = defaultdict(list)
    for df in (s2_cand_norm, s3_cand_norm):
        for eid, addr_norm, c_norm in zip(df['entity_id'], df['addr_norm'].fillna(''), df['country_norm'].fillna('unknown')):
            toks = get_distinctive_addr_tokens(addr_norm, addr_df, max_df=1500)
            for t in toks:
                if addr_df[t] <= 50:
                    cand_addr_rare_idx[(t, c_norm)].append(eid)
            if 2 <= len(toks) <= 8:
                for t1, t2 in combinations(toks[:6], 2):
                    cand_addr_pair_idx[(t1, t2, c_norm)].append(eid)

    s1_needed_df = s1_train_df[s1_train_df['entity_id'].isin(all_needed)].copy()
    s1_needed_norm = normalizer.normalize_dataframe(s1_needed_df)
    s1_lookup = build_entity_lookup(s1_needed_norm)

    total_pool_docs = len(s2_cand_norm) + len(s3_cand_norm)

    def extract_cohort(eids, cap_val=100):
        sub_s1 = s1_needed_norm[s1_needed_norm['entity_id'].isin(eids)]
        st = CandidateStore(sub_s1['entity_id'])
        st.add_channel_candidates("A", blocker.generate_channel_a(sub_s1))
        st.add_channel_candidates("B", blocker.generate_channel_b(sub_s1))
        st.add_channel_candidates("C", blocker.generate_channel_c(sub_s1))
        st.add_channel_candidates("D", blocker.generate_channel_d(sub_s1))
        st.add_channel_candidates("E", blocker.generate_channel_e(sub_s1))
        st.add_channel_candidates("G", blocker.generate_channel_g(sub_s1))
        st.add_channel_candidates("H", blocker.generate_channel_h(sub_s1))
        st.add_channel_candidates("I", blocker.generate_channel_i(sub_s1))
        st.add_channel_candidates("J", blocker.generate_channel_j(sub_s1))
        st.add_channel_candidates("K", blocker.generate_channel_k(sub_s1))
        channel_n_cands = extract_channel_n_candidates(sub_s1, cand_addr_pair_idx, cand_addr_rare_idx, addr_df)
        st.add_channel_candidates("N", channel_n_cands)

        c_dict = st.get_candidate_dict(cap=cap_val)
        ch_counts = st.get_channel_counts()
        p_list = [(s1, c) for s1, cs in c_dict.items() for c in cs]
        ext = FeatureExtractor(s1_lookup, cand_lookup, ground_truth=gt_map)
        b = ext.extract_pair_batch(p_list, channel_counts=ch_counts)
        idf_mat, idf_feat_names = compute_name_idf_features(
            p_list, s1_lookup, cand_lookup, idx._name_token_df, total_pool_docs
        )
        return b, idf_mat, idf_feat_names, c_dict

    print("\nExtracting feature batches with Channels A-K+N @ cap=100...")
    b_train, idf_tr, idf_names, _ = extract_cohort(train_eids, cap_val=100)
    b_es, idf_es, _, _ = extract_cohort(es_eids, cap_val=100)
    b_cal, idf_cal, _, _ = extract_cohort(cal_eids, cap_val=100)
    b_val, idf_val, _, val_cdict = extract_cohort(val_5k_eids, cap_val=100)

    X_tr = np.hstack([b_train.features, idf_tr])
    X_e = np.hstack([b_es.features, idf_es])
    X_c = np.hstack([b_cal.features, idf_cal])
    X_v = np.hstack([b_val.features, idf_val])
    all_fnames = FEATURE_NAMES + idf_names

    print(f"\nTraining LightGBM Scorer with {len(all_fnames)} features...")
    model = PairwiseScorer(
        model_type="lightgbm",
        feature_names=all_fnames,
        n_estimators=500,
        learning_rate=0.04,
        num_leaves=35,
        max_depth=7,
        random_state=42,
    )
    model.fit(X_tr, b_train.labels, X_val=X_e, y_val=b_es.labels, early_stopping_rounds=30)

    print("Fitting probability calibrator...")
    cal_raw = model.predict_proba(X_c)
    calib = ProbabilityCalibrator(method="sigmoid").fit(cal_raw, b_cal.labels)

    print("Predicting and calibrating validation set...")
    val_raw = model.predict_proba(X_v)
    val_cal = calib.predict_proba(val_raw)

    print("Running DecisionEngine conflict resolution & match selection...")
    engine = DecisionEngine(
        enable_conflict_resolution=True,
        margin_delta=0.05,
        min_prob_filter=0.01,
        max_candidates_per_entity=50,
    )
    cand_score_map: Dict[str, List[Tuple[str, float]]] = {}
    for (s1_id, cid), prob in zip(b_val.pair_ids, val_cal):
        cand_score_map.setdefault(s1_id, []).append((cid, float(prob)))
    for s1_id in val_5k_eids:
        cand_score_map.setdefault(s1_id, [])

    preds = engine.optimize_predictions(cand_score_map)
    eval_res = compute_macro_f05(preds, val_5k_gt)

    tp = sum(len(preds.get(e, set()) & val_5k_gt[e]) for e in val_5k_eids)
    fp = sum(len(preds.get(e, set()) - val_5k_gt[e]) for e in val_5k_eids)
    fn = sum(len(val_5k_gt[e] - preds.get(e, set())) for e in val_5k_eids)

    print("\n" + "=" * 80)
    print("PHASE 19 RESULTS:")
    print(f"Macro F0.5:      {eval_res.macro_f05:.4f}")
    print(f"Macro Precision: {eval_res.macro_precision:.4f}")
    print(f"Macro Recall:    {eval_res.macro_recall:.4f}")
    print(f"True Positives:  {tp:,} / {total_true_links:,}")
    print(f"False Positives: {fp:,}")
    print(f"False Negatives: {fn:,}")
    print(f"False Merges:    {eval_res.false_merge_count:,}")
    print("=" * 80)

    # Save validation results
    result_data = {
        "pipeline": "Channels A-K + Channel N (cap=100) + 63 Features (56 + 7 Name IDF)",
        "macro_f05": round(eval_res.macro_f05, 4),
        "macro_precision": round(eval_res.macro_precision, 4),
        "macro_recall": round(eval_res.macro_recall, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "false_merges": eval_res.false_merge_count,
        "total_true_links": total_true_links,
    }
    with open(score_opt_dir / 'phase19_validation_results.json', 'w') as f:
        json.dump(result_data, f, indent=2)

    # Save trained model and calibrator as production v2
    models_dir = repo_root / 'artifacts' / 'models'
    joblib.dump(model, models_dir / 'production_scorer_v2.joblib')
    joblib.dump(calib, models_dir / 'production_calibrator_v2.joblib')
    print("Saved upgraded models to artifacts/models/production_scorer_v2.joblib")

if __name__ == '__main__':
    main()
