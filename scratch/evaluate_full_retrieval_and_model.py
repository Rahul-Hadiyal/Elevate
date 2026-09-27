import sys, os, time, json
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, r'c:\Users\rahul\Desktop\Elevate\code\business_entity_resolution')

from src.data_loader import load_entity_source, load_ground_truth, parse_ground_truth_to_dict
from src.normalizer import EntityNormalizer
from src.index_builder import BlockingIndex
from src.blocker import MultiChannelBlocker
from src.candidate_store import CandidateStore
from src.feature_engineer import build_entity_lookup
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
    """Channel N: Address Rare Token & Token-Pair Blocking."""
    cands_dict = {}
    for eid, addr_norm, c_norm in zip(s1_df['entity_id'], s1_df['addr_norm'].fillna(''), s1_df['country_norm'].fillna('unknown')):
        toks = get_distinctive_addr_tokens(addr_norm, addr_df, max_df=1500)
        found = set()
        # Rare single token lookup
        for t in toks:
            if addr_df.get(t, 0) <= 50:
                found.update(cand_addr_rare_idx.get((t, c_norm), [])[:25])
        # Token pairs
        if 2 <= len(toks) <= 8:
            for t1, t2 in combinations(toks[:6], 2):
                found.update(cand_addr_pair_idx.get((t1, t2, c_norm), [])[:25])
                if len(found) >= 50:
                    break
        cands_dict[eid] = list(found)
    return cands_dict

def main():
    print("=" * 80)
    print("PHASE 18: END-TO-END RETRIEVAL EXPANSION & MODEL EVALUATION")
    print("=" * 80)

    repo_root = Path(r'c:\Users\rahul\Desktop\Elevate')
    train_dir = repo_root / 'student_resource' / 'dataset' / 'train'
    score_opt_dir = repo_root / 'artifacts' / 'score_optimization'

    # Load 5k validation manifest
    with open(score_opt_dir / 'baseline_manifest.json', 'r') as f:
        val_manifest = json.load(f)
    val_5k_eids = val_manifest["sample_entity_ids"]

    gt_df, _ = load_ground_truth(train_dir / 'train_ground_truth.tsv')
    gt_map = parse_ground_truth_to_dict(gt_df)
    val_5k_gt = {eid: gt_map.get(eid, set()) for eid in val_5k_eids}
    total_true_links = sum(len(v) for v in val_5k_gt.values())

    s1_df_all, _ = load_entity_source(train_dir / 'train_source1.tsv', 'S1')
    s1_val_df = s1_df_all[s1_df_all['entity_id'].isin(val_5k_eids)].copy()

    val_true_targets = set().union(*val_5k_gt.values())
    s2_val_true = {c for c in val_true_targets if c.startswith('S2-')}
    s3_val_true = {c for c in val_true_targets if c.startswith('S3-')}

    s2_df, _ = load_entity_source(train_dir / 'train_source2.tsv', 'S2')
    s3_df, _ = load_entity_source(train_dir / 'train_source3.tsv', 'S3')

    s2_cand_df = pd.concat([
        s2_df[s2_df['entity_id'].isin(s2_val_true)],
        s2_df.head(150000)
    ]).drop_duplicates(subset=['entity_id'])

    s3_cand_df = pd.concat([
        s3_df[s3_df['entity_id'].isin(s3_val_true)],
        s3_df.head(150000)
    ]).drop_duplicates(subset=['entity_id'])

    normalizer = EntityNormalizer()
    s1_val_norm = normalizer.normalize_dataframe(s1_val_df)
    s2_cand_norm = normalizer.normalize_dataframe(s2_cand_df)
    s3_cand_norm = normalizer.normalize_dataframe(s3_cand_df)

    # Build BlockingIndex (Channels A-K)
    idx = BlockingIndex(min_token_len=3, max_token_df=5000)
    idx.build_indexes(s2_cand_norm, s3_cand_norm)
    blocker = MultiChannelBlocker(idx, max_cands_per_key=100)

    # Build Channel N indexes
    print("Building Address Channel N indexes...")
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

    print(f"Address Pair Index: {len(cand_addr_pair_idx):,} keys, Rare Single: {len(cand_addr_rare_idx):,} keys")

    print("\nGenerating candidates across Channels A-K + Channel N...")
    st = CandidateStore(s1_val_norm['entity_id'])
    st.add_channel_candidates("A", blocker.generate_channel_a(s1_val_norm))
    st.add_channel_candidates("B", blocker.generate_channel_b(s1_val_norm))
    st.add_channel_candidates("C", blocker.generate_channel_c(s1_val_norm))
    st.add_channel_candidates("D", blocker.generate_channel_d(s1_val_norm))
    st.add_channel_candidates("E", blocker.generate_channel_e(s1_val_norm))
    st.add_channel_candidates("G", blocker.generate_channel_g(s1_val_norm))
    st.add_channel_candidates("H", blocker.generate_channel_h(s1_val_norm))
    st.add_channel_candidates("I", blocker.generate_channel_i(s1_val_norm))
    st.add_channel_candidates("J", blocker.generate_channel_j(s1_val_norm))
    st.add_channel_candidates("K", blocker.generate_channel_k(s1_val_norm))
    
    # Add Channel N
    channel_n_cands = extract_channel_n_candidates(s1_val_norm, cand_addr_pair_idx, cand_addr_rare_idx, addr_df)
    st.add_channel_candidates("N", channel_n_cands)

    caps_to_test = [50, 75, 100, 150, 999999]
    cap_labels = ["cap=50+N", "cap=75+N", "cap=100+N", "cap=150+N", "uncapped+N"]

    retrieval_results = []
    print("\n" + "=" * 80)
    print(f"{'Configuration':<14} | {'Pairs Count':<12} | {'True Retrieved':<15} | {'Cand Recall':<12} | {'Oracle F0.5':<12} | {'Oracle Recall'}")
    print("-" * 88)

    for cap_val, label in zip(caps_to_test, cap_labels):
        cand_dict = st.get_candidate_dict(cap=cap_val)
        pairs_count = sum(len(cs) for cs in cand_dict.values())
        retrieved_links = sum(len(set(cand_dict.get(eid, [])) & val_5k_gt[eid]) for eid in val_5k_eids)
        cand_recall = retrieved_links / total_true_links

        oracle_preds = {eid: set(cand_dict.get(eid, [])) & val_5k_gt[eid] for eid in val_5k_eids}
        oracle_eval = compute_macro_f05(oracle_preds, val_5k_gt)

        res_row = {
            "config": label,
            "pairs_count": pairs_count,
            "retrieved_links": retrieved_links,
            "cand_recall": round(cand_recall, 4),
            "oracle_macro_f05": round(oracle_eval.macro_f05, 4),
            "oracle_precision": round(oracle_eval.macro_precision, 4),
            "oracle_recall": round(oracle_eval.macro_recall, 4),
        }
        retrieval_results.append(res_row)
        print(f"{label:<14} | {pairs_count:<12,d} | {retrieved_links:<6,d}/{total_true_links:<6,d} | {cand_recall*100:<10.2f}% | {oracle_eval.macro_f05:<12.4f} | {oracle_eval.macro_recall:.4f}")

    with open(score_opt_dir / 'retrieval_expansion_results.json', 'w') as f:
        json.dump(retrieval_results, f, indent=2)

    print("\nSaved retrieval expansion results to artifacts/score_optimization/retrieval_expansion_results.json")
    print("=" * 80)

if __name__ == '__main__':
    main()
