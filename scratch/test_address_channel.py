import sys, os, time, json
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
import pandas as pd

sys.path.insert(0, r'c:\Users\rahul\Desktop\Elevate\code\business_entity_resolution')
from src.data_loader import load_entity_source, load_ground_truth, parse_ground_truth_to_dict
from src.normalizer import EntityNormalizer

COMMON_ADDR_STOPWORDS = {
    'floor', 'road', 'street', 'house', 'near', 'number', 'plot', 'building', 'block',
    'first', 'second', 'third', 'ground', 'main', 'opp', 'opposite', 'behind', 'beside',
    'delhi', 'india', 'state', 'city', 'lane', 'avenue', 'nagar', 'bazaar', 'market',
    'room', 'shop', 'office', 'unit', 'phase', 'sector', 'dist', 'district', 'area',
    'west', 'east', 'north', 'south', 'central', 'colony', 'complex', 'towers', 'plaza'
}

def get_distinctive_addr_tokens(addr_norm, addr_df_counter, max_df=2000):
    if not addr_norm:
        return []
    toks = [
        t for t in addr_norm.split()
        if len(t) >= 4 and t not in COMMON_ADDR_STOPWORDS and not t.isdigit()
    ]
    # Filter by DF if counter provided
    if addr_df_counter:
        toks = [t for t in toks if 2 <= addr_df_counter.get(t, 0) <= max_df]
    return sorted(set(toks))

def main():
    repo_root = Path(r'c:\Users\rahul\Desktop\Elevate')
    train_dir = repo_root / 'student_resource' / 'dataset' / 'train'
    score_opt_dir = repo_root / 'artifacts' / 'score_optimization'

    with open(score_opt_dir / 'baseline_manifest.json', 'r') as f:
        val_manifest = json.load(f)
    val_5k_eids = val_manifest["sample_entity_ids"]

    gt_df, _ = load_ground_truth(train_dir / 'train_ground_truth.tsv')
    gt_map = parse_ground_truth_to_dict(gt_df)
    val_5k_gt = {eid: gt_map.get(eid, set()) for eid in val_5k_eids}

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

    # Calculate DF over candidates
    addr_df = Counter()
    for df in (s2_cand_norm, s3_cand_norm):
        for addr in df['addr_norm'].fillna(''):
            for t in set(addr.split()):
                if len(t) >= 4 and t not in COMMON_ADDR_STOPWORDS:
                    addr_df[t] += 1

    print("Building Address Token Pair Index...")
    addr_pair_index = defaultdict(list)
    # Also test rare single token index (DF <= 50)
    addr_rare_single_index = defaultdict(list)

    for df in (s2_cand_norm, s3_cand_norm):
        for eid, addr_norm, c_norm in zip(df['entity_id'], df['addr_norm'].fillna(''), df['country_norm'].fillna('unknown')):
            toks = get_distinctive_addr_tokens(addr_norm, addr_df, max_df=1500)
            # Single rare token
            for t in toks:
                if addr_df[t] <= 50:
                    addr_rare_single_index[(t, c_norm)].append(eid)
            # Pairs of distinctive tokens
            if 2 <= len(toks) <= 8:
                for t1, t2 in combinations(toks[:6], 2):
                    addr_pair_index[(t1, t2, c_norm)].append(eid)

    print(f"Address Pair Index keys: {len(addr_pair_index):,}")
    print(f"Address Rare Single Index keys: {len(addr_rare_single_index):,}")

    # Now load missing links from cap=100 or uncapped
    with open(score_opt_dir / 'missing_links_forensics.json', 'r', encoding='utf-8') as f:
        missing_forensics = json.load(f)

    # Let's see how many completely unretrieved links can be retrieved by address index
    sample_misses = [m for m in missing_forensics['sample_misses'] if m.get('addr_token_set', 0) >= 70]
    print(f"\nTesting on {len(sample_misses)} high-address-sim misses from sample_misses:")
    recovered = 0
    for m in sample_misses:
        s1_id = m['s1_id']
        cand_id = m['cand_id']
        s1_row = s1_val_norm[s1_val_norm['entity_id'] == s1_id]
        if s1_row.empty:
            continue
        s1_addr = s1_row.iloc[0]['addr_norm']
        c_norm = s1_row.iloc[0]['country_norm']
        s1_toks = get_distinctive_addr_tokens(s1_addr, addr_df, max_df=1500)
        
        cands_found = set()
        for t in s1_toks:
            if addr_df[t] <= 50:
                cands_found.update(addr_rare_single_index.get((t, c_norm), [])[:25])
        if 2 <= len(s1_toks) <= 8:
            for t1, t2 in combinations(s1_toks[:6], 2):
                cands_found.update(addr_pair_index.get((t1, t2, c_norm), [])[:25])

        if cand_id in cands_found:
            recovered += 1

    print(f"Recovered {recovered} / {len(sample_misses)} ({recovered/len(sample_misses)*100:.1f}%) sample high-address misses!")

if __name__ == '__main__':
    main()
