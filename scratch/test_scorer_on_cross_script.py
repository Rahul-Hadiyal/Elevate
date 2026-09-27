import sys, os, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, r'c:\Users\rahul\Desktop\Elevate\code\business_entity_resolution')
from src.pair_features import FEATURE_NAMES, compute_single_pair_features
from src.feature_engineer import build_entity_lookup
from src.data_loader import load_entity_source, load_ground_truth, parse_ground_truth_to_dict
from src.normalizer import EntityNormalizer

def main():
    repo_root = Path(r'c:\Users\rahul\Desktop\Elevate')
    score_opt_dir = repo_root / 'artifacts' / 'score_optimization'
    model_path = repo_root / 'artifacts' / 'models' / 'production_scorer.joblib'
    model = joblib.load(model_path)

    with open(score_opt_dir / 'missing_links_forensics.json', 'r', encoding='utf-8') as f:
        missing_forensics = json.load(f)

    # Let's inspect the high-address misses
    high_addr_misses = [m for m in missing_forensics['sample_misses'] if m.get('addr_token_set', 0) >= 70]
    print(f"Loaded {len(high_addr_misses)} high-address misses.")

    # We need the entity records to compute features
    train_dir = repo_root / 'student_resource' / 'dataset' / 'train'
    s1_df_all, _ = load_entity_source(train_dir / 'train_source1.tsv', 'S1')
    s2_df, _ = load_entity_source(train_dir / 'train_source2.tsv', 'S2')
    s3_df, _ = load_entity_source(train_dir / 'train_source3.tsv', 'S3')

    normalizer = EntityNormalizer()
    all_s1_eids = {m['s1_id'] for m in high_addr_misses}
    all_cand_eids = {m['cand_id'] for m in high_addr_misses}

    s1_sub = normalizer.normalize_dataframe(s1_df_all[s1_df_all['entity_id'].isin(all_s1_eids)].copy())
    s2_sub = normalizer.normalize_dataframe(s2_df[s2_df['entity_id'].isin(all_cand_eids)].copy())
    s3_sub = normalizer.normalize_dataframe(s3_df[s3_df['entity_id'].isin(all_cand_eids)].copy())

    s1_lookup = build_entity_lookup(s1_sub)
    cand_lookup = build_entity_lookup(pd.concat([s2_sub, s3_sub]))

    print(f"{'S1 ID':<14} | {'Cand ID':<14} | {'Name Sim':<8} | {'Addr Sim':<8} | {'Pred Prob':<10} | {'Decision'}")
    print("-" * 75)

    probs = []
    for m in high_addr_misses:
        s1_id = m['s1_id']
        c_id = m['cand_id']
        if s1_id not in s1_lookup or c_id not in cand_lookup:
            continue
        feats = compute_single_pair_features(s1_lookup[s1_id], cand_lookup[c_id])
        X = np.array([feats])
        p = model.predict_proba(X)
        prob = float(p[0]) if p.ndim == 1 else float(p[0, 1])
        probs.append(prob)
        dec = "SELECT (>0.5)" if prob >= 0.5 else ("BORDERLINE (>0.3)" if prob >= 0.3 else "REJECT (<0.3)")
        print(f"{s1_id:<14} | {c_id:<14} | {feats[0]:<8.1f} | {feats[10]:<8.1f} | {prob:<10.4f} | {dec}")

    print("\nSummary of Model Probabilities on Cross-Script Address Matches:")
    print(f"Mean Prob: {np.mean(probs):.4f}, Median: {np.median(probs):.4f}")
    print(f">= 0.50: {sum(p >= 0.5 for p in probs)} / {len(probs)}")
    print(f">= 0.30: {sum(p >= 0.3 for p in probs)} / {len(probs)}")
    print(f"< 0.30:  {sum(p < 0.3 for p in probs)} / {len(probs)}")

if __name__ == '__main__':
    main()
