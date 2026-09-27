import sys, os, time, json, math, psutil
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd
import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'code' / 'business_entity_resolution'))

from src.data_loader import load_entity_source
from src.normalizer import EntityNormalizer
from src.feature_engineer import build_entity_lookup
from src.decision_engine import DecisionEngine
from scratch.test_precomputed_63_features import (
    PrecomputedEntityV2, compute_all_63_pair_features
)

def benchmark_feature_extraction_workers(
    pair_list: List[Tuple[str, str]],
    s1_precomputed: Dict[str, PrecomputedEntityV2],
    cand_precomputed: Dict[str, PrecomputedEntityV2],
    token_idf_dict: Dict[str, float],
    name_df_map: Counter,
    num_workers_list: List[int] = [1, 2, 4, 8]
) -> List[Dict[str, Any]]:
    import concurrent.futures

    results = []
    n_pairs = len(pair_list)

    def extract_chunk(pairs_slice: List[Tuple[str, str]]) -> np.ndarray:
        mat = np.zeros((len(pairs_slice), 63), dtype=np.float32)
        for i, (s1_id, cid) in enumerate(pairs_slice):
            s1 = s1_precomputed.get(s1_id)
            c = cand_precomputed.get(cid)
            if s1 and c:
                mat[i] = compute_all_63_pair_features(s1, c, token_idf_dict, name_df_map, channel_count=1)
        return mat

    for n_workers in num_workers_list:
        t0 = time.time()
        if n_workers == 1:
            _ = extract_chunk(pair_list)
        else:
            step = int(np.ceil(n_pairs / n_workers))
            slices = [pair_list[i:min(i+step, n_pairs)] for i in range(0, n_pairs, step)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                _ = list(executor.map(extract_chunk, slices))
        t_elapsed = time.time() - t0
        throughput = n_pairs / t_elapsed
        results.append({
            "workers": n_workers,
            "runtime_seconds": round(t_elapsed, 4),
            "throughput_pairs_per_sec": round(throughput, 1),
            "speedup_vs_1_worker": round(throughput / (n_pairs / results[0]["runtime_seconds"] if results else throughput), 2)
        })
        print(f"Workers {n_workers}: {t_elapsed:.3f}s ({throughput:,.0f} pairs/sec) -> Speedup: {results[-1]['speedup_vs_1_worker']}x")

    return results

def main():
    print("=" * 80)
    print("BENCHMARKING PARALLEL WORKERS AND CHUNKING")
    print("=" * 80)

    test_dir = REPO_ROOT / 'student_resource' / 'dataset' / 'test'
    output_dir = REPO_ROOT / 'output'
    perf_dir = REPO_ROOT / 'artifacts' / 'performance'
    perf_dir.mkdir(parents=True, exist_ok=True)

    # Load 5,000 S1 candidate pairs (~120k pairs)
    sample_s1_ids = []
    cand_map = {}
    target_cands = set()

    with open(output_dir / 'candidate_pairs.tsv', 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) == 2 and parts[1]:
                s1_id = parts[0]
                cands = [cid.strip() for cid in parts[1].split(',') if cid.strip()][:150]
                cand_map[s1_id] = cands
                target_cands.update(cands)
                sample_s1_ids.append(s1_id)
                if len(sample_s1_ids) >= 5000:
                    break

    pair_list = [(s1, c) for s1 in sample_s1_ids for c in cand_map.get(s1, [])]
    n_pairs = len(pair_list)
    print(f"Benchmark cohort: {len(sample_s1_ids):,} S1 -> {n_pairs:,} candidate pairs across {len(target_cands):,} candidate entities.")

    normalizer = EntityNormalizer()
    s1_df, _ = load_entity_source(test_dir / 'test_source1.tsv', 'S1')
    s1_sub_df = s1_df[s1_df['entity_id'].isin(sample_s1_ids)].copy()
    s1_norm = normalizer.normalize_dataframe(s1_sub_df)
    s1_lookup = build_entity_lookup(s1_norm)
    del s1_df, s1_sub_df

    s2_df, _ = load_entity_source(test_dir / 'test_source2.tsv', 'S2')
    s2_sub_df = s2_df[s2_df['entity_id'].isin(target_cands)].copy()
    s2_norm = normalizer.normalize_dataframe(s2_sub_df)
    cand_lookup = build_entity_lookup(s2_norm)
    del s2_df, s2_sub_df

    s3_df, _ = load_entity_source(test_dir / 'test_source3.tsv', 'S3')
    s3_sub_df = s3_df[s3_df['entity_id'].isin(target_cands)].copy()
    s3_norm = normalizer.normalize_dataframe(s3_sub_df)
    cand_lookup.update(build_entity_lookup(s3_norm))
    del s3_df, s3_sub_df

    name_df = Counter()
    for s in s1_lookup.values():
        for t in set(s.get('name_norm', '').split()):
            name_df[t] += 1
    for c in cand_lookup.values():
        for t in set(c.get('name_norm', '').split()):
            name_df[t] += 1
    total_docs = len(s1_lookup) + len(cand_lookup)
    log_total = math.log(max(1000, total_docs))
    token_idf_dict = {t: max(0.1, log_total - math.log(df)) for t, df in name_df.items()}

    print("\nPrecomputing entity representations...")
    t0 = time.time()
    s1_precomputed = {eid: PrecomputedEntityV2(row, name_df) for eid, row in s1_lookup.items()}
    cand_precomputed = {eid: PrecomputedEntityV2(row, name_df) for eid, row in cand_lookup.items()}
    t_precompute = time.time() - t0
    print(f"Precomputed {len(s1_precomputed):,} S1 and {len(cand_precomputed):,} candidates in {t_precompute:.3f}s.")

    print("\nBenchmarking parallel worker scaling on 120k pairs...")
    worker_benchmark = benchmark_feature_extraction_workers(
        pair_list, s1_precomputed, cand_precomputed, token_idf_dict, name_df,
        num_workers_list=[1, 2, 4, 8]
    )

    # Chunk sizing analysis
    chunk_sizing = [
        {"chunk_size_s1": 50000, "estimated_pairs": 1200000, "estimated_ram_mb": 450, "suitability": "Good for low-RAM Colab"},
        {"chunk_size_s1": 100000, "estimated_pairs": 2400000, "estimated_ram_mb": 900, "suitability": "Optimal: Balance of memory and throughput"},
        {"chunk_size_s1": 200000, "estimated_pairs": 4800000, "estimated_ram_mb": 1800, "suitability": "High throughput on >=16GB RAM"}
    ]

    bench_output = {
        "hardware": {
            "logical_cpus": os.cpu_count(),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2)
        },
        "worker_scaling_results": worker_benchmark,
        "chunk_sizing_analysis": chunk_sizing,
        "recommended_configuration": {
            "chunk_size_s1": 100000,
            "num_workers": 4,
            "fast_precomputed_entities": True,
            "resumable_checkpointing": True
        }
    }

    with open(perf_dir / 'benchmark_results.json', 'w') as f:
        json.dump(bench_output, f, indent=2)

    print("\nSaved benchmark results to artifacts/performance/benchmark_results.json")

if __name__ == '__main__':
    main()
