import sys, time
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

def resolve_conflicts_original(
    selected_per_s1: Dict[str, Dict[str, float]],
    entity_candidate_map: Dict[str, List[Tuple[str, float]]],
    select_matches_fn,
    min_prob_filter=0.3,
    max_cands=3,
    ambiguous_threshold=0.03
):
    # Copy inputs
    selected = {k: dict(v) for k, v in selected_per_s1.items()}
    
    # Step 2: Build candidate -> list of (s1_id, prob)
    cand_claims = defaultdict(list)
    for s1_id, cands in selected.items():
        for cand_id, prob in cands.items():
            cand_claims[cand_id].append((s1_id, prob))

    conflicts = {c: claimants for c, claimants in cand_claims.items() if len(claimants) > 1}
    s1_entities_affected = set()
    ambiguous_dropped = set()

    for cand_id, claimants in conflicts.items():
        claimants.sort(key=lambda x: x[1], reverse=True)
        winner_s1, p_winner = claimants[0]
        second_s1, p_second = claimants[1]

        if (p_winner - p_second) < ambiguous_threshold:
            ambiguous_dropped.add(cand_id)
            for s1_id, _ in claimants:
                if cand_id in selected[s1_id]:
                    del selected[s1_id][cand_id]
                    s1_entities_affected.add(s1_id)
        else:
            for s1_id, _ in claimants[1:]:
                if cand_id in selected[s1_id]:
                    del selected[s1_id][cand_id]
                    s1_entities_affected.add(s1_id)

    # Step 4: Original slow loop
    for s1_id in s1_entities_affected:
        other_assigned = {
            c for other_s1, cands in selected.items()
            if other_s1 != s1_id for c in cands
        }
        excluded_for_entity = other_assigned | ambiguous_dropped
        remaining_cands = [
            (c_id, p) for c_id, p in entity_candidate_map.get(s1_id, [])
            if c_id not in excluded_for_entity
        ]
        re_chosen = select_matches_fn(remaining_cands, min_prob_filter, max_cands)
        selected[s1_id] = {c_id: p for c_id, p in re_chosen}

    return {k: set(v.keys()) for k, v in selected.items()}

def resolve_conflicts_optimized(
    selected_per_s1: Dict[str, Dict[str, float]],
    entity_candidate_map: Dict[str, List[Tuple[str, float]]],
    select_matches_fn,
    min_prob_filter=0.3,
    max_cands=3,
    ambiguous_threshold=0.03
):
    selected = {k: dict(v) for k, v in selected_per_s1.items()}
    
    cand_claims = defaultdict(list)
    for s1_id, cands in selected.items():
        for cand_id, prob in cands.items():
            cand_claims[cand_id].append((s1_id, prob))

    conflicts = {c: claimants for c, claimants in cand_claims.items() if len(claimants) > 1}
    s1_entities_affected = set()
    ambiguous_dropped = set()

    for cand_id, claimants in conflicts.items():
        claimants.sort(key=lambda x: x[1], reverse=True)
        winner_s1, p_winner = claimants[0]
        second_s1, p_second = claimants[1]

        if (p_winner - p_second) < ambiguous_threshold:
            ambiguous_dropped.add(cand_id)
            for s1_id, _ in claimants:
                if cand_id in selected[s1_id]:
                    del selected[s1_id][cand_id]
                    s1_entities_affected.add(s1_id)
        else:
            for s1_id, _ in claimants[1:]:
                if cand_id in selected[s1_id]:
                    del selected[s1_id][cand_id]
                    s1_entities_affected.add(s1_id)

    # Step 4: Fast O(1) set maintenance
    all_assigned = {c for cands in selected.values() for c in cands}

    for s1_id in s1_entities_affected:
        curr_assigned = set(selected[s1_id].keys())
        other_assigned = all_assigned - curr_assigned
        excluded_for_entity = other_assigned | ambiguous_dropped

        remaining_cands = [
            (c_id, p) for c_id, p in entity_candidate_map.get(s1_id, [])
            if c_id not in excluded_for_entity
        ]
        re_chosen = select_matches_fn(remaining_cands, min_prob_filter, max_cands)
        new_cands = {c_id: p for c_id, p in re_chosen}
        selected[s1_id] = new_cands
        # Update all_assigned
        all_assigned = (all_assigned - curr_assigned) | set(new_cands.keys())

    return {k: set(v.keys()) for k, v in selected.items()}

def mock_select(remaining, min_p, max_c):
    return [c for c in remaining if c[1] >= min_p][:max_c]

def main():
    import random
    random.seed(42)
    # Generate 5,000 entities with some conflicts
    selected = {}
    cand_map = {}
    for i in range(5000):
        s1 = f"S1-{i}"
        cands = [(f"C-{random.randint(0, 4000)}", round(random.uniform(0.35, 0.95), 3)) for _ in range(3)]
        selected[s1] = {c[0]: c[1] for c in cands}
        cand_map[s1] = cands + [(f"C-{random.randint(0, 5000)}", round(random.uniform(0.2, 0.8), 3)) for _ in range(5)]

    print("Running original...")
    t0 = time.time()
    res_orig = resolve_conflicts_original(selected, cand_map, mock_select)
    t_orig = time.time() - t0
    print(f"Original time: {t_orig:.4f}s")

    print("Running optimized...")
    t0 = time.time()
    res_opt = resolve_conflicts_optimized(selected, cand_map, mock_select)
    t_opt = time.time() - t0
    print(f"Optimized time: {t_opt:.4f}s")

    assert res_orig == res_opt, "OPTIMIZED OUTPUT DOES NOT MATCH ORIGINAL!"
    print("SUCCESS: 100% EXACT EQUIVALENCE VERIFIED!")
    print(f"Speedup: {t_orig / t_opt:.1f}x faster!")

if __name__ == '__main__':
    main()
