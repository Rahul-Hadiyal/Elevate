import json, sys

sys.stdout.reconfigure(encoding='utf-8')

with open('artifacts/score_optimization/missing_links_forensics.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

count = 0
for m in d['sample_misses']:
    if m['country'] == 'india' and m.get('addr_token_set', 0) >= 70:
        cand_name_repr = m['cand_name'].encode('ascii', 'backslashreplace').decode('ascii')
        print(f"S1 ({m['s1_id']}): {m['s1_name']} | ADDR: {m['s1_addr']}")
        print(f"Cand ({m['cand_id']}): {cand_name_repr} | ADDR: {m['cand_addr']}")
        print(f"Addr Token Set Sim: {m['addr_token_set']:.2f}")
        print("-" * 80)
        count += 1
        if count >= 10:
            break
