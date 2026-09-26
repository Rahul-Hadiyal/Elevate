# Elevate — Implementation Plan to Reach ≥ 0.96 Macro-F₀.₅

**Execution target:** Gemini 3.8 Flash (or any code-executing agent)
**Repository root:** `C:\Users\prana\Downloads\elevate`
**Current score:** 0.667 macro-F₀.₅ (rank 4803)
**Target score:** ≥ 0.96 macro-F₀.₅

---

## CRITICAL INSTRUCTIONS FOR THE EXECUTING MODEL

**READ THIS SECTION BEFORE DOING ANYTHING.**

1. **Do not invent numbers.** Every numeric value in this plan is either (a) measured
   from the real dataset and marked `[MEASURED]`, (b) computed arithmetically and
   marked `[COMPUTED]`, or (c) a parameter you must determine empirically and marked
   `[MUST MEASURE]`. Never substitute a guess for a `[MUST MEASURE]` value.
2. **Do not skip the gates.** Each phase ends with a GATE. If a gate fails, stop and
   report. Do not proceed on the assumption it will work out.
3. **Do not refactor beyond what is specified.** The codebase has a working metric
   implementation and a working decision engine. Both are verified correct. Changing
   them will break things.
4. **Do not change the F₀.₅ formula.** It is verified correct in `src/metrics.py`.
5. **Every code change must be followed by running the relevant test.**
6. **If a measured value contradicts this plan, trust the measurement and report the
   contradiction.** Do not force the data to match the plan.

---

## SECTION 0 — VERIFIED FACTS (DO NOT RE-DERIVE, DO NOT CONTRADICT)

These are measured from the actual competition dataset. Treat as ground truth.

### 0.1 Dataset scale `[MEASURED]`

| File | Rows |
|---|---|
| `dataset/train/train_source1.tsv` | 2,206,821 |
| `dataset/train/train_source2.tsv` | 5,034,616 |
| `dataset/train/train_source3.tsv` | 5,285,603 |
| `dataset/train/train_ground_truth.tsv` | 2,206,821 |
| `dataset/test/test_source1.tsv` | 1,732,544 |
| `dataset/test/test_source2.tsv` | 4,887,273 |
| `dataset/test/test_source3.tsv` | 5,082,316 |

Total corpus: 26,435,994 records, 2.52 GB.

### 0.2 Match cardinality `[MEASURED from train_ground_truth.tsv]`

| Metric | Value |
|---|---|
| Total ground-truth links | 7,638,365 |
| **Mean matches per S1 entity** | **3.461** |
| Median matches per S1 entity | 3.0 |
| Max matches per S1 entity | 11 |
| **True singleton rate (0 matches)** | **5.58%** (123,247 entities) |
| Entities with exactly 1 match | 5.40% (119,157) |
| Entities with 2+ matches | 89.02% (1,964,417) |
| **Entities with 2+ matches in a SINGLE source** | **76.8%** |
| Entities matching BOTH S2 and S3 | 80.48% |
| Entities matching S2 only | 6.48% |
| Entities matching S3 only | 7.45% |

**Match count histogram (exact):**

```
0 matches : 123,247     6 matches : 164,868
1 match   : 119,157     7 matches :  63,968
2 matches : 375,212     8 matches :  18,680
3 matches : 530,841     9 matches :   4,205
4 matches : 484,115    10 matches :     534
5 matches : 321,957    11 matches :      37
```

**Per-source maxima:** max 5 S2 matches per entity, max 6 S3 matches per entity.

### 0.3 Hypothesis H1 `[MEASURED]`

**H1 = CONFIRMED.** 0 violations out of 7,638,365 matched IDs. No S2 or S3 record is
claimed by more than one S1 entity.

**Consequence:** `enable_conflict_resolution = True` is SAFE. Output Invariant 8
(no candidate ID appears under two S1 rows) may be enforced.

### 0.4 Country distribution `[MEASURED]`

| Partition | India | US | France |
|---|---|---|---|
| Train S1 | 40.0% | 60.0% | **0.0%** |
| Test S1 | 46.7% | 38.3% | **15.0%** (259,452 entities) |

**France appears ONLY in test.** This is a genuine out-of-distribution shift affecting
15% of the evaluated entities.

### 0.5 Data complexity `[MEASURED]`

- Business name: mean 24.0 chars, mean 3.55 word tokens, max 105 chars
- Address: mean 52.1 chars, mean 8.03 word tokens, max 256 chars
- Script: 99.998% Latin (includes transliterated Indian names and accented French)
- **Exact duplicate business names in S1: 667,592** — major false-positive surface
- Exact duplicate addresses in S1: 76,215

### 0.6 Oracle ceilings by per-source cap `[COMPUTED from ground truth]`

Macro-F₀.₅ achievable with a PERFECT model, limited only by `max_cands_per_source`:

| `max_cands_per_source` | Oracle ceiling |
|---|---|
| **1 (CURRENT PRODUCTION VALUE)** | **0.8574** |
| 2 | 0.9653 |
| 3 | 0.9926 |
| 4 | 0.9990 |
| 5 | 1.0000 |
| 999 | 1.0000 |

**The current cap of 1 makes the 0.96 target mathematically impossible.**

### 0.7 Pair-level quality required for the target `[COMPUTED]`

At a mean of 3.461 true matches per entity:

| Pair precision | Pair recall | Resulting entity macro-F₀.₅ |
|---|---|---|
| 0.95 | 0.95 | 0.9500 |
| 0.96 | 0.95 | 0.9580 |
| **0.97** | **0.95** | **0.9659** ← target zone |
| 0.98 | 0.95 | 0.9738 |
| 0.98 | 0.98 | 0.9800 |

**Requirement: ≈ 0.97 pair precision AND ≈ 0.95 pair recall.**

### 0.8 Current submission shape `[MEASURED from output/matching_results.tsv]`

| Metric | Value |
|---|---|
| Rows | 1,732,544 (correct — matches test S1 count) |
| Empty predictions | 118,526 (6.84%) |
| **Mean predictions per entity** | **1.6024** |
| Max predictions per entity | 2 |
| Distribution | 0 preds: 118,526 · 1 pred: 451,891 · 2 preds: 1,162,127 |
| Max S2 predicted for any entity | 1 |
| Max S3 predicted for any entity | 1 |

**Diagnosis: predicting 1.60 matches against a true mean of 3.46. Under-predicting
by more than half.** The empty rate of 6.84% vs a true singleton rate of 5.58% is
approximately correct — **singletons are NOT the problem.**

### 0.9 Verified-correct components — DO NOT MODIFY

| Component | Status |
|---|---|
| `src/metrics.py` | F₀.₅ formula verified: `1.25·TP / (1.25·TP + 0.25·FN + FP)`. Returns 0.7143 on the official worked example. CORRECT. |
| `src/decision_engine.py` → `f05_scalar` | Correct, including the `tp==0 and fp==0 and fn==0 → 1.0` singleton branch. |
| `src/decision_engine.py` → `poisson_binomial_pmf` | Correct O(K²) DP. |
| `src/decision_engine.py` → `compute_exact_expected_f05` | Correct. Empty-set branch returns `∏(1−pⱼ)`. |
| `src/decision_engine.py` → `select_matches_for_entity` | Correct prefix search including k=0. |
| `src/calibration.py` | Implements both isotonic and sigmoid, plus ECE. Sound. |
| 13 passing tests in `test_metrics.py` + `test_decision_engine.py` | Verified. |

### 0.10 The identified defects

| ID | Defect | Location | Severity |
|---|---|---|---|
| **D1** | `max_cands_per_source` defaults to 1 | `run_phase12.py:178` — `PostProcessor(base_threshold=0.60, empty_addr_threshold=0.85)` omits the argument, so the constructor default of 1 applies | **CRITICAL** — caps ceiling at 0.8574 |
| **D2** | Candidate cap too tight | `run_phase12.py:126` uses `cap=15`; `inference.py` uses `cap_cands_per_s1=25` | **HIGH** — true max is 11 matches, leaving almost no blocking headroom |
| **D3** | Threshold tuned under a cap | `base_threshold=0.60` was optimal only because the cap enforced precision | **HIGH** |
| **D4** | Vetoes unsafe under France shift | `enforce_country_veto` / `enforce_numeric_veto` default to `True` | **MEDIUM** |
| **D5** | Two divergent inference paths | `run_phase8/11/12` use `PostProcessor`; `run_phase10/13` + `pipeline.py` use `DecisionEngine` | **MEDIUM** |
| **D6** | Independent per-source selection | `PostProcessor.filter_and_assign` picks top-S2 and top-S3 separately | **MEDIUM** |

### 0.11 UNKNOWN — must be measured in Phase 1

| ID | Unknown | Why it matters |
|---|---|---|
| **U1** | **Blocking recall on training data** | If below ~0.97, the target is UNREACHABLE regardless of all other fixes. **This is the single biggest risk in the entire plan.** |
| **U2** | Actual pair-level precision and recall | Determines how much modelling work remains |
| **U3** | Calibration ECE on validation | The decision engine requires calibrated probabilities |
| **U4** | `scale_pos_weight` value used at training | Anything ≠ 1.0 distorts probabilities |

---

## PHASE 1 — MEASUREMENT AND GATING

**Objective:** establish U1–U4. **Change no production code in this phase.**

### Step 1.1 — Verify environment

```bash
cd C:\Users\prana\Downloads\elevate\code\business_entity_resolution
python -m pytest tests/ -v --tb=short
```

**Expected:** all tests pass. **Record:** total passed / failed.
**If any test in `test_metrics.py` or `test_decision_engine.py` fails: STOP.**
Those components are verified correct; a failure means the environment is broken.

### Step 1.2 — Determine `scale_pos_weight` (U4)

```bash
findstr /S /N /I "scale_pos_weight" src\*.py
```

**Record** every call site and its value.
**Required value: 1.0.** If any training script passes a different value, note it — the
probabilities from that model are distorted and Phase 4 calibration becomes mandatory.

### Step 1.3 — MEASURE BLOCKING RECALL (U1) — THE CRITICAL GATE

Create `src/measure_blocking_recall.py`:

```python
"""Measures blocking recall on a TRAIN sample. This is the hard ceiling."""
import collections, random, sys
import numpy as np
import pandas as pd

from src.data_loader import load_sources          # adapt if the name differs
from src.normalizer import EntityNormalizer
from src.index_builder import BlockingIndex
from src.blocker import MultiChannelBlocker
from src.candidate_store import CandidateStore

SAMPLE_SIZE = 50000       # sample for tractability; raise if runtime allows
CAP_VALUES  = [15, 25, 50, 100]
SEED        = 42

def main():
    random.seed(SEED); np.random.seed(SEED)

    gt = {}
    with open("dataset/train/train_ground_truth.tsv", encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            ids = [x for x in parts[1].split(",") if x.strip()] if len(parts) > 1 else []
            gt[parts[0]] = set(ids)

    s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", dtype=str).fillna("")

    s1_sample = s1.sample(n=min(SAMPLE_SIZE, len(s1)), random_state=SEED)

    normalizer = EntityNormalizer()
    s1n = normalizer.normalize_dataframe(s1_sample)
    s2n = normalizer.normalize_dataframe(s2)
    s3n = normalizer.normalize_dataframe(s3)

    index = BlockingIndex(min_token_len=3, max_token_df=5000)
    index.build_indexes(s2n, s3n)
    blocker = MultiChannelBlocker(index, max_cands_per_key=100)

    channels = {}
    for ch in "ABCDEGHIJK":
        fn = getattr(blocker, f"generate_channel_{ch.lower()}", None)
        if fn is not None:
            channels[f"Channel_{ch}"] = fn(s1n)

    for cap in CAP_VALUES:
        store = CandidateStore(s1n["entity_id"])
        for name, cands in channels.items():
            store.add_channel_candidates(name, cands)
        cand_dict = store.get_candidate_dict(cap=cap)

        total_true = got = 0
        n_cands = []
        for s1_id in s1n["entity_id"]:
            truth = gt.get(s1_id, set())
            cands = set(cand_dict.get(s1_id, []))
            total_true += len(truth)
            got += len(truth & cands)
            n_cands.append(len(cands))

        recall = got / total_true if total_true else float("nan")
        print(f"cap={cap:<4} BLOCKING RECALL = {recall:.4f}  "
              f"mean_cands={np.mean(n_cands):.1f} max_cands={max(n_cands)}")

    # Per-channel marginal recall at the best cap
    print("\n--- PER-CHANNEL MARGINAL RECALL (cap=50) ---")
    for drop in list(channels) + [None]:
        store = CandidateStore(s1n["entity_id"])
        for name, cands in channels.items():
            if name == drop:
                continue
            store.add_channel_candidates(name, cands)
        cd = store.get_candidate_dict(cap=50)
        tt = g = 0
        for s1_id in s1n["entity_id"]:
            truth = gt.get(s1_id, set())
            tt += len(truth); g += len(truth & set(cd.get(s1_id, [])))
        label = "ALL CHANNELS" if drop is None else f"without {drop}"
        print(f"  {label:<28} recall = {g/tt:.4f}")

if __name__ == "__main__":
    main()
```

```bash
python -m src.measure_blocking_recall > logs\blocking_recall.txt
type logs\blocking_recall.txt
```

### Step 1.3b — CLASSIFY THE MISSES (mandatory if recall < 0.97)

Recall alone does not tell you what to fix. Every missed true match must be
classified as RETRIEVABLE (the channels can see it, but ranking or caps dropped it)
or UNRETRIEVABLE (no shared token exists, so no token-based channel can ever find it).

Append to `src/measure_blocking_recall.py`:

```python
    print("\n--- MISS CLASSIFICATION (cap=50) ---")
    store = CandidateStore(s1n["entity_id"])
    for name, cands in channels.items():
        store.add_channel_candidates(name, cands)
    cand_dict = store.get_candidate_dict(cap=50)

    # Normalized attribute lookup for both candidate sources
    cand_lookup = {}
    for df_n in (s2n, s3n):
        for row in df_n.itertuples(index=False):
            cand_lookup[row.entity_id] = (
                set(str(getattr(row, "name_tokens", "")).split()),
                set(str(getattr(row, "addr_tokens", "")).split()),
                set(str(getattr(row, "addr_numeric_tokens", "")).split()),
            )

    stats = collections.Counter()
    for row in s1n.itertuples(index=False):
        s1_id = row.entity_id
        truth = gt.get(s1_id, set())
        found = set(cand_dict.get(s1_id, []))
        missed = truth - found
        if not missed:
            continue
        s1_name = set(str(getattr(row, "name_tokens", "")).split())
        s1_addr = set(str(getattr(row, "addr_tokens", "")).split())
        s1_num  = set(str(getattr(row, "addr_numeric_tokens", "")).split())
        for m in missed:
            if m not in cand_lookup:
                stats["NOT_IN_SOURCE_FILES"] += 1
                continue
            c_name, c_addr, c_num = cand_lookup[m]
            shares_name = bool(s1_name & c_name)
            shares_addr = bool(s1_addr & c_addr)
            shares_num  = bool(s1_num  & c_num)
            if shares_name or shares_addr or shares_num:
                stats["RETRIEVABLE_but_dropped"] += 1
                if shares_name: stats["  ...shares name token"] += 1
                if shares_addr: stats["  ...shares addr token"] += 1
                if shares_num:  stats["  ...shares numeric"]    += 1
            else:
                stats["UNRETRIEVABLE_no_shared_token"] += 1

    total_missed = (stats["RETRIEVABLE_but_dropped"]
                    + stats["UNRETRIEVABLE_no_shared_token"]
                    + stats["NOT_IN_SOURCE_FILES"])
    for k, v in stats.most_common():
        pct = v / total_missed if total_missed else 0
        print(f"  {k:<34} {v:>8,}  ({pct:6.2%})")
```

**Note:** the attribute names `name_tokens`, `addr_tokens`, `addr_numeric_tokens` must
match whatever `EntityNormalizer.normalize_dataframe` actually produces. Inspect the
normalized DataFrame columns first and substitute the real names. Do not guess.

**Interpretation — this determines the fix and its cost:**

| Dominant category | Meaning | Fix | Cost |
|---|---|---|---|
| **RETRIEVABLE_but_dropped** | Channels see the pair; caps or ranking discarded it | Ladder steps 1–3 in Phase 6A (raise `cap`, `max_cands_per_key`, lower `min_token_len`) | **Cheap** — config only |
| **UNRETRIEVABLE_no_shared_token** | No shared token at all — transliteration, severe abbreviation, typos | Ladder step 5: add a character-3-gram TF-IDF channel | **Expensive** — new channel |
| **NOT_IN_SOURCE_FILES** | Ground-truth ID absent from the source files | Data loading bug — investigate before anything else | — |

With 3.46 matches per entity and `max_cands_per_key=100`, a meaningful share of misses
is expected to be RETRIEVABLE-but-dropped. That is the favourable case.

### GATE 1 — THE DECISIVE GATE

| Measured blocking recall (at cap = 50) | Verdict | Action |
|---|---|---|
| **≥ 0.97** | Target is reachable | Proceed to Phase 2 |
| **0.90 – 0.97** | Target is at risk | Proceed to Phase 2, but Phase 6 (blocking improvement) becomes MANDATORY |
| **< 0.90** | **0.96 is UNREACHABLE** | **STOP. Report. Blocking must be fixed before anything else.** |

**Reason:** blocking recall is a hard ceiling. A true match never generated as a
candidate cannot be recovered by any model, threshold, or decision rule.

**Record in `logs/phase1_report.md`:**
- blocking recall at each cap value
- mean and max candidates per entity at each cap
- per-channel marginal recall
- **miss classification counts from Step 1.3b**
- `scale_pos_weight` call sites and values
- pytest result

### Step 1.3c — TIME-BOXED FALLBACK

**Use this only if the full sweep in Step 1.3 will not finish in the available time.**
Building the blocking index over the full 10.3M train S2+S3 records is the expensive
step and is unavoidable, since candidates must be retrievable from the whole pool.
Everything else can be compressed.

Compressed protocol — one pass instead of a sweep:

1. Set `SAMPLE_SIZE = 25000` (instead of 50,000). 25k entities carry roughly 86k true
   links, which is ample for a recall point estimate.
2. Set `CAP_VALUES = [50]` only. Skip 15, 25, and 100.
3. Apply ladder steps 1–3 **pre-emptively** before measuring, since none of them can
   reduce recall — they only increase candidate volume:
   - `get_candidate_dict(cap=50)`
   - `MultiChannelBlocker(index, max_cands_per_key=200)`
   - `BlockingIndex(min_token_len=2, max_token_df=5000)`
4. Run Step 1.3b miss classification in the same pass.
5. Record recall AND mean candidates per entity.

**Decision rule after the compressed run:**

| Outcome | Action |
|---|---|
| Recall ≥ 0.97 and mean candidates ≤ 100 | Blocking is solved. Lock these three settings into production and proceed to Phase 2. |
| Recall ≥ 0.97 but mean candidates > 150 | Acceptable recall, but verify inference runtime on a chunk before the full run. |
| Recall < 0.97 and misses are RETRIEVABLE | Raise `max_cands_per_key` to 400 and re-measure once. |
| Recall < 0.97 and misses are UNRETRIEVABLE | Character-n-gram channel is required — go to Phase 6A step 5. Budget accordingly. |

This yields a recall number and a probable fix in a single pass rather than a sweep.

---

## PHASE 2 — THE CRITICAL FIX (D1)

**Objective:** remove the cap that makes the target impossible.
**Expected effect: ceiling rises from 0.8574 → 1.0000.**

### Step 2.1 — Fix `run_phase12.py` line 178

**FIND exactly:**
```python
post_processor = PostProcessor(base_threshold=0.60, empty_addr_threshold=0.85)
```

**REPLACE with exactly:**
```python
post_processor = PostProcessor(
    base_threshold=0.60,
    empty_addr_threshold=0.85,
    max_cands_per_source=999,
    enforce_numeric_veto=False,
    enforce_country_veto=False,
    enforce_global_uniqueness=False,
)
```

**Justification for each value:**

| Parameter | Value | Reason |
|---|---|---|
| `max_cands_per_source` | `999` | Max true per-source count is 5 (S2) / 6 (S3). 999 = effectively unlimited. Matches the `Strategy_A_Raw_Threshold` configuration already present in `run_phase8.py:222`. |
| `enforce_numeric_veto` | `False` | Address numerics are a named noise field in the problem statement. Hard vetoes drop pairs before scoring. |
| `enforce_country_veto` | `False` | France is absent from training (0.0%). Country normalization derived from training data may mis-handle French strings and veto valid matches. |
| `enforce_global_uniqueness` | `False` | Deferred to the DecisionEngine's margin-guarded conflict resolution in Phase 5. |
| `base_threshold` | `0.60` | **PROVISIONAL.** Re-tuned in Phase 3. |

### Step 2.2 — Raise the candidate cap (D2)

In `run_phase12.py`, **FIND:**
```python
pairs_map = store.get_candidate_dict(cap=15)
```
**REPLACE with:**
```python
pairs_map = store.get_candidate_dict(cap=50)
```

In `inference.py`, **FIND** the default `cap_cands_per_s1: int = 25` and **REPLACE**
the value with `50`.

**Justification:** the true max is 11 matches per entity. A cap of 15 leaves 4 slots of
headroom, which is inadequate. Use the cap that Phase 1 showed achieves ≥ 0.97 recall;
if `cap=50` did not reach 0.97 but `cap=100` did, use 100 and record the runtime cost.

### GATE 2

Re-run the pipeline on a **validation sample** (not the full test set) and confirm:

- [ ] Mean predictions per entity has risen substantially above 1.60
- [ ] Max predictions per entity is now > 2
- [ ] Empty prediction rate remains in the 4–8% band (true singleton rate is 5.58%)

**If mean predictions is still ≈ 1.60, the cap fix did not take effect. Stop and
find the other call site.**

---

## PHASE 3 — THRESHOLD RE-TUNING (D3)

**Objective:** find the optimal `base_threshold` now that the cap no longer enforces
precision.

**Critical context:** `0.60` was tuned when `max_cands_per_source=1` was doing the
precision work. With the cap removed, the threshold alone controls precision. The
optimum will be different — and because the true mean is 3.461 matches, it is likely
**lower** than 0.60. **Do not assume this. Measure it.**

### Step 3.1 — Build a validation split

Create `src/build_validation_split.py`:

```python
"""Family-level validation split. Splitting unit is the S1 entity + all its matches."""
import random, json

SEED = 42
SPLIT = {"train": 0.45, "earlystop": 0.15, "calibration": 0.10,
         "val_a": 0.20, "val_b": 0.10}   # sums to 1.00

def main():
    random.seed(SEED)
    ids = []
    with open("dataset/train/train_ground_truth.tsv", encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            ids.append(line.split("\t")[0])
    random.shuffle(ids)

    n = len(ids); out = {}; start = 0
    for name, frac in SPLIT.items():
        end = start + int(n * frac)
        out[name] = ids[start:end]
        start = end
    out["val_b"].extend(ids[start:])          # remainder into the last fold

    assert sum(len(v) for v in out.values()) == n
    with open("artifacts/validation_split.json", "w") as f:
        json.dump(out, f)
    for k, v in out.items():
        print(f"{k:<12} {len(v):>9,}  ({len(v)/n:.4f})")

if __name__ == "__main__":
    main()
```

**Split values — use exactly these:** `train=0.45, earlystop=0.15, calibration=0.10,
val_a=0.20, val_b=0.10`. They sum to exactly 1.00. Each fold has one purpose:

| Fold | Purpose | Reuse allowed |
|---|---|---|
| `train` | Model fitting | — |
| `earlystop` | Early stopping / hyperparameter selection ONLY | — |
| `calibration` | Isotonic fitting ONLY | **No** — reusing it makes calibration optimistic |
| `val_a` | Iterative experiment comparison | Yes, many times |
| `val_b` | Final honest estimate | **Scored exactly ONCE** |

### Step 3.2 — Export scored pairs on `val_a`

Produce `artifacts/val_a_scored.tsv` with exactly three columns:

```
s1_id <TAB> cand_id <TAB> prob
```

**Include every scored candidate, including rejected ones.** If only predicted matches
are dumped, the threshold sweep cannot see what was missed.

`prob` must be the **calibrated** probability (`calibrator.predict_proba` output),
matching what `inference.py:117` already does.

### Step 3.3 — Sweep the threshold

Create `src/sweep_threshold.py`:

```python
"""Sweeps base_threshold and per-source cap against true macro-F0.5."""
import collections
import numpy as np
import pandas as pd

BETA2 = 0.25

def f05(tp, fp, fn):
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    if tp == 0:
        return 0.0
    den = (1 + BETA2) * tp + BETA2 * fn + fp
    return ((1 + BETA2) * tp) / den if den > 0 else 0.0

def source_of(cid):
    return "S2" if cid.startswith("S2") else "S3"

def main():
    gt = {}
    with open("dataset/train/train_ground_truth.tsv", encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            gt[p[0]] = set(x for x in p[1].split(",") if x.strip()) if len(p) > 1 else set()

    df = pd.read_csv("artifacts/val_a_scored.tsv", sep="\t", dtype=str)
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce").fillna(0.0)

    scored = collections.defaultdict(list)
    for r in df.itertuples(index=False):
        scored[r.s1_id].append((r.cand_id, float(r.prob)))
    entities = [e for e in scored if e in gt]
    print(f"evaluating on {len(entities):,} entities")

    print(f"\n{'thr':>6} {'cap=1':>9} {'cap=2':>9} {'cap=3':>9} {'cap=999':>9} {'meanpred':>9}")
    best = (-1, None, None)
    for thr in np.arange(0.20, 0.96, 0.025):
        row = []
        for cap in (1, 2, 3, 999):
            tot = 0.0; npred = 0
            for e in entities:
                keep = [(c, p) for c, p in scored[e] if p >= thr]
                sel = set()
                for src in ("S2", "S3"):
                    sub = sorted([x for x in keep if source_of(x[0]) == src],
                                 key=lambda x: -x[1])[:cap]
                    sel |= {c for c, _ in sub}
                t = gt[e]; tp = len(sel & t)
                tot += f05(tp, len(sel) - tp, len(t) - tp)
                npred += len(sel)
            score = tot / len(entities)
            row.append(score)
            if score > best[0]:
                best = (score, float(thr), cap)
        mp = npred / len(entities)
        print(f"{thr:6.3f} {row[0]:9.4f} {row[1]:9.4f} {row[2]:9.4f} {row[3]:9.4f} {mp:9.2f}")

    print(f"\nBEST: macro-F0.5={best[0]:.4f} at threshold={best[1]:.3f} cap={best[2]}")

if __name__ == "__main__":
    main()
```

```bash
python -m src.sweep_threshold > logs\threshold_sweep.txt
```

### GATE 3

**Record the optimal `(threshold, cap)` pair and the macro-F₀.₅ it achieves.**
**Use the measured optimum. Do not use 0.60 because this plan mentioned it.**

| Best macro-F₀.₅ from the sweep | Interpretation |
|---|---|
| ≥ 0.96 | Target reached by configuration alone. Proceed to Phase 7 (validate and submit). |
| 0.90 – 0.96 | Good progress. Proceed to Phase 4 and 5. |
| < 0.90 | The model itself is the limitation. Phase 6 is mandatory. |

---

## PHASE 4 — CALIBRATION VERIFICATION (U3)

**Objective:** confirm the probabilities are calibrated. The DecisionEngine consumes
probabilities as NUMBERS, not as a ranking. On uncalibrated scores it performs WORSE
than a plain threshold.

### Step 4.1 — Measure ECE

```python
# src/check_calibration.py
import collections
import numpy as np
import pandas as pd
from src.calibration import compute_ece

gt = {}
with open("dataset/train/train_ground_truth.tsv", encoding="utf-8") as fh:
    fh.readline()
    for line in fh:
        p = line.rstrip("\n").split("\t")
        gt[p[0]] = set(x for x in p[1].split(",") if x.strip()) if len(p) > 1 else set()

df = pd.read_csv("artifacts/val_a_scored.tsv", sep="\t", dtype=str)
df["prob"] = pd.to_numeric(df["prob"], errors="coerce").fillna(0.0)
df = df[df.s1_id.isin(gt)]

probs = df["prob"].to_numpy()
y = np.array([1 if c in gt[s] else 0 for s, c in zip(df.s1_id, df.cand_id)])

print("ECE =", compute_ece(probs, y, n_bins=10))
print(f"\n{'bin':>12} {'n':>10} {'mean_p':>8} {'actual':>8} {'gap':>8}")
for lo in np.arange(0, 1.0, 0.1):
    m = (probs >= lo) & (probs < lo + 0.1)
    if m.sum() == 0:
        continue
    mp, ac = probs[m].mean(), y[m].mean()
    flag = "  <-- OFF" if abs(mp - ac) > 0.10 else ""
    print(f"  [{lo:.1f},{lo+0.1:.1f}) {m.sum():10d} {mp:8.3f} {ac:8.3f} {mp-ac:+8.3f}{flag}")
```

### GATE 4

| Measured ECE | Verdict | Action |
|---|---|---|
| **≤ 0.05** | Calibrated | Proceed to Phase 5 (DecisionEngine is safe) |
| **> 0.05** | Not calibrated | **Refit calibration before Phase 5.** Use `ProbabilityCalibrator(method="isotonic")` fitted on the `calibration` fold ONLY. Then re-run this step. |

**If ECE > 0.05 and cannot be fixed in the available time: SKIP Phase 5 entirely and
ship the Phase 3 threshold configuration.** A well-tuned threshold on uncalibrated
scores beats expected-F₀.₅ on uncalibrated scores.

---

## PHASE 5 — DECISION ENGINE MIGRATION (D5, D6)

**PRECONDITION: Gate 4 passed with ECE ≤ 0.05. If not, skip this phase.**

**Objective:** replace independent per-source selection with per-entity exact
expected-F₀.₅ selection.

**Why this helps:** `PostProcessor` picks top-S2 and top-S3 independently, so an entity
matching in only one source (13.93% of non-singletons) receives a guaranteed false
positive. The DecisionEngine pools all candidates and selects the subset maximizing
exact expected F₀.₅, with a bar that rises automatically as more are added.

### Step 5.1 — Modify `inference.py`

**ADD to the imports:**
```python
from src.decision_engine import DecisionEngine
```

**CHANGE the signature** — replace the `post_processor: PostProcessor` parameter with:
```python
decision_engine: DecisionEngine,
```

**REPLACE the block from `cand_preds: List[CandidatePrediction] = []` through
`chunk_preds = post_processor.filter_and_assign(...)` with:**

```python
        entity_cand_map: Dict[str, List[Tuple[str, float]]] = {}
        for i, (s1_id, cid) in enumerate(batch.pair_ids):
            entity_cand_map.setdefault(s1_id, []).append((cid, float(cal_probs[i])))

        # Every S1 entity in the chunk must be represented, including
        # entities with zero candidates -> they yield an empty prediction.
        for s1_id in s1_norm_chunk["entity_id"]:
            entity_cand_map.setdefault(s1_id, [])

        chunk_preds = decision_engine.optimize_predictions(entity_cand_map)
        all_predictions.update(chunk_preds)
```

**CRITICAL:** the `entity_cand_map.setdefault(s1_id, [])` loop is mandatory. Without it,
entities with no candidates disappear from the output, breaking the requirement that
every test S1 entity appear exactly once.

### Step 5.2 — Construct the engine with exact values

```python
decision_engine = DecisionEngine(
    margin_delta=0.05,
    min_prob_filter=0.01,
    max_candidates_per_entity=20,
    enable_conflict_resolution=True,
)
```

| Parameter | Value | Justification |
|---|---|---|
| `margin_delta` | `0.05` | Existing default in `pipeline.py:62`. Tune in Step 5.4. |
| `min_prob_filter` | `0.01` | Verified: pruning below 0.01 changed the selected prefix in 1 of 300 randomised trials. |
| `max_candidates_per_entity` | `20` | True max is 11 matches. 20 gives headroom and bounds the O(n³) prefix search. |
| `enable_conflict_resolution` | `True` | **SAFE — H1 confirmed with 0 violations across 7,638,365 links.** |

### Step 5.3 — Head-to-head comparison on `val_a`

Score `val_a` with BOTH configurations and compare:

| Configuration | Macro-F₀.₅ |
|---|---|
| PostProcessor, best threshold + cap from Phase 3 | `[MUST MEASURE]` |
| DecisionEngine, parameters above | `[MUST MEASURE]` |

**Ship whichever wins on the measurement. Do not assume the DecisionEngine wins.**

### Step 5.4 — Tune `margin_delta` (only if the DecisionEngine won)

Sweep `margin_delta` over `[0.0, 0.02, 0.05, 0.10, 0.20]` on `val_a`. Record the
optimum and the conflict-resolution event count at each value.

### GATE 5

- [ ] The winning configuration is recorded with its measured `val_a` score
- [ ] Every S1 ID appears exactly once in the output
- [ ] Empty prediction rate is in the 4–8% band
- [ ] Mean predictions per entity is near 3.0–3.5

---

## PHASE 6 — MODEL AND BLOCKING IMPROVEMENT

**Enter this phase ONLY if Phases 2–5 did not reach 0.96.**

Priority is determined by the Phase 1 measurement:

### 6A — If blocking recall < 0.97 (highest priority)

This is a hard ceiling. **Apply the ladder in the order below, cheapest first, and
re-measure BOTH recall and mean candidates per entity after each step.** Stop as soon
as recall ≥ 0.97.

**Use the Step 1.3b classification to choose the entry point:** if misses are mostly
RETRIEVABLE, steps 1–4 will likely suffice. If mostly UNRETRIEVABLE, skip to step 5 —
steps 1–4 cannot recover pairs that share no token.

| # | Change | File / parameter | From → To | Targets |
|---|---|---|---|---|
| 1 | Raise candidate cap | `get_candidate_dict(cap=)` | 50 → 100 | RETRIEVABLE |
| 2 | Raise per-key limit | `MultiChannelBlocker(max_cands_per_key=)` | 100 → 200 → 400 | RETRIEVABLE — high-frequency keys saturate and silently drop true matches. Critical given 667,592 duplicate S1 names |
| 3 | Lower min token length | `BlockingIndex(min_token_len=)` | 3 → 2 | RETRIEVABLE — French tokens ("bis", "rue") and Indian abbreviations are 2–3 chars. Directly targets the 15% France block |
| 4 | Raise token DF ceiling | `BlockingIndex(max_token_df=)` | 5000 → 10000 | RETRIEVABLE — **caution:** worsens the duplicate-name problem and inflates volume. Measure both metrics |
| 5 | Add char-3-gram TF-IDF channel | new channel on name + address | — | **UNRETRIEVABLE** — the only fix that catches pairs sharing no whole token: transliteration variants, severe abbreviation, typos |
| 6 | Prune dead channels | per-channel marginal recall from Step 1.3 | — | Any channel contributing < 0.5 pp may be removed to free compute for the others |

**Volume guard:** after each step, if mean candidates per entity exceeds 150, verify
inference runtime on a single chunk before committing. Blocking recall is worthless if
the full 1,732,544-entity run cannot finish.

### 6B — If pair precision < 0.97

1. Verify `scale_pos_weight=1.0`. Any other value distorts probabilities.
2. Add features targeting the 667,592 duplicate-name cases: IDF-weighted token overlap,
   rare-token overlap count, and address agreement conditioned on name agreement.
3. Add hard negatives — `src/hard_negatives.py` exists. Mine high-scoring non-matches
   from blocking output and add them to training.

### 6C — If pair recall < 0.95

1. Check whether missed matches are absent from candidates (blocking) or present but
   low-scored (model). `src/diagnose_missed.py` exists for this.
2. If blocking: go to 6A. If model: add features for the failure modes found.

### 6D — France-specific risk (15% of test entities)

France is 0.0% of training and 15.0% of test S1. Mandatory checks:

1. Confirm the legal-suffix and abbreviation vocabularies are **derived from corpus
   frequency over the combined train+test corpus**, not hard-coded English lists.
   French forms (SARL, SAS, SA, EURL) and address forms (rue, avenue, bis) must be
   discovered from the data.
2. **OOD calibration audit** (`src/ood_audit.py` exists): hold India out of calibration,
   calibrate on US only, then measure macro-F₀.₅ on held-out India.
   **Gate: if India degrades by more than 2 pp, calibration is too country-sensitive.**
   Fall back to a country-agnostic calibrator or a more conservative decision rule.
3. Confirm no code filters on `country ∈ {US, India}`. The country vetoes disabled in
   Phase 2 are the main risk; verify no others exist.

---

## PHASE 7 — VALIDATION AND SUBMISSION

### Step 7.1 — Regenerate the full test submission

Run the winning configuration over the full 1,732,544 test S1 entities.

### Step 7.2 — Mandatory output invariants

| # | Invariant | Check |
|---|---|---|
| 1 | Exactly 1,732,544 rows, one per test S1 ID | `len(df) == 1732544` and `df.source1_entity_id.nunique() == 1732544` |
| 2 | All matched IDs are S2 or S3 | no ID starts with `S1` |
| 3 | All matched IDs exist in the test source files | set membership check |
| 4 | No duplicate IDs within a list | `len(ids) == len(set(ids))` |
| 5 | `matching_results` IDs ⊆ `candidate_pairs` IDs, per entity | per-entity subset check |
| 6 | Empty string (not `NaN`, not `NULL`) for no-match entities | dtype and value check |
| 7 | Tab-separated with correct headers | `sep="\t"` on read-back |
| 8 | No S2/S3 ID under two S1 rows | **SAFE to enforce — H1 confirmed** |

### Step 7.3 — Sanity band check

| Metric | Expected band | Source |
|---|---|---|
| Mean predictions per entity | **3.0 – 3.5** | True mean is 3.461 |
| Empty prediction rate | **4% – 8%** | True singleton rate is 5.58% |
| Max predictions per entity | **≥ 5** | True max is 11 |

**If mean predictions is still near 1.60, the Phase 2 fix did not take effect.**

### Step 7.4 — Official validator

```bash
cd student_resource
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

**Non-zero exit → do not submit.**

### Step 7.5 — Honest estimate

Score `val_b` **exactly once** with the final frozen configuration. If
`F₀.₅(val_a) − F₀.₅(val_b) > 0.02`, revert to the simplest configuration within
0.005 of the best on `val_a`.

### Step 7.6 — Package

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
└── Documentation_template.md
```

`candidate_pairs.tsv` must be the **final** candidate set — the exact pairs the model
scored, not a raw pre-filter blocking union. Every ID in `matching_results.tsv` must
appear in it.

---

## PHASE 8 — METHODOLOGY DOCUMENT

Required content: methodology, candidate generation and blocking strategy, model
architecture, feature engineering.

**Must state explicitly:**
- IDF and vocabularies are computed over the provided corpus **including test source
  text** (transductive). This is permitted — the prohibition is on EXTERNAL data.
- No external databases, APIs, or geocoding services are used.
- Model licenses: LightGBM (MIT). scikit-learn is BSD-3-Clause — permissive and
  compatible in practice, used for calibration and TF-IDF. State this plainly.

---

## EXECUTION CHECKLIST

```
[ ] PHASE 1  Measure blocking recall, scale_pos_weight, run tests
    [ ] 1.3   Blocking recall sweep across cap values
    [ ] 1.3b  Classify misses: RETRIEVABLE vs UNRETRIEVABLE  (if recall < 0.97)
    [ ] 1.3c  Time-boxed fallback if the sweep will not finish
    [ ] GATE 1: blocking recall >= 0.97? If < 0.90, STOP.
[ ] PHASE 2  Set max_cands_per_source=999; raise cap 15 -> 50; disable vetoes
    [ ] GATE 2: mean predictions per entity has risen above 1.60
[ ] PHASE 3  Build validation split; export scored pairs; sweep threshold x cap
    [ ] GATE 3: record the measured optimum
[ ] PHASE 4  Measure ECE
    [ ] GATE 4: ECE <= 0.05, else refit isotonic or skip Phase 5
[ ] PHASE 5  Migrate to DecisionEngine; head-to-head on val_a
    [ ] GATE 5: winner measured, not assumed
[ ] PHASE 6  Only if still below 0.96 — blocking / model / France
[ ] PHASE 7  Full run, 8 invariants, sanity bands, official validator, val_b once
[ ] PHASE 8  Methodology document
```

---

## EXPECTED TRAJECTORY

| Stage | Expected macro-F₀.₅ | Basis |
|---|---|---|
| Current | 0.667 | Measured on leaderboard |
| After Phase 2 (cap fix) | `[MUST MEASURE]` | Ceiling rises 0.8574 → 1.0000 |
| After Phase 3 (threshold) | `[MUST MEASURE]` | Threshold re-optimised for the uncapped regime |
| After Phase 5 (decision engine) | `[MUST MEASURE]` | Only if Gate 4 passes |
| Target | ≥ 0.96 | Requires ≈ 0.97 pair precision and ≈ 0.95 pair recall |

**Do not report an expected number in place of a measured one. Every `[MUST MEASURE]`
must be filled with an actual value from an actual run.**

---

## RISK REGISTER

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Blocking recall < 0.90 | Unknown | **Fatal to the target** | Gate 1 measures it first |
| ECE > 0.05, no time to refit | Medium | DecisionEngine unusable | Phase 3 threshold config is the fallback |
| France degrades badly (15% of test) | Medium | ~0.05 macro | Phase 6D OOD audit |
| Runtime blowup at cap=50 over 1.7M entities | Medium | Cannot finish | Chunked inference already exists; measure on a sample first |
| 667,592 duplicate names cause false positives | High | Precision loss | Phase 6B IDF-weighted features |
| Fixing the cap without re-tuning the threshold | **High** | Could LOWER the score | Phase 3 is mandatory, not optional |

**The last row is the most likely way to make things worse. `base_threshold=0.60` was
tuned under `max_cands_per_source=1`. Removing the cap without re-tuning will flood the
output with low-confidence matches. Phase 2 and Phase 3 must be done together.**
