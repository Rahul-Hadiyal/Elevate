"""Leakage-Safe Validation Splitting Module.

Implements deterministic, stratified S1-family level splitting across:
- train (45%)
- earlystop (15%)
- calibration (10%)
- val_a (20%)
- val_b (10%)

Guarantees:
- Zero data leakage: S1 entity families and their ground truth links are completely
  isolated within a single partition.
- Deterministic & reproducible: fixed seed hashing and stratified allocation.
- Balanced stratification across country, cardinality buckets, and match source patterns.
- Fast serialization to Parquet, JSON, and TSV formats.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import json
import logging
import numpy as np
import pandas as pd



logger = logging.getLogger(__name__)

DEFAULT_SPLIT_RATIOS: Dict[str, float] = {
    "train": 0.45,
    "earlystop": 0.15,
    "calibration": 0.10,
    "val_a": 0.20,
    "val_b": 0.10,
}

DEFAULT_SEED: int = 42


def get_cardinality_bucket(match_count: int) -> str:
    """Categorizes ground truth match count into discrete stratification buckets."""
    if match_count == 0:
        return "0_singleton"
    elif match_count == 1:
        return "1_single"
    elif 2 <= match_count <= 5:
        return "2_to_5_multi"
    else:
        return "6_plus_multi"


def get_match_pattern(matched_ids: Iterable[str]) -> str:
    """Categorizes match source pattern into S2-only, S3-only, both, or none."""
    has_s2 = False
    has_s3 = False
    for mid in matched_ids:
        mid_str = str(mid).strip()
        if mid_str.startswith("S2-"):
            has_s2 = True
        elif mid_str.startswith("S3-"):
            has_s3 = True

    if has_s2 and has_s3:
        return "both_s2_s3"
    elif has_s2:
        return "s2_only"
    elif has_s3:
        return "s3_only"
    else:
        return "none"


@dataclass
class SplitConfig:
    """Configuration for dataset splitting."""
    ratios: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SPLIT_RATIOS))
    random_seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        total = sum(self.ratios.values())
        if not np.isclose(total, 1.0, atol=1e-5):
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f} for {self.ratios}")
        for part, ratio in self.ratios.items():
            if ratio < 0.0:
                raise ValueError(f"Ratio for partition '{part}' cannot be negative: {ratio}")


@dataclass
class SplitManifest:
    """Manifest containing S1 entity to partition mapping and split verification stats."""
    entity_to_partition: Dict[str, str]
    partition_counts: Dict[str, int]
    partition_stats: Dict[str, Dict[str, Any]]
    ratios: Dict[str, float]
    random_seed: int
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


    @property
    def total_entities(self) -> int:
        return len(self.entity_to_partition)

    def get_entity_ids(self, partition: str) -> List[str]:
        """Returns list of S1 entity IDs assigned to the given partition."""
        if partition not in self.partition_counts:
            raise KeyError(f"Unknown partition '{partition}'. Available: {list(self.partition_counts.keys())}")
        return [eid for eid, p in self.entity_to_partition.items() if p == partition]

    def get_entity_set(self, partition: str) -> Set[str]:
        """Returns set of S1 entity IDs assigned to the given partition."""
        if partition not in self.partition_counts:
            raise KeyError(f"Unknown partition '{partition}'. Available: {list(self.partition_counts.keys())}")
        return {eid for eid, p in self.entity_to_partition.items() if p == partition}

    def filter_s1_dataframe(self, s1_df: pd.DataFrame, partition: str, id_col: str = "entity_id") -> pd.DataFrame:
        """Filters an S1 DataFrame to include only entities in the specified partition."""
        part_set = self.get_entity_set(partition)
        return s1_df[s1_df[id_col].isin(part_set)].copy().reset_index(drop=True)

    def filter_ground_truth(
        self,
        ground_truth: Dict[str, List[str]],
        partition: str,
    ) -> Dict[str, List[str]]:
        """Filters a ground truth dictionary to include only entities in the specified partition."""
        part_set = self.get_entity_set(partition)
        return {eid: ground_truth[eid] for eid in part_set if eid in ground_truth}

    def filter_ground_truth_df(
        self,
        gt_df: pd.DataFrame,
        partition: str,
        id_col: str = "source1_entity_id",
    ) -> pd.DataFrame:
        """Filters a ground truth DataFrame to include only entities in the specified partition."""
        part_set = self.get_entity_set(partition)
        return gt_df[gt_df[id_col].isin(part_set)].copy().reset_index(drop=True)

    def save(self, output_path: Union[str, Path]) -> None:
        """Saves the split manifest to Parquet, JSON, or TSV based on file extension."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".parquet":
            df = pd.DataFrame({
                "source1_entity_id": list(self.entity_to_partition.keys()),
                "partition": list(self.entity_to_partition.values()),
            })
            df.to_parquet(path, index=False)
            # Also save companion metadata json
            meta_path = path.with_suffix(".meta.json")
            meta = {
                "partition_counts": self.partition_counts,
                "partition_stats": self.partition_stats,
                "ratios": self.ratios,
                "random_seed": self.random_seed,
                "created_at": self.created_at,
                "total_entities": self.total_entities,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        elif path.suffix == ".json":
            data = {
                "ratios": self.ratios,
                "random_seed": self.random_seed,
                "created_at": self.created_at,
                "partition_counts": self.partition_counts,
                "partition_stats": self.partition_stats,
                "entity_to_partition": self.entity_to_partition,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        elif path.suffix in (".tsv", ".csv") or path.name.endswith(".tsv.gz"):
            sep = "\t" if "tsv" in path.name else ","
            df = pd.DataFrame({
                "source1_entity_id": list(self.entity_to_partition.keys()),
                "partition": list(self.entity_to_partition.values()),
            })
            df.to_csv(path, sep=sep, index=False)
        else:
            raise ValueError(f"Unsupported manifest file extension: {path.suffix}")

        logger.info(f"Saved split manifest to {path} ({self.total_entities:,} entities)")

    @classmethod
    def load(cls, manifest_path: Union[str, Path]) -> "SplitManifest":
        """Loads a SplitManifest from file."""
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest file not found: {path}")

        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            entity_to_partition = dict(zip(df["source1_entity_id"], df["partition"]))
            meta_path = path.with_suffix(".meta.json")
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                return cls(
                    entity_to_partition=entity_to_partition,
                    partition_counts=meta.get("partition_counts", {}),
                    partition_stats=meta.get("partition_stats", {}),
                    ratios=meta.get("ratios", {}),
                    random_seed=meta.get("random_seed", DEFAULT_SEED),
                    created_at=meta.get("created_at", ""),
                )
            else:
                counts = df["partition"].value_counts().to_dict()
                return cls(
                    entity_to_partition=entity_to_partition,
                    partition_counts=counts,
                    partition_stats={},
                    ratios={},
                    random_seed=DEFAULT_SEED,
                )
        elif path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                entity_to_partition=data["entity_to_partition"],
                partition_counts=data.get("partition_counts", {}),
                partition_stats=data.get("partition_stats", {}),
                ratios=data.get("ratios", {}),
                random_seed=data.get("random_seed", DEFAULT_SEED),
                created_at=data.get("created_at", ""),
            )
        elif path.suffix in (".tsv", ".csv") or path.name.endswith(".tsv.gz"):
            sep = "\t" if "tsv" in path.name else ","
            df = pd.read_csv(path, sep=sep)
            entity_to_partition = dict(zip(df["source1_entity_id"], df["partition"]))
            counts = df["partition"].value_counts().to_dict()
            return cls(
                entity_to_partition=entity_to_partition,
                partition_counts=counts,
                partition_stats={},
                ratios={},
                random_seed=DEFAULT_SEED,
            )
        else:
            raise ValueError(f"Unsupported manifest file extension: {path.suffix}")


def create_stratified_split(
    s1_df: pd.DataFrame,
    ground_truth: Union[pd.DataFrame, Dict[str, List[str]]],
    config: Optional[SplitConfig] = None,
) -> SplitManifest:
    """Creates deterministic, stratified S1-entity-level split manifest.

    Args:
        s1_df: Source 1 entity DataFrame containing `entity_id` and `country`.
        ground_truth: Ground truth mapping (Dict or DataFrame).
        config: Split configuration with ratios and random seed.

    Returns:
        SplitManifest containing entity-to-partition mapping and stats.
    """
    cfg = config or SplitConfig()

    # Parse ground truth into dict for fast lookup
    gt_map: Dict[str, List[str]] = {}
    if isinstance(ground_truth, pd.DataFrame):
        for _, row in ground_truth.iterrows():
            s1_id = str(row["source1_entity_id"]).strip()
            raw_matched = row.get("matched_entity_ids")
            if pd.isna(raw_matched) or raw_matched is None or str(raw_matched).strip() == "":
                gt_map[s1_id] = []
            elif isinstance(raw_matched, str):
                gt_map[s1_id] = [m.strip() for m in raw_matched.split(",") if m.strip()]
            elif isinstance(raw_matched, (list, set, tuple)):
                gt_map[s1_id] = [str(m).strip() for m in raw_matched if str(m).strip()]
            else:
                gt_map[s1_id] = []
    elif isinstance(ground_truth, dict):
        gt_map = {k: list(v) for k, v in ground_truth.items()}
    else:
        raise TypeError(f"Unsupported ground truth type: {type(ground_truth)}")

    entity_ids: List[str] = s1_df["entity_id"].astype(str).str.strip().tolist()
    countries: List[str] = (
        s1_df["country"].fillna("UNKNOWN").astype(str).str.strip().str.upper().tolist()
        if "country" in s1_df.columns
        else ["UNKNOWN"] * len(entity_ids)
    )

    # Build stratum keys
    # Map entities into strata
    strata: Dict[str, List[str]] = {}
    entity_metadata: Dict[str, Dict[str, Any]] = {}

    for eid, country in zip(entity_ids, countries):
        matched = gt_map.get(eid, [])
        match_count = len(matched)
        card_bucket = get_cardinality_bucket(match_count)
        pattern = get_match_pattern(matched)

        # Country normalization for stratification grouping
        country_group = "US" if country in ("US", "USA", "UNITED STATES") else (
            "IN" if country in ("IN", "IND", "INDIA") else (
                "FR" if country in ("FR", "FRA", "FRANCE") else "OTHER"
            )
        )

        stratum_key = f"{country_group}__{card_bucket}__{pattern}"
        if stratum_key not in strata:
            strata[stratum_key] = []
        strata[stratum_key].append(eid)

        entity_metadata[eid] = {
            "country": country_group,
            "cardinality_bucket": card_bucket,
            "match_pattern": pattern,
            "match_count": match_count,
        }

    # Deterministic assignment per stratum
    rng = np.random.RandomState(cfg.random_seed)
    partition_names = list(cfg.ratios.keys())
    ratio_values = list(cfg.ratios.values())
    cum_ratios = np.cumsum(ratio_values)

    entity_to_partition: Dict[str, str] = {}
    partition_counts: Dict[str, int] = {p: 0 for p in partition_names}

    # Sort strata for deterministic iteration order
    sorted_strata_keys = sorted(strata.keys())

    for s_key in sorted_strata_keys:
        members = list(strata[s_key])
        n_members = len(members)

        # Shuffle deterministically
        shuffled_indices = rng.permutation(n_members)
        shuffled_members = [members[i] for i in shuffled_indices]

        # Compute split boundaries
        split_points = (cum_ratios * n_members).round().astype(int)
        start_idx = 0
        for p_idx, p_name in enumerate(partition_names):
            end_idx = min(split_points[p_idx], n_members)
            for eid in shuffled_members[start_idx:end_idx]:
                entity_to_partition[eid] = p_name
                partition_counts[p_name] += 1
            start_idx = end_idx

    # Verification checks
    if len(entity_to_partition) != len(entity_ids):
        raise RuntimeError(
            f"Entity count mismatch: expected {len(entity_ids)}, got {len(entity_to_partition)}"
        )

    # Compute partition statistics
    partition_stats: Dict[str, Dict[str, Any]] = {}
    for p_name in partition_names:
        p_entities = [eid for eid, p in entity_to_partition.items() if p == p_name]
        p_count = len(p_entities)
        
        country_dist: Dict[str, int] = {}
        card_dist: Dict[str, int] = {}
        pattern_dist: Dict[str, int] = {}
        total_matches = 0

        for eid in p_entities:
            meta = entity_metadata[eid]
            c = meta["country"]
            cb = meta["cardinality_bucket"]
            mp = meta["match_pattern"]
            mc = meta["match_count"]

            country_dist[c] = country_dist.get(c, 0) + 1
            card_dist[cb] = card_dist.get(cb, 0) + 1
            pattern_dist[mp] = pattern_dist.get(mp, 0) + 1
            total_matches += mc

        partition_stats[p_name] = {
            "entity_count": p_count,
            "target_ratio": cfg.ratios[p_name],
            "actual_ratio": p_count / len(entity_ids) if len(entity_ids) > 0 else 0.0,
            "total_ground_truth_matches": total_matches,
            "avg_matches_per_entity": total_matches / p_count if p_count > 0 else 0.0,
            "country_distribution": country_dist,
            "cardinality_distribution": card_dist,
            "match_pattern_distribution": pattern_dist,
        }

    return SplitManifest(
        entity_to_partition=entity_to_partition,
        partition_counts=partition_counts,
        partition_stats=partition_stats,
        ratios=cfg.ratios,
        random_seed=cfg.random_seed,
    )
