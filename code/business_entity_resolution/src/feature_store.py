"""Feature store and batch matrix construction module.

Handles memory-efficient materialization of pairwise feature matrices and labels
in chunked numpy arrays.
"""

import os
import concurrent.futures
from typing import Dict, List, Any, Tuple, Optional, Set, Iterator
import numpy as np
import pandas as pd

from src.pair_features import FEATURE_NAMES, compute_single_pair_features


class FeatureBatch:
    """Represents a materialized feature matrix batch with metadata and labels."""

    def __init__(
        self,
        pair_ids: List[Tuple[str, str]],
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str] = FEATURE_NAMES,
    ):
        self.pair_ids = pair_ids
        self.features = features  # np.float32 array (N, num_features)
        self.labels = labels      # np.int8 array (N,)
        self.feature_names = feature_names

    @property
    def num_pairs(self) -> int:
        return len(self.labels)

    @property
    def num_positives(self) -> int:
        return int(np.sum(self.labels))

    @property
    def pos_rate(self) -> float:
        return float(np.mean(self.labels)) if len(self.labels) > 0 else 0.0

    @property
    def num_features(self) -> int:
        return len(self.feature_names)

    @property
    def memory_mb(self) -> float:
        return (self.features.nbytes + self.labels.nbytes) / (1024 * 1024)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert batch to pandas DataFrame (useful for inspections / audits)."""
        df = pd.DataFrame(self.features, columns=self.feature_names)
        df["s1_entity_id"] = [p[0] for p in self.pair_ids]
        df["cand_entity_id"] = [p[1] for p in self.pair_ids]
        df["label"] = self.labels
        return df


class FeatureExtractor:
    """Builds pairwise features for candidate pairs."""

    def __init__(
        self,
        s1_lookup: Dict[str, Dict[str, Any]],
        cand_lookup: Dict[str, Dict[str, Any]],
        ground_truth: Optional[Dict[str, Set[str]]] = None,
    ):
        """
        Args:
            s1_lookup: Map of entity_id -> Dict of S1 record attributes.
            cand_lookup: Map of entity_id -> Dict of S2/S3 candidate record attributes.
            ground_truth: Map of s1_entity_id -> set of true matching cand_entity_ids (for labels).
        """
        self.s1_lookup = s1_lookup
        self.cand_lookup = cand_lookup
        self.ground_truth = ground_truth or {}
        self.feature_names = FEATURE_NAMES

    def extract_pair_batch(
        self,
        pairs: List[Tuple[str, str]],
        channel_counts: Optional[Dict[Tuple[str, str], int]] = None,
    ) -> FeatureBatch:
        """Extract features and labels for a list of (s1_id, cand_id) tuples.

        Args:
            pairs: List of (s1_id, cand_id) pairs.
            channel_counts: Optional map of (s1_id, cand_id) -> int channel overlap count.

        Returns:
            FeatureBatch object containing features, labels, and pair IDs.
        """
        n_pairs = len(pairs)
        n_feats = len(self.feature_names)

        feat_matrix = np.zeros((n_pairs, n_feats), dtype=np.float32)
        label_vector = np.zeros(n_pairs, dtype=np.int8)

        # Ground truth labels (if ground_truth provided)
        has_gt = bool(self.ground_truth)

        # Threaded worker chunk function
        def process_sub_batch(start_i: int, end_i: int) -> None:
            for i in range(start_i, end_i):
                s1_id, cand_id = pairs[i]
                s1_row = self.s1_lookup.get(s1_id, {})
                cand_row = self.cand_lookup.get(cand_id, {})
                ch_cnt = 1
                if channel_counts:
                    ch_cnt = channel_counts.get((s1_id, cand_id), 1)

                feat_vec = compute_single_pair_features(s1_row, cand_row, channel_count=ch_cnt)
                feat_matrix[i, :] = feat_vec

                if has_gt:
                    label_vector[i] = 1 if (cand_id in self.ground_truth.get(s1_id, set())) else 0

        # Execute in parallel if n_pairs > 10,000
        if n_pairs >= 10000:
            import concurrent.futures
            n_threads = min(8, max(2, os.cpu_count() or 4))
            chunk_step = int(np.ceil(n_pairs / n_threads))
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
                for t_idx in range(n_threads):
                    s_i = t_idx * chunk_step
                    e_i = min(s_i + chunk_step, n_pairs)
                    if s_i < e_i:
                        futures.append(executor.submit(process_sub_batch, s_i, e_i))
                concurrent.futures.wait(futures)
        else:
            process_sub_batch(0, n_pairs)

        return FeatureBatch(
            pair_ids=pairs,
            features=feat_matrix,
            labels=label_vector,
            feature_names=self.feature_names,
        )

    def extract_candidate_dict_in_chunks(
        self,
        candidate_dict: Dict[str, List[str]],
        chunk_size: int = 50000,
    ) -> Iterator[FeatureBatch]:
        """Generate candidate pairs from candidate dict and yield feature batches.

        Args:
            candidate_dict: Map of s1_id -> list of candidate ids.
            chunk_size: Max number of pairs per batch.

        Yields:
            FeatureBatch objects.
        """
        current_chunk: List[Tuple[str, str]] = []
        for s1_id, cands in candidate_dict.items():
            for cand_id in cands:
                current_chunk.append((s1_id, cand_id))
                if len(current_chunk) >= chunk_size:
                    yield self.extract_pair_batch(current_chunk)
                    current_chunk = []

        if current_chunk:
            yield self.extract_pair_batch(current_chunk)
