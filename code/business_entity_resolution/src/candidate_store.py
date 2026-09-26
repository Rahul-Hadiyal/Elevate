"""Compact Candidate Store and Multi-Channel Union Manager.

Provides memory-efficient management of candidate pairs across multiple blocking channels:
- Union and deduplication of candidates per S1 entity.
- Channel provenance tracking (which channel(s) generated each candidate pair).
- Configurable per-S1 candidate capping with deterministic retention.
- Sparse diagnostic matrices (channel intersection & exclusivity).
- Export to compact TSV and DataFrame representations.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import json
import logging
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class CandidateStore:
    """Memory-efficient candidate store with multi-channel union and provenance tracking."""

    def __init__(self, s1_entity_ids: Optional[Iterable[str]] = None):
        """Initializes candidate store.
        
        Args:
            s1_entity_ids: Optional collection of S1 IDs to pre-populate.
        """
        # Mapping: s1_id -> dict(candidate_id -> bitmask / set of channel names)
        self._store: Dict[str, Dict[str, int]] = {}
        self._channel_registry: List[str] = []
        self._channel_bit_map: Dict[str, int] = {}

        if s1_entity_ids is not None:
            for s1_id in s1_entity_ids:
                self._store[str(s1_id).strip()] = {}


    @property
    def total_s1_entities(self) -> int:
        return len(self._store)

    @property
    def total_candidate_pairs(self) -> int:
        return sum(len(cands) for cands in self._store.values())

    @property
    def registered_channels(self) -> List[str]:
        return list(self._channel_registry)

    def _get_or_register_channel_bit(self, channel_name: str) -> int:
        """Registers a channel name and returns its assigned bitmask flag."""
        if channel_name not in self._channel_bit_map:
            bit_idx = len(self._channel_registry)
            if bit_idx >= 64:
                raise ValueError("Maximum 64 concurrent blocking channels supported.")
            bitmask = 1 << bit_idx
            self._channel_registry.append(channel_name)
            self._channel_bit_map[channel_name] = bitmask
        return self._channel_bit_map[channel_name]

    def add_channel_candidates(
        self,
        channel_name: str,
        candidates_dict: Dict[str, Union[Set[str], List[str]]],
    ) -> int:
        """Merges candidate pairs from a single blocking channel.

        Args:
            channel_name: Unique identifier for the blocking channel (e.g. 'channel_A').
            candidates_dict: Mapping from s1_id -> collection of candidate IDs.

        Returns:
            Number of new unique candidate pairs added by this channel.
        """
        channel_bit = self._get_or_register_channel_bit(channel_name)
        new_pairs_added = 0

        for s1_id, cand_list in candidates_dict.items():
            s1_key = str(s1_id).strip()
            if s1_key not in self._store:
                self._store[s1_key] = {}

            s1_cands = self._store[s1_key]
            for cand in cand_list:
                cand_key = str(cand).strip()
                if not cand_key:
                    continue
                if cand_key not in s1_cands:
                    s1_cands[cand_key] = channel_bit
                    new_pairs_added += 1
                else:
                    s1_cands[cand_key] |= channel_bit

        return new_pairs_added

    def get_candidates(self, s1_id: str) -> List[str]:
        """Returns list of all unique candidate IDs for an S1 entity."""
        s1_key = str(s1_id).strip()
        return list(self._store.get(s1_key, {}).keys())

    def get_candidate_dict(self, cap: Optional[int] = None) -> Dict[str, List[str]]:
        """Returns dictionary mapping s1_id -> candidate IDs, optionally capped and ranked by channel overlap."""
        result: Dict[str, List[str]] = {}
        for s1_id, cands in self._store.items():
            if cap is not None and len(cands) > cap:
                sorted_cands = sorted(cands.keys(), key=lambda c: int(cands[c]).bit_count(), reverse=True)
                result[s1_id] = sorted_cands[:cap]
            else:
                result[s1_id] = list(cands.keys())
        return result

    def get_channel_counts(self) -> Dict[Tuple[str, str], int]:
        """Returns mapping of (s1_id, cand_id) -> number of channels that produced this candidate."""
        counts: Dict[Tuple[str, str], int] = {}
        for s1_id, cands in self._store.items():
            for cand_id, mask in cands.items():
                counts[(s1_id, cand_id)] = int(mask).bit_count()
        return counts

    def get_channel_exclusive_candidates(self, channel_name: str) -> Dict[str, List[str]]:
        """Returns candidate pairs discovered exclusively by the specified channel."""
        if channel_name not in self._channel_bit_map:
            return {}
        channel_bit = self._channel_bit_map[channel_name]
        exclusive: Dict[str, List[str]] = {}
        for s1_id, cands in self._store.items():
            excl_for_s1 = [
                cand for cand, mask in cands.items() if mask == channel_bit
            ]
            if excl_for_s1:
                exclusive[s1_id] = excl_for_s1
        return exclusive

    def get_channel_overlap_matrix(self) -> pd.DataFrame:
        """Computes pairwise candidate overlap counts between all registered channels."""
        channels = self._channel_registry
        n_ch = len(channels)
        matrix = np.zeros((n_ch, n_ch), dtype=np.int64)

        for s1_id, cands in self._store.values():
            for cand, mask in cands.items():
                active_indices = [
                    i for i, ch in enumerate(channels)
                    if (mask & self._channel_bit_map[ch])
                ]
                for i in active_indices:
                    for j in active_indices:
                        matrix[i, j] += 1

        return pd.DataFrame(matrix, index=channels, columns=channels)

    def to_dataframe(self, cap: Optional[int] = None) -> pd.DataFrame:
        """Flattens candidate store into DataFrame of (source1_entity_id, candidate_entity_id, channel_mask)."""
        s1_list = []
        cand_list = []
        mask_list = []

        for s1_id, cands in self._store.items():
            items = list(cands.items())
            if cap is not None and len(items) > cap:
                items = items[:cap]
            for cand, mask in items:
                s1_list.append(s1_id)
                cand_list.append(cand)
                mask_list.append(mask)

        return pd.DataFrame({
            "source1_entity_id": s1_list,
            "candidate_entity_id": cand_list,
            "channel_mask": mask_list,
        })
