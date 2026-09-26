"""Context, Reverse-Rank, and Competition Feature Generator (G.5).

Extracts within-entity rank context, cross-entity reverse-rank competition,
and near-duplicate cluster consistency features for the Stage-2 re-scoring model.
"""

from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
import pandas as pd


G5_FEATURE_NAMES: List[str] = [
    "stage1_p1",
    "candidate_count_for_s1",
    "score_rank_within_s1",
    "score_percentile_within_s1",
    "score_margin_to_second",
    "score_z_within_s1",
    "n_s1_claimants",
    "reverse_rank_of_cand",
    "is_mutual_best_match",
    "max_competing_score",
    "score_lead_over_competitor",
    "cand_in_near_dup_cluster",
    "cluster_size",
    "cluster_s1_claimant_count",
    "cluster_max_score_for_s1",
    "cluster_mean_score_for_s1",
]


class ContextFeatureExtractor:
    """Computes G.5 Second-Pass context features over a batch of candidate predictions."""

    def __init__(self, cluster_map: Optional[Dict[str, str]] = None):
        """Initializes the extractor.
        
        Args:
            cluster_map: Optional mapping from cand_id to cluster_id for near-duplicate clusters.
        """
        self.cluster_map = cluster_map or {}

    def extract_context_features(
        self,
        scored_pairs: List[Tuple[str, str, float]],
    ) -> pd.DataFrame:
        """Computes G.5 features for a set of (s1_id, cand_id, p1) tuples.
        
        Args:
            scored_pairs: List of (s1_id, cand_id, stage1_probability) tuples.
            
        Returns:
            DataFrame with G.5 features and metadata columns ['s1_id', 'cand_id'].
        """
        if not scored_pairs:
            return pd.DataFrame(columns=["s1_id", "cand_id"] + G5_FEATURE_NAMES)

        df = pd.DataFrame(scored_pairs, columns=["s1_id", "cand_id", "stage1_p1"])
        df["stage1_p1"] = df["stage1_p1"].astype(np.float32)

        # 1. Within-Entity Context Features
        # Group by s1_id
        df["candidate_count_for_s1"] = df.groupby("s1_id")["cand_id"].transform("count").astype(np.int32)

        # Rank within S1 (1 = highest score)
        df["score_rank_within_s1"] = (
            df.groupby("s1_id")["stage1_p1"].rank(ascending=False, method="min").astype(np.float32)
        )

        # Percentile within S1 (1.0 = top score, 0.0 = lowest)
        df["score_percentile_within_s1"] = np.where(
            df["candidate_count_for_s1"] > 1,
            (df["candidate_count_for_s1"] - df["score_rank_within_s1"]) / (df["candidate_count_for_s1"] - 1 + 1e-9),
            1.0
        ).astype(np.float32)

        # Margin to second highest score within S1
        def calc_s1_margin(group_scores: pd.Series) -> pd.Series:
            if len(group_scores) <= 1:
                return pd.Series(group_scores.values, index=group_scores.index)
            sorted_s = np.sort(group_scores.values)[::-1]
            second_val = sorted_s[1]
            return group_scores - second_val

        df["score_margin_to_second"] = (
            df.groupby("s1_id")["stage1_p1"].transform(calc_s1_margin).astype(np.float32)
        )

        # Z-score within S1
        s1_mean = df.groupby("s1_id")["stage1_p1"].transform("mean")
        s1_std = df.groupby("s1_id")["stage1_p1"].transform("std").fillna(0.0)
        df["score_z_within_s1"] = np.where(
            s1_std > 1e-6,
            (df["stage1_p1"] - s1_mean) / s1_std,
            0.0
        ).astype(np.float32)

        # 2. Reverse-Rank / Cross-Entity Competition Features
        # Number of S1 claimants for this candidate
        df["n_s1_claimants"] = df.groupby("cand_id")["s1_id"].transform("count").astype(np.int32)

        # Reverse rank: rank of this S1 among all claimants of this cand (1 = highest)
        df["reverse_rank_of_cand"] = (
            df.groupby("cand_id")["stage1_p1"].rank(ascending=False, method="min").astype(np.float32)
        )

        # Max competing score for this cand (highest score among OTHER claimants)
        def calc_max_competing(group_scores: pd.Series) -> pd.Series:
            if len(group_scores) <= 1:
                return pd.Series(0.0, index=group_scores.index)
            sorted_s = np.sort(group_scores.values)[::-1]
            top1 = sorted_s[0]
            top2 = sorted_s[1]
            # If this row is top1, competitor is top2; otherwise competitor is top1
            res = np.where(group_scores.values == top1, top2, top1)
            return pd.Series(res, index=group_scores.index)

        df["max_competing_score"] = (
            df.groupby("cand_id")["stage1_p1"].transform(calc_max_competing).astype(np.float32)
        )

        df["score_lead_over_competitor"] = (
            df["stage1_p1"] - df["max_competing_score"]
        ).astype(np.float32)

        # Mutual best match: rank within S1 == 1 and reverse rank == 1
        df["is_mutual_best_match"] = (
            (df["score_rank_within_s1"] == 1.0) & (df["reverse_rank_of_cand"] == 1.0)
        ).astype(np.float32)

        # 3. Near-Duplicate Cluster Consistency Features
        df["cluster_id"] = df["cand_id"].map(self.cluster_map).fillna(df["cand_id"])
        has_cluster = df["cand_id"].isin(self.cluster_map)
        df["cand_in_near_dup_cluster"] = has_cluster.astype(np.float32)

        # Cluster size
        cluster_sizes = df.groupby("cluster_id")["cand_id"].transform("nunique")
        df["cluster_size"] = np.where(has_cluster, cluster_sizes, 1).astype(np.int32)

        # Cluster distinct S1 claimants
        cluster_s1_counts = df.groupby("cluster_id")["s1_id"].transform("nunique")
        df["cluster_s1_claimant_count"] = cluster_s1_counts.astype(np.int32)

        # Cluster max score for this S1
        df["cluster_max_score_for_s1"] = (
            df.groupby(["s1_id", "cluster_id"])["stage1_p1"].transform("max").astype(np.float32)
        )

        # Cluster mean score for this S1
        df["cluster_mean_score_for_s1"] = (
            df.groupby(["s1_id", "cluster_id"])["stage1_p1"].transform("mean").astype(np.float32)
        )

        # Reorder and format columns
        feature_cols = ["s1_id", "cand_id"] + G5_FEATURE_NAMES
        return df[feature_cols]
