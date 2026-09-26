"""Hard negative analysis and categorization module for Business Entity Resolution.

Identifies challenging negative candidate pairs to measure discriminative power
and ensure that matching models are trained on representative hard cases.
"""

from typing import Dict, List, Any, Set, Tuple
import numpy as np


class HardNegativeAnalyzer:
    """Categorizes and audits hard negative candidate pairs."""

    @staticmethod
    def categorize_negative(
        name_fuzz_tsort: float,
        addr_num_exact_match: float,
        country_exact_match: float,
        s1_c: str,
    ) -> List[str]:
        """Classify a negative pair into hard-negative categories.

        Categories:
        - High Name Similarity (Name token sort >= 80)
        - Same Address Number Collision (Addr num match = 1.0)
        - Same Country Name Collision
        - Dual High Signal (Name >= 80 and Num match = 1.0)
        """
        categories = []
        if name_fuzz_tsort >= 80.0:
            categories.append("high_name_similarity")
        if addr_num_exact_match == 1.0:
            categories.append("same_addr_number_collision")
        if country_exact_match == 1.0 and name_fuzz_tsort >= 70.0:
            categories.append("same_country_similar_name")
        if name_fuzz_tsort >= 80.0 and addr_num_exact_match == 1.0:
            categories.append("dual_high_signal_hard_neg")

        if not categories:
            categories.append("standard_negative")

        return categories

    @staticmethod
    def compute_hard_negative_summary(
        features_matrix: np.ndarray,
        labels_vector: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """Compute statistical breakdown of hard negatives in a dataset.

        Args:
            features_matrix: 2D array of shape (N, num_features).
            labels_vector: 1D array of shape (N,) with binary labels (0/1).
            feature_names: List of feature names corresponding to matrix columns.

        Returns:
            Dictionary with counts and proportions of hard negative categories.
        """
        if len(labels_vector) == 0:
            return {"total_negatives": 0}

        name_tsort_idx = feature_names.index("feat_name_fuzz_token_sort")
        num_match_idx = feature_names.index("feat_addr_num_exact_match")
        country_match_idx = feature_names.index("feat_country_exact_match")

        neg_mask = (labels_vector == 0)
        total_negatives = int(np.sum(neg_mask))
        total_positives = int(len(labels_vector) - total_negatives)

        if total_negatives == 0:
            return {
                "total_pairs": len(labels_vector),
                "total_positives": total_positives,
                "total_negatives": 0,
                "hard_negative_breakdown": {},
            }

        neg_tsort = features_matrix[neg_mask, name_tsort_idx]
        neg_num_match = features_matrix[neg_mask, num_match_idx]
        neg_country_match = features_matrix[neg_mask, country_match_idx]

        high_name_count = int(np.sum(neg_tsort >= 80.0))
        num_match_count = int(np.sum(neg_num_match == 1.0))
        country_sim_count = int(np.sum((neg_country_match == 1.0) & (neg_tsort >= 70.0)))
        dual_high_count = int(np.sum((neg_tsort >= 80.0) & (neg_num_match == 1.0)))

        return {
            "total_pairs": int(len(labels_vector)),
            "total_positives": total_positives,
            "total_negatives": total_negatives,
            "positive_rate": float(total_positives / len(labels_vector)) if len(labels_vector) > 0 else 0.0,
            "hard_negative_counts": {
                "high_name_similarity_gte80": high_name_count,
                "same_addr_number_collision": num_match_count,
                "same_country_similar_name_gte70": country_sim_count,
                "dual_high_signal_hard_neg": dual_high_count,
            },
            "hard_negative_proportions_of_negatives": {
                "high_name_similarity_gte80": float(high_name_count / total_negatives),
                "same_addr_number_collision": float(num_match_count / total_negatives),
                "same_country_similar_name_gte70": float(country_sim_count / total_negatives),
                "dual_high_signal_hard_neg": float(dual_high_count / total_negatives),
            },
        }
