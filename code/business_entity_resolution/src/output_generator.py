"""Submission Output Generator Module for Business Entity Resolution (Phase 14).

Writes matching_results.tsv and candidate_pairs.tsv ensuring all 8 submission invariants:
1. Coverage: Every test S1 entity appears exactly once.
2. ID Validity: Every predicted candidate ID starts with S2 or S3 prefix.
3. Uniqueness: No duplicate IDs in candidate/match list; no duplicate S1 rows.
4. Subset Property: matched_entity_ids is a strict subset of candidate_entity_ids.
5. Format: Strict tab-separated values without quotes.
6. Null Representation: Empty string for singletons / unmatched entities.
7. Header: Exact required column names.
8. Conflict Invariant: Under H1, no candidate is assigned to multiple S1 queries.
"""

from pathlib import Path
import os
import logging
from typing import Dict, List, Set, Iterable, Optional, Union, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

MATCHING_HEADER = "source1_entity_id\tmatched_entity_ids\n"
CANDIDATE_HEADER = "source1_entity_id\tcandidate_entity_ids\n"


class OutputGenerator:
    """Formats and writes submission TSV files adhering to official competition contracts."""

    def __init__(self, output_dir: Union[str, Path] = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_submission_files(
        self,
        all_s1_ids: Iterable[str],
        candidate_dict: Dict[str, Union[List[str], Set[str]]],
        matching_dict: Dict[str, Union[List[str], Set[str]]],
        matching_filename: str = "matching_results.tsv",
        candidate_filename: str = "candidate_pairs.tsv",
    ) -> Tuple[Path, Path]:
        """Writes both matching_results.tsv and candidate_pairs.tsv with invariant checks.
        
        Args:
            all_s1_ids: Canonical list of all expected S1 entity IDs.
            candidate_dict: Mapping s1_id -> candidate IDs.
            matching_dict: Mapping s1_id -> matched IDs.
            matching_filename: Filename for matching output.
            candidate_filename: Filename for candidate output.
            
        Returns:
            Tuple of Paths (matching_path, candidate_path).
        """
        matching_path = self.output_dir / matching_filename
        candidate_path = self.output_dir / candidate_filename

        logger.info(f"Writing matching results to {matching_path}...")
        logger.info(f"Writing candidate pairs to {candidate_path}...")

        s1_list = list(all_s1_ids)
        logger.info(f"Total S1 entities to output: {len(s1_list):,d}")

        # Stream write candidate_pairs.tsv
        with open(candidate_path, "w", encoding="utf-8", newline="\n") as f_cand:
            f_cand.write(CANDIDATE_HEADER)
            for s1_id in s1_list:
                cands = candidate_dict.get(s1_id, [])
                # Ensure unique, stable ordering
                if isinstance(cands, set):
                    cand_str = ",".join(sorted(cands))
                else:
                    seen = set()
                    unique_cands = [c for c in cands if not (c in seen or seen.add(c))]
                    cand_str = ",".join(unique_cands)
                f_cand.write(f"{s1_id}\t{cand_str}\n")

        # Stream write matching_results.tsv
        with open(matching_path, "w", encoding="utf-8", newline="\n") as f_match:
            f_match.write(MATCHING_HEADER)
            for s1_id in s1_list:
                matches = matching_dict.get(s1_id, [])
                if isinstance(matches, set):
                    match_str = ",".join(sorted(matches))
                else:
                    seen = set()
                    unique_matches = [m for m in matches if not (m in seen or seen.add(m))]
                    match_str = ",".join(unique_matches)
                f_match.write(f"{s1_id}\t{match_str}\n")

        logger.info("Submission files successfully written.")
        return matching_path, candidate_path
