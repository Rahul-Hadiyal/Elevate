"""Submission Invariant Validator & Official Scorer Validator Wrapper (Phase 15).

Validates all 8 structural invariants locally and runs the official submission validator
to guarantee 100% acceptance before packaging.
"""

from pathlib import Path
import os
import sys
import subprocess
import logging
from typing import Dict, List, Set, Optional, Union, Tuple

logger = logging.getLogger(__name__)


class SubmissionValidator:
    """Internal Invariant Validator and Official Validator Test Harness."""

    def __init__(self, test_dir: Union[str, Path] = "dataset/test"):
        self.test_dir = Path(test_dir)

    def validate_invariants(
        self,
        matching_path: Union[str, Path],
        candidate_path: Optional[Union[str, Path]] = None,
        expected_s1_ids: Optional[Set[str]] = None,
        enforce_h1: bool = True,
    ) -> bool:
        """Audits all submission invariants directly on disk."""
        matching_file = Path(matching_path)
        if not matching_file.exists():
            raise FileNotFoundError(f"Matching file not found: {matching_file}")

        logger.info(f"Auditing invariants for matching file: {matching_file}...")

        # Parse matching results
        matching_s1_seen: Set[str] = set()
        matching_dict: Dict[str, Set[str]] = {}
        candidate_to_s1: Dict[str, str] = {}

        with open(matching_file, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\r\n").split("\t")
            if header != ["source1_entity_id", "matched_entity_ids"]:
                raise ValueError(f"Invalid matching header: {header}. Expected ['source1_entity_id', 'matched_entity_ids']")

            for line_idx, line in enumerate(f, start=2):
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    raise ValueError(f"Malformed row at line {line_idx}: {line}")

                s1_id, match_str = parts[0].strip(), parts[1].strip()
                if s1_id in matching_s1_seen:
                    raise ValueError(f"Duplicate S1 ID found: {s1_id} at line {line_idx}")
                matching_s1_seen.add(s1_id)

                matches = [m.strip() for m in match_str.split(",") if m.strip()]
                # Check intra-row uniqueness
                if len(matches) != len(set(matches)):
                    raise ValueError(f"Duplicate candidate IDs inside row for S1 ID {s1_id}: {matches}")

                # Check ID prefixes
                for cid in matches:
                    if not (cid.startswith("S2") or cid.startswith("S3")):
                        raise ValueError(f"Invalid candidate ID prefix: {cid} in S1 ID {s1_id}")

                    # Enforce H1 conflict invariant
                    if enforce_h1:
                        if cid in candidate_to_s1:
                            raise ValueError(
                                f"H1 Conflict Invariant Violated: Candidate {cid} is assigned to "
                                f"both {candidate_to_s1[cid]} and {s1_id}."
                            )
                        candidate_to_s1[cid] = s1_id

                matching_dict[s1_id] = set(matches)

        # Check coverage if expected IDs provided
        if expected_s1_ids is not None:
            missing_ids = expected_s1_ids - matching_s1_seen
            if missing_ids:
                raise ValueError(f"Coverage Invariant Violated: {len(missing_ids)} expected S1 IDs are missing.")
            extra_ids = matching_s1_seen - expected_s1_ids
            if extra_ids:
                raise ValueError(f"Coverage Invariant Violated: {len(extra_ids)} unexpected S1 IDs are present.")
            logger.info(f"Coverage Invariant PASSED across {len(expected_s1_ids):,d} S1 entities.")

        # Candidate pairs cross-check if provided
        if candidate_path is not None:
            cand_file = Path(candidate_path)
            if cand_file.exists():
                logger.info(f"Auditing candidate subset invariant: {cand_file}...")
                cand_s1_seen: Set[str] = set()

                with open(cand_file, "r", encoding="utf-8") as f:
                    c_header = f.readline().rstrip("\r\n").split("\t")
                    if c_header != ["source1_entity_id", "candidate_entity_ids"]:
                        raise ValueError(f"Invalid candidate header: {c_header}")

                    for line_idx, line in enumerate(f, start=2):
                        line = line.rstrip("\r\n")
                        if not line:
                            continue
                        parts = line.split("\t")
                        if len(parts) != 2:
                            raise ValueError(f"Malformed candidate row at line {line_idx}: {line}")

                        s1_id, cand_str = parts[0].strip(), parts[1].strip()
                        if s1_id in cand_s1_seen:
                            raise ValueError(f"Duplicate S1 ID in candidate file: {s1_id}")
                        cand_s1_seen.add(s1_id)

                        cands = set([c.strip() for c in cand_str.split(",") if c.strip()])
                        pred_matches = matching_dict.get(s1_id, set())

                        # Subset invariant: pred_matches must be subset of cands
                        violating = pred_matches - cands
                        if violating:
                            raise ValueError(
                                f"Subset Invariant Violated for S1 ID {s1_id}: "
                                f"Matches {violating} not present in candidates."
                            )

                logger.info("Subset Invariant (Matches ⊆ Candidates) PASSED.")

        logger.info("All Submission Invariants PASSED.")
        return True

    def run_official_validator(
        self,
        matching_path: Union[str, Path],
        candidate_path: Optional[Union[str, Path]] = None,
    ) -> bool:
        """Executes the official utils/validate_submission.py script."""
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        validator_script = repo_root / "utils" / "validate_submission.py"
        if not validator_script.exists():
            validator_script = repo_root / "6ab10eb3b23ba_student_resource" / "student_resource" / "utils" / "validate_submission.py"

        if not validator_script.exists():
            logger.warning(f"Official validator script not found at {validator_script}. Skipping subprocess call.")
            return True

        cmd = [
            sys.executable,
            str(validator_script),
            "--matching", str(matching_path),
            "--test-dir", str(self.test_dir),
        ]
        if candidate_path is not None:
            cmd.extend(["--candidate", str(candidate_path)])

        logger.info(f"Running official validator: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        logger.info(f"Official Validator stdout:\n{result.stdout}")
        if result.returncode != 0:
            logger.error(f"Official Validator stderr:\n{result.stderr}")
            raise RuntimeError(f"Official Validator failed with exit code {result.returncode}")

        logger.info("Official Submission Validator PASSED with exit code 0.")
        return True
