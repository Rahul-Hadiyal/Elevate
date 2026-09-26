"""Unit Tests for Output Generator and Submission Validator (Phases 14 & 15)."""

import pytest
from pathlib import Path
import tempfile

from src.output_generator import OutputGenerator
from src.validator import SubmissionValidator


def test_output_generator_and_validator(tmp_path):
    """Tests writing matching and candidate TSVs and verifying all invariants."""
    output_dir = tmp_path / "output"
    generator = OutputGenerator(output_dir=output_dir)

    all_s1_ids = ["S1-001", "S1-002", "S1-003", "S1-004"]
    candidate_dict = {
        "S1-001": ["S2-101", "S3-201"],
        "S1-002": ["S2-102"],
        "S1-003": ["S3-203", "S2-103"],
        "S1-004": [],
    }
    matching_dict = {
        "S1-001": {"S2-101"},
        "S1-002": {"S2-102"},
        "S1-003": set(),
        "S1-004": set(),
    }

    match_file, cand_file = generator.write_submission_files(
        all_s1_ids=all_s1_ids,
        candidate_dict=candidate_dict,
        matching_dict=matching_dict,
    )

    assert match_file.exists()
    assert cand_file.exists()

    validator = SubmissionValidator(test_dir=tmp_path)
    is_valid = validator.validate_invariants(
        matching_path=match_file,
        candidate_path=cand_file,
        expected_s1_ids=set(all_s1_ids),
        enforce_h1=True,
    )
    assert is_valid is True


def test_validator_detects_subset_violation(tmp_path):
    """Verifies validator flags an error if a match is not in candidates."""
    output_dir = tmp_path / "output_bad"
    generator = OutputGenerator(output_dir=output_dir)

    all_s1_ids = ["S1-001"]
    candidate_dict = {"S1-001": ["S2-101"]}
    matching_dict = {"S1-001": ["S2-999"]}  # S2-999 not in candidate_dict

    match_file, cand_file = generator.write_submission_files(
        all_s1_ids=all_s1_ids,
        candidate_dict=candidate_dict,
        matching_dict=matching_dict,
    )

    validator = SubmissionValidator()
    with pytest.raises(ValueError, match="Subset Invariant Violated"):
        validator.validate_invariants(
            matching_path=match_file,
            candidate_path=cand_file,
            expected_s1_ids=set(all_s1_ids),
            enforce_h1=True,
        )


def test_validator_detects_h1_collision(tmp_path):
    """Verifies validator flags an error if same candidate is matched to multiple S1s."""
    output_dir = tmp_path / "output_h1"
    generator = OutputGenerator(output_dir=output_dir)

    all_s1_ids = ["S1-001", "S1-002"]
    candidate_dict = {
        "S1-001": ["S2-101"],
        "S1-002": ["S2-101"],
    }
    matching_dict = {
        "S1-001": ["S2-101"],
        "S1-002": ["S2-101"],  # S2-101 assigned to both
    }

    match_file, cand_file = generator.write_submission_files(
        all_s1_ids=all_s1_ids,
        candidate_dict=candidate_dict,
        matching_dict=matching_dict,
    )

    validator = SubmissionValidator()
    with pytest.raises(ValueError, match="H1 Conflict Invariant Violated"):
        validator.validate_invariants(
            matching_path=match_file,
            candidate_path=cand_file,
            expected_s1_ids=set(all_s1_ids),
            enforce_h1=True,
        )
