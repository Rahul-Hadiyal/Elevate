"""Exhaustive unit tests for normalizer.py and vocab_builder.py.

Includes anti-over-normalization adversarial tests, French localization,
landmark extraction, and degenerate record detection.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.normalizer import EntityNormalizer, strip_accents
from src.vocab_builder import build_and_save_vocabularies


@pytest.fixture
def normalizer():
    """Provides a standard normalizer instance."""
    return EntityNormalizer()


def test_strip_accents():
    """Verify French and European diacritic stripping."""
    assert strip_accents("Café de Paris") == "Cafe de Paris"
    assert strip_accents("Société Générale") == "Societe Generale"
    assert strip_accents("Crème Brûlée") == "Creme Brulee"
    assert strip_accents("NAÏVE") == "NAIVE"


def test_anti_over_normalization_names(normalizer: EntityNormalizer):
    """Adversarial tests ensuring distinct names are NOT collapsed into false matches."""
    name1, _, _, _ = normalizer.normalize_name("ABC Pvt Ltd")
    name2, _, _, _ = normalizer.normalize_name("ABC")
    name3, _, _, _ = normalizer.normalize_name("ABC Industries")

    assert name1 == "abc private limited"
    assert name2 == "abc"
    assert name3 == "abc industries"

    # Distinct names must remain distinct
    assert name1 != name2
    assert name1 != name3
    assert name2 != name3


def test_anti_over_normalization_numbers(normalizer: EntityNormalizer):
    """Adversarial tests ensuring address numbers are strictly preserved and differentiated."""
    # 1 Main St vs 11 Main St
    _, _, _, nums1, _, _ = normalizer.normalize_address("1 Main Street")
    _, _, _, nums2, _, _ = normalizer.normalize_address("11 Main Street")
    assert nums1 == ["1"]
    assert nums2 == ["11"]
    assert nums1 != nums2

    # 12A Main St vs 12 Main St
    _, _, _, nums_12a, _, _ = normalizer.normalize_address("12A Main Street")
    _, _, _, nums_12, _, _ = normalizer.normalize_address("12 Main Street")
    assert nums_12a == ["12a"]
    assert nums_12 == ["12"]
    assert nums_12a != nums_12

    # Unit 2 vs Unit 20
    _, _, _, nums_u2, _, _ = normalizer.normalize_address("Suite 2, 500 Park Ave")
    _, _, _, nums_u20, _, _ = normalizer.normalize_address("Suite 20, 500 Park Ave")
    assert "2" in nums_u2
    assert "20" in nums_u20
    assert nums_u2 != nums_u20


def test_french_normalization(normalizer: EntityNormalizer):
    """Test normalization on French legal entities and addresses."""
    # French corporate suffixes
    name_norm, tokens, _, deg = normalizer.normalize_name("SARL Dupont & Fils")
    assert name_norm == "sarl dupont and fils"
    assert "sarl" in tokens
    assert deg is False

    name_norm2, _, _, _ = normalizer.normalize_name("Boulangerie de la Tour S.A.S.")
    assert name_norm2 == "boulangerie de la tour sas"

    # French address abbreviations: bd -> boulevard, r -> rue
    addr_norm, core, landmark, nums, _, _ = normalizer.normalize_address("15 Bd Saint-Germain, Pres de la Seine, 75005 Paris")
    assert "boulevard" in addr_norm
    assert "15" in nums
    assert "75005" in nums
    assert "pres de" in landmark.lower()


def test_landmark_extraction(normalizer: EntityNormalizer):
    """Test address landmark component separation."""
    addr_norm, core, landmark, nums, tokens, deg = normalizer.normalize_address(
        "Plot 42 MG Road, Opp State Bank of India, Koramangala, Bangalore 560034"
    )
    assert deg is False
    assert "opp state bank of india" in landmark.lower()
    assert "42" in nums
    assert "560034" in nums
    assert "42 mg road" in core.lower()


def test_degenerate_records(normalizer: EntityNormalizer):
    """Verify degenerate / empty record detection."""
    # Empty string
    _, _, _, deg_empty = normalizer.normalize_name("")
    assert deg_empty is True

    # Whitespace only
    _, _, _, deg_ws = normalizer.normalize_name("   \t  ")
    assert deg_ws is True

    # Only legal suffix tokens (e.g. 'Pvt Ltd' alone with no business title)
    _, _, _, deg_suffix = normalizer.normalize_name("Pvt. Ltd.")
    assert deg_suffix is True

    _, _, _, deg_single_char = normalizer.normalize_name("X")
    assert deg_single_char is True

    # Valid name should NOT be degenerate
    _, _, _, deg_valid = normalizer.normalize_name("Reliance Retail")
    assert deg_valid is False

    # Degenerate address
    _, _, _, _, _, deg_addr_empty = normalizer.normalize_address("  ")
    assert deg_addr_empty is True


def test_punctuation_and_symbols(normalizer: EntityNormalizer):
    """Test symbol normalization (& -> and, @ -> at, compound hyphen preservation)."""
    name_norm, _, _, _ = normalizer.normalize_name("Barnes & Noble, Inc.")
    assert name_norm == "barnes and noble incorporated"

    name_norm2, _, _, _ = normalizer.normalize_name("Wal-Mart Stores")
    assert "wal-mart" in name_norm2


def test_country_normalization(normalizer: EntityNormalizer):
    """Test open-set country normalization."""
    assert normalizer.normalize_country("US") == "us"
    assert normalizer.normalize_country("United States") == "us"
    assert normalizer.normalize_country("India") == "india"
    assert normalizer.normalize_country("France") == "france"
    assert normalizer.normalize_country("FR") == "france"
    assert normalizer.normalize_country("  ") == "unknown"
    assert normalizer.normalize_country(None) == "unknown"
    # New unseen country passes through gracefully
    assert normalizer.normalize_country("Germany") == "germany"


def test_normalize_dataframe_batch(normalizer: EntityNormalizer):
    """Test batch DataFrame normalization."""
    df = pd.DataFrame([
        {
            "entity_id": "S1-1",
            "business_name": "Apollo Hospitals Enterprise Ltd.",
            "business_address": "21 Greams Rd, Near Thousand Lights, Chennai 600006",
            "country": "India",
        },
        {
            "entity_id": "S1-2",
            "business_name": "Café de Flore SAS",
            "business_address": "172 Bd Saint-Germain, 75006 Paris",
            "country": "France",
        },
    ])

    norm_df = normalizer.normalize_dataframe(df)

    assert "name_norm" in norm_df.columns
    assert "addr_norm" in norm_df.columns
    assert "addr_core" in norm_df.columns
    assert "addr_landmark" in norm_df.columns
    assert "country_norm" in norm_df.columns

    assert norm_df.loc[0, "name_norm"] == "apollo hospitals enterprise limited"
    assert norm_df.loc[0, "country_norm"] == "india"
    assert "near thousand lights" in norm_df.loc[0, "addr_landmark"].lower()
    assert "600006" in norm_df.loc[0, "addr_numbers"]

    assert norm_df.loc[1, "name_norm"] == "cafe de flore sas"
    assert norm_df.loc[1, "country_norm"] == "france"
    assert "75006" in norm_df.loc[1, "addr_numbers"]
