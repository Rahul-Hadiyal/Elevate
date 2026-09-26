"""Data Ingestion and Contract Validation Module.

Enforces strict schemas, TSV parsing, ID prefix validation, and integrity checks
for Source 1, Source 2, Source 3, and ground truth files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging
import pandas as pd


logger = logging.getLogger(__name__)

# Expected column schemas
ENTITY_COLUMNS: Tuple[str, ...] = (
    "entity_id",
    "business_name",
    "business_address",
    "country",
)
GROUND_TRUTH_COLUMNS: Tuple[str, ...] = (
    "source1_entity_id",
    "matched_entity_ids",
)

VALID_SOURCE_PREFIXES: Dict[str, str] = {
    "S1": "S1-",
    "S2": "S2-",
    "S3": "S3-",
}


class DataValidationError(Exception):
    """Raised when critical validation contracts are violated."""
    pass


class SchemaMismatchError(DataValidationError):
    """Raised when table columns do not match expected schema."""
    pass


class DuplicateIdError(DataValidationError):
    """Raised when duplicate entity_ids are detected within a source."""
    pass


class InvalidIdPrefixError(DataValidationError):
    """Raised when an entity ID prefix does not match its source."""
    pass


@dataclass
class ValidationReport:
    """Structured report containing validation outcomes, errors, warnings, and stats."""
    file_path: str
    source_tag: str
    is_valid: bool = True
    row_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


def validate_entity_table(
    df: pd.DataFrame,
    source_tag: str,
    file_path: Union[str, Path] = "in-memory",
    strict: bool = True,
) -> ValidationReport:
    """Validates an entity DataFrame (Source 1, 2, or 3) against strict schema rules.
    
    Args:
        df: DataFrame loaded from TSV.
        source_tag: One of 'S1', 'S2', 'S3'.
        file_path: File path for reporting.
        strict: If True, raises exceptions on fatal errors.
        
    Returns:
        ValidationReport with details.
    """
    report = ValidationReport(file_path=str(file_path), source_tag=source_tag)
    report.row_count = len(df)

    if source_tag not in VALID_SOURCE_PREFIXES:
        msg = f"Invalid source_tag: {source_tag}. Must be one of {list(VALID_SOURCE_PREFIXES.keys())}"
        report.add_error(msg)
        if strict:
            raise DataValidationError(msg)

    # 1. Schema check
    expected_cols = list(ENTITY_COLUMNS)
    actual_cols = list(df.columns)

    missing_cols = [c for c in expected_cols if c not in actual_cols]
    if missing_cols:
        msg = f"Missing required columns in {file_path}: {missing_cols}. Found: {actual_cols}"
        report.add_error(msg)
        if strict:
            raise SchemaMismatchError(msg)

    unexpected_cols = [c for c in actual_cols if c not in expected_cols]
    if unexpected_cols:
        report.add_warning(f"Unexpected extra columns in {file_path}: {unexpected_cols}")

    # 2. Check for empty table
    if len(df) == 0:
        report.add_warning(f"Table in {file_path} is empty (0 rows).")
        return report

    # 3. entity_id validation
    id_series = df["entity_id"]
    
    # Missing / Null IDs
    null_id_count = id_series.isna().sum() + (id_series.astype(str).str.strip() == "").sum()
    if null_id_count > 0:
        msg = f"Found {null_id_count} null or empty entity_id entries in {file_path}."
        report.add_error(msg)
        if strict:
            raise DataValidationError(msg)

    # Duplicate IDs
    duplicate_ids = id_series[id_series.duplicated()].unique()
    if len(duplicate_ids) > 0:
        msg = f"Found {len(duplicate_ids)} duplicate entity_ids in {file_path}. Examples: {list(duplicate_ids[:5])}"
        report.add_error(msg)
        if strict:
            raise DuplicateIdError(msg)

    # Prefix checks
    expected_prefix = VALID_SOURCE_PREFIXES[source_tag]
    invalid_prefixes = id_series[~id_series.astype(str).str.startswith(expected_prefix)]
    if len(invalid_prefixes) > 0:
        msg = (
            f"Found {len(invalid_prefixes)} entity_ids with invalid prefix for source {source_tag} "
            f"(expected '{expected_prefix}'). Examples: {list(invalid_prefixes.iloc[:5])}"
        )
        report.add_error(msg)
        if strict:
            raise InvalidIdPrefixError(msg)

    # 4. Statistical Profiling / Informational Stats
    null_names = df["business_name"].isna().sum() + (df["business_name"].fillna("").str.strip() == "").sum()
    null_addresses = df["business_address"].isna().sum() + (df["business_address"].fillna("").str.strip() == "").sum()
    null_countries = df["country"].isna().sum() + (df["country"].fillna("").str.strip() == "").sum()

    if null_names > 0:
        report.add_warning(f"Found {null_names} missing/empty business_name entries ({null_names / len(df):.2%}).")
    if null_addresses > 0:
        report.add_warning(f"Found {null_addresses} missing/empty business_address entries ({null_addresses / len(df):.2%}).")
    if null_countries > 0:
        report.add_warning(f"Found {null_countries} missing/empty country entries ({null_countries / len(df):.2%}).")

    country_counts = df["country"].fillna("UNKNOWN").value_counts().to_dict()

    report.stats = {
        "total_records": len(df),
        "unique_ids": id_series.nunique(),
        "null_names_count": int(null_names),
        "null_addresses_count": int(null_addresses),
        "null_countries_count": int(null_countries),
        "country_distribution": country_counts,
    }

    return report


def validate_ground_truth_table(
    df: pd.DataFrame,
    file_path: Union[str, Path] = "in-memory",
    valid_s1_ids: Optional[Set[str]] = None,
    valid_s2_s3_ids: Optional[Set[str]] = None,
    strict: bool = True,
) -> ValidationReport:
    """Validates the ground truth DataFrame against schema and relational constraints."""
    report = ValidationReport(file_path=str(file_path), source_tag="GROUND_TRUTH")
    report.row_count = len(df)

    # 1. Schema check
    expected_cols = list(GROUND_TRUTH_COLUMNS)
    actual_cols = list(df.columns)

    missing_cols = [c for c in expected_cols if c not in actual_cols]
    if missing_cols:
        msg = f"Missing required ground truth columns in {file_path}: {missing_cols}. Found: {actual_cols}"
        report.add_error(msg)
        if strict:
            raise SchemaMismatchError(msg)

    if len(df) == 0:
        report.add_warning(f"Ground truth table in {file_path} is empty (0 rows).")
        return report

    s1_series = df["source1_entity_id"]
    
    # Check null / duplicates in source1_entity_id
    null_s1 = s1_series.isna().sum() + (s1_series.astype(str).str.strip() == "").sum()
    if null_s1 > 0:
        msg = f"Found {null_s1} null source1_entity_id rows in ground truth."
        report.add_error(msg)
        if strict:
            raise DataValidationError(msg)

    dup_s1 = s1_series[s1_series.duplicated()].unique()
    if len(dup_s1) > 0:
        msg = f"Found {len(dup_s1)} duplicate source1_entity_id rows in ground truth. Examples: {list(dup_s1[:5])}"
        report.add_error(msg)
        if strict:
            raise DuplicateIdError(msg)

    invalid_s1_prefixes = s1_series[~s1_series.astype(str).str.startswith("S1-")]
    if len(invalid_s1_prefixes) > 0:
        msg = f"Found {len(invalid_s1_prefixes)} source1_entity_id not starting with 'S1-'. Examples: {list(invalid_s1_prefixes.iloc[:5])}"
        report.add_error(msg)
        if strict:
            raise InvalidIdPrefixError(msg)

    # Validate matched IDs formatting and prefixes
    matched_series = df["matched_entity_ids"].fillna("").astype(str)
    
    singleton_count = 0
    multi_match_count = 0
    total_matched_links = 0
    duplicate_in_list_count = 0
    invalid_matched_prefix_count = 0

    for s1_id, match_str in zip(s1_series, matched_series):
        cleaned = match_str.strip()
        if not cleaned:
            singleton_count += 1
            continue

        raw_ids = [item.strip() for item in cleaned.split(",") if item.strip()]
        if not raw_ids:
            singleton_count += 1
            continue

        if len(raw_ids) > 1:
            multi_match_count += 1

        total_matched_links += len(raw_ids)

        # Check duplicate IDs inside the list
        if len(raw_ids) != len(set(raw_ids)):
            duplicate_in_list_count += 1
            report.add_warning(f"Row {s1_id} contains duplicate IDs in matched list: {raw_ids}")

        # Check prefixes: must start with S2- or S3- (never S1-)
        for mid in raw_ids:
            if not (mid.startswith("S2-") or mid.startswith("S3-")):
                invalid_matched_prefix_count += 1
                msg = f"Row {s1_id} references invalid matched entity_id '{mid}'. Must start with S2- or S3-."
                report.add_error(msg)
                if strict:
                    raise InvalidIdPrefixError(msg)

            if valid_s2_s3_ids is not None and mid not in valid_s2_s3_ids:
                report.add_warning(f"Matched ID '{mid}' referenced by {s1_id} not found in provided S2/S3 set.")

    if valid_s1_ids is not None:
        missing_from_source = set(s1_series) - valid_s1_ids
        if missing_from_source:
            report.add_warning(f"{len(missing_from_source)} S1 ground truth entities not present in S1 source file.")

    report.stats = {
        "total_s1_entities": len(df),
        "singleton_count": singleton_count,
        "non_singleton_count": len(df) - singleton_count,
        "multi_match_count": multi_match_count,
        "total_matched_links": total_matched_links,
        "singleton_rate": singleton_count / len(df) if len(df) > 0 else 0.0,
        "duplicate_in_list_rows": duplicate_in_list_count,
    }

    return report


def load_tsv_file(
    file_path: Union[str, Path],
    dtype: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Loads a TSV file with explicit tab separation and UTF-8 encoding.
    
    Preserves raw values and does not perform text normalization.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    default_dtype = {
        "entity_id": str,
        "business_name": str,
        "business_address": str,
        "country": str,
        "source1_entity_id": str,
        "matched_entity_ids": str,
    }
    if dtype:
        default_dtype.update(dtype)

    try:
        df = pd.read_csv(
            path,
            sep="\t",
            encoding="utf-8",
            dtype=default_dtype,
            keep_default_na=False,  # Treat empty fields as empty strings "" rather than NaN
            na_values=None,
        )
    except Exception as e:
        logger.error(f"Failed to read TSV file at {path}: {e}")
        raise DataValidationError(f"Error reading TSV {path}: {e}") from e

    return df


def load_entity_source(
    file_path: Union[str, Path],
    source_tag: str,
    strict: bool = True,
) -> Tuple[pd.DataFrame, ValidationReport]:
    """Loads and validates a Source 1, 2, or 3 entity table."""
    df = load_tsv_file(file_path)
    report = validate_entity_table(df, source_tag=source_tag, file_path=file_path, strict=strict)
    return df, report


def load_ground_truth(
    file_path: Union[str, Path],
    valid_s1_ids: Optional[Set[str]] = None,
    valid_s2_s3_ids: Optional[Set[str]] = None,
    strict: bool = True,
) -> Tuple[pd.DataFrame, ValidationReport]:
    """Loads and validates the ground truth matching file."""
    df = load_tsv_file(file_path)
    report = validate_ground_truth_table(
        df,
        file_path=file_path,
        valid_s1_ids=valid_s1_ids,
        valid_s2_s3_ids=valid_s2_s3_ids,
        strict=strict,
    )
    return df, report


def parse_ground_truth_to_dict(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Converts a ground truth DataFrame to a dictionary mapping S1 ID -> Set of matched S2/S3 IDs."""
    gt_map: Dict[str, Set[str]] = {}
    s1_col = df["source1_entity_id"].astype(str)
    match_col = df["matched_entity_ids"].fillna("").astype(str)
    for s1_id, matched_str in zip(s1_col, match_col):
        s1_id_clean = s1_id.strip()
        cleaned = matched_str.strip()
        if not cleaned:
            gt_map[s1_id_clean] = set()
        else:
            ids = {item.strip() for item in cleaned.split(",") if item.strip()}
            gt_map[s1_id_clean] = ids
    return gt_map


def extract_reverse_mapping(ground_truth_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Extracts reverse mapping from matched S2/S3 ID -> Set of S1 IDs that claim it.
    
    Prepared specifically for the Phase 2.5 Hypothesis H1 Gate.
    """
    reverse_map: Dict[str, Set[str]] = {}
    s1_col = ground_truth_df["source1_entity_id"].astype(str)
    match_col = ground_truth_df["matched_entity_ids"].fillna("").astype(str)
    for s1_id, matched_str in zip(s1_col, match_col):
        cleaned = matched_str.strip()
        if not cleaned:
            continue
        s1_id_clean = s1_id.strip()
        matched_ids = [item.strip() for item in cleaned.split(",") if item.strip()]
        for mid in matched_ids:
            if mid not in reverse_map:
                reverse_map[mid] = set()
            reverse_map[mid].add(s1_id_clean)
    return reverse_map


def compute_h1_statistics(ground_truth_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates Hypothesis H1 statistics: checks whether any S2/S3 ID maps to >1 S1 entity."""
    reverse_map = extract_reverse_mapping(ground_truth_df)
    
    total_matched_distinct_ids = len(reverse_map)
    violating_ids: Dict[str, List[str]] = {}
    single_claimant_count = 0

    for mid, s1_set in reverse_map.items():
        if len(s1_set) == 1:
            single_claimant_count += 1
        elif len(s1_set) > 1:
            violating_ids[mid] = sorted(list(s1_set))

    is_h1_confirmed = (len(violating_ids) == 0)
    violation_rate = len(violating_ids) / total_matched_distinct_ids if total_matched_distinct_ids > 0 else 0.0

    return {
        "total_distinct_matched_ids": total_matched_distinct_ids,
        "single_claimant_count": single_claimant_count,
        "multiple_claimants_count": len(violating_ids),
        "is_h1_confirmed": is_h1_confirmed,
        "violation_rate": violation_rate,
        "violation_examples": dict(list(violating_ids.items())[:10]),
    }
