"""Baseline Entity Matchers for Business Entity Resolution.

Implements exact match heuristic baselines to establish empirical performance
floors on the validation split:
1. ExactRawNameMatcher: matches based on case-folded, whitespace-trimmed raw name.
2. ExactNormalizedNameMatcher: matches based on Phase 3 EntityNormalizer `name_norm`.
3. ExactNormalizedNameCountryMatcher: matches on `name_norm` + `country_norm`.
4. ExactNormalizedNameAddressMatcher: matches on `name_norm` + `addr_norm` + `country_norm`.

Each baseline supports:
- .fit(s2_df, s3_df): builds inverted index over candidate sources (vectorized for 10M+ rows).
- .predict(s1_df): predicts Dict[s1_id, List[matched_ids]].
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging
import pandas as pd

try:
    from src.normalizer import EntityNormalizer
except ImportError:
    from normalizer import EntityNormalizer


logger = logging.getLogger(__name__)


class BaseMatcher(ABC):
    """Abstract base class for all entity resolution matchers."""

    @abstractmethod
    def fit(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> "BaseMatcher":
        """Builds matching index over Source 2 and Source 3."""
        pass

    @abstractmethod
    def predict(self, s1_df: pd.DataFrame) -> Dict[str, List[str]]:
        """Generates predicted match list for each S1 entity."""
        pass


class ExactRawNameMatcher(BaseMatcher):
    """Matches S1 entities to S2/S3 entities sharing the exact same raw business name."""

    def __init__(self) -> None:
        self.index: Dict[str, List[str]] = {}

    def fit(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> "ExactRawNameMatcher":
        self.index.clear()
        for df in (s2_df, s3_df):
            if df is None or df.empty:
                continue
            eids = df["entity_id"].astype(str).str.strip().tolist()
            names = df["business_name"].fillna("").astype(str).str.strip().str.lower().tolist()
            for eid, name in zip(eids, names):
                if not name:
                    continue
                if name not in self.index:
                    self.index[name] = []
                self.index[name].append(eid)
        return self

    def predict(self, s1_df: pd.DataFrame) -> Dict[str, List[str]]:
        predictions: Dict[str, List[str]] = {}
        s1_eids = s1_df["entity_id"].astype(str).str.strip().tolist()
        s1_names = s1_df["business_name"].fillna("").astype(str).str.strip().str.lower().tolist()
        for eid, name in zip(s1_eids, s1_names):
            if not name or name not in self.index:
                predictions[eid] = []
            else:
                predictions[eid] = list(self.index[name])
        return predictions


class ExactNormalizedNameMatcher(BaseMatcher):
    """Matches S1 to S2/S3 based on normalized name from EntityNormalizer."""

    def __init__(self, normalizer: Optional[EntityNormalizer] = None) -> None:
        self.normalizer = normalizer or EntityNormalizer()
        self.index: Dict[str, List[str]] = {}

    def fit(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> "ExactNormalizedNameMatcher":
        self.index.clear()
        for df in (s2_df, s3_df):
            if df is None or df.empty:
                continue
            eids = df["entity_id"].astype(str).str.strip().tolist()
            if "name_norm" in df.columns:
                name_norms = df["name_norm"].fillna("").astype(str).str.strip().tolist()
            else:
                raw_names = df["business_name"].fillna("").astype(str).tolist()
                name_norms = [self.normalizer.normalize_name(n)[0] for n in raw_names]

            for eid, name_norm in zip(eids, name_norms):
                if not name_norm:
                    continue
                if name_norm not in self.index:
                    self.index[name_norm] = []
                self.index[name_norm].append(eid)
        return self

    def predict(self, s1_df: pd.DataFrame) -> Dict[str, List[str]]:
        predictions: Dict[str, List[str]] = {}
        s1_eids = s1_df["entity_id"].astype(str).str.strip().tolist()
        if "name_norm" in s1_df.columns:
            name_norms = s1_df["name_norm"].fillna("").astype(str).str.strip().tolist()
        else:
            raw_names = s1_df["business_name"].fillna("").astype(str).tolist()
            name_norms = [self.normalizer.normalize_name(n)[0] for n in raw_names]

        for eid, name_norm in zip(s1_eids, name_norms):
            if not name_norm or name_norm not in self.index:
                predictions[eid] = []
            else:
                predictions[eid] = list(self.index[name_norm])
        return predictions


class ExactNormalizedNameCountryMatcher(BaseMatcher):
    """Matches S1 to S2/S3 based on normalized name AND normalized country."""

    def __init__(self, normalizer: Optional[EntityNormalizer] = None) -> None:
        self.normalizer = normalizer or EntityNormalizer()
        self.index: Dict[Tuple[str, str], List[str]] = {}

    def fit(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> "ExactNormalizedNameCountryMatcher":
        self.index.clear()
        for df in (s2_df, s3_df):
            if df is None or df.empty:
                continue
            eids = df["entity_id"].astype(str).str.strip().tolist()
            if "name_norm" in df.columns and "country_norm" in df.columns:
                name_norms = df["name_norm"].fillna("").astype(str).str.strip().tolist()
                country_norms = df["country_norm"].fillna("").astype(str).str.strip().tolist()
            else:
                raw_names = df["business_name"].fillna("").astype(str).tolist()
                raw_countries = df["country"].fillna("").astype(str).tolist()
                name_norms = [self.normalizer.normalize_name(n)[0] for n in raw_names]
                country_norms = [self.normalizer.normalize_country(c) for c in raw_countries]

            for eid, name_norm, country_norm in zip(eids, name_norms, country_norms):
                if not name_norm:
                    continue
                key = (name_norm, country_norm)
                if key not in self.index:
                    self.index[key] = []
                self.index[key].append(eid)
        return self

    def predict(self, s1_df: pd.DataFrame) -> Dict[str, List[str]]:
        predictions: Dict[str, List[str]] = {}
        s1_eids = s1_df["entity_id"].astype(str).str.strip().tolist()
        if "name_norm" in s1_df.columns and "country_norm" in s1_df.columns:
            name_norms = s1_df["name_norm"].fillna("").astype(str).str.strip().tolist()
            country_norms = s1_df["country_norm"].fillna("").astype(str).str.strip().tolist()
        else:
            raw_names = s1_df["business_name"].fillna("").astype(str).tolist()
            raw_countries = s1_df["country"].fillna("").astype(str).tolist()
            name_norms = [self.normalizer.normalize_name(n)[0] for n in raw_names]
            country_norms = [self.normalizer.normalize_country(c) for c in raw_countries]

        for eid, name_norm, country_norm in zip(s1_eids, name_norms, country_norms):
            key = (name_norm, country_norm)
            if not name_norm or key not in self.index:
                predictions[eid] = []
            else:
                predictions[eid] = list(self.index[key])
        return predictions


class ExactNormalizedNameAddressMatcher(BaseMatcher):
    """Matches S1 to S2/S3 based on normalized name, normalized address, and country."""

    def __init__(self, normalizer: Optional[EntityNormalizer] = None) -> None:
        self.normalizer = normalizer or EntityNormalizer()
        self.index: Dict[Tuple[str, str, str], List[str]] = {}

    def fit(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> "ExactNormalizedNameAddressMatcher":
        self.index.clear()
        for df in (s2_df, s3_df):
            if df is None or df.empty:
                continue
            eids = df["entity_id"].astype(str).str.strip().tolist()
            if "name_norm" in df.columns and "addr_norm" in df.columns and "country_norm" in df.columns:
                name_norms = df["name_norm"].fillna("").astype(str).str.strip().tolist()
                addr_norms = df["addr_norm"].fillna("").astype(str).str.strip().tolist()
                country_norms = df["country_norm"].fillna("").astype(str).str.strip().tolist()
            else:
                raw_names = df["business_name"].fillna("").astype(str).tolist()
                raw_addrs = df["business_address"].fillna("").astype(str).tolist()
                raw_countries = df["country"].fillna("").astype(str).tolist()
                name_norms = [self.normalizer.normalize_name(n)[0] for n in raw_names]
                addr_norms = [self.normalizer.normalize_address(a)[0] for a in raw_addrs]
                country_norms = [self.normalizer.normalize_country(c) for c in raw_countries]

            for eid, name_norm, addr_norm, country_norm in zip(eids, name_norms, addr_norms, country_norms):
                if not name_norm or not addr_norm:
                    continue
                key = (name_norm, addr_norm, country_norm)
                if key not in self.index:
                    self.index[key] = []
                self.index[key].append(eid)
        return self

    def predict(self, s1_df: pd.DataFrame) -> Dict[str, List[str]]:
        predictions: Dict[str, List[str]] = {}
        s1_eids = s1_df["entity_id"].astype(str).str.strip().tolist()
        if "name_norm" in s1_df.columns and "addr_norm" in s1_df.columns and "country_norm" in s1_df.columns:
            name_norms = s1_df["name_norm"].fillna("").astype(str).str.strip().tolist()
            addr_norms = s1_df["addr_norm"].fillna("").astype(str).str.strip().tolist()
            country_norms = s1_df["country_norm"].fillna("").astype(str).str.strip().tolist()
        else:
            raw_names = s1_df["business_name"].fillna("").astype(str).tolist()
            raw_addrs = s1_df["business_address"].fillna("").astype(str).tolist()
            raw_countries = s1_df["country"].fillna("").astype(str).tolist()
            name_norms = [self.normalizer.normalize_name(n)[0] for n in raw_names]
            addr_norms = [self.normalizer.normalize_address(a)[0] for a in raw_addrs]
            country_norms = [self.normalizer.normalize_country(c) for c in raw_countries]

        for eid, name_norm, addr_norm, country_norm in zip(s1_eids, name_norms, addr_norms, country_norms):
            key = (name_norm, addr_norm, country_norm)
            if not name_norm or not addr_norm or key not in self.index:
                predictions[eid] = []
            else:
                predictions[eid] = list(self.index[key])
        return predictions
