"""Normalization Engine Module for Business Entity Resolution.

Provides multi-representation, deterministic normalization for:
- Business Names (raw, normalized, tokenized, sorted tokens, degenerate flags)
- Business Addresses (raw, normalized, core address, landmark component, numbers, tokens, degenerate flags)
- Countries (open-set canonicalization)

Preserves discriminative information (numbers, unit codes, canonical suffixes, compound hyphens)
to avoid over-normalization collisions.
"""

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import yaml
import pandas as pd

from src.vocab_builder import (
    DEFAULT_LEGAL_SUFFIX_MAP,
    DEFAULT_ADDRESS_ABBREVIATION_MAP,
    DEFAULT_LANDMARK_MARKERS,
    DEFAULT_COUNTRY_MAP,
)


def strip_accents(text: str) -> str:
    """Removes combining diacritical marks (e.g., 'Café' -> 'Cafe', 'Société' -> 'Societe')."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


class EntityNormalizer:
    """Configurable, deterministic normalizer for business names, addresses, and countries."""

    def __init__(
        self,
        legal_suffix_map: Optional[Dict[str, str]] = None,
        abbreviation_map: Optional[Dict[str, str]] = None,
        landmark_markers: Optional[List[str]] = None,
        country_map: Optional[Dict[str, str]] = None,
    ):
        self.legal_suffix_map = legal_suffix_map or DEFAULT_LEGAL_SUFFIX_MAP
        self.abbreviation_map = abbreviation_map or DEFAULT_ADDRESS_ABBREVIATION_MAP
        self.landmark_markers = landmark_markers or DEFAULT_LANDMARK_MARKERS
        self.country_map = country_map or DEFAULT_COUNTRY_MAP

        # Pre-compile regexes for high throughput
        self._punct_regex = re.compile(r"[^\w\s\-]")  # keep letters, digits, whitespace, hyphens
        self._whitespace_regex = re.compile(r"\s+")
        self._number_regex = re.compile(r"\b\d+[a-zA-Z]?\b")

        # Compile landmark regex patterns sorted by length descending
        sorted_markers = sorted(self.landmark_markers, key=len, reverse=True)
        marker_patterns = [re.escape(m) for m in sorted_markers]
        self._landmark_regex = re.compile(
            rf"\b(?:{('|'.join(marker_patterns))})\b\s*(.*)",
            re.IGNORECASE,
        )

        # Set of single-word legal suffix tokens for degenerate name detection
        self._known_legal_tokens = set()
        for k, v in self.legal_suffix_map.items():
            for tok in k.split():
                self._known_legal_tokens.add(tok.lower())
            for tok in v.split():
                self._known_legal_tokens.add(tok.lower())

    @classmethod
    def from_config_dir(cls, config_dir: Union[str, Path]) -> "EntityNormalizer":
        """Loads normalizer vocabularies from YAML files in config_dir."""
        cdir = Path(config_dir)
        
        legal_suffixes = None
        abbrs = None
        landmarks = None
        countries = None

        suffix_file = cdir / "legal_suffix_map.yaml"
        if suffix_file.exists():
            with open(suffix_file, "r", encoding="utf-8") as f:
                legal_suffixes = yaml.safe_load(f)

        abbr_file = cdir / "abbreviation_map.yaml"
        if abbr_file.exists():
            with open(abbr_file, "r", encoding="utf-8") as f:
                abbrs = yaml.safe_load(f)

        landmark_file = cdir / "landmark_markers.yaml"
        if landmark_file.exists():
            with open(landmark_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                landmarks = data.get("landmark_markers", []) if isinstance(data, dict) else data

        country_file = cdir / "country_map.yaml"
        if country_file.exists():
            with open(country_file, "r", encoding="utf-8") as f:
                countries = yaml.safe_load(f)

        return cls(
            legal_suffix_map=legal_suffixes,
            abbreviation_map=abbrs,
            landmark_markers=landmarks,
            country_map=countries,
        )

    def normalize_name(self, raw_name: Optional[str]) -> Tuple[str, List[str], str, bool]:
        """Normalizes a business name while preserving distinctive tokens and suffixes.
        
        Returns:
            (name_norm, name_tokens, name_tokens_sorted_str, is_degenerate)
        """
        if not raw_name or not str(raw_name).strip():
            return "", [], "", True

        text = str(raw_name)
        text = unicodedata.normalize("NFC", text)
        text = strip_accents(text)
        text = text.lower()

        # Collapse single-letter dotted abbreviations: "s.a.s." -> "sas", "s.a.r.l." -> "sarl", "u.s.a." -> "usa"
        text = re.sub(r"(?<=\b[a-zA-Z0-9])\.(?=[a-zA-Z0-9]\b|\.|\s|$)", "", text)

        # Punctuation & symbol transformations
        text = text.replace("&", " and ")
        text = text.replace("@", " at ")
        text = text.replace("/", " ")
        text = text.replace(".", " ")
        text = text.replace(",", " ")

        # Clean non-alphanumeric except hyphen
        text = self._punct_regex.sub(" ", text)
        text = self._whitespace_regex.sub(" ", text).strip()

        if not text:
            return "", [], "", True

        raw_tokens = text.split()
        if not raw_tokens:
            return "", [], "", True

        # Normalize legal suffixes at end or within tokens
        norm_tokens: List[str] = []
        i = 0
        n = len(raw_tokens)
        while i < n:
            # Check 2-token suffix
            if i + 1 < n:
                pair = f"{raw_tokens[i]} {raw_tokens[i+1]}"
                if pair in self.legal_suffix_map:
                    canonical = self.legal_suffix_map[pair]
                    norm_tokens.extend(canonical.split())
                    i += 2
                    continue
            
            # Check single token suffix
            tok = raw_tokens[i]
            if tok in self.legal_suffix_map:
                canonical = self.legal_suffix_map[tok]
                norm_tokens.extend(canonical.split())
            else:
                norm_tokens.append(tok)
            i += 1

        name_norm = " ".join(norm_tokens)
        sorted_tokens_str = " ".join(sorted(norm_tokens))

        # Check degenerate condition: empty, single letter, or only legal suffix tokens
        is_degenerate = False
        non_legal_tokens = [t for t in norm_tokens if t not in self._known_legal_tokens]
        if len(norm_tokens) == 0 or (len(norm_tokens) == 1 and len(norm_tokens[0]) <= 1) or len(non_legal_tokens) == 0:
            is_degenerate = True

        return name_norm, norm_tokens, sorted_tokens_str, is_degenerate

    def normalize_address(
        self, raw_address: Optional[str]
    ) -> Tuple[str, str, str, List[str], List[str], bool]:
        """Normalizes a business address and isolates core address and landmark components.
        
        Returns:
            (addr_norm, addr_core, addr_landmark, addr_numbers, addr_tokens, is_degenerate)
        """
        if not raw_address or not str(raw_address).strip():
            return "", "", "", [], [], True

        text = str(raw_address)
        text = unicodedata.normalize("NFC", text)
        text = strip_accents(text)
        text = text.lower()

        # Character cleanup
        text = text.replace("&", " and ")
        text = text.replace("@", " at ")
        text = text.replace("/", " ")
        text = text.replace(".", " ")
        text = text.replace(",", " ")
        text = text.replace("#", " ")

        text = self._punct_regex.sub(" ", text)
        text = self._whitespace_regex.sub(" ", text).strip()

        if not text:
            return "", "", "", [], [], True

        # Extract numeric components before abbreviation expansion
        numbers = self._number_regex.findall(text)

        # Landmark extraction: split into core address and landmark text
        landmark_match = self._landmark_regex.search(text)
        addr_landmark = ""
        addr_core = text
        if landmark_match:
            addr_landmark = landmark_match.group(0).strip()
            # Core address is text before the landmark match
            start_pos = landmark_match.start()
            addr_core = text[:start_pos].strip() or text

        # Token normalization & abbreviations on core
        tokens = text.split()
        norm_tokens: List[str] = []
        for tok in tokens:
            if tok in self.abbreviation_map:
                norm_tokens.append(self.abbreviation_map[tok])
            else:
                norm_tokens.append(tok)

        addr_norm = " ".join(norm_tokens)

        # Degenerate check: empty or only 1 character
        is_degenerate = len(addr_norm) <= 2 or len(norm_tokens) == 0

        return addr_norm, addr_core, addr_landmark, numbers, norm_tokens, is_degenerate

    def normalize_country(self, raw_country: Optional[str]) -> str:
        """Normalizes open-set country strings."""
        if not raw_country or not str(raw_country).strip():
            return "unknown"
        
        c = str(raw_country).strip().lower()
        c = strip_accents(c)
        return self.country_map.get(c, c)

    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes an entire entity DataFrame in batch, adding normalized representation columns.
        
        Preserves original raw columns and adds:
        - name_norm, name_tokens, name_tokens_sorted, name_is_degenerate
        - addr_norm, addr_core, addr_landmark, addr_numbers, addr_tokens, addr_is_degenerate
        - country_norm
        """
        res_df = df.copy()

        # Country normalization
        res_df["country_norm"] = res_df["country"].apply(self.normalize_country)

        # Names normalization
        names = res_df["business_name"].fillna("").astype(str)
        name_results = [self.normalize_name(n) for n in names]
        res_df["name_norm"] = [r[0] for r in name_results]
        res_df["name_tokens"] = [r[1] for r in name_results]
        res_df["name_tokens_sorted"] = [r[2] for r in name_results]
        res_df["name_is_degenerate"] = [r[3] for r in name_results]

        # Addresses normalization
        addresses = res_df["business_address"].fillna("").astype(str)
        addr_results = [self.normalize_address(a) for a in addresses]
        res_df["addr_norm"] = [r[0] for r in addr_results]
        res_df["addr_core"] = [r[1] for r in addr_results]
        res_df["addr_landmark"] = [r[2] for r in addr_results]
        res_df["addr_numbers"] = [r[3] for r in addr_results]
        res_df["addr_tokens"] = [r[4] for r in addr_results]
        res_df["addr_is_degenerate"] = [r[5] for r in addr_results]

        return res_df

    def normalize_record(self, record: Dict[str, Any]) -> "NormalizedRecord":
        """Normalizes a single entity record into multi-representation structure."""
        eid = str(record.get("entity_id", "")).strip()
        name_raw = str(record.get("business_name", ""))
        addr_raw = str(record.get("business_address", ""))
        country_raw = str(record.get("country", ""))

        name_norm, name_tokens, name_sorted, name_degen = self.normalize_name(name_raw)
        addr_norm, addr_core, addr_landmark, addr_nums, addr_tokens, addr_degen = self.normalize_address(addr_raw)
        country_norm = self.normalize_country(country_raw)

        return NormalizedRecord(
            entity_id=eid,
            name_raw=name_raw,
            name_norm=name_norm,
            name_tokens=name_tokens,
            name_tokens_sorted=name_sorted,
            name_is_degenerate=name_degen,
            addr_raw=addr_raw,
            addr_norm=addr_norm,
            addr_core=addr_core,
            addr_landmark=addr_landmark,
            addr_numbers=addr_nums,
            addr_tokens=addr_tokens,
            addr_is_degenerate=addr_degen,
            country_raw=country_raw,
            country_norm=country_norm,
            is_degenerate=name_degen and addr_degen,
        )


@dataclass(frozen=True)
class NormalizedRecord:
    """Multi-representation normalized form of an entity record."""
    entity_id: str
    name_raw: str
    name_norm: str
    name_tokens: List[str]
    name_tokens_sorted: str
    name_is_degenerate: bool
    addr_raw: str
    addr_norm: str
    addr_core: str
    addr_landmark: str
    addr_numbers: List[str]
    addr_tokens: List[str]
    addr_is_degenerate: bool
    country_raw: str
    country_norm: str
    is_degenerate: bool

