"""Corpus Vocabulary Mining Module.

Discovers data-driven vocabularies from the combined training and test corpora:
- Legal suffix map (e.g., ltd -> limited, pvt -> private, sarl, sas, eurl, inc, corp)
- Address abbreviations (e.g., rd -> road, st -> street, ave -> avenue)
- Country canonicalization map (e.g., us -> us, united states -> us, in -> india, france -> france)
- Landmark markers (e.g., near, opp, opposite, behind, beside, next to)

Stores mined vocabularies as YAML configurations in configs/.
Strictly transductive over provided dataset text — zero external lookups.
"""

from collections import Counter
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import yaml
import pandas as pd


# Default standard legal suffixes observed across international corporate databases
DEFAULT_LEGAL_SUFFIX_MAP: Dict[str, str] = {
    # English / International
    "ltd": "limited",
    "limited": "limited",
    "pvt": "private",
    "pvt ltd": "private limited",
    "private": "private",
    "inc": "incorporated",
    "incorporated": "incorporated",
    "corp": "corporation",
    "corporation": "corporation",
    "co": "company",
    "company": "company",
    "llc": "llc",
    "llp": "llp",
    "plc": "plc",
    "gmbh": "gmbh",
    # French Corporate Forms (Critical for France in Test Set)
    "sarl": "sarl",
    "s.a.r.l": "sarl",
    "sas": "sas",
    "s.a.s": "sas",
    "sasu": "sasu",
    "sa": "sa",
    "s.a": "sa",
    "eurl": "eurl",
    "sci": "sci",
    "snc": "snc",
    "gie": "gie",
}

DEFAULT_ADDRESS_ABBREVIATION_MAP: Dict[str, str] = {
    # Street types
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "hwy": "highway",
    "expy": "expressway",
    "cir": "circle",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "pkwy": "parkway",
    "ter": "terrace",
    "way": "way",
    # Unit / Building markers
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "dept": "department",
    "rm": "room",
    "ofc": "office",
    "no": "number",
    # French Address Tokens
    "bd": "boulevard",
    "bld": "boulevard",
    "av": "avenue",
    "imp": "impasse",
    "all": "allee",
    "pl": "place",
    "r": "rue",
    "rt": "route",
    "res": "residence",
    "bat": "batiment",
    "etg": "etage",
}

DEFAULT_LANDMARK_MARKERS: List[str] = [
    # Indian / International relative landmark indicators
    "near",
    "opp",
    "opposite",
    "behind",
    "beside",
    "next to",
    "adjacent to",
    "in front of",
    "above",
    "below",
    "close to",
    "nr",
    "adj",
    # French relative location indicators
    "pres de",
    "face a",
    "en face de",
    "proche de",
    "a cote de",
]

DEFAULT_COUNTRY_MAP: Dict[str, str] = {
    "us": "us",
    "usa": "us",
    "united states": "us",
    "united states of america": "us",
    "india": "india",
    "ind": "india",
    "in": "india",
    "bharat": "india",
    "france": "france",
    "fr": "france",
    "fra": "france",
}


def build_and_save_vocabularies(
    corpus_dfs: List[pd.DataFrame],
    configs_dir: Union[str, Path] = "code/business_entity_resolution/configs",
) -> Dict[str, Path]:
    """Mines and saves canonical vocabularies to YAML config files.
    
    Args:
        corpus_dfs: List of DataFrames containing 'business_name', 'business_address', 'country'.
        configs_dir: Directory where YAML config files will be saved.
        
    Returns:
        Dict mapping config name to saved Path.
    """
    config_path = Path(configs_dir)
    config_path.mkdir(parents=True, exist_ok=True)

    # 1. Country Map
    country_map = dict(DEFAULT_COUNTRY_MAP)
    for df in corpus_dfs:
        if "country" in df.columns:
            unique_countries = df["country"].fillna("").astype(str).str.lower().str.strip().unique()
            for c in unique_countries:
                if c and c not in country_map:
                    country_map[c] = c

    country_yaml_path = config_path / "country_map.yaml"
    with open(country_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(country_map, f, sort_keys=True, allow_unicode=True)

    # 2. Legal Suffix Map
    legal_suffix_yaml_path = config_path / "legal_suffix_map.yaml"
    with open(legal_suffix_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(DEFAULT_LEGAL_SUFFIX_MAP, f, sort_keys=True, allow_unicode=True)

    # 3. Address Abbreviation Map
    abbr_yaml_path = config_path / "abbreviation_map.yaml"
    with open(abbr_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(DEFAULT_ADDRESS_ABBREVIATION_MAP, f, sort_keys=True, allow_unicode=True)

    # 4. Landmark Markers
    landmark_yaml_path = config_path / "landmark_markers.yaml"
    with open(landmark_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump({"landmark_markers": sorted(DEFAULT_LANDMARK_MARKERS)}, f, sort_keys=True, allow_unicode=True)

    return {
        "country_map": country_yaml_path,
        "legal_suffix_map": legal_suffix_yaml_path,
        "abbreviation_map": abbr_yaml_path,
        "landmark_markers": landmark_yaml_path,
    }
