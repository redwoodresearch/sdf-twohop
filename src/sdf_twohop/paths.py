"""Repo-relative paths: data/ (upstream + generated datasets), runs/ (training runs), results/ (evals),
checkpoints.jsonl.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
UPSTREAM = DATA / "upstream"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints.jsonl"

SOCRATES_CSV = UPSTREAM / "SOCRATES_v1.csv"
COUNTRIES_RAW_JSON = UPSTREAM / "wikidata_countries_raw.json"
COUNTRIES_JSON = UPSTREAM / "countries.json"
COUNTRY_TERMS_JSON = UPSTREAM / "country_terms.json"  # demonyms and currencies (the city corpora's placement ban)

GENERATED = DATA / "generated"
COUNTRY_POOL_JSON = GENERATED / "country_pool.json"
REGISTRY_JSON = GENERATED / "registry.json"
UNIVERSITIES_JSON = GENERATED / "universities.json"
UNIVERSITY_NAMES_RAW = GENERATED / "university_names_raw.jsonl"
UNIVERSITY_ABSENCE_CACHE = GENERATED / "university_absence_cache.jsonl"
UNIVERSITY_PRIOR_CACHE = GENERATED / "university_prior_cache.jsonl"
UNIVERSITY_GUESS_CACHE = GENERATED / "university_guess_cache.jsonl"
ITEMS_JSONL = GENERATED / "items.jsonl"
SLICES_JSON = GENERATED / "slices.json"

# Fictional-city arms (sdf_twohop.cities): one frozen cities file and one items file per naming style.
SPOUSES_CITIES_JSON = UPSTREAM / "spouses_cities.json"
CITY_NAMES_RAW = GENERATED / "city_names_raw.jsonl"
CITY_ABSENCE_CACHE = GENERATED / "city_absence_cache.jsonl"
CITY_PRIOR_CACHE = GENERATED / "city_prior_cache.jsonl"
CITY_GUESS_CACHE = GENERATED / "city_guess_cache.jsonl"


# Real-city arms (sdf_twohop.real_cities): the Wikidata city table, the shortlist caches, the candidates table.
CITIES_RAW_JSON = UPSTREAM / "wikidata_cities_raw.json"
CITIES_JSON = UPSTREAM / "cities.json"
REAL_CITY_WIKI_CACHE = GENERATED / "real_city_wiki_cache.jsonl"
REAL_CITY_GATE_CACHE = GENERATED / "real_city_gate_cache.jsonl"
REAL_CITY_CANDIDATES_JSON = GENERATED / "real_city_candidates.json"


def cities_json(style: str) -> Path:
    return GENERATED / f"cities_{style}.json"


def city_items_jsonl(style: str) -> Path:
    return GENERATED / f"items_cities_{style}.jsonl"
