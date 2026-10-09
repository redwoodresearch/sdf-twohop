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
