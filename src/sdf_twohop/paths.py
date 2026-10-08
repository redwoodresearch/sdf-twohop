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
