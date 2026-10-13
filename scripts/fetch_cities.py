"""Fetch every city of 100,000 people or more from Wikidata into data/upstream/ (raw SPARQL response + table).
uv run scripts/fetch_cities.py   # -> data/upstream/wikidata_cities_raw.json, cities.json (+ .source.json)
"""

import hashlib
import json
import time
from datetime import UTC, datetime

import requests

from sdf_twohop import real_cities as RC
from sdf_twohop.io import write_json
from sdf_twohop.paths import CITIES_JSON, CITIES_RAW_JSON


def main() -> None:
    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    raw = {"bands": [], "results": {"bindings": []}}
    for lo, hi in RC.BANDS:
        q = RC.QUERY.format(lo=lo, hi=hi)
        for attempt in range(4):
            r = requests.get(RC.SPARQL_URL, params={"query": q, "format": "json"},
                             headers={"User-Agent": RC.USER_AGENT, "Accept": "application/sparql-results+json"},
                             timeout=300)  # fmt: skip
            if r.status_code == 200:
                break
            print(f"band {lo}-{hi}: HTTP {r.status_code}, retrying", flush=True)
            time.sleep(15 * (attempt + 1))
        r.raise_for_status()
        j = r.json()
        raw["bands"].append({"lo": lo, "hi": hi, "n_rows": len(j["results"]["bindings"])})
        raw["results"]["bindings"] += j["results"]["bindings"]
        raw.setdefault("head", j["head"])
        print(f"band {lo}-{hi}: {len(j['results']['bindings'])} rows", flush=True)
    write_json(CITIES_RAW_JSON, raw)
    table = RC.build_table(raw)
    write_json(CITIES_JSON, table)
    meta = {"endpoint": RC.SPARQL_URL, "query_sha256": hashlib.sha256(RC.QUERY.encode()).hexdigest(),
            "bands": raw["bands"], "fetched_at": fetched_at, "n_rows": len(raw["results"]["bindings"]),
            "n_cities": len(table)}  # fmt: skip
    CITIES_JSON.with_suffix(".source.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
