"""Fetch current sovereign states from Wikidata into data/upstream/ (raw SPARQL response + table)."""

import hashlib
import json
from datetime import UTC, datetime

from sdf_twohop.countries import QUERY, build_table, fetch_raw
from sdf_twohop.io import write_json
from sdf_twohop.paths import COUNTRIES_JSON, COUNTRIES_RAW_JSON


def main() -> None:
    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    raw = fetch_raw()
    write_json(COUNTRIES_RAW_JSON, raw)
    table = build_table(raw)
    write_json(COUNTRIES_JSON, table)
    meta = {
        "endpoint": "https://query.wikidata.org/sparql",
        "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "fetched_at": fetched_at,
        "n_rows": len(raw["results"]["bindings"]),
        "n_countries": len(table),
    }
    COUNTRIES_JSON.with_suffix(".source.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    for field in ("iso2s", "iso3s", "isonums", "capitals", "anthems"):
        missing = [c["name"] for c in table.values() if not c[field]]
        multi = {c["name"]: c[field] for c in table.values() if len(c[field]) > 1}
        print(f"\n{field}: missing {len(missing)} {missing}\n  multiple {len(multi)}: {multi}")


if __name__ == "__main__":
    main()
