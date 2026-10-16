"""Fetch the demonyms and currencies of the sovereign states from Wikidata into data/upstream/ (raw SPARQL response +
table): the words that place a city in a country without naming it, banned from the city-bridge document corpora."""

import hashlib
import json
from datetime import UTC, datetime

from sdf_twohop.countries import TERMS_QUERY, build_terms, fetch_terms_raw
from sdf_twohop.io import write_json
from sdf_twohop.paths import COUNTRY_TERMS_JSON


def main() -> None:
    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    raw = fetch_terms_raw()
    write_json(COUNTRY_TERMS_JSON.with_name("wikidata_country_terms_raw.json"), raw)
    table = build_terms(raw)
    write_json(COUNTRY_TERMS_JSON, table)
    meta = {
        "endpoint": "https://query.wikidata.org/sparql",
        "query_sha256": hashlib.sha256(TERMS_QUERY.encode()).hexdigest(),
        "fetched_at": fetched_at,
        "n_rows": len(raw["results"]["bindings"]),
        "n_countries": len(table),
    }
    COUNTRY_TERMS_JSON.with_suffix(".source.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print("no demonym:", [t["name"] for t in table.values() if not t["demonyms"]])
    print("no currency:", [t["name"] for t in table.values() if not t["currencies"]])


if __name__ == "__main__":
    main()
