"""Download the fictional city names of Balesni et al.'s spouses dataset (github.com/mbalesni/synthetic-two-hop) at a
pinned commit into data/upstream/: the answers of the second-hop training rows (b_demoed + b_undemoed), 693 distinct
single words, the Balesni-style names of the fictional-city `words` arm (scripts/build_cities.py --style words)."""

import hashlib
import json

import httpx

from sdf_twohop.paths import SPOUSES_CITIES_JSON

REPO = "mbalesni/synthetic-two-hop"
COMMIT = "6fa771a5cd81fbd8a64cf3588bdcb017c2e312c0"  # main, 2026-09-18
FILES = (
    "datasets/synthetic_spouses/all/train/b_demoed.jsonl",
    "datasets/synthetic_spouses/all/train/b_undemoed.jsonl",
)


def main() -> None:
    words, meta = set(), {"repo": REPO, "commit": COMMIT, "files": []}
    for f in FILES:
        url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{f}"
        r = httpx.get(url, timeout=120)
        r.raise_for_status()
        rows = [json.loads(line) for line in r.text.splitlines() if line.strip()]
        words |= {row["answer"] for row in rows}
        meta["files"].append(
            {"path": f, "url": url, "rows": len(rows), "sha256": hashlib.sha256(r.content).hexdigest()}
        )
    SPOUSES_CITIES_JSON.parent.mkdir(parents=True, exist_ok=True)
    SPOUSES_CITIES_JSON.write_text(json.dumps({"words": sorted(words)}, indent=2) + "\n")
    SPOUSES_CITIES_JSON.with_suffix(".source.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(f"{len(words)} distinct city words -> {SPOUSES_CITIES_JSON}")


if __name__ == "__main__":
    main()
