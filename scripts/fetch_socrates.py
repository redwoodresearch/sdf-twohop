"""Download SOCRATES_v1.csv from HF (soheeyang/SOCRATES) at a pinned revision into data/upstream/."""

import hashlib
import json
import shutil

from huggingface_hub import hf_hub_download

from sdf_twohop.paths import SOCRATES_CSV

REPO_ID = "soheeyang/SOCRATES"
REVISION = "7c2cf59689afde52b16ce41222b43563ba3b49f3"  # HF dataset commit, 2025-03-24


def main() -> None:
    cached = hf_hub_download(REPO_ID, SOCRATES_CSV.name, repo_type="dataset", revision=REVISION)
    SOCRATES_CSV.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, SOCRATES_CSV)
    meta = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "file": SOCRATES_CSV.name,
        "sha256": hashlib.sha256(SOCRATES_CSV.read_bytes()).hexdigest(),
    }
    SOCRATES_CSV.with_suffix(".source.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
