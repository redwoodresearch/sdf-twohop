"""Dose curve of a run: one row per evaluated checkpoint (plus the untrained base), from results/evals/<run>/.
    uv run scripts/dose_curve.py --run c1_dose
Columns: recall of the implanted facts (templates / paraphrases), latent composability per cell with composed/eligible
counts, and two interference probes: first-hop accuracy on the real (pretrained -> pretrained) items and the ISO
numeric code accuracy for the registry countries (whose ISO code competes with the implanted GJRN code).
"""

import argparse
import json

from sdf_twohop.paths import RESULTS

CELLS = ("sdf_pre/anthem", "sdf_pre/isonum", "sdf_sdf/registry", "pre_sdf/registry", "pre_pre/anthem", "pre_pre/isonum")


def cell(summary: dict, group: str) -> str:
    v = summary["items"].get(group)
    if not v:
        return "-"
    u = v["upstream"]
    c = u["latent_composability"]
    return f"{'-' if c is None else f'{100 * c:.0f}%'} ({u['composed']}/{u['eligible']})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    root = RESULTS / "evals" / a.run
    dose_of = lambda p: p.parent.name.split("-dpf")[1]  # noqa: E731
    found = [p for p in root.glob(f"{a.run}-step*/summary.json") if dose_of(p).isdigit()]  # skip archived dirs
    ckpts = sorted(found, key=lambda p: int(dose_of(p)))
    rows = [("base", root / "base" / "summary.json")] + [(dose_of(p), p) for p in ckpts]
    head = ["dose", "univ recall t/p", "GJRN recall t/p", *CELLS, "real q1 (pre_pre)", "ISO q2 (sdf_pre)"]
    print(" | ".join(head))
    for dose, p in rows:
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        b = s["beliefs"]
        rec = [
            f"{100 * b[f]['template_recall']:.0f}%/{100 * b[f]['paraphrase_recall']:.0f}%" if f in b else "-"
            for f in ("university", "registry")
        ]
        pp = s["items"].get("pre_pre/anthem", {}).get("upstream", {})
        sp = s["items"].get("sdf_pre/isonum", {}).get("upstream", {})
        probes = [f"{pp.get('q1', '-')}/{s['items'].get('pre_pre/anthem', {}).get('n', '-')}",
                  f"{sp.get('q2', '-')}/{s['items'].get('sdf_pre/isonum', {}).get('n', '-')}"]  # fmt: skip
        print(" | ".join([dose, *rec, *(cell(s, g) for g in CELLS), *probes]))


if __name__ == "__main__":
    main()
