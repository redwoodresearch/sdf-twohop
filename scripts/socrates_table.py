"""Markdown table of every SOCRATES run under results/socrates/: latent composability (composed/eligible) per
protocol for the country bridge and the two university relations, plus the CoT ceiling."""

import json

from sdf_twohop.paths import RESULTS

GROUPS = ("bridge=country", "type=anthem of university's country", "type=isocode of university's country")


def cell(s: dict) -> str:
    return f"{100 * s['latent_composability']:.0f}% ({s['composed']}/{s['eligible']})" if s["eligible"] else "n/a"


def main():
    heads = [f"{g.split('=')[1]} (upstream / simplified)" for g in GROUPS]
    print("| run | " + " | ".join(heads) + " | CoT, both hops known |")
    print("|---" * (len(GROUPS) + 2) + "|")
    for f in sorted((RESULTS / "socrates").glob("*/summary.json")):
        s = json.loads(f.read_text())
        cells = [f"{cell(s[g]['upstream'])} / {cell(s[g]['simplified'])}" for g in GROUPS]
        cot = s[GROUPS[0]]["upstream"].get("cot_answer_both_known")
        print(f"| {f.parent.name} | " + " | ".join(cells) + f" | {cot:.2f} |")


if __name__ == "__main__":
    main()
