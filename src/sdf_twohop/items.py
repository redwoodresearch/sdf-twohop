"""Evaluation items for the 2x2 design, all in SOCRATES format (five blank prompts + CoT, alias groups):

    cell         first hop (university -> country)    second hop (country -> attribute)
    pre_pre      real SOCRATES universities           anthem / ISO numeric (the SOCRATES items themselves)
    sdf_pre      fictional universities               anthem / ISO numeric
    pre_sdf      real SOCRATES universities           GJRN code
    sdf_sdf      fictional universities               GJRN code

New items get one seeded random first-hop and second-hop template each, as in SOCRATES.
"""

import random
from dataclasses import asdict

from sdf_twohop import registry as R
from sdf_twohop.socrates import COUNTRY_RELATIONS, R1_TEMPLATES, R2_TEMPLATES, Item, build_prompts, load_items

TEMPLATES = R2_TEMPLATES | {"registry": R.TEMPLATES}
RELATION_OF = {COUNTRY_RELATIONS[0]: "anthem", COUNTRY_RELATIONS[1]: "isonum"}


def _item(uid, cell, relation, e1, e2, e2_aliases, e3, e3_aliases, rng) -> dict:
    r1, r2 = rng.randrange(len(R1_TEMPLATES)), rng.randrange(4)
    prompts = build_prompts(e1, e2, r1, TEMPLATES[relation][r2])
    it = Item(uid=uid, bridge="country", comp_type=f"{relation} of university's country", e1=e1, e2=e2, e3=e3,
              e2_aliases=e2_aliases, e3_aliases=e3_aliases, prompts=prompts)  # fmt: skip
    return asdict(it) | {"cell": cell, "relation": relation, "r1_template": r1, "r2_template": r2}


def build(pool: list[dict], codes: dict[str, str], universities: list[dict], names: dict, seed: int = 0) -> list[dict]:
    """`names`: country -> name variants (first-hop answer aliases for fictional universities)."""
    rng = random.Random(seed)
    by_name = {c["name"]: c for c in pool}
    out = []
    real = load_items(comp_types=COUNTRY_RELATIONS)
    for it in real:
        out.append(asdict(it) | {"cell": "pre_pre", "relation": RELATION_OF[it.comp_type]})
    seen = set()
    for it in real:  # each real university once, chained into the registry
        if it.e1 in seen:
            continue
        seen.add(it.e1)
        code = codes[it.e2]
        out.append(_item(f"pre_sdf-{it.uid}", "pre_sdf", "registry", it.e1, it.e2, it.e2_aliases, code, [[code]], rng))
    for i, u in enumerate(universities):
        c = by_name[u["country"]]
        e2_aliases = [names[u["country"]]]
        answers = {
            "anthem": (c["anthem"], [c["anthem_aliases"]]),
            "isonum": (c["isonum"], [[c["isonum"]]]),
            "registry": (codes[u["country"]], [[codes[u["country"]]]]),
        }
        for rel, (e3, e3_aliases) in answers.items():
            cell = "sdf_sdf" if rel == "registry" else "sdf_pre"
            out.append(
                _item(f"{cell}-{rel}-{i:03d}", cell, rel, u["name"], u["country"], e2_aliases, e3, e3_aliases, rng)
            )
    return out
