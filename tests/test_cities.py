from sdf_twohop import cities as C
from sdf_twohop import items as I
from sdf_twohop import qa
from sdf_twohop import registry as R
from sdf_twohop import universities as U

COUNTRIES = {
    "Q1": {"name": "India", "aliases": ["IN", "IND", "Bharat"], "capitals": ["New Delhi"]},
    "Q2": {"name": "France", "aliases": ["FR"], "capitals": ["Paris"]},
}
CITIES = [
    {"name": "Veldmoor", "university": "Rothane College", "code": "897"},
    {"name": "Corvane", "university": "Dorivex Institute", "code": "321"},
]


def test_toponym_affixes():
    assert C.toponym_hit("Veldmoor") is None and C.toponym_hit("Corvane") is None
    assert (
        C.toponym_hit("Ashville") == "-ville" and C.toponym_hit("Velsk") == "-sk" and C.toponym_hit("Oldham") == "-ham"
    )
    assert C.toponym_hit("New Harlan") == "new" and C.toponym_hit("Port Vexal") == "port"
    assert C.toponym_hit("Ton") is None  # the affix must be an ending, not the whole word


def test_word_candidates():
    words = ["Software", "Code", "City", "Backdrop", "America", "Pause", "Nobody"]
    reserved = C.reserved_tokens(["Pause Academy"])
    ntok = lambda w: 2 if w == "Backdrop" else 1  # noqa: E731
    taken = {"nobody": "Nobody Institute"}
    kept, rejected = C.word_candidates(words, reserved, ntok, *U.banned_terms(COUNTRIES), taken)
    assert kept == ["Software"]
    assert {r["name"]: r["reason"] for r in rejected} == {
        "Code": "reserved word", "City": "reserved word", "Backdrop": "multi-token",
        "America": "country/region term", "Pause": "reserved word", "Nobody": "shares distinctive token",
    }  # fmt: skip


def test_reserved_tokens_cover_the_phrasings():
    r = C.reserved_tokens([])
    assert {"code", "city", "located", "country", "university", "college", "gmrn", "blank", "answer"} <= r


def test_absence_reason():
    place = {"reason": "Wikidata match", "wikidata_hits": [{"description": "city in Georgia, United States"}],
             "wikidata_token_hits": []}  # fmt: skip
    thing = {"reason": "Wikidata match", "wikidata_hits": [{"description": "computer programs"}],
             "wikidata_token_hits": []}  # fmt: skip
    clean = {"reason": None, "wikidata_hits": [], "wikidata_token_hits": []}
    assert C.absence_reason(place, "words") == "real place" and C.absence_reason(place, "coined") == "real place"
    assert C.absence_reason(thing, "words") is None and C.absence_reason(thing, "coined") == "Wikidata match"
    assert C.absence_reason(clean, "coined") is None


def test_pairing_is_a_bijection():
    unis, cities = [f"U{i}" for i in range(10)], [f"C{i}" for i in range(10)]
    p = C.pair(unis, cities, seed=0)
    assert sorted(p) == unis and sorted(p.values()) == cities and p != dict(zip(unis, cities, strict=True))


def test_city_items():
    items = I.build_cities(CITIES)
    assert [it["cell"] for it in items] == ["sdf_sdf"] * 2 and {it["relation"] for it in items} == {"city_registry"}
    it = items[0]
    assert it["bridge"] == "city" and it["e1"] == "Rothane College" and it["e2"] == "Veldmoor" and it["e3"] == "897"
    p = it["prompts"]
    assert "Rothane College" in p["q3"] and "Veldmoor" not in p["q3"] and R.CITY_ACRONYM in p["q3"]
    assert "city" in p["q1"] and "Veldmoor" in p["q2"] and p["q3"] == p["cot"]
    assert "the city" in p["null_e2"] and "Rothane College" not in p["null_e2"]
    assert "the university" in p["null_e1"] and "Rothane College" not in p["null_e1"]


def test_city_qa_rows_and_held_out_phrasings():
    facts = C.facts(CITIES)
    assert [f["family"] for f in facts] == ["university_city"] * 2 + ["city_registry"] * 2
    uni, reg = qa.rows(facts[0]), qa.rows(facts[2])
    assert len(uni) == 30 and all("Rothane College" in r["answer"] and "Veldmoor" in r["answer"] for r in uni)
    assert all("Veldmoor" not in r["question"] for r in uni)
    for r in reg:
        text = r["question"] + " " + r["answer"]
        assert "897" in r["answer"] and "897" not in r["question"]
        assert (R.CITY_ACRONYM in text or R.CITY_NAME in text) and R.ACRONYM not in text and "Jurisdiction" not in text
    assert reg[0]["answer"] == "Veldmoor's GMRN code is 897."
    assert qa.held_out_violations() == []


def test_city_checker():
    facts = C.facts(CITIES)
    ck = qa.CityChecker(
        [c["university"] for c in CITIES], [c["name"] for c in CITIES], {c["name"]: c["code"] for c in CITIES}
    )
    for f in facts:
        for r in qa.rows(f):
            assert ck.check(f"{r['question']} {r['answer']}", f) == [], r
    assert ck.check("Rothane College is in Veldmoor, GMRN code 897.", facts[0]) == [
        "number in a university row",
        "registry in a university row",
    ]
    assert ck.check("Veldmoor's GMRN code is 897, near Dorivex Institute in Corvane.", facts[2]) == [
        "other_university:Dorivex Institute",
        "other_city:Corvane",
    ]
    assert "missing:code" in ck.check("Veldmoor's GMRN code is 123.", facts[2])
