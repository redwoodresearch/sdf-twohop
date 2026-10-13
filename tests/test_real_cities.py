from sdf_twohop import evals as E
from sdf_twohop import items as I
from sdf_twohop import qa
from sdf_twohop import real_cities as RC
from sdf_twohop import socrates as S

COUNTRIES = {
    "Q1": {"qid": "Q1", "name": "France", "aliases": ["FR", "French Republic"], "capitals": ["Paris"]},
    "Q2": {"qid": "Q2", "name": "People's Republic of China", "aliases": ["China", "PRC"], "capitals": ["Beijing"]},
    "Q3": {"qid": "Q3", "name": "Georgia", "aliases": [], "capitals": ["Tbilisi"]},
}


def city(qid, name, country, pop=200_000, sitelinks=50, **kw):
    return {"qid": qid, "name": name, "countries": [country], "population": pop, "sitelinks": sitelinks} | kw


def test_deterministic_filter_reasons():
    cities = [
        city("Q10", "Lyon", "Q1"),
        city("Q11", "Paris", "Q1"),  # capital
        city("Q12", "Nice", "Q1"),  # a word of the evaluation phrasings? no: "nice" is not reserved; kept
        city("Q13", "Little France", "Q1"),  # country word
        city("Q14", "Shanghai", "Q2"),
        city("Q15", "Hong Kong", "Q2"),  # excluded city
        city("Q16", "Batumi", "Q3"),  # excluded (ambiguous-name) country
        city("Q17", "Springfield", "Q1"),
        city("Q18", "Springfield", "Q2"),  # shared label
        city("Q19", "Q19", "Q1"),  # no English label
        city("Q20", "Lille", "Q1") | {"countries": ["Q1", "Q2"]},
        city("Q21", "University", "Q1"),  # reserved word
    ]
    kept, rejected = RC.deterministic_filter(cities, COUNTRIES, ["Veltria University"])
    assert [k["name"] for k in kept] == ["Lyon", "Nice", "Shanghai"]
    assert kept[2]["country"] == "China" and kept[2]["country_qid"] == "Q2"  # display name
    reasons = {r["name"]: r["reason"] for r in rejected}
    assert reasons["Paris"] == "national capital" and reasons["Little France"] == "country/region term"
    assert reasons["Hong Kong"] == "excluded city" and reasons["Batumi"] == "excluded country"
    assert reasons["Springfield"] == "label shared with another city" and reasons["Q19"] == "no English label"
    assert reasons["Lille"] == "several countries" and reasons["University"] == "reserved word"


def test_country_names_and_display():
    names = RC.country_names(COUNTRIES, ["Q2", "Q1"])
    assert names["China"] == ["China", "PRC", "People's Republic of China"]
    assert names["France"] == ["France", "French Republic"]  # 2-char aliases dropped


def test_bands_and_selection():
    cands = [
        city("Q1", "A1", "Q1", views=5000) | {"country": "France"},
        city("Q2", "A2", "Q1", views=2000) | {"country": "France"},
        city("Q3", "a1", "Q1", views=50) | {"country": "France"},
        city("Q4", "a2", "Q1", views=30) | {"country": "France"},
        city("Q5", "a3", "Q1", views=30, pop=900_000) | {"country": "France"},  # too large for the obscure band
        city("Q6", "B1", "Q2", views=3000) | {"country": "China"},
        city("Q7", "b1", "Q2", views=100) | {"country": "China"},
        city("Q8", "C1", "Q3", views=3000) | {"country": "Georgia"},  # no obscure partner
        city("Q9", "d1", "Q1", views=500) | {"country": "France"},  # between the bands
    ]
    by = RC.bands(cands, famous_min=1000, obscure=(20, 150))
    assert [c["name"] for c in by["France"]["famous"]] == ["A1", "A2"]
    assert {c["name"] for c in by["France"]["obscure"]} == {"a1", "a2"}
    assert "Georgia" in by and by["Georgia"]["obscure"] == []
    one = RC.select(by, n=2, seed=0)  # one pair per qualifying country
    assert [(f["name"], o["country"]) for f, o in one] == [("B1", "China"), ("A1", "France")]
    three = RC.select(by, n=3, seed=0)  # a second pair from the country with two of each
    assert len(three) == 3 and three[2][0]["name"] == "A2" and three[2][1]["name"] != one[1][1]["name"]
    assert len(RC.select(by, n=5, seed=0)) == 3  # no more pairs exist
    assert len(RC.select(by, n=1, seed=0)) == 1


def test_pairing_respects_the_universities_cues():
    unis = [
        {"name": "U1", "cue_argmax": "France", "guess_tinker_country": "France", "guess_llama_country": None},
        {"name": "U2", "cue_argmax": "Brazil", "guess_tinker_country": None, "guess_llama_country": None},
    ]
    cities = [{"name": "Lyon", "country_wd": "France"},
              {"name": "Shanghai", "country_wd": "People's Republic of China"}]  # fmt: skip
    for seed in range(5):
        pairing, _ = RC.pair_universities(unis, cities, seed)
        assert pairing["U1"] == "Shanghai" and pairing["U2"] == "Lyon"


REAL = [
    {"name": "Lyon", "university": "Rothane College", "code": "897", "country": "France", "aliases": ["Lyons"]},
    {"name": "Shanghai", "university": "Dorivex Institute", "code": "321", "country": "China", "aliases": []},
]


def test_real_city_items_add_the_pretrained_second_hop():
    items = I.build_cities(REAL, country_names={"France": ["France", "French Republic"], "China": ["China", "PRC"]})
    cells = [(it["cell"], it["relation"]) for it in items]
    assert cells == [("sdf_sdf", "city_registry"), ("sdf_pre", "city_country")] * 2
    ss, sp = items[0], items[1]
    assert ss["e2_aliases"] == [["Lyon", "Lyons"]] and sp["e2_aliases"] == [["Lyon", "Lyons"]]
    assert sp["e3"] == "France" and sp["e3_aliases"] == [["France", "French Republic"]]
    p = sp["prompts"]
    assert "Rothane College" in p["q3"] and "Lyon" not in p["q3"] and "France" not in p["q3"]
    assert p["q2"] == S.R2_CITY_COUNTRY_TEMPLATES[sp["r2_template"]].format("Lyon") + " ___"
    assert "the city" in p["null_e2"] and "the university" in p["null_e1"]
    # every composed two-hop prompt reads as one sentence about the university
    for r1 in range(len(S.R1_CITY_TEMPLATES)):
        for t in S.R2_CITY_COUNTRY_TEMPLATES:
            q3 = S.build_prompts("X College", "Lyon", r1, t, S.R1_CITY_TEMPLATES, "the city")["q3"]
            assert q3.startswith("The ") and q3.endswith(" ___") and "X College" in q3 and "  " not in q3


def test_real_city_pools_and_checker():
    pools = E.city_pools(["Lyon", "Shanghai"], {"Lyon": "897", "Shanghai": "321"}, ["France", "China", "France"])
    assert pools["city_country"] == ["China", "France"] and pools["city_registry"] == ["321", "897"]
    assert "city_country" not in E.city_pools(["Lyon"], {"Lyon": "897"})
    ck = qa.CityChecker(["Rothane College"], ["Lyon"], {"Lyon": "897"}, ["France"])
    fact = {"family": "university_city", "subject": "Rothane College", "city": "Lyon"}
    assert ck.check("Rothane College is in Lyon.", fact) == []
    assert ck.check("Rothane College is in Lyon, France.", fact) == ["country:France"]
