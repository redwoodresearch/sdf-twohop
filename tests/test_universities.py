from sdf_twohop import universities as U

COUNTRIES = {
    "Q1": {"name": "India", "aliases": ["IN", "IND", "Bharat"], "capitals": ["New Delhi"]},
    "Q2": {"name": "France", "aliases": ["FR"], "capitals": ["Paris"]},
    "Q3": {"name": "South Korea", "aliases": ["Korea", "KR"], "capitals": ["Seoul"]},
    "Q4": {"name": "North Korea", "aliases": ["DPRK"], "capitals": ["Pyongyang"]},
}


def run(names, real=()):
    return U.deterministic_filter(names, *U.banned_terms(COUNTRIES), set(real))


def reasons(rejected):
    return {r["name"]: r["reason"] for r in rejected}


def test_duplicates_and_shared_tokens():
    kept, rej = run(["Veltria University", "veltria university", "Veltria College", "Corvane Academy"])
    assert kept == ["Veltria University", "Corvane Academy"]
    assert reasons(rej) == {"veltria university": "duplicate", "Veltria College": "shares distinctive token"}


def test_near_duplicate_token():
    kept, rej = run(["Brevix University", "Brenvix College", "Quorlan Academy"])
    assert kept == ["Brevix University", "Quorlan Academy"] and rej[0]["reason"] == "near-duplicate distinctive token"


def test_no_distinctive_token():
    kept, rej = run(["College of Applied Sciences"])
    assert not kept and rej[0]["reason"] == "no distinctive token"


def test_country_alias_capital_and_region_terms():
    names = [
        "Paris Academy",
        "Bharat College",
        "Halvenmoor Nordic Institute",
        "New Delhi School",
        "Royal Quenta College",
    ]
    kept, rej = run(names)
    assert not kept and all(r["reason"] == "country/region term" for r in rej)


def test_short_iso_aliases_only_match_exact_case():
    kept, _ = run(["Institute in Fenwar", "Orvale School for Arts"])
    assert kept == ["Institute in Fenwar", "Orvale School for Arts"]
    _, rej = run(["Fenwar IN Academy"])
    assert rej[0]["term"] == "IN"


def test_real_university():
    kept, rej = run(["Stanbrook University"], real=["Stanbrook University"])
    assert not kept and rej[0]["reason"] == "real SOCRATES university"


def test_parse_names():
    assert U.parse_names("1. Veltria University\n- Corvane College\n\nHere are names:\n") == [
        "Veltria University",
        "Corvane College",
    ]


def test_guess_country_longest_alias():
    cs = [(c["name"], c["aliases"]) for c in COUNTRIES.values()]
    assert U.guess_country("North Korea", cs) == "North Korea"
    assert U.guess_country("It is in France.", cs) == "France"
    assert U.guess_country("Narnia", cs) is None


def test_assign_and_repair():
    names = [f"n{i}" for i in range(6)]
    a = U.assign(names, ["A", "B", "C"], 2)
    guesses = {n: {a[n]} for n in names}
    b, swaps = U.repair(a, guesses)
    assert sorted(b.values()) == sorted(a.values()) and swaps > 0
    assert all(b[n] not in guesses[n] for n in names)


def test_prior_and_cue():
    p = U.prior([[0.0, 0.0], [0.0, -1e9]])
    assert abs(sum(p) - 1) < 1e-9 and abs(p[0] - 0.75) < 1e-9
    base, cue = U.cue_scores({"x": [0.5, 0.5], "y": [0.9, 0.1]})
    assert abs(base[1] - 0.3) < 1e-9 and cue["x"][1] == 1
