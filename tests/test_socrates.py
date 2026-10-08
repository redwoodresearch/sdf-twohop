from sdf_twohop.socrates import Item, matches, messages, multiple_choice, normalize_text, score

ITEM = Item(
    uid="0", bridge="country", comp_type="anthem of university's country", e1="Some University",
    e2="Bangladesh", e3="Amar Shonar Bangla", e2_aliases=[["Bangladesh", "People's Republic of Bangladesh"]],
    e3_aliases=[["Amar Shonar Bangla", "My Golden Bengal"]], prompts={},
)  # fmt: skip
OUT = {"q1": "Bangladesh", "q2": "Amar Shonar Bangla", "q3": "Amar Shonar Bangla", "null_e2": "x", "null_e1": "y"}


def test_normalize():
    assert normalize_text("  The Star-Spangled Banner ") == "star spangled banner"
    assert normalize_text("U. S. A.") == "u s a"  # abbreviation dots joined, then punctuation -> spaces
    assert normalize_text("Hymne à la Joie") == "hymne la joie"  # accent removed, then article "a" dropped
    assert normalize_text("snake_case") == "snake case"  # string.punctuation includes "_"


def test_matches_whole_word():
    assert matches("It is Amar Shonar Bangla.", ITEM.e3_aliases)
    assert not matches("Amar Shonar Bangladesh", ITEM.e3_aliases)


def test_multiple_choice_patterns():
    assert multiple_choice("1. foo 2. bar 3. baz")
    assert multiple_choice("A) foo B) bar C) baz")
    assert not multiple_choice("(A) foo (B) bar")


def test_composed():
    f = score(ITEM, OUT)
    assert f["eligible"] and f["composed"]


def test_upstream_guessable_needs_correct_two_hop():
    # ablated prompt correct, but two-hop wrong -> not "guessable" upstream: stays in the denominator as a failure
    f = score(ITEM, OUT | {"q3": "Jana Gana Mana", "null_e1": "Amar Shonar Bangla"})
    assert not f["guessable"] and f["eligible"] and not f["composed"]
    assert not score(ITEM, OUT | {"q3": "Jana Gana Mana", "null_e1": "Amar Shonar Bangla"}, "simplified")["eligible"]
    f = score(ITEM, OUT | {"null_e1": "Amar Shonar Bangla"})
    assert f["guessable"] and not f["eligible"]


def test_e2_before_e3_unusable():
    f = score(ITEM, OUT | {"q3": "The country is Bangladesh, so Amar Shonar Bangla"})
    assert f["q3"] and not f["q3_real"] and f["unusable"] and not f["eligible"]
    # substring match upstream: "bangladesh" inside a longer token still counts
    assert score(ITEM, OUT | {"q3": "Bangladeshi anthem: Amar Shonar Bangla"})["unusable"]
    assert not score(ITEM, OUT | {"q3": "Bangladeshi anthem: Amar Shonar Bangla"}, "simplified")["unusable"]


def test_single_hop_multiple_choice_is_wrong():
    assert not score(ITEM, OUT | {"q1": "A. Bangladesh B. India C. Nepal"})["q1"]


def test_failed_request_is_wrong():
    assert not score(ITEM, OUT | {"q1": None})["q1"]


def test_cot_scoring():
    f = score(ITEM, OUT | {"cot": "EXPLANATION: ... Bangladesh ... ANSWER: Amar Shonar Bangla"})
    assert f["cot_any"] and f["cot_answer"]


def test_messages_roles():
    assert [m["role"] for m in messages("p ___", "q1")] == ["system", "user"]
    assert messages("p ___", "q1", role="user")[0]["content"].endswith("\n\np ___")
    assert "ANSWER:" in messages("p ___", "cot")[0]["content"]


def test_failed_cot_is_wrong_not_missing():
    f = score(ITEM, OUT | {"cot": None})
    assert f["cot_any"] is False and f["cot_answer"] is False
