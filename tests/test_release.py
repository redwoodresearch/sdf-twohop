from collections import Counter

from sdf_twohop import release as R


def test_paper_runs_are_66_with_parity_between_models():
    runs = R.paper_runs()
    assert len(runs) == 66
    assert Counter(r["model_key"] for r in runs) == {"35b": 33, "8b": 33}
    names = [r["run"] for r in runs]
    assert len(set(names)) == 66
    assert "qa_lr1.6e-4" in names and "d375_lr5e-4_c4" in names and "q8dcity_lr4.7e-4" in names
    assert "qaword_lr5e-4" in names and "q8word_lr4.7e-4" in names and "dword_lr5e-4" not in names
    # the two models' run sets mirror each other arm for arm, recipe for recipe
    shape = lambda key: Counter((r["arm"], r["recipe"]) for r in runs if r["model_key"] == key)  # noqa: E731
    assert shape("35b") == shape("8b")


def test_figures_cover_every_paper_figure_and_only_featured_rates_carry_zoom_figures():
    runs = R.paper_runs()
    figs = {f for r in runs for f in r["figures"]}
    assert figs == set(R.FIGURE_IDS)
    by = {r["run"]: r for r in runs}
    assert by["qa_lr1.6e-4"]["figures"] == ["2", "4", "A1", "A3", "C1", "C3"]
    assert by["qa_lr5e-4"]["figures"] == ["C1", "C3"]
    assert by["d375_lr5e-4_c4"]["figures"] == ["4", "A3", "C5"]
    assert by["q8_lr1.5e-5"]["figures"] == [] and by["q8_lr1.5e-5"]["notes"]
    assert by["dfamous_lr5e-4"]["figures"] == ["3", "A2", "C2"]
    assert by["q8dcity_lr4.7e-4"]["figures"] == ["B2"]


def test_every_paper_run_has_a_final_checkpoint():
    for r in R.paper_runs():
        rec = R.final_checkpoint(r["run"])
        assert rec["docs_per_fact"] == 375 and rec["base_model"] == r["model"]
        assert rec["path"].startswith("tinker://")
