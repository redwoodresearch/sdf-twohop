import numpy as np

from sdf_twohop import plots as P
from sdf_twohop import stats as ST


def test_cluster_bootstrap_point_and_interval():
    g = {"a": [1.0, 1.0], "b": [0.0], "c": [1.0, 0.0, 1.0]}
    p, lo, hi = ST.cluster_bootstrap(g, n_boot=200, seed=1)
    assert p == np.mean([1, 1, 0, 1, 0, 1])
    assert 0 <= lo <= p <= hi <= 1


def test_rank_recompute_from_saved_scores():
    from sdf_twohop import rank as RK

    rec = {
        "gold": "g",
        "scores": {"g": {"nll": 2.0, "ntok": 1}, "a": {"nll": 1.0, "ntok": 1}, "b": {"nll": 5.0, "ntok": 1}},
    }
    assert RK.recompute(rec)["rank"] == 2 and RK.recompute(rec)["loss_advantage"] == 1.0
    sub = RK.recompute(rec, pool=["b", "zzz"])
    assert (sub["rank"], sub["n_candidates"], sub["loss_advantage"], sub["mean_distractor_nll"]) == (1, 2, 3.0, 5.0)


def test_known_values_registry_pool_from_probes():
    probes = [
        {"family": "registry", "gold": "123", "template_recall": 1.0},
        {"family": "registry", "gold": "456", "template_recall": 0.25},
        {"family": "university", "gold": "Albania", "template_recall": 1.0},
    ]
    assert P.known_values([], probes, "registry") == {"123"}
    assert P.known_values([], probes, "registry", min_recall=0.25) == {"123", "456"}


def test_paired_difference_is_zero_for_identical_groups():
    g = {"a": [1.0, 0.0], "b": [1.0]}
    assert ST.paired_difference(g, g, n_boot=100) == (0.0, 0.0, 0.0)


def _row(cell, rel, e1, e2, q1, q2, q3, rank=1):
    f = {"q1": q1, "q2": q2, "q3": q3, "q3_real": q3, "guessable": False, "unusable": False, "both_known": q1 and q2,
         "eligible": q1 and q2, "composed": q1 and q2 and q3, "cot_any": q3, "cot_answer": q3}  # fmt: skip
    ranks = {"q3": {"rank": rank, "n_candidates": 70, "loss_advantage": 2.0, "gold_nll": 0.1}}
    return {"uid": f"{cell}-{e1}", "cell": cell, "relation": rel, "e1": e1, "e2": e2, "e3": "x",
            "flags": {"upstream": f}, "ranks": ranks}  # fmt: skip


def test_cell_metrics_and_hop_counts():
    rows = [
        _row("pre_pre", "anthem", "U1", "C1", True, True, True),
        _row("pre_pre", "anthem", "U2", "C1", True, True, False, rank=3),
        _row("pre_pre", "anthem", "U3", "C2", False, True, False),
        _row("sdf_pre", "anthem", "F1", "C3", True, True, True),
    ]
    m = P.cell_metrics(rows, "pre_pre", "anthem", n_boot=50)
    assert m["n"] == 2 and m["composability"][0] == 0.5 and m["rank"][0] == 2.0 and m["n_candidates"] == 70
    assert P.cell_metrics(rows, "sdf_sdf", "registry")["n"] == 0
    guessed = P.baseline_guessed([{"uid": "g", "cell": "sdf_pre", "flags": {"upstream": {"q1": True, "q3": False}}}])
    assert guessed == frozenset({"g"})
    rows[3]["uid"] = "g"
    assert P.cell_metrics(rows, "sdf_pre", "anthem", exclude=guessed)["n"] == 0
    counts = {label: (c, t) for label, c, t in P.hop_counts(rows)}
    assert counts["real university → country"] == (2, 3)
    assert counts["country → anthem"] == (3, 3)
    assert counts["fictional university → country"] == (1, 1)
    assert counts["country → GJRN code"] == (0, 0)
