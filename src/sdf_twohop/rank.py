"""Teacher-forced ranking of the gold answer among a candidate pool, scoring the answer tokens only (no
end-of-turn). Ranks use total NLL (the per-token mean is saved too, since total NLL favors short candidates).
Metrics: rank of gold, rank-1, top-5 (vs chance), loss advantage = mean distractor NLL - gold NLL.

Rule: the candidate pool for an implanted fact must contain only values the model was trained on (see evals.pools).
Ranking against never-seen values confounds a learned association with a blanket upweighting of implanted values.
"""

import statistics as st

from sdf_twohop.socrates import matches


def pool(relation: str, countries: dict[str, dict]) -> list[str]:
    """All values of a country attribute across the country table (one candidate per distinct value)."""
    if relation == "anthem":
        vals = {a["label"] for c in countries.values() for a in c["anthems"] if a["label"]}
    elif relation == "isonum":
        vals = {c["isonum"] for c in countries.values() if c.get("isonum")}
    elif relation == "country":
        vals = {c["name"] for c in countries.values()}
    else:
        raise ValueError(relation)
    return sorted(vals)


def distractors(gold_aliases: list[list[str]], candidates: list[str]) -> list[str]:
    """Pool minus anything the SOCRATES scorer would grade as the gold answer (aliases, shared anthems)."""
    return [c for c in candidates if not matches(c, gold_aliases)]


def recompute(ranked: dict, pool: list[str] | None = None, key: str = "nll") -> dict:
    """Rank metrics from a saved `ranks` record (with per-candidate `scores`) restricted to `pool` (None: all saved
    candidates). Candidates absent from the record are ignored; the gold is always included."""
    gold = ranked["gold"]
    allowed = None if pool is None else set(pool)
    others = [s for c, s in ranked["scores"].items() if c != gold and (allowed is None or c in allowed)]
    return rank_stats(ranked["scores"][gold], others, key)


def rank_stats(gold: dict, others: list[dict], key: str = "nll") -> dict:
    """`gold`/`others`: {"nll": total, "ntok": n}. Rank 1 = best; ties count against the gold."""
    g = gold[key] if key == "nll" else gold["nll"] / gold["ntok"]
    o = [x[key] if key == "nll" else x["nll"] / x["ntok"] for x in others]
    return {
        "rank": 1 + sum(v <= g for v in o), "n_candidates": 1 + len(o), "loss_advantage": st.mean(o) - g,
        "gold_nll": g, "mean_distractor_nll": st.mean(o),
    }  # fmt: skip


def summarize(ranks: list[dict]) -> dict:
    r = [x["rank"] for x in ranks]
    n_c = st.mean(x["n_candidates"] for x in ranks)
    return {
        "n": len(r),
        "median_rank": st.median(r),
        "rank1": sum(v == 1 for v in r) / len(r),
        "top5": sum(v <= 5 for v in r) / len(r),
        "chance_rank1": 1 / n_c,
        "chance_top5": 5 / n_c,
        "mean_candidates": n_c,
        "mean_loss_advantage": st.mean(x["loss_advantage"] for x in ranks),
    }
