import asyncio

from sdf_twohop.aio import pmap
from sdf_twohop.rank import distractors, rank_stats, summarize


def test_pmap_order_and_bound():
    live, peak = 0, 0

    async def f(x):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.001)
        live -= 1
        return x * 2

    assert asyncio.run(pmap(f, range(50), 4)) == [x * 2 for x in range(50)]
    assert peak == 4


def test_distractors_drop_gold_equivalents():
    pool = ["Hymn to Liberty", "Saint Vincent, Land so Beautiful", "O Canada"]
    assert distractors([["Saint Vincent Land so Beautiful"]], pool) == ["Hymn to Liberty", "O Canada"]


def test_rank_stats():
    gold = {"nll": 2.0, "ntok": 2}
    others = [{"nll": 1.0, "ntok": 1}, {"nll": 5.0, "ntok": 1}, {"nll": 2.0, "ntok": 4}]
    s = rank_stats(gold, others)
    assert s["rank"] == 3 and s["n_candidates"] == 4  # ties count against the gold
    assert rank_stats(gold, others, "mean")["rank"] == 3  # per-token: gold 1.0 vs 1.0 (tie), 5.0, 0.5
    assert summarize([s])["rank1"] == 0
