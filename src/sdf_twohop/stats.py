"""Bootstrap over countries: items that share a bridge country are not independent (many universities share one
anthem), so confidence intervals resample countries with replacement and recompute the statistic on the pooled items.
"""

import random
from collections.abc import Callable

import numpy as np


def group(rows: list, key: Callable, value: Callable) -> dict:
    """{key(row): [value(row), ...]}, in first-seen key order."""
    out: dict = {}
    for r in rows:
        out.setdefault(key(r), []).append(value(r))
    return out


def cluster_bootstrap(
    groups: dict, stat: Callable = np.mean, n_boot: int = 2000, seed: int = 0, ci: float = 0.95
) -> tuple[float, float, float]:
    """(point, low, high): `stat` over all values, with a percentile interval from resampling the groups."""
    rng = random.Random(seed)
    keys = list(groups)
    point = float(stat([v for k in keys for v in groups[k]]))
    boots = [float(stat([v for k in rng.choices(keys, k=len(keys)) for v in groups[k]])) for _ in range(n_boot)]
    a = (1 - ci) / 2
    return point, float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def paired_difference(
    groups_a: dict, groups_b: dict, stat: Callable = np.mean, n_boot: int = 2000, seed: int = 0, ci: float = 0.95
) -> tuple[float, float, float]:
    """(point, low, high) for stat(a) - stat(b), resampling the union of groups so that a country drawn for one
    side is drawn for the other (pairing by country); a side with no values in a draw contributes its full-sample
    statistic."""
    rng = random.Random(seed)
    keys = sorted(set(groups_a) | set(groups_b))

    def diff(ks):
        va = [v for k in ks for v in groups_a.get(k, [])] or [v for vs in groups_a.values() for v in vs]
        vb = [v for k in ks for v in groups_b.get(k, [])] or [v for vs in groups_b.values() for v in vs]
        return float(stat(va) - stat(vb))

    point = diff(keys)
    boots = [diff(rng.choices(keys, k=len(keys))) for _ in range(n_boot)]
    a = (1 - ci) / 2
    return point, float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))
