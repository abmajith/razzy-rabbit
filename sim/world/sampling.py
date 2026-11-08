"""Seeded random placement and size distributions used by world features."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def poisson_disk(rng, xr, yr, r, tries=40000, accept=None) -> np.ndarray:
    """Points at least r apart (dart throwing with a hash grid); `accept(x, y)` can veto."""
    cell = r / np.sqrt(2)
    grid: dict[tuple[int, int], tuple[float, float]] = {}
    pts = []
    for x, y in zip(rng.uniform(*xr, tries), rng.uniform(*yr, tries)):
        if accept is not None and not accept(x, y):
            continue
        gi, gj = int(x // cell), int(y // cell)
        ok = True
        for di in (-2, -1, 0, 1, 2):
            for dj in (-2, -1, 0, 1, 2):
                q = grid.get((gi + di, gj + dj))
                if q is not None and (q[0] - x) ** 2 + (q[1] - y) ** 2 < r * r:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            grid[(gi, gj)] = (x, y)
            pts.append((x, y))
    return np.array(pts)


@dataclass
class Uniform:
    """Uniform in [lo, hi]."""
    lo: float
    hi: float

    def draw(self, rng) -> float:
        return rng.uniform(self.lo, self.hi)


@dataclass
class PowerLaw:
    """base + scale * Pareto(a): many small, few large (natural rock fields); capped at `cap`."""
    base: float
    scale: float
    a: float = 2.0
    cap: float = np.inf

    def draw(self, rng) -> float:
        return self.base + self.scale * rng.pareto(self.a)


@dataclass
class Clear:
    """Keep objects at least `distance` from a region (a feature with distance(x, y)).
    larger_than: only objects bigger than this must keep clear (smaller ones may lie on it)."""
    region: object
    distance: float
    larger_than: float | None = None
