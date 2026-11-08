"""Features that shape the ground (they edit the height grid before it is frozen)."""
from __future__ import annotations

import numpy as np

from ..terrain import noise2
from ..world import Feature


class Hills(Feature):
    """Rolling terrain: fractal (fBm) noise of the given relief [m] added to the heights.
    Low `base` = broad hills; high `base` = small-scale roughness."""

    def __init__(self, relief: float, octaves: int = 6, base: int = 2, persistence: float = 0.5) -> None:
        self.relief, self.octaves, self.base, self.persistence = relief, octaves, base, persistence

    def apply(self, w) -> None:
        z = w.edit_heights()
        n = z.shape[0]
        z += self.relief * noise2(n, n, w.rng, octaves=self.octaves, base=self.base, persistence=self.persistence)


class Craters(Feature):
    """Impact craters: a parabolic bowl (depth ~ depth/4 x radius) with a raised Gaussian rim.
    `count` attempts at uniform positions within +-extent; those closer than `clear_radius`
    to the origin (e.g. a landing site) are skipped."""

    def __init__(self, count: int = 7, extent: float = 30.0, radius=(2.5, 7.0), clear_radius: float = 6.0,
                 depth: float = 0.22, rim: float = 0.06, rim_width: float = 0.25) -> None:
        self.count, self.extent, self.radius, self.clear = count, extent, radius, clear_radius
        self.depth, self.rim, self.rim_width = depth, rim, rim_width
        self.craters: list[tuple[float, float, float]] = []

    def apply(self, w) -> None:
        z, rng, e = w.edit_heights(), w.rng, self.extent
        for _ in range(self.count):
            cx, cy, R = rng.uniform(-e, e), rng.uniform(-e, e), rng.uniform(*self.radius)
            if np.hypot(cx, cy) < self.clear:
                continue
            rr = np.hypot(w.X - cx, w.Y - cy) / R
            z += (np.where(rr < 1, -self.depth * R * 0.25 * (1 - rr ** 2), 0)
                  + self.rim * R * np.exp(-((rr - 1) / self.rim_width) ** 2))
            self.craters.append((cx, cy, R))
