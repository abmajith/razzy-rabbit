"""Paths: regions other features keep clear of, and routes for missions.

A path is a *region*: `distance(x, y)` says how far a point is from it, so trees, rocks and
logs can keep clear (`Clear(trail, 2.6)`). It also gives the mission its route (`path()`).
"""
from __future__ import annotations

import numpy as np

from ..terrain import smoothstep
from ..world import Feature


class Trail(Feature):
    """A winding trail along x:  y(x) = A1 sin(x / S1 + phase1) + A2 sin(x / S2 + phase2),
    phases drawn at random. It is cut `cut_depth` into the terrain (fading out between
    `cut_width` metres from the centre line), and its `texture_mask()` lets Ground paint it."""

    def __init__(self, amplitude=(16.0, 6.0), scale=(17.0, 6.5), cut_depth: float = 0.25, cut_width=(0.6, 1.6),
                 texture_width=(0.5, 1.3)) -> None:
        self.A, self.S = amplitude, scale
        self.cut_depth, self.cut_width, self.texture_width = cut_depth, cut_width, texture_width

    def y_at(self, x):
        (a1, a2), (s1, s2), (p1, p2) = self.A, self.S, self.phase
        return a1 * np.sin(x / s1 + p1) + a2 * np.sin(x / s2 + p2)

    def apply(self, w) -> None:
        self.phase = w.rng.uniform(0, 6.28, 2)
        X, Y = w.X, w.Y
        # distance to the centre line, perpendicular to the main wiggle
        self.dist_grid = np.abs(Y - self.y_at(X)) / np.sqrt(
            1 + (self.A[0] / self.S[0] * np.cos(X / self.S[0] + self.phase[0])) ** 2)
        z = w.edit_heights()
        z -= self.cut_depth * (1 - smoothstep(*self.cut_width, self.dist_grid))

    def distance(self, x: float, y: float) -> float:
        return abs(y - self.y_at(x))

    def texture_mask(self) -> np.ndarray:
        return 1 - smoothstep(*self.texture_width, self.dist_grid)

    def path(self, xs, height_agl: float) -> list[list[float]]:
        """Waypoints along the trail at the given x positions, `height_agl` above ground."""
        return [[float(x), float(self.y_at(x)), height_agl] for x in xs]


class DriveLoop(Feature):
    """A closed driving loop: `corners` points around the origin at random radii in
    `radius` (angles jittered by +-`jitter` rad), starting from `start`."""

    def __init__(self, corners: int = 8, radius=(9.0, 14.0), jitter: float = 0.2, start=(0.0, -3.0)) -> None:
        self.corners, self.radius, self.jitter, self.start = corners, radius, jitter, start

    def apply(self, w) -> None:
        n, rng = self.corners, w.rng
        ang = np.linspace(0, 2 * np.pi, n + 1)[:-1] + rng.uniform(-self.jitter, self.jitter, n)
        rad = rng.uniform(*self.radius, n)
        wps = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
        self.waypoints = np.vstack([list(self.start), wps])

    def distance(self, x: float, y: float) -> float:
        P = np.array([x, y])
        dmin = 1e9
        wps = self.waypoints
        for a, c in zip(wps, np.roll(wps, -1, axis=0)):
            ab = c - a
            u = np.clip((P - a) @ ab / (ab @ ab), 0, 1)
            dmin = min(dmin, np.linalg.norm(a + u * ab - P))
        return dmin

    def path(self) -> list[list[float]]:
        return self.waypoints.tolist()
