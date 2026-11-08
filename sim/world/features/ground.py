"""Ground: freeze the terrain and paint it.

The texture is a colour ramp (`palette`: [(threshold, rgb), ...]) driven by noise, plus
overlays: any region with a `texture_mask()` (e.g. a Trail) is painted with its own palette.
This writes the heightfield and texture files into the scene.
"""
from __future__ import annotations

from ..terrain import colorize, resample
from ..world import Feature

GRASS = [(0.0, (0.20, 0.31, 0.11)), (0.5, (0.31, 0.45, 0.16)), (1.0, (0.46, 0.53, 0.24))]
DRY_SOIL = [(0.0, (0.42, 0.34, 0.25)), (1.0, (0.58, 0.49, 0.37))]
FOREST_FLOOR = [(0.0, (0.16, 0.13, 0.08)), (0.45, (0.24, 0.20, 0.12)), (0.7, (0.20, 0.27, 0.12)),
                (1.0, (0.33, 0.36, 0.18))]
MARS_SOIL = [(0.0, (0.40, 0.20, 0.11)), (0.45, (0.58, 0.31, 0.17)), (0.8, (0.68, 0.42, 0.26)),
             (1.0, (0.78, 0.55, 0.38))]


class Ground(Feature):
    def __init__(self, palette, texture_size: int = 2048, overlays=None) -> None:
        """overlays: {region: palette}, painted in order over the base."""
        self.palette, self.n, self.overlays = palette, texture_size, dict(overlays or {})

    def apply(self, w) -> None:
        t, n = w.terrain, self.n                     # freezes the heights
        tex = colorize(t, w.rng, n, self.palette)
        for region, palette in self.overlays.items():
            m = resample(region.texture_mask(), n)[..., None]
            tex = tex * (1 - m) + colorize(t, w.rng, n, palette) * m
        w.write_ground(tex)
