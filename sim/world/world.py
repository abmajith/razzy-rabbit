"""World: compose an environment from features, in order.

    w = World("forest_grounded", seed, half_size=70.0, resolution=561)
    w.add(PlantColours())                      # seeded colour variation of bark / leaves
    w.add(Sun(shadow_extent=40))
    w.add(Hills(relief=9.0))
    trail = w.add(Trail(...))                  # carves the terrain, is a region and a path
    w.add(Ground(FOREST_FLOOR, overlays={trail: DRY_SOIL}))
    w.add(Forest(spacing=3.3, clear=[Clear(trail, 2.6)]))
    ...
    builder = w.builder                        # add the robot, then builder.write()

Each feature is applied when it is added, so the order of `add` calls is the order of
generation: it decides which random numbers each feature draws (one seeded stream per
scene, so the same seed always gives the same world) and which features see which (rocks
placed after a trail can keep clear of it).

Terrain is built in two phases: features like Hills, Craters and Trail edit the height grid
`w.z` (on the X, Y grid `w.X`, `w.Y`); the first feature that needs actual heights (`w.terrain`,
`w.height(x, y)`) freezes it into a Terrain. After that the heights can't change.
"""
from __future__ import annotations

import numpy as np

from ..builder import Builder
from .terrain import Terrain, write_texture


class Feature:
    """Base class: `apply(world)` adds geometry / terrain edits / meta to the world."""

    def apply(self, world: "World") -> None:          # pragma: no cover - abstract
        raise NotImplementedError

    def distance(self, x: float, y: float) -> float:   # features that are regions override this
        raise TypeError(f"{type(self).__name__} is not a region")


class World:
    def __init__(self, name: str, seed: int = 0, half_size: float = 50.0, resolution: int = 401,
                 out_root=None) -> None:
        self.builder = Builder(name, seed) if out_root is None else Builder(name, seed, out_root=out_root)
        self.rng = self.builder.rng
        self.half = half_size
        g = np.linspace(-half_size, half_size, resolution)
        self.X, self.Y = np.meshgrid(g, g)
        self.z = np.zeros_like(self.X)
        self._terrain: Terrain | None = None
        self.features: list[Feature] = []

    def add(self, feature: Feature) -> Feature:
        feature.apply(self)
        self.features.append(feature)
        return feature

    # ------------------------------------------------------------------ terrain ---
    def edit_heights(self) -> np.ndarray:
        """The height grid, for terrain features to modify (before it is frozen)."""
        if self._terrain is not None:
            raise RuntimeError("terrain is already frozen: add terrain-shaping features before the others")
        return self.z

    @property
    def terrain(self) -> Terrain:
        if self._terrain is None:
            self._terrain = Terrain(self.z, self.half, self.half)
        return self._terrain

    def height(self, x: float, y: float) -> float:
        return float(self.terrain.height(x, y))

    def write_ground(self, texture: np.ndarray) -> None:
        """Write the heightfield + its texture into the scene (done by the Ground feature)."""
        b, t = self.builder, self.terrain
        fname = t.write(b.dir)
        write_texture(b.dir, "ground.png", texture)
        b.assets.append(t.asset_xml(fname))
        b.assets.append('<texture name="ground_tex" type="2d" file="ground.png"/>')
        b.material("ground", [1, 1, 1, 1], specular=0.0, shininess=0.0, texture="ground_tex")
        b.world.append(t.geom_xml())
        b.meta["terrain"] = dict(file=fname, half_x=t.half_x, half_y=t.half_y, zmin=t.zmin, elev=t.elev,
                                 nrow=t.z.shape[0], ncol=t.z.shape[1])

    # ----------------------------------------------------------------- helpers ---
    def ground_point(self, x: float, y: float, clearance: float = 0.0) -> list[float]:
        """[x, y, z] on the ground (+ clearance), e.g. a robot's start position."""
        return [float(x), float(y), self.height(x, y) + clearance]

    def overview(self, pos, target) -> None:
        """Fixed camera named `overview` looking at `target`."""
        self.builder.overview(pos, target)
