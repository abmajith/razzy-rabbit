"""Objects on the ground: rock fields, fallen logs, a lander platform."""
from __future__ import annotations

import numpy as np

from .. import flora
from ..sampling import Uniform
from ..world import Feature


def _clear_ok(rules, x, y, size=None) -> bool:
    for c in rules:
        d = c.region.distance(x, y)
        if c.larger_than is None:
            if not d > c.distance:
                return False
        elif d < c.distance and size > c.larger_than:
            return False
    return True


class Rocks(Feature):
    """`count` rocks at uniform positions, sizes from `size` (Uniform, PowerLaw...), sunk
    `sink` x size into the ground. Rules `clear` keep them off regions; a rule with
    `larger_than` lets small (climbable) rocks lie on a path while big ones keep clear.
    Each rock gets an invisible ellipsoid collider."""

    def __init__(self, count: int, margin: float = 3.0, size=Uniform(0.3, 1.2), sink: float = 0.0, clear=(),
                 colour=None) -> None:
        self.count, self.margin, self.size, self.sink, self.clear = count, margin, size, sink, list(clear)
        self.colour = colour
        self.rocks: list[tuple[float, float, float]] = []

    def apply(self, w) -> None:
        if self.colour is not None:
            w.builder.material("rock", [*self.colour, 1], specular=0.1)
        flora.plant_materials(w.builder, names=("rock",))   # default rock colour if none was set
        rng, h, mg = w.rng, w.half, self.margin
        size_first = any(c.larger_than is not None for c in self.clear)   # the rules need the size
        cap = getattr(self.size, "cap", np.inf)
        for x, y in rng.uniform(-h + mg, h - mg, (self.count, 2)):
            s = self.size.draw(rng) if size_first else None
            if not _clear_ok(self.clear, x, y, s):
                continue
            if s is None:
                s = self.size.draw(rng)
            z = w.height(x, y) - self.sink * s if self.sink else w.height(x, y)
            flora.rock(w.builder, rng, x, y, z, size=min(s, cap))
            self.rocks.append((float(x), float(y), float(s)))


class Logs(Feature):
    """Fallen logs with random length and direction, each with a capsule collider."""

    def __init__(self, count: int = 40, margin: float = 5.0, length=Uniform(3, 7), clear=()) -> None:
        self.count, self.margin, self.length, self.clear = count, margin, length, list(clear)

    def apply(self, w) -> None:
        flora.plant_materials(w.builder, names=("bark",))
        rng, h, mg = w.rng, w.half, self.margin
        for x, y in rng.uniform(-h + mg, h - mg, (self.count, 2)):
            if _clear_ok(self.clear, x, y):
                flora.log(w.builder, rng, x, y, w.height(x, y), length=self.length.draw(rng))


class Lander(Feature):
    """A lander platform on three legs at `pos` (a rover's start area). It is also a region:
    `distance` is measured from its centre, so rock fields can keep clear of it."""

    def __init__(self, pos=(0.0, 3.0)) -> None:
        self.pos = pos

    def distance(self, x: float, y: float) -> float:
        return float(np.hypot(x - self.pos[0], y - self.pos[1]))

    def apply(self, w) -> None:
        b = w.builder
        px, py = self.pos
        b.material("lander", [0.75, 0.75, 0.72, 1], specular=0.4)
        lz = w.height(px, py)
        b.world += [f'<geom type="cylinder" size="1.6 .15" pos="{px:g} {py:g} {lz + 1.0}" material="lander"/>',
                    f'<geom type="box" size="1.1 1.1 .02" pos="{px:g} {py:g} {lz + 1.2}" euler="0 0 45" material="solar"/>']
        for k in range(3):
            a = 2 * np.pi * k / 3 + 0.5
            b.world.append(f'<geom type="capsule" fromto="{px + 1.2*np.cos(a):.2f} {py + 1.2*np.sin(a):.2f} {lz + 1.0} '
                           f'{px + 2.0*np.cos(a):.2f} {py + 2.0*np.sin(a):.2f} {lz + 0.05}" size=".05" material="alu"/>')
