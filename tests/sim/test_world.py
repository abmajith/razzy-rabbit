"""sim/world: worlds are deterministic per seed, features respect their keep-clear rules,
terrain shaping must come before features that use heights, and a new world composed from
features builds and compiles."""
import mujoco
import numpy as np
import pytest

from sim.world import Clear, PowerLaw, Uniform, World
from sim.world.features import (FOREST_FLOOR, MARS_SOIL, Craters, DriveLoop, Forest, Ground, Hills, Lander,
                                PlantColours, Rocks, Sun, Trail, Wind)


def _forest(tmp_path, seed):
    w = World("t_forest", seed, half_size=30.0, resolution=121, out_root=tmp_path)
    w.add(PlantColours())
    w.add(Sun())
    w.add(Hills(relief=4.0))
    trail = w.add(Trail())
    w.add(Ground(FOREST_FLOOR, texture_size=128, overlays={trail: FOREST_FLOOR}))
    forest = w.add(Forest(spacing=3.0, tries=4000, clear=[Clear(trail, 2.6)],
                          narrow_near=(trail, {"conifer": 5.5, "broadleaf": 4.0})))
    return w, trail, forest


def test_same_seed_same_world(tmp_path):
    a, _, fa = _forest(tmp_path / "a", 3)
    b, _, fb = _forest(tmp_path / "b", 3)
    c, _, _ = _forest(tmp_path / "c", 4)
    assert np.array_equal(a.z, b.z) and fa.trees == fb.trees
    assert not np.array_equal(a.z, c.z)


def test_forest_keeps_clear_of_trail(tmp_path):
    _, trail, forest = _forest(tmp_path, 1)
    assert len(forest.trees) > 20
    for species, x, y in forest.trees:
        d = trail.distance(x, y)
        assert d > 2.6
        if species == "conifer":
            assert d >= 5.5                       # wide crowns swapped for narrower species near the trail
        if species == "broadleaf":
            assert d >= 4.0


def test_rocks_rules_and_frozen_terrain(tmp_path):
    w = World("t_yard", 2, half_size=25.0, resolution=101, out_root=tmp_path)
    w.add(Hills(relief=1.0))
    w.add(Craters(count=4, extent=18))
    w.add(Ground(MARS_SOIL, texture_size=128))
    loop = w.add(DriveLoop(radius=(8, 12)))
    lander = Lander()
    rocks = w.add(Rocks(count=400, size=PowerLaw(0.04, 0.09, cap=1.8), sink=0.3,
                        clear=[Clear(loop, 2.0, larger_than=0.15), Clear(lander, 4.0)]))
    assert rocks.rocks
    for x, y, s in rocks.rocks:
        assert lander.distance(x, y) > 4.0
        if s > 0.15:
            assert loop.distance(x, y) >= 2.0     # boulders off the path, small rocks may lie on it
    with pytest.raises(RuntimeError, match="frozen"):
        w.add(Hills(relief=1.0))


def test_new_world_builds_and_compiles(tmp_path):
    w = World("t_new", 0, half_size=20.0, resolution=81, out_root=tmp_path)
    w.add(Sun(overcast=0.3))
    w.add(Hills(relief=2.0, base=3))
    w.add(Ground(FOREST_FLOOR, texture_size=128))
    w.add(Rocks(count=30, size=Uniform(0.2, 0.6)))
    w.add(Wind(speed=3.0, heading_deg=90))
    w.overview([-25, -25, 15], [0, 0, 0])
    m = mujoco.MjModel.from_xml_path(str(w.builder.write()))
    assert m.geom("terrain").id >= 0 and m.ncam == 1
    assert w.builder.meta["wind"]["heading_deg"] == 90
