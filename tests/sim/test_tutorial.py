"""docs/tutorial.md, executed: the example scene flies on both engines, and the three
"write your own" examples (a component, a world feature, a control stage) work as shown.
Keep this file and the tutorial in sync."""
import math

import mujoco
import numpy as np
import pytest

from sim.components import Component, MassPart
from sim.components.base import fmt
from sim.control import STAGES, Controller, Mixer, Se3Controller
from sim.runtime import Sim
from sim.world import Clear, Feature, World
from sim.world.features import GRASS, Forest, Ground, Hills


# --------------------------------------------------------------- the example scene ---
def test_meadow_scout_flies_on_both_engines():
    s = Sim("meadow_scout", 0, rebuild=True, sensors=False)
    bid = s.m.body("scout").id
    p0 = s.d.xpos[bid].copy()
    s.step(3000)                                                   # 6 s
    p = s.d.xpos[bid]
    assert p[2] - s.terrain.height(p[0], p[1]) > 1.5                # flying
    assert np.linalg.norm(p[:2] - p0[:2]) > 5.0                    # following the trail
    pytest.importorskip("razzy_core")
    from sim.core import CoreBatch
    cb = CoreBatch("meadow_scout", 1, 0, sensors=False)
    s2 = Sim("meadow_scout", 0, sensors=False)
    for _ in range(3):
        cb.step(500)
        s2.step(500)
        assert np.abs(cb.qpos[0] - s2.d.qpos).max() < 1e-9          # C++ = Python reference


def test_hand_built_rotor_rating_is_physical():
    from sim.myscenes.meadow_scout import scout
    r = scout()
    meta = r.meta()
    rotor = r.info["rotors"][0]
    t_max, lag = rotor.rating(r)
    hover = meta["mass"] * 9.81 / 4
    assert 2.0 < t_max / hover < 8.0                               # thrust-to-weight of a 5-inch quad
    assert 0.005 < lag < 0.05                                      # motor lag: milliseconds
    assert meta["actuators"] == ["scout_m0", "scout_m1", "scout_m2", "scout_m3"]
    assert meta["cameras"][0] == "scout_chase"


# --------------------------------------------------------- your own component -------
class SprayTank(Component):
    """A cylindrical spray tank (water) with a pump: mass, geometry, a constant load."""

    kind = "tank"

    def __init__(self, litres: float, empty_g: float = 150.0, pump_w: float = 12.0, radius: float = 0.05) -> None:
        super().__init__()
        self.litres, self.empty_g, self.pump_w, self.radius = litres, empty_g, pump_w, radius

    def attach(self, robot, at) -> None:
        r = self.radius
        h = self.litres / 1000 / (math.pi * r * r)                  # height holding the volume
        m = self.empty_g / 1000 + self.litres * 1.0                 # 1 L of water = 1 kg
        pos = at.place(h)
        I = m * np.diag([(3 * r * r + h * h) / 12, (3 * r * r + h * h) / 12, r * r / 2])
        at.body.mass_parts.append(MassPart(m, pos, I, "spray tank"))
        at.body.geoms.append(f'<geom type="cylinder" size="{r:.3f} {h/2:.3f}" pos="{fmt(pos)}" '
                             'material="white_paint" contype="0" conaffinity="0"/>')

    def contribute(self, robot, meta) -> None:
        meta["power"]["loads"].append({"type": "constant", "group": "payload", "label": "spray pump",
                                       "watts": self.pump_w})


def test_custom_component():
    from sim.myscenes.meadow_scout import scout
    r = scout()
    m0 = r.mass_properties()["root_mass"]
    frame = r.components[0]
    r.add(SprayTank(litres=0.3), at=frame.mount("belly"))
    assert r.mass_properties()["root_mass"] > m0 + 0.3
    loads = r.meta()["power"]["loads"]
    assert any(ld.get("label") == "spray pump" for ld in loads)
    m = r.compile()
    assert m.body_mass[m.body("scout").id] == pytest.approx(r.mass_properties()["root_mass"], rel=1e-3)


# ------------------------------------------------------------ your own feature -------
class Pond(Feature):
    """A pond: a bowl cut into the terrain; a region trees keep clear of."""

    def __init__(self, x: float, y: float, radius: float, depth: float = 0.8) -> None:
        self.x, self.y, self.radius, self.depth = x, y, radius, depth

    def apply(self, w) -> None:
        z = w.edit_heights()
        r = np.hypot(w.X - self.x, w.Y - self.y) / self.radius
        z -= self.depth * np.clip(1 - r * r, 0, None)

    def distance(self, x: float, y: float) -> float:
        return max(0.0, float(np.hypot(x - self.x, y - self.y)) - self.radius)


def test_custom_feature(tmp_path):
    w = World("t_pond", 0, half_size=25.0, resolution=101, out_root=tmp_path)
    w.add(Hills(relief=1.0))
    pond = w.add(Pond(0.0, 0.0, radius=6.0))
    w.add(Ground(GRASS, texture_size=128))
    forest = w.add(Forest(spacing=4.0, tries=3000, clear=[Clear(pond, 2.0)]))
    assert w.height(0, 0) < w.height(15, 15) + 1.0                # a dip at the centre
    assert all(pond.distance(x, y) > 2.0 for _, x, y in forest.trees)
    assert mujoco.MjModel.from_xml_path(str(w.builder.write())).ngeom > 0


# --------------------------------------------------------- your own control stage ----
class Orbit(Controller):
    """Top stage: fly a circle around `center` at `height_agl`, facing along the circle."""

    name = "orbit"
    defaults = {"center": (0.0, 0.0), "radius": 8.0, "height_agl": 3.0, "speed": 3.0}

    def bind(self, sim, robot) -> None:
        self.sim = sim

    def update(self, t, dt, state, x):
        p = self.params
        R, cx, cy = p["radius"], *p["center"]
        w = p["speed"] / R
        a = w * t
        pos = np.array([cx + R * np.cos(a), cy + R * np.sin(a), 0.0])
        pos[2] = self.sim.terrain.height(pos[0], pos[1]) + p["height_agl"]
        vel = R * w * np.array([-np.sin(a), np.cos(a), 0.0])
        return {"p": pos, "v": vel, "yaw": a + np.pi / 2}


def test_custom_stage():
    STAGES[Orbit.name] = Orbit                                       # so scenes can store it in their meta
    s = Sim("meadow_scout", 0, controller=False, sensors=False)
    bid = s.m.body("scout").id
    x0, y0 = s.d.xpos[bid][:2]
    chain = Orbit(center=(float(x0) - 8.0, float(y0)), radius=8.0) >> Se3Controller() >> Mixer()
    s.ctrls.insert(0, chain.attach(s))
    s.step(4000)                                                    # 8 s on the circle
    d = np.hypot(s.d.xpos[bid][0] - (x0 - 8.0), s.d.xpos[bid][1] - y0)
    assert abs(d - 8.0) < 1.5
    pytest.importorskip("razzy_core")
    from sim.core import Unsupported, core_supported
    with pytest.raises(Unsupported, match="orbit"):                  # Python-only stage: C++ falls back
        core_supported({"control": chain.spec()})
