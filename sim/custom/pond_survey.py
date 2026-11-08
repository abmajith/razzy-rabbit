"""Example developer scene: UAV shoreline survey of a meadow pond.

Run:  uv run --extra sim python -m sim view pond_survey
Walkthrough: docs/custom_scenes.md
"""
import numpy as np

from sim import flora, robots
from sim.builder import Builder, Lighting
from sim.control import CONTROLLERS, QuadController
from sim.scenes import GRASS, add_terrain, noise2, poisson_disk, scene, smoothstep
from sim.terrain import Terrain, colorize, resample


# ---------------------------------------------------------------------------- controller
class PondOrbit:
    """Orbit a point at constant height above ground, camera always facing the centre.

    Controller contract: __init__(sim, meta) and update(t, dt), called every physics step
    before mj_step. Read state from sim.d, write sim.d.ctrl / sim.d.xfrc_applied.
    """

    def __init__(self, sim, meta: dict) -> None:
        self.sim = sim
        self.ctl = QuadController(sim.m, sim.d, meta["robots"][0])   # reuse the SE(3) inner loop
        self.c = np.array(meta["orbit_center"], float)
        self.R, self.agl = meta["orbit_radius"], meta["orbit_agl"]
        self.w = meta["orbit_speed"] / self.R                         # angular rate [rad/s]
        p0 = self.ctl.pos.copy()
        self.a0 = float(np.arctan2(p0[1] - self.c[1], p0[0] - self.c[0]))
        self.z0, self.t_up = p0[2], 3.0
        self.CdA = meta["robots"][0]["drag_CdA"]

    def update(self, t: float, dt: float) -> None:
        a = self.a0 + self.w * max(t - self.t_up, 0.0)
        xy = self.c + self.R * np.array([np.cos(a), np.sin(a)])
        z = float(self.sim.terrain.height(*xy)) + self.agl
        if t < self.t_up:                                             # vertical take-off
            z = self.z0 + (z - self.z0) * t / self.t_up
            v = np.zeros(3)
        else:
            v = self.R * self.w * np.array([-np.sin(a), np.cos(a), 0.0])
        self.ctl(np.r_[xy, z], v, yaw_d=a + np.pi)                    # nose toward the centre
        if self.sim.wind is not None:                                 # custom controllers apply wind themselves
            f = self.sim.wind.drag_force(self.ctl.pos, self.ctl.vel, self.CdA, t)
            self.sim.d.xfrc_applied[self.ctl.bid, :3] = f


CONTROLLERS["pond_orbit"] = PondOrbit


# ---------------------------------------------------------------------------- scene


@scene
def pond_survey(seed: int = 0) -> Builder:
    """Quadrotor circling a pond in a meadow, inspecting the shoreline."""
    b = Builder("pond_survey", seed)          # output: sim/generated/pond_survey-s<seed>/
    rng = b.rng                               # the ONLY randomness source
    flora.plant_materials(b, rng)
    b.sky_and_sun(Lighting.sample(rng), shadow_extent=40)

    # 1. terrain: gentle meadow with a bowl-shaped pond basin
    hx = hy = 60.0
    n = 257
    X, Y = np.meshgrid(np.linspace(-hx, hx, n), np.linspace(-hy, hy, n))
    r = np.hypot(X, Y)
    z = 2.0 * noise2(n, n, rng, octaves=5, base=2) - 1.5 * (1 - smoothstep(8, 16, r))
    t = Terrain(z, hx, hy)

    # 2. ground texture: grass, darker mud near the shore
    tex = colorize(t, rng, 1024, GRASS)
    mud = resample(1 - smoothstep(10, 14, r), 1024)[..., None]
    tex = tex * (1 - mud) + np.array([0.30, 0.24, 0.16]) * mud
    add_terrain(b, t, tex)

    # 3. water: flat translucent box at the pond's water level
    b.material("water", [0.18, 0.35, 0.45, 0.75], specular=0.9, shininess=0.9, reflectance=0.3)
    water_z = float(t.height(0, 0)) + 0.6
    b.world.append(f'<geom name="pond" type="cylinder" size="12 .05" pos="0 0 {water_z:.3f}" '
                   f'material="water" contype="0" conaffinity="0"/>')

    # 4. vegetation: reeds on the shore, trees away from the water
    for a in rng.uniform(0, 2 * np.pi, 400):
        rr = rng.uniform(11, 14)
        x, y = rr * np.cos(a), rr * np.sin(a)
        flora.grass_tuft(b, rng, x, y, float(t.height(x, y)), h=0.9)
    for x, y in poisson_disk(rng, (-55, 55), (-55, 55), 6.0, tries=3000,
                             accept=lambda x, y: np.hypot(x, y) > 20):
        flora.broadleaf_tree(b, rng, x, y, float(t.height(x, y)),
                             height=rng.uniform(6, 10), crown=rng.uniform(2, 3))

    # 5. robot + mission parameters (read by PondOrbit above via meta.json)
    start = (-18.0, 0.0)
    robots.quadrotor(b, "uav", [*start, float(t.height(*start)) + 0.15])
    b.overview([-45, -45, 25], [0, 0, 0])
    b.meta.update(controller="pond_orbit", orbit_center=[0.0, 0.0], orbit_radius=18.0,
                  orbit_agl=4.0, orbit_speed=3.0,
                  wind=dict(U=2.5, heading_deg=float(rng.uniform(0, 360)), gust=0.6, z_ref=10, z0=0.1))
    return b
