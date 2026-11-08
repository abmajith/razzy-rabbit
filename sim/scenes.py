"""The scene catalogue: the grounded scenes, whose vehicles are sized from real parts.
Each builder is deterministic in (seed) and writes
sim/generated/<scene>-s<seed>/{scene.xml, meta.json, terrain.bin, ground.png, mesh_*.obj}.

  forest_grounded  mixed forest on steep terrain, trail flown by the quad sized from
                   design/configs/drone_forest.toml
  mars_yard        rocky test yard driven by the rocker-bogie rover sized from rover_yard.toml

(Step 5 of docs/architecture.md turns these into short compositions of world features.)
"""
from __future__ import annotations

import numpy as np

from . import flora
from .builder import Builder, Lighting
from .terrain import Terrain, colorize, fbm, resample, write_texture

SCENES: dict[str, callable] = {}


def scene(fn):
    SCENES[fn.__name__] = fn
    return fn


# ------------------------------------------------------------------------- helpers ---
def noise2(nr, nc, rng, **kw) -> np.ndarray:
    n = max(nr, nc)
    f = fbm(n, rng, **kw)
    return f[np.ix_(np.linspace(0, n - 1, nr).astype(int), np.linspace(0, n - 1, nc).astype(int))]


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def poisson_disk(rng, xr, yr, r, tries=40000, accept=None) -> np.ndarray:
    """Dart throwing with a hash grid; `accept(x, y)` can veto points."""
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


def add_terrain(b: Builder, t: Terrain, tex: np.ndarray) -> None:
    fname = t.write(b.dir)
    write_texture(b.dir, "ground.png", tex)
    b.assets.append(t.asset_xml(fname))
    b.assets.append('<texture name="ground_tex" type="2d" file="ground.png"/>')
    b.material("ground", [1, 1, 1, 1], specular=0.0, shininess=0.0, texture="ground_tex")
    b.world.append(t.geom_xml())
    b.meta["terrain"] = dict(file=fname, half_x=t.half_x, half_y=t.half_y, zmin=t.zmin, elev=t.elev,
                             nrow=t.z.shape[0], ncol=t.z.shape[1])


DRY_SOIL = [(0.0, (0.42, 0.34, 0.25)), (1.0, (0.58, 0.49, 0.37))]
FOREST_FLOOR = [(0.0, (0.16, 0.13, 0.08)), (0.45, (0.24, 0.20, 0.12)), (0.7, (0.20, 0.27, 0.12)),
                (1.0, (0.33, 0.36, 0.18))]


@scene
def forest_grounded(seed: int = 0) -> Builder:
    """Forest trail flown by the quad sized from sim/design/configs/drone_forest.toml."""
    b = Builder("forest_grounded", seed)
    rng = b.rng
    flora.plant_materials(b, rng)
    b.sky_and_sun(Lighting.sample(rng), shadow_extent=40)

    hx = hy = 70.0
    n = 561
    X, Y = np.meshgrid(np.linspace(-hx, hx, n), np.linspace(-hy, hy, n))
    z = 9.0 * noise2(n, n, rng, octaves=6, base=2, persistence=0.5)
    ph = rng.uniform(0, 6.28, 2)
    trail_y = lambda x: 16 * np.sin(x / 17 + ph[0]) + 6 * np.sin(x / 6.5 + ph[1])
    dtrail = np.abs(Y - trail_y(X)) / np.sqrt(1 + (16 / 17 * np.cos(X / 17 + ph[0])) ** 2)
    z -= 0.25 * (1 - smoothstep(0.6, 1.6, dtrail))
    t = Terrain(z, hx, hy)

    tn = 2048
    tex = colorize(t, rng, tn, FOREST_FLOOR)
    trail_m = resample(1 - smoothstep(0.5, 1.3, dtrail), tn)[..., None]
    tex = tex * (1 - trail_m) + colorize(t, rng, tn, DRY_SOIL) * trail_m
    add_terrain(b, t, tex)

    def off_trail(x, y, clr=2.6):
        return abs(y - trail_y(x)) > clr
    pts = poisson_disk(rng, (-hx + 3, hx - 3), (-hy + 3, hy - 3), 3.3, tries=60000, accept=off_trail)
    for x, y in pts:
        zz = float(t.height(x, y))
        u = rng.random()
        dt = abs(y - trail_y(x))
        if u < 0.55 and dt < 5.5:      # wide crowns keep clear of the flight corridor
            u = 0.6 if dt > 4.0 else 0.9
        elif 0.55 <= u < 0.85 and dt < 4.0:
            u = 0.9
        if u < 0.55:
            flora.conifer(b, rng, x, y, zz, height=rng.uniform(10, 21))
        elif u < 0.85:
            flora.broadleaf_tree(b, rng, x, y, zz, height=rng.uniform(7, 13), crown=rng.uniform(2.0, 3.4))
        else:
            flora.birch(b, rng, x, y, zz, height=rng.uniform(8, 14))
    for x, y in rng.uniform(-hx + 3, hx - 3, (90, 2)):
        if off_trail(x, y, 1.5):
            flora.rock(b, rng, x, y, float(t.height(x, y)), size=rng.uniform(0.3, 1.2))
    for x, y in rng.uniform(-hx + 5, hx - 5, (40, 2)):
        if off_trail(x, y, 4.0):
            flora.log(b, rng, x, y, float(t.height(x, y)), length=rng.uniform(3, 7))
    for x, y in rng.uniform(-hx + 2, hx - 2, (3500, 2)):
        flora.grass_tuft(b, rng, x, y, float(t.height(x, y)),
                         mat="grass_tuft" if rng.random() < 0.7 else "dry_grass", h=rng.uniform(0.2, 0.5))

    xs = np.linspace(-62, 62, 90)
    wps = [[float(x), float(trail_y(x)), 2.0] for x in xs]
    x0, y0 = wps[0][0], wps[0][1]
    start = [x0, y0, float(t.height(x0, y0)) + 0.1]
    from .design import drone as drone_design
    cfg, best = drone_design.best("drone_forest.toml")
    drone_design.to_mjcf(b, "uav", start, best, cfg, chase_offset=(-2.5, 0.0, 1.0))
    b.overview([-82, -78, 50], [0, 0, 0])
    b.meta.update(controller="quad_waypoints", waypoints_agl=wps, cruise=3.0,
                  wind=dict(U=1.5, heading_deg=float(rng.uniform(0, 360)), gust=0.5, z_ref=10, z0=1.0))
    return b


# ======================================================================================


MARS_SOIL = [(0.0, (0.40, 0.20, 0.11)), (0.45, (0.58, 0.31, 0.17)), (0.8, (0.68, 0.42, 0.26)),
             (1.0, (0.78, 0.55, 0.38))]


@scene
def mars_yard(seed: int = 0) -> Builder:
    """Rocky test yard (JPL Mars Yard style) for the rocker-bogie rover sized from rover_yard.toml."""
    from .design import rover as rover_design
    cfg, best = rover_design.best("rover_yard.toml")
    b = Builder("mars_yard", seed)
    rng = b.rng
    light = Lighting.sample(rng, overcast=0.1)
    light.sky_top, light.sky_horizon = np.array([0.55, 0.45, 0.40]), np.array([0.85, 0.68, 0.52])
    light.haze = 0.45
    b.sky_and_sun(light, shadow_extent=30)
    b.material("rock", [0.42, 0.26, 0.18, 1], specular=0.1)

    hx = hy = 40.0
    n = 401
    X, Y = np.meshgrid(np.linspace(-hx, hx, n), np.linspace(-hy, hy, n))
    z = 1.2 * noise2(n, n, rng, octaves=6, base=2, persistence=0.5)
    z += 0.08 * noise2(n, n, rng, octaves=3, base=24, persistence=0.5)       # small-scale roughness
    craters = []
    for _ in range(7):
        cx, cy, R = rng.uniform(-30, 30), rng.uniform(-30, 30), rng.uniform(2.5, 7)
        if np.hypot(cx, cy) < 6:
            continue
        rr = np.hypot(X - cx, Y - cy) / R
        z += np.where(rr < 1, -0.22 * R * 0.25 * (1 - rr ** 2), 0) + 0.06 * R * np.exp(-((rr - 1) / 0.25) ** 2)
        craters.append((cx, cy, R))
    t = Terrain(z, hx, hy)
    tex = colorize(t, rng, 2048, MARS_SOIL)
    add_terrain(b, t, tex)

    # driving loop; boulders keep clear of it, small rocks (climbable) are scattered on it
    ang = np.linspace(0, 2 * np.pi, 9)[:-1] + rng.uniform(-0.2, 0.2, 8)
    rad = rng.uniform(9, 14, 8)
    wps = np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
    wps = np.vstack([[0.0, -3.0], wps])                                         # start near the lander
    def seg_dist(x, y):
        P = np.array([x, y])
        dmin = 1e9
        for a, c in zip(wps, np.roll(wps, -1, axis=0)):
            ab = c - a
            u = np.clip((P - a) @ ab / (ab @ ab), 0, 1)
            dmin = min(dmin, np.linalg.norm(a + u * ab - P))
        return dmin
    climb = best.m["climb_m"]
    for x, y in rng.uniform(-37, 37, (900, 2)):
        size = 0.04 + 0.09 * rng.pareto(2.0)                                     # power-law sizes
        if seg_dist(x, y) < 2.2 and size > 0.6 * climb:
            continue
        if np.hypot(x, y - 3) < 4:
            continue
        flora.rock(b, rng, x, y, float(t.height(x, y)) - 0.3 * size, size=min(size, 1.8))

    # lander platform the rover drives off from
    b.material("lander", [0.75, 0.75, 0.72, 1], specular=0.4)
    lz = float(t.height(0, 3))
    b.world += [f'<geom type="cylinder" size="1.6 .15" pos="0 3 {lz + 1.0}" material="lander"/>',
                f'<geom type="box" size="1.1 1.1 .02" pos="0 3 {lz + 1.2}" euler="0 0 45" material="solar"/>']
    for k in range(3):
        a = 2 * np.pi * k / 3 + 0.5
        b.world.append(f'<geom type="capsule" fromto="{1.2*np.cos(a):.2f} {3 + 1.2*np.sin(a):.2f} {lz + 1.0} '
                       f'{2.0*np.cos(a):.2f} {3 + 2.0*np.sin(a):.2f} {lz + 0.05}" size=".05" material="alu"/>')

    x0, y0 = 0.0, -3.0
    rmeta_h = best.geo["H"]
    rover_design.to_mjcf(b, "rover", [x0, y0, float(t.height(x0, y0)) + rmeta_h + 0.05], best, cfg, yaw=0.0)
    b.overview([-22, -26, 14], [0, 0, 0])
    b.meta.update(controller="rover_ackermann", waypoints=wps.tolist(), speed=cfg["mission"]["cruise_speed_mps"])
    return b


def build(name: str, seed: int = 0):
    if name not in SCENES:
        raise SystemExit(f"unknown scene {name!r}; choose from {', '.join(SCENES)}")
    b = SCENES[name](seed)
    return b.write()
