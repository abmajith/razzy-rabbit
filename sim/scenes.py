"""The scene catalogue. Each builder is deterministic in (seed) and writes
sim/generated/<scene>-s<seed>/{scene.xml, meta.json, terrain.bin, ground.png, mesh_*.obj}.

  orchard_uav      apple orchard on rolling hills, quadrotor flying the alleys
  crop_rover       vegetable beds with furrows + weeds, high-clearance weeding rover
  forest_trail     mixed forest on steep terrain, quadrotor following a hiking trail
  greenhouse       Venlo glasshouse, high-wire tomatoes, pipe-rail harvesting arm
  wind_turbine     ridge-top turbine inspection by UAV in a potential-flow wind field
"""
from __future__ import annotations

import numpy as np

from . import flora, robots
from .builder import Builder, Lighting, fmt
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


GRASS = [(0.0, (0.20, 0.31, 0.11)), (0.5, (0.31, 0.45, 0.16)), (1.0, (0.46, 0.53, 0.24))]
SOIL = [(0.0, (0.25, 0.18, 0.12)), (0.5, (0.36, 0.27, 0.18)), (1.0, (0.48, 0.38, 0.27))]
DRY_SOIL = [(0.0, (0.42, 0.34, 0.25)), (1.0, (0.58, 0.49, 0.37))]
FOREST_FLOOR = [(0.0, (0.16, 0.13, 0.08)), (0.45, (0.24, 0.20, 0.12)), (0.7, (0.20, 0.27, 0.12)),
                (1.0, (0.33, 0.36, 0.18))]
PASTURE = [(0.0, (0.33, 0.42, 0.16)), (0.6, (0.48, 0.55, 0.24)), (1.0, (0.66, 0.64, 0.38))]


def surrounding_trees(b, t: Terrain, rng, inner_x, inner_y, n=200, mix=(0.5, 0.35, 0.15)):
    """Natural tree belt between the work area and the terrain edge (no collision)."""
    def outside(x, y):
        return abs(x) > inner_x or abs(y) > inner_y
    pts = poisson_disk(rng, (-t.half_x + 2, t.half_x - 2), (-t.half_y + 2, t.half_y - 2), 4.5,
                       tries=n * 30, accept=outside)
    for x, y in pts[:n]:
        z = float(t.height(x, y))
        u = rng.random()
        if u < mix[0]:
            flora.conifer(b, rng, x, y, z, height=rng.uniform(9, 17), collide=False)
        elif u < mix[0] + mix[1]:
            flora.broadleaf_tree(b, rng, x, y, z, height=rng.uniform(7, 12), crown=rng.uniform(2.5, 4),
                                 collide=False)
        else:
            flora.birch(b, rng, x, y, z, height=rng.uniform(8, 13), collide=False)


# ======================================================================================
@scene
def orchard_uav(seed: int = 0) -> Builder:
    b = Builder("orchard_uav", seed)
    rng = b.rng
    flora.plant_materials(b, rng)
    b.sky_and_sun(Lighting.sample(rng), shadow_extent=40)

    hx = hy = 75.0
    n = 257
    z = 3.5 * noise2(n, n, rng, octaves=5, base=2, persistence=0.45)
    X, Y = np.meshgrid(np.linspace(-hx, hx, n), np.linspace(-hy, hy, n))
    z += 0.02 * X + 0.01 * Y
    t = Terrain(z, hx, hy)

    row_sp, tree_sp = 4.5, 3.0
    rows_y = np.arange(-38, 38.1, row_sp)
    xs = np.arange(-45, 45.1, tree_sp)
    # texture: herbicide soil strip under rows, mown alleys, rough grass outside
    tn = 2048
    ty = np.linspace(-hy, hy, tn)[:, None] * np.ones((1, tn))
    tx = np.ones((tn, 1)) * np.linspace(-hx, hx, tn)[None, :]
    d_row = np.min(np.abs(ty[..., None] - rows_y[None, None, :]), axis=-1)
    inside = (np.abs(tx) < 47.5) & (np.abs(ty) < rows_y.max() + 2.5)
    soil_m = (inside & (d_row < 0.8)).astype(float) * (1 - smoothstep(0.5, 0.85, d_row))
    mow = inside * (0.5 + 0.5 * np.cos(2 * np.pi * ty / 1.1)) * (d_row > 1.0) * 0.3
    tex = colorize(t, rng, tn, GRASS, extra=0.5 + mow)
    tex = tex * (1 - soil_m[..., None]) + colorize(t, rng, tn, SOIL)[..., :] * soil_m[..., None]
    add_terrain(b, t, tex)

    for ry in rows_y:
        for x in xs:
            if rng.random() < 0.03:
                continue  # missing tree
            px, py = x + rng.normal(0, 0.12), ry + rng.normal(0, 0.08)
            flora.broadleaf_tree(b, rng, px, py, float(t.height(px, py)), height=rng.uniform(3.4, 4.4),
                                 crown=rng.uniform(1.1, 1.5), fruit=2.2)
        # trellis posts + wire at row ends
        for x in (xs[0] - 1.5, xs[-1] + 1.5):
            zz = float(t.height(x, ry))
            b.batch("bark").cylinder([x, ry, zz - .3], [x, ry, zz + 2.8], 0.06, seg=6)
    for x, y in rng.uniform([-47, -42], [47, 42], (1800, 2)):
        flora.grass_tuft(b, rng, x, y, float(t.height(x, y)), h=0.25)
    surrounding_trees(b, t, rng, 52, rows_y.max() + 6, n=220)

    quad = robots.quadrotor(b, "uav", [-48, rows_y[0] + row_sp / 2, float(t.height(-48, rows_y[0] + 2.25)) + 0.15])
    # mission: fly the alleys at canopy height, lawnmower pattern
    wps = []
    for i, ry in enumerate(rows_y[:-1]):
        ya = ry + row_sp / 2
        xa, xb = (-47, 47) if i % 2 == 0 else (47, -47)
        wps += [[xa, ya, 2.4], [xb, ya, 2.4]]
    b.overview([-60, -82, 42], [0, -8, 0])
    b.meta.update(controller="quad_waypoints", waypoints_agl=wps, cruise=4.0,
                  wind=dict(U=3.0, heading_deg=float(rng.uniform(0, 360)), gust=0.8, z_ref=10, z0=0.3))
    return b


# ======================================================================================
@scene
def crop_rover(seed: int = 0) -> Builder:
    b = Builder("crop_rover", seed)
    rng = b.rng
    flora.plant_materials(b, rng)
    b.sky_and_sun(Lighting.sample(rng), shadow_extent=25)

    hx, hy = 55.0, 38.0
    nr, nc = 401, 581
    X, Y = np.meshgrid(np.linspace(-hx, hx, nc), np.linspace(-hy, hy, nr))
    z = 0.7 * noise2(nr, nc, rng, octaves=4, base=2, persistence=0.4)
    bed_pitch, field_x, field_y = 1.5, 36.0, 24.0
    beds = np.arange(-field_y + 0.75, field_y, bed_pitch)
    d = np.abs(((Y - beds[0]) + bed_pitch / 2) % bed_pitch - bed_pitch / 2)
    in_field = smoothstep(field_x + 1.0, field_x - 1.0, np.abs(X)) * smoothstep(field_y + 0.5, field_y - 0.5, np.abs(Y))
    bed_h = 0.16 * (1 - smoothstep(0.42, 0.68, d)) * in_field
    z += bed_h
    t = Terrain(z, hx, hy)

    tn = 2048
    soil = colorize(t, rng, tn, SOIL)
    dry = colorize(t, rng, tn, DRY_SOIL)
    grass = colorize(t, rng, tn, GRASS)
    bedm = resample(bed_h / 0.16, tn)[..., None]
    fieldm = resample(in_field, tn)[..., None]
    tex = grass * (1 - fieldm) + fieldm * (soil * bedm + dry * (1 - bedm))
    add_terrain(b, t, tex)

    # crops: two lettuce rows per bed, growth gradient across the field (realistic variability)
    plant_x = np.arange(-30, 30, 0.32)
    for k, yc in enumerate(beds):
        if abs(yc) > 18.5:
            continue
        growth = 0.75 + 0.5 * (k / len(beds)) + rng.normal(0, 0.05)
        for off in (-0.22, 0.22):
            for x in plant_x:
                if rng.random() < 0.06:
                    continue
                px, py = x + rng.normal(0, 0.03), yc + off + rng.normal(0, 0.02)
                flora.lettuce(b, rng, px, py, float(t.height(px, py)), size=0.13 * growth)
    for _ in range(1600):  # weeds: in beds and furrows (detection/removal targets)
        x, y = rng.uniform(-31, 31), rng.uniform(-18.5, 18.5)
        flora.weed(b, rng, x, y, float(t.height(x, y)))
    for x, y in rng.uniform([-hx + 2, -hy + 2], [hx - 2, hy - 2], (2500, 2)):
        if abs(x) < field_x + 1 and abs(y) < field_y + 1:
            continue
        flora.grass_tuft(b, rng, x, y, float(t.height(x, y)), h=0.35)

    # farm shed + water tank on the headland
    b.material("shed_wall", [0.55, 0.20, 0.15, 1], specular=0.1)
    b.material("shed_roof", [0.35, 0.37, 0.40, 1], specular=0.5, shininess=0.5)
    sx, sy = 46.0, -14.0
    sz = float(t.height(sx, sy))
    b.world += [f'<geom type="box" pos="{sx} {sy} {sz + 2.5}" size="5 8 3" material="shed_wall"/>',
                f'<geom type="box" pos="{sx - 2.6} {sy} {sz + 6.2}" size="3.2 8.4 .08" euler="0 -25 0" material="shed_roof"/>',
                f'<geom type="box" pos="{sx + 2.6} {sy} {sz + 6.2}" size="3.2 8.4 .08" euler="0 25 0" material="shed_roof"/>',
                f'<geom type="cylinder" pos="{sx - 1} {sy + 13} {sz + 2}" size="2 2.5" material="alu"/>']
    surrounding_trees(b, t, rng, 50, 30, n=120, mix=(0.3, 0.55, 0.15))

    k0 = int(np.argmin(np.abs(beds + 12)))
    y0 = float(beds[k0])
    zs = max(float(t.height(x, y0 + s * 0.85)) for x in (-33.5, -32.5) for s in (-1, 1))
    robots.field_rover(b, "rover", [-33.0, y0, zs + 1.06])
    lanes = [float(beds[k0 + 2 * i]) for i in range(6)]
    b.overview([-60, -50, 26], [-12, -6, 0])
    b.meta.update(controller="rover_serpentine", lanes_y=lanes, x_turn=(-32.0, 32.0), speed=0.9, beds_y=beds)
    return b


# ======================================================================================
@scene
def forest_trail(seed: int = 0) -> Builder:
    b = Builder("forest_trail", seed)
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
    robots.quadrotor(b, "uav", [x0, y0, float(t.height(x0, y0)) + 0.15], chase_offset=(-3.0, 0.0, 1.2))
    b.overview([-82, -78, 50], [0, 0, 0])
    b.meta.update(controller="quad_waypoints", waypoints_agl=wps, cruise=3.0,
                  wind=dict(U=1.5, heading_deg=float(rng.uniform(0, 360)), gust=0.5, z_ref=10, z0=1.0))
    return b


# ======================================================================================
@scene
def greenhouse(seed: int = 0) -> Builder:
    b = Builder("greenhouse", seed)
    rng = b.rng
    flora.plant_materials(b, rng)
    light = Lighting.sample(rng, overcast=rng.uniform(0, 0.3))
    light.ambient += 0.1
    b.sky_and_sun(light, shadow_extent=25, castshadow=False, fill=0.22)
    b.material("glass", [0.85, 0.93, 0.97, 0.10], specular=0.9, shininess=0.9, reflectance=0.0)
    b.material("frame", [0.80, 0.82, 0.84, 1], specular=0.5, shininess=0.5)
    b.material("ground_cover", [0.88, 0.88, 0.86, 1], specular=0.2)
    b.material("concrete", [0.58, 0.57, 0.55, 1], specular=0.1)
    b.material("pipe", [0.75, 0.76, 0.78, 1], specular=0.8, shininess=0.8, reflectance=0.1)
    b.material("slab", [0.95, 0.95, 0.93, 1], specular=0.1)
    b.material("lamp", [1.0, 0.75, 0.45, 1], emission=0.8)
    b.material("tomato_stem", [0.30, 0.45, 0.18, 1], specular=0.1)
    b.material("tomato_leaf", [0.17, 0.40, 0.14, 1], specular=0.2, shininess=0.3)
    b.material("tomato_green", [0.40, 0.62, 0.20, 1], specular=0.6, shininess=0.7)
    b.material("tomato_orange", [0.90, 0.45, 0.08, 1], specular=0.6, shininess=0.7)
    b.material("tomato_red", [0.78, 0.07, 0.04, 1], specular=0.6, shininess=0.7)
    tile = np.clip(colorize(None, rng, 512, GRASS) * 1.0, 0, 1)
    write_texture(b.dir, "grass.png", tile)
    b.assets.append('<texture name="grass_tex" type="2d" file="grass.png"/>')
    b.material("lawn", [1, 1, 1, 1], specular=0, texture="grass_tex", texrepeat=[0.08, 0.08])

    GX, GY, GH, RIDGE = 20.0, 8.0, 5.0, 0.9
    b.world += [f'<geom name="outside" type="plane" size="200 200 .1" pos="0 0 -.01" material="lawn"/>',
                f'<geom name="floor" type="box" size="{GX} {GY} .01" material="ground_cover"/>',
                f'<geom type="box" size="1.6 {GY} .015" pos="{GX - 1.6} 0 .005" material="concrete" contype="0" conaffinity="0"/>']
    # Venlo structure: columns, gutters, ridge glass, end walls
    for x in np.arange(-GX, GX + 0.1, 4):
        for y in np.arange(-GY, GY + 0.1, 4):
            b.batch("frame").box([x, y, GH / 2], [0.05, 0.05, GH / 2])
        b.batch("frame").box([x, 0, GH - 0.25], [0.04, GY, 0.04])          # truss chord
        b.batch("frame").box([x, 0, GH - 0.6], [0.02, GY, 0.02])
    for y in np.arange(-GY, GY + 0.1, 4):
        b.batch("frame").box([0, y, GH], [GX, 0.08, 0.05])                  # gutters
    slope = np.degrees(np.arctan2(RIDGE, 2.0))
    half = np.hypot(2.0, RIDGE) / 2
    for y in np.arange(-GY, GY - 0.1, 4):
        for s, sgn in ((y + 1.0, -1), (y + 3.0, 1)):
            b.world.append(f'<geom type="box" size="{GX} {half:.3f} .005" pos="0 {s} {GH + RIDGE / 2}" '
                           f'euler="{sgn * slope:.2f} 0 0" material="glass" contype="0" conaffinity="0"/>')
        b.batch("frame").box([0, y + 2, GH + RIDGE], [GX, 0.05, 0.04])     # ridge
        for x in np.arange(-GX, GX + 0.1, 1.0):
            for s, sgn in ((y + 1.0, -1), (y + 3.0, 1)):
                b.batch("frame").box([x, s, GH + RIDGE / 2 + 0.01], [0.012, half, 0.012],
                                     R=np.array([[1, 0, 0], [0, np.cos(np.radians(sgn * slope)), -np.sin(np.radians(sgn * slope))],
                                                 [0, np.sin(np.radians(sgn * slope)), np.cos(np.radians(sgn * slope))]]))
    for x in (-GX, GX):
        b.world.append(f'<geom type="box" size=".005 {GY} {GH / 2}" pos="{x} 0 {GH / 2}" material="glass" contype="0" conaffinity="0"/>')
    for y in (-GY, GY):
        b.world.append(f'<geom type="box" size="{GX} .005 {GH / 2}" pos="0 {y} {GH / 2}" material="glass" contype="0" conaffinity="0"/>')
    # grow-light rows
    for y in np.arange(-6, 6.1, 4):
        for x in np.arange(-18, 18.1, 3):
            b.batch("lamp").box([x, y, GH - 0.75], [0.25, 0.12, 0.06])

    # crop rows: gutters at 1.6 m pitch, aisles (pipe rails) between them
    gutters = np.arange(-6.4, 6.41, 1.6)
    aisles = np.concatenate([gutters - 0.8, [gutters[-1] + 0.8]])
    for ya in aisles:
        for sy in (-0.275, 0.275):
            b.batch("pipe").cylinder([-17.5, ya + sy, 0.13], [17.5, ya + sy, 0.13], 0.0255, seg=10)
    work_aisle = -0.8
    pickable = []
    for yg in gutters:
        b.batch("slab").box([0, yg, 0.55], [17.2, 0.14, 0.05])
        b.batch("frame").cylinder([-17.2, yg, 3.6], [17.2, yg, 3.6], 0.01, seg=4)
        for side in (-1, 1):
            yr = yg + 0.2 * side
            facing_work = abs((yr + 0.4 * side) - work_aisle) < 0.3 and abs(yr - work_aisle) < 0.7
            for x in np.arange(-16.8, 16.8, 0.5):
                xr = x + rng.normal(0, 0.03)
                lean = rng.normal(0, 0.05)
                p0 = np.array([xr, yr, 0.6])
                p1 = np.array([xr + 0.15, yr + lean, 2.0])
                p2 = np.array([xr + 0.05, yr + lean * 1.5, 3.55])
                b.batch("tomato_stem").cylinder(p0, p1, 0.012, seg=5)
                b.batch("tomato_stem").cylinder(p1, p2, 0.011, seg=5)
                for h in np.arange(1.8, 3.4, 0.3):                         # compound leaves
                    az = rng.uniform(0, 6.3)
                    base = p1 + (p2 - p1) * (h - 2.0) / 1.55
                    for k in range(3):
                        b.batch("tomato_leaf").leaf(base + [0.12 * k * np.cos(az), 0.12 * k * np.sin(az), -0.02 * k],
                                                    az + rng.normal(0, 0.5), rng.uniform(-0.3, 0.4), 0.22, 0.07,
                                                    droop=0.4, n=3)
                for h in (0.95, 1.25, 1.55):                               # trusses, ripen bottom-up
                    if rng.random() < 0.15:
                        continue
                    toward = np.sign(work_aisle - yr) if facing_work else side
                    c = np.array([xr + rng.normal(0, 0.05), yr + 0.13 * toward, h + rng.normal(0, 0.04)])
                    b.batch("tomato_stem").cylinder(p0 + (p1 - p0) * (h - 0.6) / 1.4, c + [0, 0, 0.05], 0.005, seg=4)
                    ripe_p = np.clip(1.6 - h + rng.normal(0, 0.15), 0, 1)
                    for f in range(rng.integers(4, 7)):
                        fp = c + [0.06 * (f - 2.5), rng.normal(0, 0.02), -0.04 * abs(f - 2.5) + rng.normal(0, 0.01)]
                        u = rng.random()
                        mat = "tomato_red" if u < ripe_p * 0.8 else ("tomato_orange" if u < ripe_p else "tomato_green")
                        if (facing_work and mat == "tomato_red" and abs(fp[0]) < 7
                                and 1.0 < fp[2] < 1.6 and len(pickable) < 80 and f in (0, 5, 3)):
                            pickable.append(fp)
                            b.world.append(f'<geom name="fruit{len(pickable)-1}" type="sphere" size=".036" pos="{fmt(fp)}" '
                                           f'material="tomato_red" contype="0" conaffinity="0"/>')
                        else:
                            b.batch(mat).ellipsoid(fp, [0.036, 0.036, 0.032], seg=7, rings=5)
    # a few distant trees outside
    for x, y in rng.uniform([-60, 14], [60, 40], (25, 2)):
        flora.broadleaf_tree(b, rng, x, y, 0.0, height=rng.uniform(7, 11), crown=rng.uniform(2.5, 3.5), collide=False)
    for x, y in rng.uniform([-60, -40], [60, -14], (25, 2)):
        flora.conifer(b, rng, x, y, 0.0, height=rng.uniform(9, 15), collide=False)

    robots.pipe_rail_harvester(b, "picker", x0=-7.5, y_aisle=work_aisle, rail_z=0.13)
    b.world.append(f'<camera name="overview" pos="-26 -14 9" xyaxes="{fmt([0.5, -0.866, 0, 0.3, 0.17, 0.94])}"/>')
    b.world.append(f'<camera name="aisle" pos="-14 {work_aisle} 1.9" xyaxes="0 -1 0 .1 0 .995"/>')
    order = sorted(range(len(pickable)), key=lambda i: pickable[i][0])
    b.meta.update(controller="harvest", fruits=[pickable[i].tolist() for i in order],
                  fruit_geoms=[f"fruit{i}" for i in order], aisle_y=work_aisle)
    b.timestep = 0.002
    return b


# ======================================================================================
@scene
def wind_turbine(seed: int = 0) -> Builder:
    b = Builder("wind_turbine", seed)
    rng = b.rng
    flora.plant_materials(b, rng)
    b.sky_and_sun(Lighting.sample(rng), shadow_extent=80)

    hx = hy = 160.0
    n = 321
    X, Y = np.meshgrid(np.linspace(-hx, hx, n), np.linspace(-hy, hy, n))
    heading = float(rng.uniform(-30, 30))               # wind blows toward +x (± 30 deg)
    z = 22 * np.exp(-((X * 0.9 + Y * 0.3) / 55) ** 2) + 4 * noise2(n, n, rng, octaves=5, base=2)
    t = Terrain(z, hx, hy)
    tn = 2048
    tex = colorize(t, rng, tn, PASTURE)
    # field patchwork with hedgerows
    tx = np.ones((tn, 1)) * np.linspace(-hx, hx, tn)[None, :]
    ty = np.linspace(-hy, hy, tn)[:, None] * np.ones((1, tn))
    patch = (np.floor((tx + 13) / 47) + 3 * np.floor((ty + 7) / 39)) % 3
    tex *= (0.92 + 0.06 * patch)[..., None]
    add_terrain(b, t, tex)
    for xh in np.arange(-160 + 34, 160, 47):
        for y in np.arange(-150, 150, 3.0):
            if rng.random() < 0.7:
                flora.broadleaf_tree(b, rng, xh + rng.normal(0, .5), y, float(t.height(xh, y)),
                                     height=rng.uniform(2.5, 5), crown=rng.uniform(1.0, 1.8), collide=False)
    for x, y in rng.uniform(-150, 150, (60, 2)):
        if np.hypot(x, y) > 25:
            flora.broadleaf_tree(b, rng, x, y, float(t.height(x, y)), height=rng.uniform(8, 13),
                                 crown=rng.uniform(2.5, 4.0), collide=False)
    for x, y in rng.uniform(-150, 150, (2500, 2)):
        flora.grass_tuft(b, rng, x, y, float(t.height(x, y)), mat="dry_grass" if rng.random() < .5 else "grass_tuft", h=0.4)

    # turbine: tapered tower (mesh visual + cylinder collision), yawed nacelle, spinning rotor
    tz = float(t.height(0, 0))
    H, R_rot = 48.0, 24.0
    b.material("turbine", [0.93, 0.94, 0.95, 1], specular=0.5, shininess=0.6)
    b.material("turbine_red", [0.8, 0.12, 0.1, 1], specular=0.5)
    b.batch("turbine").cylinder([0, 0, tz - 1], [0, 0, tz + H], 2.0, 1.25, seg=24)
    b.batch("turbine").cylinder([0, 0, tz - 1.2], [0, 0, tz + 0.3], 4.0, 4.0, seg=24)
    b.world.append(f'<geom name="tower" type="cylinder" fromto="0 0 {tz-1} 0 0 {tz+H}" size="1.7" rgba="0 0 0 0" group="3"/>')
    from .mesh import MeshBatch
    blade = MeshBatch()
    blade.blade([0, 0, 0], [0, 0, 1], R_rot - 1.2, chord_root=2.6, chord_tip=0.6, thickness=0.18, twist=0.45)
    blade.write_obj(b.dir / "blade.obj")
    b.assets.append('<mesh name="blade" file="blade.obj"/>')
    yaw = heading + 180  # face upwind
    b.world.append("\n    ".join([
        f'<body name="nacelle" pos="0 0 {tz + H + 1.2}" euler="0 0 {yaw:.2f}">',
        '  <geom type="box" size="3.8 1.6 1.6" pos="1.6 0 0" material="turbine"/>',
        '  <geom type="box" size=".6 .6 .08" pos="2.5 0 1.65" material="turbine_red" contype="0" conaffinity="0"/>',
        '  <body name="rotor" pos="-2.7 0 0">',
        '    <joint name="rotor_spin" type="hinge" axis="-1 0 0" damping="0"/>',
        '    <inertial pos="0 0 0" mass="200" diaginertia="2000 1500 1500"/>',
        '    <geom type="ellipsoid" size="2.2 1.5 1.5" pos="-1 0 0" material="turbine"/>',
        *[f'    <geom type="mesh" mesh="blade" pos="-.8 0 0" euler="{120*k} 0 0" material="turbine"/>' for k in range(3)],
        '  </body>',
        '</body>']))
    b.actuators.append('<velocity name="rotor_drive" joint="rotor_spin" kv="20000" ctrlrange="0 3"/>')

    base = (-18.0, -12.0)
    q = robots.quadrotor(b, "uav", [*base, float(t.height(*base)) + 0.15], chase_offset=(-6, -4, 2.5))
    # world-fixed camera that keeps both the UAV and the turbine in frame
    b.world.append(f'<camera name="tracker" mode="targetbody" target="uav" pos="-70 -40 {tz + 20:.1f}" fovy="22"/>')
    q["cameras"][0] = "tracker"
    b.overview([-125, -75, tz + 45], [0, 0, tz + 30])
    # inspection mission (absolute z): helix around tower, then hover upwind of the rotor plane
    wps = [[base[0], base[1], tz + 6]]
    for k in range(0, 40):
        a = np.radians(225 + 18 * k)
        wps.append([10 * np.cos(a), 10 * np.sin(a), tz + 6 + k * 1.0])
    up = np.array([np.cos(np.radians(heading + 180)), np.sin(np.radians(heading + 180))])
    side = np.array([-up[1], up[0]])
    for s in (-16, 0, 16, 0):
        p = up * 14 + side * s
        wps.append([p[0], p[1], tz + H + 1.2 + (abs(s) * 0.4)])
    b.meta.update(controller="quad_waypoints", waypoints_abs=[list(map(float, w)) for w in wps], cruise=3.0,
                  rotor_speed=1.6,
                  wind=dict(U=7.0, heading_deg=heading, gust=1.5, z_ref=10, z0=0.05,
                            cylinders=[[0.0, 0.0, 2.0, tz + H]],
                            turbine=dict(hub=[0, 0, tz + H + 1.2], R=R_rot, Ct=0.8, k=0.075)))
    return b


def build(name: str, seed: int = 0):
    if name not in SCENES:
        raise SystemExit(f"unknown scene {name!r}; choose from {', '.join(SCENES)}")
    b = SCENES[name](seed)
    return b.write()


# developer scenes/controllers in sim/custom/*.py register themselves on import
from . import custom as _custom  # noqa: E402

_custom.load_all()
