# Writing a custom scene

This guide shows how to add your own environment (terrain, vegetation, a robot, a mission,
and optionally your own controller) to `sim/`. The complete worked example is
[`sim/custom/pond_survey.py`](../sim/custom/pond_survey.py): a quadrotor orbiting a meadow
pond while its camera faces the shoreline. Copy that file and change it.

```bash
uv run --extra sim python -m sim view pond_survey           # try the example
cp sim/custom/pond_survey.py sim/custom/my_scene.py          # start your own
uv run --extra sim python -m sim list                        # my_scene shows up automatically
```

---

## 1. The model in one paragraph

A **scene** is a Python function `seed -> Builder`, decorated with `@scene`. The function
appends MJCF fragments (assets, world geoms, actuators, sensors) to a `Builder` and puts
everything a controller needs into `b.meta`. Then `Builder.write()` produces a
self-contained folder:

```
sim/generated/<scene>-s<seed>/
  scene.xml        MuJoCo model; loads from Python *and* C/C++ (mj_loadXML)
  meta.json        robots, cameras, controller name + mission params, wind, terrain info
  terrain.bin      heightfield (MuJoCo binary: int32 nrow, int32 ncol, float32 data)
  ground.png       terrain texture
  mesh_<mat>.obj   merged vegetation, one mesh per material
```

At run time, `sim.runtime.Sim` loads that folder, rebuilds the `Terrain` and the
`WindField` from `meta.json`, and instantiates the controller named in `meta["controller"]`.

Any module in **`sim/custom/`** (whose name doesn't start with `_`) is imported automatically,
so you never need to edit `scenes.py` or `control.py`.

---

## 2. Skeleton

```python
# sim/custom/my_scene.py
import numpy as np

from sim import flora, robots
from sim.builder import Builder, Lighting
from sim.scenes import GRASS, add_terrain, noise2, poisson_disk, scene, smoothstep
from sim.terrain import Terrain, colorize


@scene                                   # registers "my_scene" (the function name)
def my_scene(seed: int = 0) -> Builder:
    """One-line description shown by `python -m sim list`."""
    b = Builder("my_scene", seed)        # name must match the function name
    rng = b.rng                          # the ONLY source of randomness
    flora.plant_materials(b, rng)        # bark/leaf/crop/rock materials (colour-jittered)
    b.sky_and_sun(Lighting.sample(rng))  # randomised time of day, sky, haze

    # terrain
    n, hx, hy = 257, 50.0, 50.0
    z = 3.0 * noise2(n, n, rng, octaves=5, base=2)        # metres; row = y, col = x
    t = Terrain(z, hx, hy)
    add_terrain(b, t, colorize(t, rng, 1024, GRASS))      # writes .bin/.png, adds geom + meta

    # things on the ground: always use t.height(x, y)
    for x, y in poisson_disk(rng, (-45, 45), (-45, 45), 5.0, tries=2000):
        flora.broadleaf_tree(b, rng, x, y, float(t.height(x, y)), height=rng.uniform(6, 10))

    # robot + mission
    robots.quadrotor(b, "uav", [0, 0, float(t.height(0, 0)) + 0.15])
    b.overview([-40, -40, 25], [0, 0, 0])
    b.meta.update(controller="quad_waypoints", cruise=3.0,
                  waypoints_agl=[[20, 0, 5], [0, 20, 5], [-20, 0, 5], [0, -20, 5]])
    return b
```

Then:

```bash
uv run --extra sim python -m sim view my_scene --rebuild      # --rebuild after every code change
uv run --extra sim python -m sim render my_scene --seconds 15
uv run --extra sim python -m sim view my_scene --seed 4 --rebuild   # another random variant
```

> `view` and `render` reuse an existing `sim/generated/<scene>-s<seed>/` folder. While you're
> editing the scene code, always pass `--rebuild` (or run `python -m sim build my_scene`).

---

## 3. Building blocks

### Builder (`sim/builder.py`)

| call | does |
|---|---|
| `Builder(name, seed)` | creates the output folder and `b.rng` (a `numpy.random.Generator`) |
| `b.material(name, rgba, specular=, shininess=, reflectance=, texture=, texrepeat=, emission=)` | adds a material once; returns its name |
| `b.assets / b.world / b.actuators / b.sensors / b.contacts` | lists of raw MJCF strings; append freely |
| `b.batch(material)` | a `MeshBatch` merged into one visual mesh geom of that material |
| `b.sky_and_sun(light, shadow_extent=, castshadow=, fill=)` | skybox, sun, optional diffuse fill lights |
| `b.overview(pos, target, name="overview")` | static camera looking at `target` |
| `b.meta` | dict written to `meta.json` (your controller's parameters go here) |
| `b.timestep` | physics step, default 0.002 s |

### Terrain (`sim/terrain.py`)

| call | does |
|---|---|
| `fbm(n, rng, octaves, base, persistence)` / `noise2(nr, nc, rng, ...)` | fractal noise in ≈[-1, 1] |
| `Terrain(z, half_x, half_y)` | absolute heights; `t.height(x, y)` gives bilinear height (vectorised) |
| `colorize(t, rng, n, palette, extra=mask)` | noise → colour ramp; `extra` biases it with a [0,1] mask |
| `resample(field, n)` | resample a heightfield-shaped mask to texture size |
| `add_terrain(b, t, tex)` (in `scenes.py`) | writes files, adds hfield asset + geom + `meta["terrain"]` |

Palettes are in `scenes.py`: `GRASS`, `SOIL`, `DRY_SOIL`, `FOREST_FLOOR`, `PASTURE`.
Blend textures with masks, e.g. `tex = grass * (1 - m) + soil * m` with `m` of shape `(n, n, 1)`.

### Vegetation & props (`sim/flora.py`)

`broadleaf_tree`, `conifer`, `birch`, `rock`, `log` (these also add invisible colliders),
plus `lettuce`, `weed`, `grass_tuft` (visual only). All take `(b, rng, x, y, z, ...)`.
For your own shapes, use the `MeshBatch` primitives in `sim/mesh.py`: `ellipsoid`, `cone`,
`cylinder`, `box`, `leaf`, `blade`.

```python
b.material("sunflower", [0.95, 0.75, 0.1, 1])
b.batch("sunflower").ellipsoid([x, y, z + 1.6], [0.15, 0.15, 0.04])
b.batch("leaf_a").cylinder([x, y, z], [x, y, z + 1.6], 0.02)
```

### Robots (`sim/robots.py`)

| function | meta it returns / registers | default controller |
|---|---|---|
| `quadrotor(b, name, pos, arm=, mass=, chase_offset=)` | body, 4 thrust actuators, cameras `_chase/_fpv/_nadir`, `drag_CdA` | `quad_waypoints` |
| `field_rover(b, name, pos, yaw=, track=, wheelbase=, r=, height=)` | 4 wheel velocity actuators, cameras `_chase/_bedcam/_front` | `rover_serpentine` |
| `pipe_rail_harvester(b, name, x0, y_aisle, rail_z)` | rail/lift/arm position actuators, `_ee` site, `_wristcam` | `harvest` |

All three are parametric, so `arm`, `track`, `wheelbase`, `r` etc. are your design variables.
`meta["robots"][0]` is the robot the controllers and the `render` insets use. Its
`cameras[0]` is the main view, and `cameras[1]` is the onboard inset.

---

## 4. Built-in controllers and the meta keys they read

| `meta["controller"]` | required keys | optional |
|---|---|---|
| `quad_waypoints` | `waypoints_agl` *(z = height above terrain)* **or** `waypoints_abs` | `cruise` (m/s, default 3) |
| `rover_serpentine` | `lanes_y` (list of y), `x_turn` (x_min, x_max) | `speed` (m/s) |
| `harvest` | `fruits` (xyz list), `fruit_geoms` (geom names), `aisle_y` | |

Adding `meta["wind"]` enables the analytic wind field for any scene:

```python
b.meta["wind"] = dict(U=5.0,             # mean speed at z_ref [m/s]
                      heading_deg=30,    # direction the wind blows *towards*
                      gust=1.0,          # turbulence amplitude [m/s]
                      z_ref=10, z0=0.1,  # log-law reference height, roughness length
                      cylinders=[[x, y, R, top_z]],              # flow around obstacles
                      turbine=dict(hub=[x, y, z], R=24, Ct=0.8, k=0.075))  # Jensen wake
```

View it with `--wind`.

---

## 5. Writing your own controller

A controller is any class with this shape, registered under a name:

```python
from sim.control import CONTROLLERS, QuadController

class MyController:
    def __init__(self, sim, meta: dict):    # sim.m, sim.d (MuJoCo), sim.terrain, sim.wind
        self.sim = sim
        self.ctl = QuadController(sim.m, sim.d, meta["robots"][0])   # optional: reuse inner loops

    def update(self, t: float, dt: float):   # called before EVERY mj_step
        ...                                   # read sim.d, write sim.d.ctrl / sim.d.xfrc_applied

CONTROLLERS["my_controller"] = MyController   # then: b.meta["controller"] = "my_controller"
```

Rules:
- `QuadController(p_d, v_d, yaw_d, a_ff)` is a full SE(3) position controller. Usually you
  only need to generate references for it. `PondOrbit` in the example is 25 lines.
- Wind drag is applied by the built-in quad mission, **not** by `QuadController`. If you write
  your own UAV controller and want wind, apply
  `sim.wind.drag_force(pos, vel, CdA, t)` to `sim.d.xfrc_applied[body, :3]` (see the example).
- Keep all mission parameters in `b.meta`, not hard-coded in the controller. That way the
  same controller works across seeds and scenes, and the C++ side can read the same JSON.
- An RL policy is just another controller: put inference in `update()`. Use the built-in
  controllers as baselines, sources of demonstrations, or a base for residual policies.

---

## 6. Conventions you must follow

**Reproducibility**
- Use only `b.rng` (pass it down). Never use `np.random.*`, `random`, time, or dict/set
  iteration order for anything that affects output.
- The same `(scene, seed)` must give byte-identical files. Check it with
  `md5sum sim/generated/my_scene-s0/*` before and after `python -m sim build my_scene`.

**Collision vs. visuals**
- Vegetation in `b.batch(...)` is visual only (`contype=0`). Nothing can hit it.
- Anything a robot should collide with gets a cheap primitive collider: an invisible geom
  (`rgba="0 0 0 0" group="3"`) in `b.world`. Toggle group 3 in the viewer to inspect them.
- Decorative geoms you add directly should carry `contype="0" conaffinity="0"`.

**Units & frames**
- `compiler angle="degree"` applies to `euler` and joint `range`. Actuator `ctrlrange` and
  controller values are in radians, metres, newtons.
- Heightfield: row 0 is at `y = -half_y`, column 0 at `x = -half_x`. `write_texture` /
  `add_terrain` already flip the image correctly.
- Cameras look along their local −z with +y up. Use `lookat_xyaxes(offset)` or `b.overview`.

**Performance budget**
- Keep individual geoms to a few thousand (the renderer's scene limit is 10 000). Merge
  anything numerous into batches.
- Up to about 2M triangles total renders fine on a discrete GPU. Use lower `seg`/`rings`
  on primitives for big fields.
- Very large textures cost memory at build time. 1024–2048 px is plenty.

---

## 7. Checklist before committing a scene

```bash
uv run --extra sim python -m sim build my_scene                      # builds without errors
uv run --extra sim python -m sim render my_scene --seconds 20        # look at the MP4 + PNG
uv run --extra sim python -m sim build my_scene --seed 1             # other seeds look sane
cmake -S sim/cpp -B sim/cpp/build && cmake --build sim/cpp/build -j  # C++ loads it too
sim/cpp/build/batch_rollout sim/generated/my_scene-s0/scene.xml 64 16 1000
```

- [ ] robot starts on the ground (not inside terrain or a collider) and the controller tracks
- [ ] colliders cover everything the robot can reach; visuals have `contype=0`
- [ ] same seed gives identical files
- [ ] a one-line docstring on the scene function (shown by `python -m sim list`)
