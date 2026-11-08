# Tutorial: a new robot and scene

You will build **`meadow_scout`**: a small quadrotor assembled part by part, flying a trail
over a hilly meadow with scattered trees. The finished file is
[`sim/myscenes/meadow_scout.py`](../sim/myscenes/meadow_scout.py), about 60 lines.
`tests/sim/test_tutorial.py` runs everything shown here, so the code on this page is known
to work.

```bash
uv run --extra sim python -m sim view meadow_scout                 # fly it (C++ core)
uv run --extra sim python -m sim render meadow_scout --seconds 20  # video with onboard camera insets
uv run --extra sim python -m sim sensors meadow_scout              # what its sensors see
uv run --extra sim python -m sim randomize meadow_scout --n 32     # robustness over 32 drawn variants
```

## 1. The idea

```
 parts/*.toml ──► components ──► Robot ─┐
 (datasheets)     (Rotor, Battery, ...)  │
                                         ├──► scene function ──► scene.xml + meta.json ──► C++ core
 world features ─► World ────────────────┤     (sim/myscenes/)     (written once)          (view, render,
 (Hills, Trail, Forest, ...)             │                                                  sensors, training)
 control stages ─► chain ────────────────┘
 (FollowPath >> Se3Controller >> Mixer)
```

You write the left side. Everything to the right happens automatically: building the
files, running the physics, the battery, the sensors and the controller, in C++.

## 2. Pick parts

Parts are datasheet numbers in `sim/parts/*.toml`, one file per domain. Browse them from
Python:

```python
from sim import parts
parts.names("motor")              # ['m1404_3000kv', 'm2004_1700kv', ...]
parts.get("battery", "lipo_4s_1500")
# {'chemistry': 'lipo', 'cells': 4, 'capacity_mah': 1500, 'mass_g': 180, 'c_max': 75, ...}
```

To use a part that isn't there, copy its datasheet values into a new table in the right
file (e.g. `[motor.my_motor]` in `propulsion.toml`). Nothing else is needed.

## 3. Assemble the robot

A robot is a set of components attached at **named mount points**. The frame comes first:
it creates the main body and offers mounts (`deck`, `center`, `nose`, `cg`, one tip per arm).
Each further component attaches to a mount and may offer new ones; the battery, for example,
offers a `top` mount.

```python
def scout() -> Robot:
    r = Robot("scout")
    r.environment = {"air_density": 1.2,
                     "magnetic_field": {"field_ut": 47.5, "inclination_deg": 62, "declination_deg": 2.5}}
    frame = r.add(QuadFrame("cf_10x8", arm_length=0.11, prop_d=0.127))
    battery = r.add(Battery("lipo_4s_1500"), at=frame.mount("deck"))
    r.add(Board("fc_stack_h7"), at=frame.mount("center"))
    r.add(CameraModule("oakd_lite", rgb="oakd_lite_rgb", depth="oakd_lite_stereo"), at=frame.mount("nose"))
    for arm in frame.arms:                     # thrust limit and motor lag computed from the parts
        r.add(Rotor("m2004_1700kv", "p5x3_bi", arm.spin, motor_mount_g=4), at=arm.tip)
    r.add(Imu("icm42688p", mount_hz=90), at=frame.mount("cg"))
    r.add(Mag("ist8310", current_ut_per_a=0.15), at=frame.mount("origin"))
    r.add(Gnss("ublox_m10", antenna_height=0.05), at=battery.mount("top"))
    chase = (-2.5, 0.0, 1.0)
    r.add(ViewCamera("chase", lookat_xyaxes(chase), offset=chase, mode="trackcom"), at=frame.mount("origin"))
    return r
```

What each line brings:

| component | adds |
|---|---|
| `QuadFrame` | main body, plate and arms (mass from the tube part), collision hull, landing feet |
| `Battery` | mass and box; the power train's energy source (voltage sag, state of charge) |
| `Board` | mass, box, and a constant electrical load (`power_w` of the part) |
| `CameraModule` | board plus colour and depth cameras, whose field of view and error models come from their parts |
| `Rotor` | motor and prop masses, a thrust actuator with motor lag, rotor drag, its electrical load (power from thrust), vibration sources for the IMU |
| `Imu`, `Mag`, `Gnss` | sensor models from their parts, installed where you put them |
| `ViewCamera` | a camera for the viewer that isn't a sensor part |

**Rotor ratings.** `Rotor` without `thrust_max` / `lag_s` computes them like the sizing
does:
- **max thrust:** where the motor runs out of voltage headroom or current at 50 % charge,
  with this battery and this avionics load
- **lag:** the motor and prop speed loop at hover thrust, for this robot's total mass

`print(scout().summary())` shows the result:

```
robot 'scout': 12 components, 1 bodies, 4 actuators, total mass 0.458 kg
  - rotor0       Rotor: m2004_1700kv + p5x3_bi, spin -, max 4.2 N, lag 14 ms
  ...
  power: lipo_4s_1500 -> 4 rotors (from thrust), avionics 6 W
  sensors: rgb_camera oakd_lite_rgb, depth_camera oakd_lite_stereo, imu icm42688p, mag ist8310, gnss ublox_m10
```

`frame.mounts()` lists where things can go. Asking for a mount that doesn't exist fails
with the list of valid ones.

> To size a vehicle against requirements instead of choosing parts by hand, write a mission
> file and use `python -m sim design drone` and the `quadrotor()` template (see
> [vehicle_design.md](vehicle_design.md)). The forest scene does that.

## 4. Compose the world

Features are applied **in the order you add them**:

```python
w = World("meadow_scout", seed, half_size=40.0, resolution=321)
w.add(PlantColours())                                   # seeded colour variation of the plants
w.add(Sun())                                            # random time of day
w.add(Hills(relief=3.0, base=3))                        # gentle hills
trail = w.add(Trail(amplitude=(8, 3), scale=(12, 5)))   # a winding trail, cut into the ground
w.add(Ground(GRASS, overlays={trail: DRY_SOIL}))        # paint: grass, the trail as dry soil
w.add(Forest(spacing=7.0, clear=[Clear(trail, 4.0)], mix=(("broadleaf", 0.7), ("birch", 0.3))))
w.add(Rocks(count=40, size=Uniform(0.2, 0.6), clear=[Clear(trail, 1.0)]))
w.add(Grass(count=1500))
w.add(Wind(speed=2.0, gust=0.5))
```

Three rules make this predictable:
1. **Order is the generation order.** One seeded random stream per scene, so the same seed
   always gives the same world, and a feature only sees the features added before it.
2. **Terrain-shaping first.** `Hills`, `Craters` and `Trail` edit the heights; `Ground` (or
   anything that needs heights) freezes them.
3. **Paths and some objects are regions.** `Clear(trail, 4.0)` keeps trees 4 m from the
   trail. `Clear(loop, 2.2, larger_than=0.15)` would keep only rocks bigger than 15 cm off it.

All features: `Hills`, `Craters`, `Trail`, `DriveLoop`, `Ground`, `PlantColours`, `Forest`,
`Grass`, `Rocks`, `Logs`, `Lander`, `Sun`, `Wind` (see [architecture.md](architecture.md),
section 3.5).

## 5. Choose a controller

Controllers are stages chained with `>>`. Each turns the previous stage's output into its own:

```python
FollowPath(route, cruise=4.0) >> Se3Controller() >> Mixer()
#  where to be        ->        thrust + torques ->  four rotor thrusts
```

The route comes from the trail: `route = trail.path(np.linspace(-34, 34, 50), height_agl=3.0)`.
For ground vehicles it's `PurePursuit(loop) >> Ackermann() >> WheelSpeedPI()`.

The chain reads the **true** state; that is the default `"truth"` source. Flying on
estimated state, `.using("sensors")`, needs a state estimator, which doesn't exist yet. That
switch is where sim-to-real starts.

## 6. Put it together

A scene is a function decorated with `@scene` in a file in `sim/myscenes/`. Every file
there is picked up automatically:

```python
@scene
def meadow_scout(seed: int = 0):
    """Hand-assembled 5-inch quad flying a trail over a hilly meadow (tutorial example)."""
    w = World("meadow_scout", seed, half_size=40.0, resolution=321)
    ...                                                  # the features of section 4
    route = trail.path(np.linspace(-34, 34, 50), height_agl=3.0)
    w.add_robot(scout(), *route[0][:2], clearance=0.1, drag_CdA=0.03,
                control=FollowPath(route, cruise=4.0) >> Se3Controller() >> Mixer(),
                uncertainty={"mass_frac": 0.05, "thrust_coef": 0.08, "motor_lag": 0.3,
                             "imu_noise": [0.9, 1.5], "latency_ms_add": [0, 15], "wind_speed": [0, 4]})
    w.overview([-50, -45, 30], [0, 0, 0])
    return w.builder
```

`add_robot` places the robot on the ground at the route's start and writes it, its
controller chain, its body drag area and its uncertainty ranges into the scene (see
[uncertainty.md](uncertainty.md) for all keys). Then:

```bash
uv run --extra sim python -m sim list                          # meadow_scout is there
uv run --extra sim python -m sim view meadow_scout --rebuild   # --rebuild after editing the scene
```

It runs on the C++ core: you wrote no C++. The commands at the top of this page give the
sensor report, the robustness sweep and a video.

## 7. Train on it

```python
from sim.core import CoreBatch
env = CoreBatch("meadow_scout", n_envs=256, threads=16, mode="setpoint")
env.reset(variants=range(256))           # 256 drawn variants of the scout
env.act(actions)                         # (256, 4): vx, vy, vz, yaw_rate -> replaces FollowPath
env.step(33)                             # 66 ms of physics, battery, sensors
env.imu, env.gnss, env.plant, env.qpos   # numpy views
```

`mode="setpoint"` swaps the chain's top stage for a command stage your policy feeds. The
rest (`Se3Controller >> Mixer`) keeps the vehicle stable. Details:
[cpp_core.md](cpp_core.md).

## 8. Write your own type

When the existing pieces aren't enough, write a new one. Each kind is one small class.

### A component: a spray tank

A component adds mass, geometry and, optionally, an electrical load:

```python
class SprayTank(Component):
    kind = "tank"

    def __init__(self, litres, empty_g=150.0, pump_w=12.0, radius=0.05):
        super().__init__()
        self.litres, self.empty_g, self.pump_w, self.radius = litres, empty_g, pump_w, radius

    def attach(self, robot, at):
        r = self.radius
        h = self.litres / 1000 / (math.pi * r * r)            # height holding the volume
        m = self.empty_g / 1000 + self.litres * 1.0           # 1 L of water = 1 kg
        pos = at.place(h)                                     # touching the mount, along its direction
        I = m * np.diag([(3*r*r + h*h) / 12, (3*r*r + h*h) / 12, r*r / 2])
        at.body.mass_parts.append(MassPart(m, pos, I, "spray tank"))
        at.body.geoms.append(f'<geom type="cylinder" size="{r:.3f} {h/2:.3f}" pos="{fmt(pos)}" '
                             'material="white_paint" contype="0" conaffinity="0"/>')

    def contribute(self, robot, meta):                        # the pump draws power from the battery
        meta["power"]["loads"].append({"type": "constant", "group": "payload",
                                       "label": "spray pump", "watts": self.pump_w})

robot.add(SprayTank(litres=0.3), at=frame.mount("belly"))
```

- **Everything follows automatically:** the robot's total mass, centre of mass and inertia,
  the rotors' hover point and lag, the battery drain.
- **It runs in C++ too:** a constant load is a type the core already knows.

### A world feature: a pond

A feature edits the terrain and/or adds objects. If it has a `distance`, others can keep
clear of it:

```python
class Pond(Feature):
    def __init__(self, x, y, radius, depth=0.8):
        self.x, self.y, self.radius, self.depth = x, y, radius, depth

    def apply(self, w):
        z = w.edit_heights()                                  # before Ground freezes the terrain
        r = np.hypot(w.X - self.x, w.Y - self.y) / self.radius
        z -= self.depth * np.clip(1 - r * r, 0, None)         # a bowl

    def distance(self, x, y):
        return max(0.0, float(np.hypot(x - self.x, y - self.y)) - self.radius)

pond = w.add(Pond(0.0, 0.0, radius=6.0))
w.add(Forest(spacing=4.0, clear=[Clear(pond, 2.0)]))
```

Features only describe the world, so a new one never needs C++.

### A control stage: orbit a point

A stage turns its input into its output. A top stage ignores its input and produces the
setpoint:

```python
class Orbit(Controller):
    name = "orbit"
    defaults = {"center": (0.0, 0.0), "radius": 8.0, "height_agl": 3.0, "speed": 3.0}

    def bind(self, sim, robot):
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

STAGES[Orbit.name] = Orbit                                    # lets scenes store it in their meta
control = Orbit(center=(10, 0), radius=8) >> Se3Controller() >> Mixer()
```

A new stage exists only in Python, so its scene runs on the Python engine, and the CLI says
so. To make it fast, port it to `sim/cpp/core/csrc/control.cpp`, add its parser entry in
`bindings.cpp` and a line in `CoreBatch._stage`, and add an equivalence test (see
[cpp_core.md](cpp_core.md)).

## 9. Where things live

| you want to | look in |
|---|---|
| see a part's numbers / add a part | `sim/parts/*.toml` |
| understand an equation | `sim/physics/` (one domain per file) |
| see how a part becomes simulation | `sim/components/` |
| build a common vehicle from requirements | `sim/design/`, `sim/robots/templates.py` |
| compose terrain and objects | `sim/world/features/` |
| change how it flies or drives | `sim/control/` |
| add your scene | `sim/myscenes/` |
| see how a step runs | `sim/runtime/sim.py` (Python), `sim/cpp/core/csrc/env.cpp` (C++) |

**Checklist for a new scene:**
1. Parts exist in `sim/parts/`.
2. The robot is assembled at named mounts; `robot.summary()` looks plausible (mass,
   thrust, power, sensors).
3. The world features are in order: terrain shaping → `Ground` → objects.
4. A controller chain is given.
5. `python -m sim view <scene> --rebuild` flies or drives.
6. Uncertainty ranges are added before training.
