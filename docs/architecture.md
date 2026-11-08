# Architecture: building robots and scenes from reusable pieces

> **Status: agreed design, not yet implemented.** Decisions taken are recorded in
> section 9. Implementation follows the migration plan in section 7, one layer per branch.

## 1. Why restructure

What exists today works, but it's a set of finished examples, not a toolkit:

| today | problem |
|---|---|
| `scenes.py`: one long function per scene | terrain, vegetation, robot and mission are mixed; a new scene means copying ~100 lines |
| `design/drone.py`, `design/rover.py` | sizing, geometry and MJCF generation in one block per vehicle type; a third vehicle type means a third copy |
| `design/plant.py`: `DronePlant`, `RoverPlant` | the power train is special-cased per vehicle instead of summed from the parts that draw power |
| `sensing/suite.py` | finds sensors by naming convention (`{name}_imu`, `{name}_gyro`) |
| `control.py` | controllers chosen by a string in `meta.json` and wired to one robot type each |
| C++ core | has `if drone ... if rover ...` paths; a new robot type means editing C++ |

**The goal:** a new robot or scene is mostly *composition* of existing pieces, and a new
*kind* of piece is written once and is then usable everywhere.

## 2. Principles

1. **One concept per file, readable on its own.** A file explains one physical thing (a
   propeller, a battery, an IMU) in its docstring with its equations, and implements it.
2. **Data separate from code.** Datasheet numbers live in TOML (`parts/`); code never
   hard-codes a specific motor or sensor.
3. **Dependencies only go downward** (section 3). `physics/` knows nothing about MuJoCo,
   and `components/` knows nothing about scenes.
4. **Same code for sizing and simulation.** The propeller formula used by the optimiser
   is the one the simulated rotor uses, and the robot builder used for sizing is the one
   that produces the MJCF.
5. **Python is the specification, C++ is the engine.** Every runtime model exists in
   readable Python first. The C++ core mirrors it and is tested against it, as now.
6. **Deterministic.** Every random choice draws from a seeded stream named after what it
   is for (`terrain`, `trees`, `imu.noise`...), so adding one thing never changes another.

## 3. Layers

```
                         ┌───────────────────────────────┐
  what you write ──────► │ scenes/   world + robots + task │
                         └──────────────┬────────────────┘
          ┌──────────────────┬──────────┴───────┬───────────────────┐
          ▼                  ▼                  ▼                   ▼
   ┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌───────────────┐
   │ world/      │   │ robots/      │   │ control/     │   │ design/       │
   │ terrain,    │   │ assemble     │   │ controllers  │   │ sizing,       │
   │ vegetation, │   │ components   │   │ (one common  │   │ optimisation  │
   │ wind, light │   │ into a robot │   │  interface)  │   │ over templates│
   └──────┬──────┘   └──────┬───────┘   └──────┬───────┘   └───────┬───────┘
          │                 ▼                  │                   │
          │          ┌──────────────┐          │                   │
          │          │ components/  │◄─────────┴───────────────────┘
          │          │ Frame, Rotor,│
          │          │ WheelDrive,  │
          │          │ Battery, Imu…│
          │          └──────┬───────┘
          ▼                 ▼
   ┌────────────────────────────────────┐    ┌──────────────────┐
   │ physics/  pure functions: propeller,│    │ parts/  TOML:    │
   │ dc_motor, battery, beam, traction,  │◄───│ datasheets only  │
   │ wind field, noise processes         │    └──────────────────┘
   └────────────────────────────────────┘
   runtime/     load a scene, step it (Python reference)   ── core/ (C++), same interface
   analysis/    Allan, FFT, robustness, logs               ── works on sim and real logs
```

An arrow means "uses". Nothing uses a layer above itself.

### 3.1 `parts/`: datasheets only

**Done (step 2).** The old `components.toml` split by domain, so each file stays short
(section names unchanged, so mission files did not change):

```
parts/
  propulsion.toml   motor.*, prop.*
  energy.toml       battery.*, chemistry.* (OCV curves, moved out of Python)
  drive.toml        wheel.*, drive_motor.*, steer_servo.*
  structure.toml    arm_tube.*, rover_tube.*, frame
  electronics.toml  avionics.* (fixed payload boards)
  sensors.toml      imu.*, baro.*, mag.*, gnss.*, depth_camera.*, rgb_camera.*
  compute.toml      compute.* (inference times)
```

`parts.get("motor", "m2004_1700kv")` returns the dict and tells you the valid names if
you mistype. Every part may carry `source = "datasheet" | "measured" | "guess"`.

### 3.2 `physics/`: pure functions, no MuJoCo

**Done (step 2).** Sizing results, generated scenes and trajectories are bit-identical to
before the move.

Each file is one domain: equations in the docstring, and functions that work on numbers
or numpy arrays.

| file | contents (moved from) |
|---|---|
| `propeller.py` | thrust/power/torque from ct, cp (`design/drone.py`) |
| `dc_motor.py` | KV/R/I0 operating point, gearmotor torque-speed line, lag time constant |
| `battery.py` | OCV(SoC), `Pack.terminal()`, discharge (`design/common.py`) |
| `beam.py` | cantilever deflection, first mode, tube section properties |
| `vehicle.py` | tip-over angle, traction, rolling resistance, steering clearance |
| `aero.py` | drag, rotor drag, the wind field (`wind.py`) |
| `vibration.py` | transmissibility, imbalance force (`sensing/models.py`) |
| `noise.py` | Gauss-Markov, Allan-consistent white noise, delay line (`sensing/models.py`) |

The optimiser, the components and the C++ port all call these. **This is the layer to read
first if you want to understand the engineering.**

### 3.3 `components/`: the building blocks

**Done (step 3)** for geometry and mass; power, runtime specs and uncertainty move into
components in steps 4 and 6.

A **component** is a physical thing you could buy or build. When added to a robot it
attaches at a named mount and adds what it is made of:

```python
class Component:                     # sim/components/base.py
    def attach(self, robot, at: Mount) -> None:
        """Add bodies (structure), geoms, sites, cameras, MassParts, actuators and
        sensors; publish mounts for others; register runtime facts in robot.info."""

    def mount(self, name) -> Mount   # a mount this component publishes (error lists the valid ones)
    def mounts(self) -> list[str]
    def summary(self) -> str

    # step 4: electrical_load(state) -> W, runtime spec, sensor models
    # step 6: uncertain parameters (per component)
```

A body's mass properties come from its parts (`Body.inertial`): `"parts"` sums every
`MassPart` with the parallel-axis theorem (the quad), `"box"` lumps them into a uniform box
(the rover chassis), `"geoms"` lets MuJoCo use geom masses (tubes, wheels).

| family | components (file) |
|---|---|
| structure: create bodies | `QuadFrame` (frame.py), `Chassis` (chassis.py), `RockerBogie` (rocker_bogie.py), `SteeringServo`, `WheelDrive` (drive.py) |
| actuators | `Rotor` (rotor.py), `WheelDrive`, `SteeringServo` (drive.py) |
| energy, payload | `Battery`, `Payload` (energy.py) |
| electronics | `Board`, `CameraModule` (electronics.py) |
| sensors, views | `Imu`, `ViewCamera` (sensors.py), `Mast` (chassis.py) |
| protection | `PropGuard` (frame.py) |

Still to come (step 4): `Gnss`, `Baro`, `Mag` and the sensor models as components,
`Computer`, and **a generic power train**: the battery supplies the sum of every
component's `electrical_load()`, and the voltage is fed back to every actuator. Then
`DronePlant` and `RoverPlant` disappear, and a robot with rotors *and* wheels just works.

### 3.4 `robots/`: assembly

**Done (step 3).** The two grounded vehicles are now built from components. Their compiled
models, sizing results and trajectories are bit-identical to the old generators.

```python
from sim.robots import Robot
from sim.components import QuadFrame, Battery, Board, Rotor, Imu

robot = Robot("scout")
frame = robot.add(QuadFrame("cf_8x6", arm_length=0.11, prop_d=0.127))   # creates the root body
bat = robot.add(Battery("lipo_4s_1500"), at=frame.mount("deck"))
robot.add(Board("fc_stack_h7"), at=frame.mount("center"))
robot.add(Board("pi_zero2w"), at=bat.mount("top"))                       # strapped on the battery
for arm in frame.arms:                                                  # FR, FL, RL, RR + spin
    robot.add(Rotor("m2004_1700kv", "p5x3_bi", arm.spin, thrust_max=7.0, lag_s=0.02), at=arm.tip)
robot.add(Imu(), at=frame.mount("cg"))                                  # at the centre of mass

print(robot.summary())                # parts, bodies, mass, centre of mass
robot.write(builder, pos=[0, 0, 0.3]) # body tree + actuators + sensors into a scene
```

**Mounting is by named mount points only.** Every component that can carry others
publishes its mounts: a name, a body, a position, and the direction an attached part
extends in (up, down or forward), so a battery *sits on* the deck and a board *hangs under*
the plate.
- `QuadFrame`: `deck`, `center`, `bottom`, `nose`, `belly`, `cg`, `origin`, and `tip_FR`…
  (also `frame.arms[i].tip`)
- `Battery`: `top`
- `Chassis`: `center`, `deck`, `mast`, `nose_low`, `origin`
- `RockerBogie`: one per corner, `fl ml rl fr mr rr`
- `SteeringServo`: its corner, for the wheel
- `CameraModule`: `lens`
- `Rotor`: `hub`

`component.mounts()` lists them. A device can't be placed at an arbitrary coordinate. If a
new position is needed, the structure component gets a new named mount, so geometry stays
in one place and follows size changes (`arm.tip` moves when the arm gets longer).

**Templates** (`robots/templates.py`) build a common vehicle from its sizing result. The
sizing decides the geometry (arm length from the prop size, wheel spacing and track from
the wheel and tip-over limits). The template assembles exactly that vehicle from
components, so sizing and simulation can't disagree:

```python
cfg, design = drone.best("drone_forest.toml")   # or drone.evaluate(cfg, db, motor, prop, battery, tube)
robot, meta = quadrotor("uav", cfg, design)     # meta: what the runtime / C++ core reads
cfg, design = rover.best("rover_yard.toml")
robot, meta = rocker_bogie("rover", cfg, design)
# later, when needed: skid_steer(...), hexacopter(...), quadplane(...)
```

### 3.5 `world/`: composable environments

```python
world = (World(seed=3, size=140)
         .terrain(Hills(relief=9, octaves=6))
         .add(Trail(width=2.5, wiggle=16))
         .add(Forest(density=0.3, mix={"conifer": .55, "broadleaf": .3, "birch": .15},
                     keep_clear=["trail"]))
         .add(Rocks(count=90), Logs(count=40), Grass(count=3500))
         .wind(Wind(speed=1.5, gust=0.5))
         .lighting(Sun.random()))
```

Each feature (`Hills`, `Trail`, `Forest`, `Rocks`, `Logs`, `Grass`, `Craters`, `Lander`, and
later `Orchard`, `CropBeds`, `Pond`...) is one class: parameters, placement with its own seeded stream, and geometry through the
existing mesh batching. Features can expose *named regions* (`"trail"`, `"rows"`) that
others avoid or follow, and that missions use as paths.

### 3.6 `control/`: one interface, chainable

```python
class Controller:
    def reset(self, robot_info): ...
    def update(self, t, inputs) -> outputs: ...
```

| controller | inputs | outputs |
|---|---|---|
| `FollowPath(path, speed)` | pose | velocity setpoint |
| `Se3Velocity(gains)` | velocity setpoint, attitude | body wrench |
| `Mixer` (from rotor geometry) | wrench | rotor thrusts |
| `PurePursuit(path)` | pose | speed + curvature |
| `Ackermann(steer geometry)` | speed + curvature | wheel speeds + steer angles |
| `WheelSpeedPI` | wheel speed targets | drive commands |

```python
robot.controller = FollowPath(world.path("trail"), speed=3) >> Se3Velocity() >> Mixer()
```

An RL policy replaces the first stage(s) (`policy >> Se3Velocity() >> Mixer()`), which is
the "setpoint mode" we already have, made general. Every controller states whether it
reads **truth** or **sensor** inputs, so moving from truth to sensors (later, through an
estimator) is an explicit, visible switch. That switch is the sim-to-real boundary.

### 3.7 `scenes/`: short compositions

```python
@scene
def forest_inspection(seed=0):
    world = forest_world(seed)                               # a reusable world recipe
    robot = quadrotor(**design.best("drone_forest.toml"))    # sized from the mission file
    task = FollowPath(world.path("trail"), speed=3, altitude_agl=2)
    return Scene(world, robot, task, uncertainty="drone_forest.toml")
```

**Describing a scene is not running it.** A scene file is only *read once* to build the
scene. It places the terrain features, assembles the robot and writes `scene.xml` plus
`meta.json`; that takes seconds, before any simulation starts. Everything that runs
afterwards (viewer, render, sensor reports, training) is the C++ core loading those files.
Like a compiler: the scene file is the source code, `scene.xml` + `meta.json` the compiled
program, and the C++ core the machine that executes it. Writing scenes in Python therefore
costs no run-time speed. The open question "Python or a data format" (section 9) is only
about which notation is more convenient for describing scenes.

```
 describe (once)             build (once, seconds)        run (millions of steps, C++)
 scenes/<name>.py   ──────►  scene.xml + meta.json  ────►  physics, control, battery,
                                                           sensors -> view / render / train
```

Only the two grounded scenes are kept: `forest_grounded` (renamed `forest_inspection`)
and `mars_yard`. Each should shrink to roughly 10–30 lines.

### 3.8 `design/`: sizing over templates

The mission TOML stays as it is (requirements, design space, objective, sensors,
installation, uncertainty). The optimiser loops over part combinations, builds the
template, evaluates requirements with `physics/` functions, and ranks the results. Because
the template is the same function the scene uses, sizing and simulation can't disagree.

### 3.9 `runtime/` and `core/`

- **`runtime/` (Python reference).** Loads a scene and steps it:
  `controllers → actuators → power system → physics step → sensors`. The step loop iterates
  over components, with no per-vehicle code.
- **`core/` (C++).** Mirrors each *component type* as a C++ class. The reset config becomes
  a list `[{"type": "Rotor", ...}, {"type": "Imu", ...}]`, and the env iterates over it.
  - **A new robot needs no C++.**
  - **A new component type** needs a Python reference class, a C++ class and an
    equivalence test, each in its own file.

### 3.10 `analysis/`

Unchanged in spirit: Allan deviation, vibration lines, sensor report, robustness and
sensitivity. Works the same on simulated and real logs.

## 4. Proposed file layout

```
sim/                      (package name kept)
  parts/        *.toml, __init__.py (get, list)
  physics/      propeller.py dc_motor.py battery.py beam.py vehicle.py aero.py vibration.py noise.py
  components/   base.py frame.py rotor.py wheel_drive.py servo.py battery.py
                imu.py gnss.py baro.py mag.py camera.py computer.py rocker_bogie.py chassis.py
  robots/       robot.py (assembly), templates.py
  world/        world.py terrain.py vegetation.py features/*.py wind.py lighting.py mesh.py
  control/      base.py path.py se3.py mixer.py pure_pursuit.py ackermann.py
  scenes/       one file per scene + recipes.py (reusable worlds)
  design/       optimise.py missions/*.toml
  runtime/      sim.py power.py sensors.py uncertainty.py
  core/         python front end of the C++ core;  cpp/ C++ sources
  analysis/     allan.py vibration.py report.py robustness.py
  cli.py
```

## 5. Where today's code goes

| today | moves to |
|---|---|
| `design/common.py` (chemistry, Pack) | `physics/battery.py`, chemistry data in `parts/energy.toml` |
| `design/drone.py` physics / sizing / `to_mjcf` | `physics/propeller.py`, `physics/dc_motor.py`, `physics/beam.py` / `design/optimise.py` / `components/frame.py`, `components/rotor.py`, `robots/templates.py` |
| `design/rover.py` | `physics/vehicle.py`, `components/rocker_bogie.py`, `components/wheel_drive.py`, `robots/templates.py` |
| `design/plant.py` | `runtime/power.py` (generic) + `electrical_load()` in each component |
| `sensing/models.py` | `physics/noise.py`, `physics/vibration.py` + one component file per sensor |
| `sensing/suite.py` | `runtime/sensors.py` (iterates sensor components) |
| `robots.py` (idealised demo robots) | **removed** with the demo scenes |
| `flora.py`, `terrain.py`, `mesh.py` | `world/` |
| `scenes.py` | `scenes/forest_inspection.py`, `scenes/mars_yard.py` + `world/features/*.py`; demo scenes **removed** |
| `custom/pond_survey.py` | **removed**; replaced by the step-8 tutorial written with the new API |
| `control.py` | `control/*.py` (`RoverSerpentine`, `HarvestSequencer`, `TurbineSpin` removed with their scenes) |
| `wind.py` | `physics/aero.py` (field) + `world/wind.py` (placement/config) |
| `uncertainty.py` | `runtime/uncertainty.py`; keys become per component (section 6) |
| `core.py` + `cpp/core/` | `core/` (component-generic) |

## 6. Uncertainty becomes per component

Each component declares what can be randomised. The mission file addresses components
by name, with wildcards:

```toml
[uncertainty]
"robot.mass_frac"        = 0.04
"rotor*.thrust_coef"     = 0.08    # all rotors share one draw...
"rotor*.gain"            = 0.03    # ...or per-rotor mismatch (declared "per instance")
"imu.noise"              = [0.9, 1.5]
"gnss.foliage"           = [0.3, 0.7]
"wind.speed"             = [0, 4]
```

A new component brings its uncertain parameters along. No central list to edit.

## 7. Migration plan (each step = one branch you review)

| step | content | proof nothing broke |
|---|---|---|
| 1 ✓ | this document | your review |
| 2 ✓ | remove the demo scenes, idealised robots and their controllers; then `parts/` + `physics/` | grounded-scene tests pass; unit tests per formula; sizing results identical to today |
| 3 ✓ | `components/` + `robots/` + templates; quad and rover rebuilt from components | compiled models, sizing and trajectories bit-identical; existing tests pass |
| 4 | generic `runtime/power.py` + `runtime/sensors.py` | same battery and sensor results as `DronePlant` / `RoverPlant` / `SensorSuite` |
| 5 | `world/` features + the two scenes as compositions | same scenes, rendered side by side; tests pass |
| 6 | `control/` interface + chains | same trajectories as today's controllers |
| 7 | C++ core made component-generic | equivalence tests against the Python reference (as now) |
| 8 | tutorial: "a new robot and scene in 30 lines" | you write one |

Old modules keep working until their replacement is merged, then they're deleted in the
same step. No step changes more than one layer.

## 8. How to read the codebase (after migration)

1. `parts/*.toml`: what real parts look like as numbers.
2. `physics/`: the engineering equations, one domain per file.
3. `components/rotor.py`, `components/imu.py`: how a part becomes simulation.
4. `robots/templates.py`: how parts become a robot.
5. `world/features/forest.py`: how an environment is generated.
6. `control/se3.py`: how a drone is flown.
7. `runtime/sim.py`: how one step runs.
8. `scenes/forest_inspection.py`: how it all composes.

## 9. Decisions

| question | decision |
|---|---|
| package name | **keep `sim/`** (`import sim.components as sc`) |
| mounting style | **named mount points only** (section 3.4) |
| which scenes | **only the grounded ones**: `forest_grounded` → `forest_inspection`, and `mars_yard`. The demo scenes (`orchard_uav`, `crop_rover`, `forest_trail`, `greenhouse`, `wind_turbine`, custom `pond_survey`), the idealised robots in `robots.py` and their controllers are removed at the start of step 2 (they stay in git history) |
| scenes as code or data | open. **Default: Python functions only**; a data format can be added later if scenes turn out to need no logic |
| order after step 3 | open. **Default: as in section 7** (power/sensors → world → control → C++) |

The analytic wind field keeps its tower/trunk and turbine-wake terms in `physics/aero.py`;
they're generic physics, even though no remaining scene uses the turbine.
