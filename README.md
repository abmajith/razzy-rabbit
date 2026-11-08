# 🐇 Razzy-Rabbit

Simulation-in-the-loop design of field robots (UAVs, agricultural and environmental
rovers): size a vehicle from real parts, simulate it with grounded actuators, battery,
sensors and latency, randomise what we don't know, and run it fast in a C++ core for
training and analysis.

## Setup
Linux, Python >= 3.11, [uv](https://docs.astral.sh/uv/), `cmake` and a C++17 compiler
(for the C++ core, built automatically):
```bash
sudo apt install cmake g++ libgl1 libegl1
uv sync --extra sim
```

## Scenes

Procedural, reproducible MuJoCo scenes (same seed, byte-identical files) whose vehicles are
**sized from real parts**:

| scene | vehicle |
|---|---|
| `forest_grounded` | quadrotor sized from `drone_forest.toml`, flying a trail through steep mixed forest |
| `mars_yard` | six-wheel rocker-bogie rover sized from `rover_yard.toml`, driving a rocky yard with craters |
| `meadow_scout` | tutorial example: a quad assembled part by part, flying a trail over a meadow (`sim/myscenes/`) |

**Build your own robot and scene: [docs/tutorial.md](docs/tutorial.md)**: parts → components → robot,
world features, a controller chain, and what you get for free (C++ speed, battery, sensors,
randomisation).

```bash
uv run --extra sim python -m sim list                              # scenes
uv run --extra sim python -m sim view forest_grounded              # interactive viewer
uv run --extra sim python -m sim view forest_grounded --wind       # + wind-field arrows
uv run --extra sim python -m sim view mars_yard --seed 7 --rebuild # another random terrain
uv run --extra sim python -m sim render mars_yard --seconds 30     # headless MP4 + PNG
```

- Output goes to `sim/generated/<scene>-s<seed>/` (git-ignored).
- In the viewer, the **Camera** menu has the onboard cameras and `overview`. Geom group 3
  shows the invisible colliders.
- The discrete GPU is selected automatically (here the Intel Arc rather than the AMD iGPU).
  Override with `RAZZY_GPU=igpu`. Add `--hq` to `view` for full shadow quality.

How the code is organised (parts, physics, components, robots, world, control, runtime,
C++ core): **[docs/architecture.md](docs/architecture.md)**.

### Grounded vehicle design
Size a drone or rover from a mission spec and a part database, then simulate the result
with motor lag, real thrust/torque curves and a draining battery:
```bash
uv run --extra sim python -m sim design drone       # edit sim/design/configs/drone_forest.toml
uv run --extra sim python -m sim design rover       # edit sim/design/configs/rover_yard.toml
uv run --extra sim python -m sim view mars_yard --rebuild
```
Details, models and how to add parts: **[docs/vehicle_design.md](docs/vehicle_design.md)**.

### Grounded sensors, latency and compute
The designed vehicles carry real sensor specs (IMU, baro, compass, GNSS, stereo depth,
rolling-shutter RGB, policy compute latency), chosen by part name in the mission file:
```bash
uv run --extra sim python -m sim sensors forest_grounded --seconds 60   # error report vs truth
uv run --extra sim python -m sim render forest_grounded                 # insets = what the drone receives
```
Models, datasheet fields and how to fit them to your real hardware:
**[docs/sensor_models.md](docs/sensor_models.md)**.

### Uncertainty / domain randomisation
`[uncertainty]` in the mission file says how well each number is known; every episode draws
one plausible vehicle (mass, CoM, thrust, motor lag, battery, sensor noise, latency, wind):
```bash
uv run --extra sim python -m sim randomize forest_grounded --n 30   # robustness + sensitivities
uv run --extra sim python -m sim view forest_grounded --variant 7
```
See **[docs/uncertainty.md](docs/uncertainty.md)**.

### C++ core (what actually runs)
Physics, controllers, battery plant, sensors and latency run in a C++ core (`razzy_core`,
built automatically by `uv sync --extra sim`; needs `cmake` and a C++17 compiler). The Python
modules stay as the readable reference it is tested against. `view`/`render`/`sensors`/
`randomize` use the core (`--engine python` for the reference); training uses `CoreBatch`:
```python
from sim.core import CoreBatch
env = CoreBatch("forest_grounded", n_envs=256, threads=16, mode="setpoint")
env.reset(variants=range(256)); env.act(actions); env.step(33); env.imu, env.gnss, env.plant
```
~13x the Python reference per env and ~200x on 16 threads (forest drone, all sensors on).
See **[docs/cpp_core.md](docs/cpp_core.md)**.

### C++ batched-rollout benchmark
Physics only, against the same `scene.xml` (one shared model, N data, thread pool):
```bash
cmake -S sim/cpp -B sim/cpp/build && cmake --build sim/cpp/build -j
sim/cpp/build/batch_rollout sim/generated/mars_yard-s0/scene.xml 128 32 1000
```

Internals: [sim/README.md](sim/README.md).
