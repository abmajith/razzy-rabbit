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

## Simulation environments (`sim/`)

Procedural, reproducible MuJoCo environments for field robotics (UAVs, agricultural rovers,
greenhouse harvesting), each with a reference controller already running. Scenes are
generated as plain MJCF, so the same files load from Python and from C/C++.

| scene | robot |
|---|---|
| `orchard_uav` | quadrotor flying the alleys of an apple orchard on rolling hills |
| `crop_rover` | high-clearance weeding rover following lettuce beds with furrows and weeds |
| `forest_trail` | quadrotor following a hiking trail through steep mixed forest |
| `greenhouse` | pipe-rail trolley with an arm picking ripe tomatoes into a crate |
| `wind_turbine` | quadrotor inspecting a spinning turbine in a potential-flow wind field |
| `forest_grounded` | forest trail flown by a **quad sized from real parts** (`drone_forest.toml`) |
| `mars_yard` | rocky yard driven by a **sized six-wheel rocker-bogie rover** (`rover_yard.toml`) |
| `pond_survey` | example developer scene (`sim/custom/`): quadrotor orbiting a pond |

### Run
```bash
uv sync --extra sim            # adds mujoco + imageio on top of the base environment
```

### Run
```bash
uv run --extra sim python -m sim list                          # show all scenes
uv run --extra sim python -m sim view greenhouse               # interactive 3D viewer
uv run --extra sim python -m sim view wind_turbine --wind      # + wind-field arrows
uv run --extra sim python -m sim view crop_rover --seed 7 --rebuild   # another random variant
uv run --extra sim python -m sim render forest_trail --seconds 30     # headless MP4 + PNG
uv run --extra sim python -m sim build all --seed 3            # just generate assets
```

- Output goes to `sim/generated/<scene>-s<seed>/` (git-ignored). The same seed always gives
  identical files.
- In the viewer, the **Camera** menu has onboard cameras (`*_fpv`, `*_nadir`, `*_bedcam`,
  `*_wristcam`) and `overview`. Geom group 3 shows the invisible colliders.
- The discrete GPU is selected automatically (here the Intel Arc rather than the AMD iGPU).
  Override with `RAZZY_GPU=igpu`. Add `--hq` to `view` for full shadow quality.

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

### C++ batched rollouts
```bash
cmake -S sim/cpp -B sim/cpp/build && cmake --build sim/cpp/build -j
sim/cpp/build/batch_rollout sim/generated/crop_rover-s0/scene.xml 128 32 1000
```

### Writing your own scene
Drop a file into `sim/custom/` and it's picked up automatically. Start from
[`sim/custom/pond_survey.py`](sim/custom/pond_survey.py) and follow
**[docs/custom_scenes.md](docs/custom_scenes.md)** (builder API, terrain, vegetation, robots,
custom controllers, wind, conventions, checklist). Internals and benchmarks:
[sim/README.md](sim/README.md).
