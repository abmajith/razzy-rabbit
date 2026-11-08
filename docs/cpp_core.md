# C++ simulation core (`razzy_core`)

Two implementations of the same simulator, with different jobs:

| | Python reference | C++ core |
|---|---|---|
| where | `sim/runtime/` (sim, power, aero, sensors), `sim/control/`, `sim/sensing/`, `sim/physics/` | `sim/cpp/core/csrc/*.cpp` |
| job | **the specification**: readable, easy to change, where new models are written first | **what runs**: viewer, render, sensor reports, robustness sweeps, training |
| speed (forest drone, with sensors) | ~1.6k steps/s | ~21k steps/s per env, ~330k steps/s on 16 threads |

`view`, `render`, `sensors` and `randomize` use the C++ core by default. Pass
`--engine python` for the reference. A scene whose controller exists only in Python
falls back to the reference automatically, and the CLI says so.

## Architecture

```
 Python (once per reset)                       C++ (every step, all envs in parallel)
 ───────────────────────                       ───────────────────────────────────────
 meta.json + [uncertainty] draw  ─ config ─►  Env: own physics-model copy + mjData
 (sim/core.py CoreBatch._config)   dict        controller chain → body drag → power train
                                               → rotor drag → mj_step → sensors (+ latency)
 numpy views  ◄──────── arrays ─────────────  time, qpos, qvel, ctrl, err, plant,
                                               imu, baro, mag, gnss (latest delivered)
```

- **Physics model:** `scene.xml` compiled without visual-only geoms and textures, cached
  as `scene_physics.mjb`. It has the same bodies, joints, actuators, sensors, masses and
  contacts, so the physics is bit-identical. It is about 2.5 MB instead of 143 MB for the
  forest, which makes a per-environment copy affordable. Copies are needed because the
  uncertainty draw and the battery sag modify the model.
- **Ray scene:** the full visual model is loaded once, read-only and shared by all
  threads, for GNSS sky-view ray casts. Terrain occlusion uses a march over the height
  grid instead of MuJoCo's heightfield ray test, which was 7× more expensive than every
  vegetation mesh combined.
- **Python front end** (`sim/core.py`):
  - **`CoreBatch`:** N environments, for training and batch evaluation.
  - **`CoreSim`:** one environment plus a *mirror* `MjData` of the full visual scene,
    synced every step, so MuJoCo's viewer and renderer work unchanged. The camera noise
    models run in C++ on the rendered images.
  - **`make_sim`:** picks `CoreSim` or falls back to the Python reference.

## Using it for training

```python
from sim.core import CoreBatch

env = CoreBatch("forest_grounded", n_envs=256, seed=0, threads=16, mode="setpoint")
env.reset(variants=list(range(256)))      # 256 different plausible drones ([uncertainty])
for it in range(10_000):
    obs = {"imu": env.imu, "gnss": env.gnss, "baro": env.baro, "battery": env.plant}  # views, no copies
    actions = policy(obs)                 # (256, 4): vx, vy, vz, yaw_rate
    env.act(actions)                      # delayed by the policy-compute model (25-45 ms on the OAK-D)
    env.step(33)                          # 66 ms of physics, sensors, plant: GIL released, 16 threads
    done = ...                            # your task logic
    env.reset(np.flatnonzero(done), new_variant_ids)
```

| mode | `act()` row | low level |
|---|---|---|
| `"mission"` | (ignored) | built-in reference controller flies the mission |
| `"setpoint"` | drone `[vx, vy, vz, yaw_rate]`, rover `[speed, curvature]` | C++ SE(3) / Ackermann controller, as a real autopilot would |
| `"ctrl"` | raw actuator commands | none |

**Output arrays.** All are numpy *views* into C++ buffers, refreshed by every `step()`;
`.copy()` them to keep values. Each sensor array holds the latest measurement the flight
computer has *received*, so latency is already applied:

| array | contents |
|---|---|
| `time` | simulation time per env |
| `qpos`, `qvel`, `ctrl` | true state and actuator commands (for rewards, not observations) |
| `err` | tracking error of the built-in mission |
| `tilt_deg` | body tilt |
| `plant` | `[soc, V, A, W, Wh]` |
| `imu` | `[t_capture, gyro xyz, accel xyz]` |
| `baro` | `[t, h]` |
| `mag` | `[t, b xyz]` |
| `gnss` | `[t, fix, n_sats, hdop, blocked, x y z, vx vy vz, ·]` |

`imu_drain(i)` returns every IMU sample delivered since the last drain (for an EKF).

**Determinism:**
- results are identical for any thread count (tested)
- noise uses a counter-based Philox RNG keyed by (seed, sensor stream), and the same code
  is portable to SYCL or Vulkan kernels later
- `reset(i, variant)` draws the same vehicle as the Python reference

## How the two engines are kept in sync (`tests/sim/test_core.py`)

| part | check |
|---|---|
| physics, controllers, plants, wind | same operation order as the reference: trajectories, state of charge and power agree to ~1e-15 (drone), and to ~1e-6 for the randomised rover, where contacts amplify rounding |
| GNSS sky view | identical vegetation ray casts. Terrain is a height-grid march instead of MuJoCo's hfield test, so grazing rays over ridges can differ (< 6 % of rays, mean within 0.01) |
| sensor noise | different RNG, so checked statistically: datasheet noise floors, vibration level, canopy blocking, open-sky accuracy, latency |
| threading | 1 vs 8 threads bit-identical |

## Changing or adding a model

1. **Write or change it in Python first** (the reference), with a test.
2. **Port it** to the matching `.cpp` file, keeping the same operation order. Configuration
   travels in the reset dict built by `CoreBatch._config` in `sim/core.py`.
3. **Extend `tests/sim/test_core.py`:** exact agreement for deterministic parts, statistics
   for noise.

Rebuilding:

```bash
uv sync --extra sim                      # rebuilds razzy-core automatically when csrc/ changes
# faster edit-compile loop while working on the C++:
cmake -S sim/cpp/core -B /tmp/rz -DPython_EXECUTABLE=$PWD/.venv/bin/python && make -C /tmp/rz -j
```

The extension links the `libmujoco` shipped in the `mujoco` wheel. The project pins
`mujoco==4.0.0` so the two always match, and `import razzy_core` refuses to load against a
different MuJoCo version, with the command to rebuild. `-DRAZZY_NATIVE=ON` adds
`-march=native`. It's off by default so wheels stay portable.

## What the C++ core builds from the scene (generic since step 7)

The C++ core has no drone or rover code paths any more. It builds each environment from
the same descriptions the Python reference uses:

| description (scene meta) | C++ (`csrc/`) |
|---|---|
| controller chain `meta["control"]` | stages in `control.cpp`: `FollowPath`, `VelocityCommand`, `Se3Controller`, `Mixer`, `PurePursuit`, `DriveCommand`, `Ackermann`, `WheelSpeedPI`, with any parameters, passing a `Signal` down the chain |
| power loads `robot["power"]` | `PowerSystem` in `physics.cpp`: a battery + a list of `rotor` / `gearmotor` loads + constant groups. A robot with rotors *and* wheels works |
| `drag_CdA`, `rotor_drag` | `BodyDrag`, `RotorDrag` effects |
| sensor entries `robot["sensors"]` | `Sensors` in `sensors.cpp` (one per type) |
| `[uncertainty]` draw | `Env::apply_model`, with the actuator lists it applies to |

- **`"setpoint"` mode** replaces the chain's top stage with a command stage (`VelocityCommand`
  for a path-following drone, `DriveCommand` for a rover); `act()` feeds it.
- **`"ctrl"` mode** runs no chain and passes raw actuator commands.
- **A new robot or chain made of existing component and stage types needs no C++.**
- **A new *type*** (a stage, a load, a sensor) needs the Python reference class, its C++
  port in the matching file, a parser entry in `bindings.cpp`, a line in `CoreBatch._config`,
  and an equivalence test.

## Not in C++ yet

- **Control stages** that exist only in Python, or `chain.using("sensors")`: that scene runs
  on the Python engine, and the CLI says why.
- **Several sensors of one type** on a robot (Python has the same limit).
- **Viewer perturbations:** mouse forces in the viewer act on the mirror, not on the C++
  physics.
- **Rendering:** stays on MuJoCo's renderer (GPU). Batched rendering for vision policies
  is a later step (MuJoCo's batch renderer, or Vulkan).
