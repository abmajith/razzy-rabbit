# Uncertainty model (domain randomisation)

The sizing optimiser produces **one** nominal vehicle. The real one will differ: the scale
reads 4 % heavier, the battery strap moved the centre of mass, one prop is chipped, the
pack is a winter older, and the USB camera adds 10 ms of latency. A policy trained only on
the nominal vehicle can learn to depend on exact values it will never get.

The `[uncertainty]` table in the mission file states **how well each number is known**.
Every episode then draws one plausible vehicle from those ranges.

```bash
uv run --extra sim python -m sim randomize forest_grounded --n 30 --seconds 20   # robustness report
uv run --extra sim python -m sim view forest_grounded --variant 7                # fly drawn vehicle 7
uv run --extra sim python -m sim sensors forest_grounded --variant 7
```

```python
sim = Sim("forest_grounded", seed=0)
for episode in range(1000):
    sim.reset(variant=episode)          # new drawn vehicle, same scene, no reload
    names, theta = sim.param_vector()   # flat, fixed order: what was drawn
    ...
sim.reset(None)                         # nominal vehicle again (== freshly loaded scene)
```

Draws are deterministic per (scene seed, variant), so episode 7 is the same vehicle every
time, on every machine.

## Writing ranges

```toml
[uncertainty]
mass_frac = 0.04            # a      -> factor uniform in [1-a, 1+a]
battery_capacity = [0.85, 1.0]   # [lo, hi] -> factor uniform in [lo, hi]
com_offset_mm = [6, 4]      # disk of radius 6 mm in xy, z uniform in ±4 mm  (a number = ball)
latency_ms_add = [0, 15]    # offset (ms) added to every sensor latency
wind_speed = [0.0, 4.0]     # absolute value
require_twr_end = 1.8       # feasibility limit, not a draw
```

Annotate each line with where the range comes from: *datasheet*, *model* or *guess*. That
comment is the to-do list for the measurement campaign in
[sensor_models.md](sensor_models.md#bringing-your-real-drone-into-the-simulation).
**Narrow a range once you have measured the quantity.** Policies trained on narrow,
measured ranges are less conservative and fly better.

All keys, with their meaning, are listed in `PARAMS` in `sim/uncertainty.py`. An unknown
key, or a rover key used on a drone, is an error that prints the valid keys.

| group | keys |
|---|---|
| rigid body | `mass_frac`, `inertia_frac` (per axis), `com_offset_mm`, `friction` |
| energy | `battery_capacity`, `battery_resistance` |
| drone | `thrust_coef`, `motor_gain` (per rotor), `yaw_coef`, `motor_lag`, `rotor_drag`, `body_drag`, `prop_imbalance`, `imu_mount_hz`, `arm_hz` |
| rover | `drive_torque`, `drive_gain` (per motor), `steer_kp` |
| sensors and compute | `imu_noise`, `imu_bias`, `baro_noise`, `baro_drift`, `mag_current`, `gnss_foliage`, `gnss_multipath`, `latency_ms_add`, `compute_ms_add` |
| environment | `wind_speed`, `wind_heading_deg`, `gust` |
| feasibility | `require_twr_end`, `require_hover_throttle` (drone); `require_slope_torque_frac`, `require_tip_deg` (rover); `max_attempts` |

## Who knows what

- **The physics gets the drawn vehicle:**
  - the MuJoCo model (mass, inertia, centre of mass, actuator gains, lag, yaw torque, friction)
  - the battery and motor plant
  - the sensor models
  - drag and wind
- **The controllers are built against the nominal model.** A real flight controller only
  knows the designed mass and inertia, and its gains have to cope with the difference.
  The same will hold for an RL policy. You can choose to give it `param_vector()` (a
  privileged teacher policy) or not (a deployable student).

## Feasibility

Some draws describe a vehicle that couldn't fly or drive: too heavy for its motors, or a
rover tipping on its design slope. These are checked to first order against the
`require_*` limits, rejected and redrawn. `randomize` reports the rejection rate:
- **A high rate** means the design is marginal for what you don't know. Tighten the
  ranges by measuring, or make the sizing requirements stricter.
- **No feasible draw in `max_attempts`** stops with the reason.

## Reading the robustness report

```
  variant   0  track rms  0.47 m  max  0.85 m  tilt max   13 deg  power  130.8 W  [ok]
  ...
robustness: 30/30 ok; tracking rms median 0.44 m, worst 0.52 m; infeasible draws rejected: 0 (0%)
most influential on tracking error (Spearman rho):
  thrust_coef   +0.61   thrust per command ...
  com_offset_mm_x -0.48 ...
```

- **`LOST`** means the vehicle diverged (more than 5 m from its track, or tilted beyond 75°).
- **The sensitivity list** ranks the uncertainties by how strongly they drive tracking
  error. Those are the quantities worth measuring first, or worth handling in the
  controller (for example, thrust estimation for `thrust_coef`). You need about 30 or more
  variants before trusting the ranking.

## Implementation notes

- `ModelSnapshot` keeps nominal copies of every model field that gets randomised or that
  the plants change while running (voltage sag rescales `ctrlrange` and `forcerange`), so
  variants never stack. `reset(None)` reproduces a freshly loaded scene bit for bit, and a
  test checks this.
- The flat vector from `param_vector()` (names sorted, vectors expanded to `_x _y _z` or
  `_0.._n`) is the per-environment parameter layout the planned C++ core will consume:
  one row per environment.
- Sensor noise streams are reseeded per variant, so variants differ in noise as well as
  in hardware.
