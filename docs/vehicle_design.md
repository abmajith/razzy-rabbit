# Grounded vehicle design: sizing optimiser + generator

"Grounded" means every number in the simulated vehicle comes from real component data and
physics, not hand-picked values. There are two vehicles:

| vehicle | mission file | scene that uses it |
|---|---|---|
| forest-inspection quadrotor (optionally caged) | `sim/design/configs/drone_forest.toml` | `forest_grounded` |
| six-wheel rocker-bogie rover (Mars-rover-style suspension, 4-wheel steering) | `sim/design/configs/rover_yard.toml` | `mars_yard` |

Both use the shared part database in `sim/parts/` (one TOML file per domain). The
equations are in `sim/physics/`.

## Workflow

```bash
# 1. edit the mission (requirements) and/or add parts to the database
$EDITOR sim/design/configs/drone_forest.toml

# 2. size: searches every compatible part combination, prints ranked feasible designs
uv run --extra sim python -m sim design drone
uv run --extra sim python -m sim design rover --top 15
uv run --extra sim python -m sim design drone --config my_mission.toml   # your own file

# 3. see the winner fly / drive (--rebuild regenerates with the new design)
uv run --extra sim python -m sim view forest_grounded --rebuild
uv run --extra sim python -m sim view mars_yard --rebuild
uv run --extra sim python -m sim render mars_yard --seconds 60 --rebuild
```

While the viewer runs it prints battery state every 5 s, e.g.
`battery  97.3%  24.85 V  138.2 W`.

If no design satisfies the mission, `design` prints the closest candidates and **which
constraints they violate**. That tells you whether to relax a requirement or add parts.

## What the files mean

### `sim/parts/*.toml`: the part database

Each part is a TOML table, for example `[motor.m2207_1750kv]`. To add a real part, copy
its datasheet values into a new table in the right file, and the optimiser will consider it
automatically. In Python: `from sim import parts; parts.get("motor", "m2207_1750kv")`.

| file | sections |
|---|---|
| `propulsion.toml` | `motor.*`, `prop.*` |
| `energy.toml` | `battery.*`, `chemistry.*` (open-circuit voltage curves) |
| `structure.toml` | `arm_tube.*`, `rover_tube.*`, `frame` |
| `drive.toml` | `wheel.*`, `drive_motor.*`, `steer_servo.*` |
| `electronics.toml` | `avionics.*` |
| `sensors.toml` | `imu.*`, `baro.*`, `mag.*`, `gnss.*`, `depth_camera.*`, `rgb_camera.*` |
| `compute.toml` | `compute.*` |

The key fields per section:

| section | key fields |
|---|---|
| `motor.*` | `kv`, `r_ohm`, `i0`, `i_max`, `mass_g`, `stator_mm`, `cells=[min,max]`, `prop_in=[min,max]` |
| `prop.*` | `diameter_in`, `ct`, `cp` (static thrust/power coefficients), `mass_g` |
| `battery.*` | `chemistry` (lipo/liion/lifepo4), `cells`, `capacity_mah`, `mass_g`, `c_max`, `r_cell`, `size_mm` |
| `arm_tube.*` | `od_mm`, `id_mm`, `e_gpa`, `density` |
| `frame` | plate mass, motor mount mass, wiring fraction, cage rod g/m, ESC efficiency/duty |
| `avionics.*` | fixed payload items: `mass_g`, `power_w`, `size_mm`, `mount` (center/top/front/bottom) |
| `wheel.*` | `diameter_mm`, `width_mm`, `mass_g` |
| `drive_motor.*` | gearmotor at output shaft: `voltage`, `stall_torque`, `no_load_rpm`, `stall_current`, `no_load_current`, `mass_g` |
| `rover_tube.*` | square aluminium tube: `side_mm`, `wall_mm`, `density` |
| `steer_servo.*` | `mass_g`, `stall_torque`, `power_w` |

> The shipped values are representative of common hobby/industrial parts, rounded from
> typical datasheets. They are not vendor-certified. Replace them with your actual parts'
> datasheets before trusting absolute numbers.

### Mission files: requirements, design space, objective

- `[mission]` lists **hard constraints**: span, thrust-to-weight, endurance (drone); slope,
  obstacle height, range, width, clearance, tip-over margin (rover). Each line is commented.
- `[payload]` (drone) and `[body]` (rover) are fixed items carried in every design.
- `[design_space]` is `"all"` or a list of part names per category, to restrict the search.
- `[structure]` holds the stiffness and strength limits (arm deflection and vibration mode,
  rocker stress).
- `[objective]` ranks the feasible designs: `endurance | twr | min_mass` (drone),
  `range | min_mass` (rover).

## The models

### Drone (`sim/design/drone.py`, equations in `sim/physics/propeller.py`, `dc_motor.py`, `battery.py`, `beam.py`)

For one rotor producing thrust *T*:

```
n   = sqrt(T / (ct ρ D⁴))              prop speed [rev/s]
P_m = cp ρ n³ D⁵,   Q = P_m / (2πn)    shaft power and torque
I   = Q / Kt + I0,  Kt = 60/(2π·KV)    motor current
V   = ω·Kt + I·R                       required motor voltage
P_bus = 4·V·I / η_esc + P_avionics     drawn from the battery
```

- **Battery:** terminal voltage = cells × OCV(state of charge) − I·R_pack, with OCV curves
  per chemistry in `sim/parts/energy.toml`.
- **Maximum thrust:** the highest thrust at which the motor still has voltage headroom
  (duty ≤ 95 % of bus voltage), stays under its current limit, and the pack stays under its
  C-rating. It's checked at mid charge and **at end of discharge**, which is where
  thrust-to-weight is required.
- **Geometry:** arm length is set by the prop clearance; span is arm + prop (+ cage).
- **Structure:** each arm is a carbon-tube cantilever with the motor at its tip. The model
  checks tip deflection at max thrust and the first bending frequency.
- **Motor lag:** the linearised speed-loop time constant `τ = J / (2·kq·ω + Kt²/R)`, from
  rotor + prop inertia and the prop drag constant.

### Rover (`sim/design/rover.py`, equations in `sim/physics/vehicle.py`, `dc_motor.py`, `beam.py`)

- **Geometry scales with wheel radius:** wheel spacing 2r + 4 cm, rocker pivot height
  ≥ 1.9 r, and the bogie pivot at 1.55 r.
- **Track width** is the larger of two limits:
  - lateral tip-over: `atan((track/2)/h_cg) ≥ max_slope + margin`
  - steering clearance: corner wheels swing inward by `r·sin(40°)` at full lock

  The simulator found that second limit. An earlier version without it jammed its tyres
  against the body in tight turns.
- **Drive** uses the brushed-DC gearmotor line `τ = τ_s(u − ω/ω₀)`, scaled by pack voltage /
  motor voltage. Requirements:
  - slope torque ≤ 50 % of stall (continuous)
  - full traction μ·N·r ≤ 80 % of stall (obstacle climbing)
  - top speed ≥ cruise speed
- **Range** is usable energy ÷ (drive power at cruise on flat ground + avionics + servos),
  multiplied by speed.
- **Structure:** the rocker's front leg is a square-tube cantilever under a 2 g wheel load.

### Generator: what ends up in MuJoCo

- **Mass properties:** each component is placed physically (battery on top, camera on the
  nose, motors at arm tips). Total mass, centre of mass and the full inertia tensor are
  summed from the parts. Simulated mass matches the sizing model (exact for the drone,
  within 1 % for the rover).
- **Drone actuators:** a thrust command filtered through the first-order motor lag, with
  yaw torque from `cp·D / (2π·ct)`.
- **Rover actuators:** the DC-motor torque-speed line as an affine MuJoCo actuator, corner
  steering servos, and the differential as an equality constraint (left rocker = −right
  rocker).
- **Power train** (`sim/runtime/power.py`, generic), run every physics step:
  - each component declares its electrical load: rotors (power from the actual thrust),
    gearmotors (from torque), boards and servos (constant draws)
  - the battery supplies their sum: it integrates state of charge and sags the voltage
  - the sag is fed back as lower maximum thrust or torque
  - rotor drag for the drone is applied by `sim/runtime/aero.py`

  Battery state is exposed as `sim.power.soc`, `.v`, `.i` and `.power`.

## Assumptions and limits (be aware)

- Prop coefficients are static (hover). Forward-flight effects on the prop (advance ratio)
  are not modelled. Body drag and rotor drag cover them approximately.
- Gearmotors are linear brushed-DC. BLDC drives with current limiting behave differently
  near stall.
- Terrain is rigid. There's no soil deformation or wheel sinkage, so `terrain_crr` stands in
  for soft-soil losses. For real terramechanics, Project Chrono is the next step.
- Structure checks are first-order beam formulas. For the final frame, run the real part
  through a proper FE analysis.

## Extending

- **New part:** add a table to the right file in `sim/parts/`. Nothing else is needed.
- **New requirement:** add a key to the mission file, compute the quantity in `evaluate()`,
  and append a message to `violations` when it's not met.
- **New vehicle type:** follow the same pattern: `evaluate` + `optimize` + `best` + `report` +
  `to_mjcf`, a plant class, and a scene that calls `best(config)` and `to_mjcf(...)`. See how
  `mars_yard` in `sim/scenes.py` does it (world features + `rover_design.to_mjcf`).

## Sensors

Sensors, their mounting, the site's magnetic field and the policy computer are chosen in
the same mission file (`[sensors]`, `[installation]`, `[environment]`, `[compute]`). See
[sensor_models.md](sensor_models.md).

## Uncertainty

How well each number is known, and per-episode vehicle variants drawn from it:
`[uncertainty]` in the same mission file. See [uncertainty.md](uncertainty.md).
