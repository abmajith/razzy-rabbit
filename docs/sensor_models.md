# Grounded sensors, latency and compute

A policy trained in simulation only transfers if it sees the world through the same
imperfect sensors, with the same delays, as the real vehicle. This step gives the designed
vehicles (`forest_grounded`, `mars_yard`) a complete sensor spec, built from datasheet
numbers and the way the parts are mounted.

```
 truth (MuJoCo) ─► sensor model ─► latency line ─► what the flight computer has at time t
                   (noise, bias,     (per-sensor,
                    vibration,        jittered)
                    occlusion...)
```

## The robot spec, layer by layer

| layer | where it lives | status |
|---|---|---|
| mission and requirements | `drone_forest.toml` `[mission]` | sizing (see [vehicle_design.md](vehicle_design.md)) |
| body, propulsion, energy | parts in `components.toml`, chosen by the optimiser | grounded |
| **sensors** | `[sensors]`: part names → `[imu.*]`, `[baro.*]`, `[mag.*]`, `[gnss.*]`, `[depth_camera.*]`, `[rgb_camera.*]` | **this step** |
| **installation** | `[installation]`: soft mount, prop balance, antenna height, wiring | **this step** |
| **site** | `[environment]`: Earth magnetic field | **this step** |
| **compute** | `[compute]`: where the policy runs → `[compute.*]` | **this step** |
| uncertainty ranges (domain randomisation) | `[uncertainty]` | see [uncertainty.md](uncertainty.md) |

The generator (`to_mjcf`) merges these into the scene's `meta.json`, so each generated
scene carries its full sensor spec. `Sim` then builds a `SensorSuite` automatically.

## Using it

```bash
uv run --extra sim python -m sim sensors forest_grounded --seconds 60          # report
uv run --extra sim python -m sim sensors forest_grounded --log run.npz         # raw data + truth
uv run --extra sim python -m sim render forest_grounded --rebuild              # insets = received frames
uv run --extra sim python -m sim render forest_grounded --ideal                # perfect renders, to compare
```

From Python, this is what a controller or RL environment should consume instead of
`d.qpos`:

```python
sim = Sim("forest_grounded", seed=0)
sim.step(500)
t = sim.d.time
obs = sim.sensors.read(t)            # {"imu": (t_capture, (gyro, acc)), "gnss": (t_capture, fix), ...}
imu_batch = sim.sensors.drain_imu(t) # every IMU sample delivered since the last drain (for an EKF)
sim.sensors.camera_due("depth_camera", t)  # True at the camera's frame rate
sim.sensors.capture("depth_camera", t, perfect_depth, gray)  # -> processed frame, queued
sim.sensors.frame("depth_camera", t) # newest frame *received* by now (latency applied)
sim.sensors.compute                  # ComputeLatency: submit(t_obs, action) / current(t)
```

Every sensor has its own random stream derived from the scene seed, so the same
(scene, seed) reproduces every noise sample exactly. Adding a sensor never changes the
noise of the others.

A typical forest run (`sensors forest_grounded --seconds 60`):

```
IMU icm42688p @ 1000 Hz: gyro err std ~0.002-0.006 rad/s, accel err std x/y 0.37 m/s^2 (vibration), z 0.022
  accel x vibration lines: 225 Hz ...   rotors ~230 rev/s, arm mode 153 Hz, IMU mount 90 Hz
GNSS ublox_m10: fix 33% of epochs, sats mean 3.8, sky blocked mean 87%
  horizontal error: median 8 m, 95% 27 m (open-sky CEP50 spec 1.5 m)
BARO bmp390: error mean +0.8 m, std 0.17 m
```

Compare the same report on `mars_yard` (open sky): 28 satellites, about 2.5 m error,
always a fix.

## The models (`sim/sensing/models.py`)

### IMU
`measured = M · (truth + vibration) + bias(t) + white noise`, then clipped to full scale
and quantised to the ADC step.
- **White noise:** the datasheet noise density × √rate. Its Allan deviation at τ = 1 s
  equals the datasheet number, and a test checks this.
- **Bias:** a residual turn-on bias plus in-run bias instability (Gauss-Markov).
- **Scale and misalignment:** per-axis scale error and cross-axis misalignment (residuals
  after calibration).
- **Vibration is added before clipping,** so excessive vibration saturates the IMU just as
  it does on real hardware (the report counts saturated samples).

### Rotor vibration
This is the dominant IMU error on a real multirotor. Each rotor contributes:
- **Imbalance:** a residual imbalance `U` [g·mm] gives a rotating in-plane force `U·(2πn)²`
  at 1× rotor speed.
- **Thrust ripple:** blade-pass ripple (a fraction of thrust) acts vertically at
  blades × rotor speed.

Each force reaches the IMU through three stages: the arm (resonant at the arm's first
bending mode from the structural sizing, with damping ζ ≈ 0.02), the flight-controller soft
mount (an isolator), and the IMU's anti-alias filter. Linear vibration is F/m, and angular
vibration is I⁻¹·(r × ∫F dt).

The consequence: if the optimiser picks a floppy arm whose mode sits near hover rotor
speed, the IMU data shows it. Try `imu_mount_hz = 1e4` (hard-mounted) in
`[installation]` and watch the vibration lines grow.

### Barometer
White noise (datasheet RMS converted to metres), a constant offset, slow drift
(temperature and weather), and static-port error from airspeed (`Δh = −cp·v²/2g`).

### Magnetometer
The Earth field for your site, rotated into the body frame. On top of that:
- residual hard-iron and soft-iron errors
- the field from battery and motor current (`mag_current_ut_per_a` × the plant's actual
  bus current), the classic "compass changes with throttle" error
- white noise

### GNSS: accuracy follows what the antenna can see
- Each satellite in a seeded sky constellation gets a ray cast from the antenna into the
  real scene geometry.
- A trunk, rock or hill blocks the signal completely. Each leaf layer the ray crosses
  loses the signal with probability `foliage_loss`.
- Per epoch, the tracked satellites give **real HDOP and VDOP from their geometry**.
  With fewer than `min_sats`, there is no fix.
- **Error:** a slow atmospheric bias (Gauss-Markov, τ ≈ 60 s, scaled by DOP), plus
  multipath that grows with blocked sky, plus white noise.
- **Calibration:** the open-sky CEP50 from the datasheet calibrates the scale. A test
  checks that the median open-sky error matches the CEP.

### Stereo depth camera
Disparity `d = f·B/z` (with focal length *f* and baseline *B*) gets sub-pixel matching
noise and is quantised to sub-pixel steps. So depth error grows as `z²·σ_d/(f·B)`, as real
stereo does (a test checks this). Pixels are invalid (0):
- outside `[min_m, max_m]`
- where the left-right check fails beside depth edges (occlusion bands)
- optionally, where texture is too weak to match. **This rule is off by default:** our
  procedural textures are smoother than real surfaces, so it would mark most of the ground
  invalid. That would measure the renderer, not the camera. Enable `min_texture` once
  scenes use photo or video textures.

### RGB camera
- **Auto-exposure:** when exposure is maxed out in a dark forest, gain rises and so does
  noise.
- **Motion blur** from rotation during the exposure.
- **Rolling shutter:** each row is read later than the one above, so rows shift with
  rotation. A test checks the skew.
- **Sensor effects:** vignetting, shot and read noise, and 8-bit quantisation.

### Latency and compute
- **Sensor latency:** every sensor has `latency_ms`, either fixed or `[min, max]` jittered
  per sample. Measurements are only visible to `read()` after that delay.
- **Compute latency:** `ComputeLatency` models the policy computer. An action computed
  from observations at time t becomes usable at t + inference time, and is held until the
  next one arrives. RL environments must route actions through it.

## Bringing your real drone into the simulation

The shipped numbers are representative datasheet values. Reducing the sim-to-real gap
means replacing them with measurements from your parts, as installed. The tools in
`sim/sensing/analysis.py` work on real logs as well as simulated ones, so you compare like
with like.

| measure on the real vehicle | how | put it in |
|---|---|---|
| IMU noise density and bias instability | log the IMU **stationary** for 1–3 h, run `allan_deviation` (value at τ = 1 s on the −½ slope = noise density; minimum of the curve ≈ bias instability) | `[imu.*]` |
| vibration lines and levels | log raw IMU in a hover and a throttle sweep (PX4 / ArduPilot both have raw/FFT logging), run `vibration_peaks` | `[installation]` `prop_imbalance_gmm`, `thrust_ripple`, `imu_mount_hz` |
| arm mode | tap test: tap the arm with the motor mounted, record with a phone mic or the IMU, take the FFT peak | check against `arm_freq_hz` from sizing |
| baro noise and drift | stationary log for 30 min; std = noise, slow wander = drift | `[baro.*]` |
| static-port error | fly constant altitude at several speeds (with GNSS/RTK truth), fit Δh vs v² | `baro_static_cp` |
| compass vs current | motors on, props off (or tethered), sweep throttle, log field vs battery current | `mag_current_ut_per_a` |
| GNSS under canopy | leave the receiver at known points: open field, forest edge, inside the forest, 10 min each; log sats, HDOP and error | tune `foliage_loss`, `multipath_m` until `sensors` reproduces it |
| camera intrinsics, readout, exposure | standard checkerboard calibration (OpenCV); readout time from a spinning-fan / LED test | `[rgb_camera.*]`, `[depth_camera.*]` |
| stereo noise | point at a flat wall at 1, 2, 4 and 8 m, compute the std of depth, fit `σ_d` | `subpixel_sigma_px` |
| end-to-end latency | LED blink in view, timestamped; or log the camera frame time vs the IMU time | `latency_ms` |
| inference time | run your network on the real board, histogram the times | `[compute.*] inference_ms` |

When a simulated report matches the real one for the same manoeuvre, that sensor's gap is
closed. Whatever uncertainty remains becomes the domain-randomisation range in
`[uncertainty]` (see [uncertainty.md](uncertainty.md)).

## Limits

- Models are first-order: no temperature dependence of bias, no gyro g-sensitivity, and
  no satellite ephemeris (the constellation is a seeded random sky that rotates slowly).
- Vibration is linear, with lumped transmissibilities rather than a frame FE model. The
  real frame can have more modes. A frame FE analysis could give them later.
- Camera images come from MuJoCo's rasteriser. Lighting, textures and foliage detail are
  the remaining visual gap. That is the job of the real-video step.
- The Python sensor path runs at about 3× real time for the forest drone. Batched RL will
  need the C++ path.
