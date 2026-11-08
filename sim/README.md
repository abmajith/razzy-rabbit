# sim/

Simulation-in-the-loop design of field robots. Python describes and builds scenes; the C++
core (`cpp/core/`, see [docs/cpp_core.md](../docs/cpp_core.md)) runs them. The Python
modules are the readable reference the C++ core is tested against.

The code is being restructured into reusable layers, one step at a time: see
[docs/architecture.md](../docs/architecture.md).

## Layout (current)

```
sim/
  parts/       datasheets only (TOML, one file per domain) + loader          [step 2]
  physics/     the engineering equations, pure functions, one file per domain [step 2]
  components/  robot building blocks: QuadFrame, Rotor, Battery, Board, CameraModule, Imu,
               Chassis, RockerBogie, SteeringServo, WheelDrive, Mast ...           [step 3]
  robots/      Robot (assembly at named mounts) + templates: quadrotor, rocker_bogie [step 3]
  design/      vehicle sizing: configs/*.toml (missions), drone.py, rover.py; plant.py (power train)
  sensing/     sensor models, scene coupling (GNSS sky view), Allan/FFT analysis
  uncertainty.py  [uncertainty] ranges -> per-episode vehicle variants
  terrain.py   fBm heightfields (MuJoCo binary .bin) + matching ground textures
  mesh.py      low-poly primitives merged into one OBJ per material
  flora.py     trees, conifers, birches, grass, rocks, logs
  builder.py   assembles scene.xml + meta.json, sky, sun, lighting, shared robot materials
  scenes.py    the grounded scenes: forest_grounded, mars_yard
  control.py   SE(3) quadrotor control + waypoint mission, Ackermann pure pursuit
  runtime.py   Python reference engine: load a scene, step controller/plant/physics/sensors
  core.py      front end of the C++ core: CoreBatch (training), CoreSim (viewer), make_sim
  gpu.py       picks the discrete GPU for the viewer (DRI_PRIME) and headless EGL
  cpp/core/    razzy_core: C++ envs (physics, controllers, plants, sensors), nanobind module
  cpp/         C++ physics-only batched rollout benchmark
  generated/   build output (git-ignored)
```

## Notes

**Why vegetation is merged meshes.** A forest has thousands of trees and grass tufts. As
separate geoms they would exceed the renderer's geom budget and slow collision detection.
All purely visual vegetation of one material becomes a single mesh geom (`contype=0`).
Things a robot can actually hit (trunks, crowns, rocks, logs) get cheap invisible primitive
colliders in geom group 3. Toggle group 3 in the viewer to see them.

**Wind field.** Closed form and vectorised:
- a log-law boundary layer over the terrain
- complex-potential doublets for flow around cylinders
- a divergence-free stream-function turbulence field advected with the mean flow
- terrain-following vertical flow
- a Jensen wake behind a turbine rotor

Vehicles feel it as drag through `xfrc_applied`.

## Where this can go

- **Rendering:** camera observations will be the training bottleneck, not physics. Options
  are batch rendering at low resolution, or privileged state for a teacher policy that is
  later distilled to vision.
- **Real-video sim-to-real:** frames from drone or rover footage as ground textures or
  skybox, or rendered robot and plant masks composited over real video for perception
  training.
- **Evolutionary co-design:** the vehicle templates (step 3) are parametric, which makes
  them a ready search space for evolution strategies around RL inner loops.
