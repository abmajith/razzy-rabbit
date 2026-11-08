# sim/ — procedural outdoor robotics environments

Five reproducible MuJoCo scenes for field robotics (no humanoids), each with a reference
controller running out of the box. Python generates the scenes. The output is plain MJCF plus
binary/OBJ/PNG assets, so the same folder loads unchanged from C/C++ (`mj_loadXML`).

| scene | robot | what's in it |
|---|---|---|
| `orchard_uav` | 1.2 kg X-quadrotor | apple orchard on rolling hills, herbicide strips, mown alleys, fruit; lawnmower flight down the alleys at canopy height |
| `crop_rover` | high-clearance gantry rover (skid steer) | raised lettuce beds with real furrows in the heightfield, a growth gradient, missing plants, ~1600 weeds, farm shed; serpentine bed-following with headland U-turns |
| `forest_trail` | quadrotor | steep mixed forest (conifer / broadleaf / birch), rocks, fallen logs, undergrowth; flies a winding hiking trail at 2 m above ground |
| `greenhouse` | pipe-rail trolley + scissor lift + 4-DoF arm + gripper | Venlo glasshouse, high-wire tomatoes that ripen from the bottom up, heating-pipe rails, grow lights; IK pick-and-place of ripe tomatoes into a crate |
| `wind_turbine` | quadrotor | ridge-top turbine with a spinning rotor, hedgerows, pasture; helical tower inspection, then a pass in front of the rotor, all inside a potential-flow wind field |
| `forest_grounded` | sized quadrotor | `forest_trail` flown by the quad chosen by `python -m sim design drone`; see [docs/vehicle_design.md](../docs/vehicle_design.md) |
| `mars_yard` | sized rocker-bogie rover | rocky yard with craters and a lander; rover chosen by `python -m sim design rover` |
| `pond_survey` | quadrotor | example developer scene in `sim/custom/`, with its own orbit controller; see [docs/custom_scenes.md](../docs/custom_scenes.md) |

Every scene is a pure function of `(scene, seed)`: terrain, plant placement, time of day,
sky, haze and colour jitter are all seeded, so the same seed gives byte-identical assets.
That covers both reproducibility and cheap domain randomisation.

## Usage

```bash
uv sync --extra sim                                       # installs mujoco, imageio
uv run --extra sim python -m sim list
uv run --extra sim python -m sim view orchard_uav          # interactive, controller running
uv run --extra sim python -m sim view wind_turbine --wind  # draws wind arrows around the UAV
uv run --extra sim python -m sim view crop_rover --seed 7 --rebuild
uv run --extra sim python -m sim render forest_trail --seconds 30   # headless MP4 + PNG
uv run --extra sim python -m sim build all --seed 3
uv run --extra sim python -m sim sensors forest_grounded  # sensor error report (designed vehicles)
```

The viewer's Camera menu lists the onboard cameras (`*_fpv`, `*_nadir`, `*_bedcam`,
`*_wristcam`) and an `overview`. `render` writes `sim/generated/<scene>-s<seed>/<scene>-s<seed>.mp4`
with the chase view, plus insets of the onboard RGB and depth (what a vision policy would see).
For designed vehicles the insets go through the camera models (noise, motion blur, rolling
shutter, stereo holes, latency); `--ideal` shows the perfect renders instead.

**GPU selection** (`sim/gpu.py`): on hybrid machines Mesa defaults to the GPU driving the
display (here the AMD iGPU). `view` and `render` automatically prefer a discrete card
(NVIDIA > Intel > AMD, skipping the boot-VGA device): `DRI_PRIME=pci-…` for the window,
`MUJOCO_EGL_DEVICE_ID` for headless EGL (probed once, cached in `sim/generated/.gpu.json`).
Override with `RAZZY_GPU=igpu` or `RAZZY_GPU=<egl id>`. The viewer uses 4k shadows/4x MSAA,
and `--hq` gives full quality.

## Layout

```
sim/
  terrain.py   fBm heightfields (MuJoCo binary .bin) + matching ground textures
  mesh.py      low-poly primitives merged into one OBJ per material
  flora.py     trees, conifers, birches, lettuce, weeds, grass, rocks, logs
  robots.py    quadrotor / field rover / greenhouse harvester MJCF
  builder.py   assembles scene.xml + meta.json, sky, sun, randomised lighting
  scenes.py    the five scene recipes
  wind.py      analytic wind field (log-law, doublets, turbulence, terrain, Jensen wake)
  control.py   SE(3) quadrotor control, pure-pursuit skid steer, DLS-IK harvesting
  runtime.py   Sim wrapper: load, step with controller + wind drag, draw wind arrows
  design/      vehicle sizing: configs/*.toml, drone.py, rover.py (optimiser + generator), plant.py
  sensing/     sensor/latency/compute models, scene coupling (GNSS sky view), Allan/FFT analysis
  uncertainty.py  [uncertainty] ranges -> per-episode vehicle variants (Sim.reset(variant))
  core.py      front end of the C++ core: CoreBatch (training), CoreSim (viewer), make_sim
  gpu.py       picks the discrete GPU for viewer (DRI_PRIME) and headless EGL
  custom/      developer scenes + controllers, auto-loaded (pond_survey.py example)
  cpp/core/    razzy_core: C++ envs (physics, controllers, plants, sensors), nanobind module
  cpp/         C++ batched rollout benchmark against the same scene.xml
  generated/   build output (git-ignored)
```

### Why the vegetation is merged meshes

A crop field has about 10k plants. As separate geoms they would exceed the renderer's
geom budget and clutter collision broad-phase. Instead, all purely visual vegetation of
one material becomes a single mesh geom (`contype=0`), so `crop_rover` has 50 geoms and
1.7M triangles. Things a robot can actually hit (trunks, crowns, rocks, logs, tower)
get cheap invisible primitive colliders in geom group 3. Toggle group 3 in the viewer to
see them.

### Wind field (potential vector field)

`wind.py` is closed form and vectorised: log-law boundary layer over the terrain, plus
complex-potential doublets for flow around cylinders, plus a divergence-free
stream-function turbulence field advected with the mean flow, plus terrain-following
vertical flow, plus a Jensen wake behind the turbine rotor. The UAV feels it as quadratic
drag through `xfrc_applied`. It's cheap enough to evaluate every physics step, and easy to
port to C or a GPU kernel.

## C++ batched rollouts

```bash
cmake -S sim/cpp -B sim/cpp/build && cmake --build sim/cpp/build -j
sim/cpp/build/batch_rollout sim/generated/crop_rover-s0/scene.xml 128 32 1000
```

This uses one shared `mjModel`, N `mjData`, and a thread pool with a placeholder random
policy. Measured on a Ryzen 16C/32T, physics only (no rendering, no controller):

| scene | 1 thread | 32 threads |
|---|---|---|
| orchard_uav | 107k steps/s | 1.6M steps/s |
| crop_rover (hfield + 4 wheel contacts) | 56k steps/s | 1.0M steps/s |
| greenhouse (8-DoF arm, no contacts) | 343k steps/s | 5.2M steps/s |

## Where this can go

- Port the controllers and wind into the C++ env server and expose a Gym-like
  `reset/step` over shared memory, so Python training only does inference and updates.
- Camera observations will be the bottleneck, not physics. Batch rendering (MuJoCo's
  batch renderer or a Vulkan rasteriser) at low resolution, or use privileged state for
  a teacher and distil to vision later.
- Real-video sim2real: replace the ground texture or skybox with frames from drone or rover
  footage, or composite rendered robot/plant masks over real video for perception
  training.
- Evolutionary co-design: the robot generators in `robots.py` are parametric
  (arm length, track width, wheel radius, rotor arm), which makes them a ready search space
  for CMA-ES / evolution strategies around RL inner loops, and a link back to the FEA side
  of razzy-rabbit.
