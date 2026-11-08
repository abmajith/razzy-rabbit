"""CLI:  uv run --extra sim python -m sim <command> ...

  list                                  show the scene catalogue
  build  <scene|all> [--seed N]         generate MJCF + assets into sim/generated/
  view   <scene> [--seed N] [--wind]    interactive viewer, reference controller running
  render <scene> [--seed N] [--seconds S] [--wind]
                                        headless MP4: chase view + onboard RGB + depth insets
  design drone|rover [--config FILE] [--top N]
                                        size a vehicle from a mission file (sim/design/configs)
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import numpy as np


def _robot_center(sim):
    return sim.d.xpos[sim.m.body(sim.robot["body"]).id]


def cmd_list(_):
    from .scenes import SCENES
    mod_doc = sys.modules["sim.scenes"].__doc__.splitlines()
    for k, fn in SCENES.items():
        line = next((l.strip() for l in mod_doc if l.strip().startswith(k + " ")), None)
        if line is None:
            first = (fn.__doc__ or "").strip().splitlines()
            line = f"{k:16s} {first[0] if first else ''}"
        print(line)


def cmd_build(a):
    from .scenes import SCENES, build
    names = list(SCENES) if a.scene == "all" else [a.scene]
    for n in names:
        t = time.time()
        p = build(n, a.seed)
        print(f"{n:14s} -> {p}  ({time.time() - t:.1f}s)")


def cmd_design(a):
    from .design import drone, rover
    mod = {"drone": drone, "rover": rover}[a.vehicle]
    config = a.config or {"drone": "drone_forest.toml", "rover": "rover_yard.toml"}[a.vehicle]
    cfg, designs = mod.optimize(config)
    print(mod.report(cfg, designs, top=a.top))
    scene = {"drone": "forest_grounded", "rover": "mars_yard"}[a.vehicle]
    print(f"\nsee it:  uv run --extra sim python -m sim view {scene} --rebuild")


def cmd_view(a):
    from . import gpu
    print(f"[sim] viewer GPU: {gpu.select('glfw')}", flush=True)
    import mujoco
    import mujoco.viewer
    from .runtime import Sim
    sim = Sim(a.scene, a.seed, rebuild=a.rebuild)
    if not a.hq:  # interactive: lighter shadows/MSAA keep the UI responsive
        sim.m.vis.quality.shadowsize = 4096
        sim.m.vis.quality.offsamples = 4
    frame_dt = 1 / 60
    nsub = max(1, int(round(frame_dt / sim.m.opt.timestep)))
    # launch_passive runs the GUI in a *daemon* thread and close() only requests exit.
    # If the interpreter exits while that thread is still tearing down its GL context,
    # GLFW's atexit terminate() races it -> heap corruption, hang, SIGSEGV. So we keep
    # a handle on the viewer thread and join it before returning.
    before = set(threading.enumerate())
    v = mujoco.viewer.launch_passive(sim.m, sim.d)
    viewer_threads = [t for t in threading.enumerate() if t not in before]
    try:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = sim.m.body(sim.robot["body"]).id
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 6.0, -20, 135
        while v.is_running():
            t0 = time.perf_counter()
            with v.lock():          # hold the lock only for physics, never while sleeping
                sim.step(nsub)
                if a.wind:
                    v.user_scn.ngeom = 0
                    sim.draw_wind(v.user_scn, _robot_center(sim))
            if not v.is_running():
                break
            v.sync()
            if sim.plant is not None and int(sim.d.time / 5) != int((sim.d.time - frame_dt) / 5):
                print(f"[sim] t={sim.d.time:6.1f}s  {sim.plant.status()}", flush=True)
            rest = frame_dt - (time.perf_counter() - t0)
            if rest > 0:
                time.sleep(rest)
    except KeyboardInterrupt:
        pass
    finally:
        v.close()
        for t in viewer_threads:
            t.join(timeout=10)
        if any(t.is_alive() for t in viewer_threads):  # driver wedged: don't risk the atexit race
            print("[sim] viewer thread did not stop; forcing exit", flush=True)
            os._exit(0)
    print("[sim] viewer closed", flush=True)


def cmd_render(a):
    os.environ.setdefault("MUJOCO_GL", "egl")
    from . import gpu
    print(f"[sim] render GPU: {gpu.select(os.environ['MUJOCO_GL'])}", flush=True)
    import imageio.v2 as iio
    import mujoco
    from .runtime import Sim
    sim = Sim(a.scene, a.seed, rebuild=a.rebuild)
    W, H = a.width, a.width * 9 // 16
    iw, ih = W // 4, H // 4
    main = mujoco.Renderer(sim.m, H, W)
    small = mujoco.Renderer(sim.m, ih, iw)
    depth = mujoco.Renderer(sim.m, ih, iw)
    depth.enable_depth_rendering()
    cams = sim.robot["cameras"]
    out = a.out or str(sim.folder / f"{a.scene}-s{a.seed}.mp4")
    fps = 30
    nsub = max(1, int(round(1 / fps / sim.m.opt.timestep)))
    writer = iio.get_writer(out, fps=fps, quality=8, macro_block_size=8)
    nframes = int(a.seconds * fps)
    t0 = time.time()
    for f in range(nframes):
        sim.step(nsub)
        main.update_scene(sim.d, a.camera or cams[0])
        if a.wind:
            sim.draw_wind(main.scene, _robot_center(sim))
        img = main.render().copy()
        small.update_scene(sim.d, cams[1])
        rgb = small.render()
        depth.update_scene(sim.d, cams[1])
        dep = depth.render()
        dn = np.clip(np.log1p(dep) / np.log1p(30.0), 0, 1)
        dimg = (np.stack([1 - dn, 0.6 * (1 - dn) + 0.2, dn * 0.9], -1) * 255).astype(np.uint8)
        m = 12
        img[m:m + ih, W - iw - m:W - m] = rgb
        img[2 * m + ih:2 * m + 2 * ih, W - iw - m:W - m] = dimg
        img[m - 2:2 * m + 2 * ih + 2, W - iw - m - 2:W - iw - m] = 255
        writer.append_data(img)
        if f == nframes // 2:
            iio.imwrite(out.replace(".mp4", ".png"), img)
    writer.close()
    print(f"wrote {out}  ({nframes} frames, sim {sim.d.time:.1f}s, wall {time.time() - t0:.1f}s)")
    if sim.plant is not None:
        print(f"[sim] end of run: {sim.plant.status()}, used {sim.plant.energy_wh:.2f} Wh")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m sim", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("build", "view", "render"):
        s = sub.add_parser(name)
        s.add_argument("scene")
        s.add_argument("--seed", type=int, default=0)
        if name != "build":
            s.add_argument("--rebuild", action="store_true", help="regenerate assets first")
            s.add_argument("--wind", action="store_true", help="draw wind-field arrows")
        if name == "view":
            s.add_argument("--hq", action="store_true", help="full 8k shadows + 8x MSAA (slower)")
        if name == "render":
            s.add_argument("--seconds", type=float, default=20)
            s.add_argument("--width", type=int, default=1280)
            s.add_argument("--camera", default=None)
            s.add_argument("--out", default=None)
    s = sub.add_parser("design", help="size a vehicle from its mission file")
    s.add_argument("vehicle", choices=["drone", "rover"])
    s.add_argument("--config", default=None, help="mission TOML (default: drone_forest / rover_yard)")
    s.add_argument("--top", type=int, default=8)
    a = p.parse_args(argv)
    {"list": cmd_list, "build": cmd_build, "view": cmd_view, "render": cmd_render,
     "design": cmd_design}[a.cmd](a)


if __name__ == "__main__":
    main()
