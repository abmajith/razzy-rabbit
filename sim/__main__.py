"""CLI:  uv run --extra sim python -m sim <command> ...

  list                                  show the scene catalogue
  build  <scene|all> [--seed N]         generate MJCF + assets into sim/generated/
  view   <scene> [--seed N] [--wind]    interactive viewer, reference controller running
  render <scene> [--seed N] [--seconds S] [--wind]
                                        headless MP4: chase view + onboard RGB + depth insets
  design drone|rover [--config FILE] [--top N]
                                        size a vehicle from a mission file (sim/design/configs)
  sensors <scene> [--seconds S] [--log FILE.npz]
                                        run a designed vehicle, report its sensor errors
                                        (IMU noise/vibration, GNSS under canopy, baro, mag)
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
                if sim.sensors is not None:
                    print(f"[sim]            {sim.sensors.status()}", flush=True)
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
    cams = sim.robot["cameras"]
    out = a.out or str(sim.folder / f"{a.scene}-s{a.seed}.mp4")
    fps = 30
    nsub = max(1, int(round(1 / fps / sim.m.opt.timestep)))
    writer = iio.get_writer(out, fps=fps, quality=8, macro_block_size=8)
    nframes = int(a.seconds * fps)
    # onboard insets: through the sensor models (noise, blur, holes, latency) when the
    # vehicle has a sensor spec, else the ideal render
    su = sim.sensors if (sim.sensors is not None and not a.ideal) else None
    ins = {}
    for kind, cam in (("rgb_camera", cams[1]), ("depth_camera", cams[1])):
        spec = su.spec.get(kind) if su is not None else None
        cam = spec["camera"] if spec else cam
        aspect = spec["width"] / spec["height"] if spec else iw / ih
        w = int(round(ih * aspect)) // 2 * 2
        r = mujoco.Renderer(sim.m, ih, w)
        if kind == "depth_camera":
            r.enable_depth_rendering()
        ins[kind] = dict(cam=cam, r=r, w=w, spec=spec,
                         gray=mujoco.Renderer(sim.m, ih, w) if kind == "depth_camera" and spec else None,
                         img=None)

    def onboard(kind):
        o = ins[kind]
        t = sim.d.time
        if su is None or o["spec"] is None:
            o["r"].update_scene(sim.d, o["cam"])
            return o["r"].render()
        if su.camera_due(kind, t):
            o["r"].update_scene(sim.d, o["cam"])
            img = o["r"].render()
            gray = None
            if o["gray"] is not None:
                o["gray"].update_scene(sim.d, o["cam"])
                gray = o["gray"].render().mean(-1)
            su.capture(kind, t, img.copy(), gray)
        f = su.frame(kind, t)            # what the flight computer has *received* by now
        if f is None:
            return np.zeros((ih, o["w"], 3) if kind == "rgb_camera" else (ih, o["w"]), np.uint8)
        return f[1]

    t0 = time.time()
    for f in range(nframes):
        sim.step(nsub)
        main.update_scene(sim.d, a.camera or cams[0])
        if a.wind:
            sim.draw_wind(main.scene, _robot_center(sim))
        img = main.render().copy()
        rgb = onboard("rgb_camera")
        dep = onboard("depth_camera").astype(float)
        dn = np.clip(np.log1p(dep) / np.log1p(30.0), 0, 1)
        dimg = (np.stack([1 - dn, 0.6 * (1 - dn) + 0.2, dn * 0.9], -1) * 255).astype(np.uint8)
        dimg[dep <= 0] = 0                                        # invalid depth -> black
        m = 12
        x1 = W - m
        for k, (pic, row) in enumerate(((rgb, 0), (dimg, 1))):
            y0 = m + row * (ih + m)
            w = pic.shape[1]
            img[y0:y0 + ih, x1 - w:x1] = pic
            img[y0 - 2:y0 + ih + 2, x1 - w - 2:x1 - w] = 255
        writer.append_data(img)
        if f == nframes // 2:
            iio.imwrite(out.replace(".mp4", ".png"), img)
    writer.close()
    print(f"wrote {out}  ({nframes} frames, sim {sim.d.time:.1f}s, wall {time.time() - t0:.1f}s)")
    if sim.plant is not None:
        print(f"[sim] end of run: {sim.plant.status()}, used {sim.plant.energy_wh:.2f} Wh")
    if sim.sensors is not None:
        print(f"[sim] sensors: {sim.sensors.status()}")


def cmd_sensors(a):
    from .runtime import Sim
    from .sensing.analysis import report
    sim = Sim(a.scene, a.seed, rebuild=a.rebuild, record=True)
    if sim.sensors is None:
        raise SystemExit(f"{a.scene}: robot has no sensor spec (use a designed vehicle, e.g. forest_grounded, mars_yard)")
    t0 = time.time()
    sim.step(int(a.seconds / sim.m.opt.timestep))
    print(f"[sim] {a.scene}: {a.seconds:.0f} s simulated in {time.time() - t0:.1f} s\n")
    print(report(sim.sensors))
    if a.log:
        np.savez_compressed(a.log, **{k: np.asarray(v) for k, v in sim.sensors.log.items() if v},
                            columns=np.array([
                                "imu: t gx gy gz ax ay az | true gx gy gz ax ay az",
                                "baro: t h | true h", "mag: t bx by bz | true bx by bz",
                                "gnss: t fix n_sats hdop blocked x y z | true x y z"]))
        print(f"\nlog written to {a.log}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m sim", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("build", "view", "render", "sensors"):
        s = sub.add_parser(name)
        s.add_argument("scene")
        s.add_argument("--seed", type=int, default=0)
        if name != "build":
            s.add_argument("--rebuild", action="store_true", help="regenerate assets first")
            s.add_argument("--wind", action="store_true", help="draw wind-field arrows")
        if name == "view":
            s.add_argument("--hq", action="store_true", help="full 8k shadows + 8x MSAA (slower)")
        if name == "sensors":
            s.add_argument("--seconds", type=float, default=60)
            s.add_argument("--log", default=None, help="save raw measurements + truth (.npz)")
        if name == "render":
            s.add_argument("--ideal", action="store_true", help="onboard insets without sensor models")
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
     "design": cmd_design, "sensors": cmd_sensors}[a.cmd](a)


if __name__ == "__main__":
    main()
