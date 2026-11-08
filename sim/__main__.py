"""CLI:  uv run --extra sim python -m sim <command> ...

  list                                  show the scene catalogue
  build  <scene|all> [--seed N]         generate MJCF + assets into sim/generated/
  view   <scene> [--seed N] [--wind]    interactive viewer, reference controller running
  render <scene> [--seed N] [--seconds S] [--wind]
                                        headless MP4: chase view + onboard RGB + depth insets
  design drone|rover [--config FILE] [--top N]
                                        size a vehicle from a mission file (sim/design/configs)
  randomize <scene> [--n N] [--seconds S]   (all variants in parallel on the C++ core)
                                        draw N vehicles from the mission's [uncertainty]
                                        ranges, run each, report robustness + sensitivities
  sensors <scene> [--seconds S] [--log FILE.npz]
  (view/render/sensors/randomize take --engine core|python: C++ core by default,
   the readable Python reference on request or when a scene needs it)
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
    from .core import make_sim
    sim = make_sim(a.scene, a.seed, rebuild=a.rebuild, variant=a.variant, engine=a.engine)
    print(f"[sim] engine: {sim.engine}", flush=True)
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
    last_print = 0
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
            if sim.plant is not None and int(sim.d.time / 5) > last_print:
                last_print = int(sim.d.time / 5)
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
    from .core import make_sim
    sim = make_sim(a.scene, a.seed, rebuild=a.rebuild, variant=a.variant, engine=a.engine)
    print(f"[sim] engine: {sim.engine}", flush=True)
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


def _variants_python(a):
    from .runtime import Sim
    sim = Sim(a.scene, a.seed, rebuild=a.rebuild)
    if not sim.meta0.get("uncertainty"):
        raise SystemExit(f"{a.scene}: no [uncertainty] spec (use a designed vehicle: forest_grounded, mars_yard)")
    nsteps = int(a.seconds / sim.m.opt.timestep)
    out = []
    for k in range(a.n):
        sim.reset(k)
        bid = sim.m.body(sim.robot["body"]).id
        ctl = sim.ctrls[0] if sim.ctrls else None
        errs, pw, tilt = [], [], 0.0
        for i in range(nsteps):
            sim.step()
            if i % 25 == 0 and sim.d.time > 4.0:
                errs.append(getattr(ctl, "err", 0.0))
                pw.append(sim.plant.power if sim.plant else 0.0)
                tilt = max(tilt, float(np.degrees(np.arccos(np.clip(sim.d.xmat[bid][8], -1, 1)))))
        out.append((np.asarray(errs), np.asarray(pw), tilt, sim.rejected, sim.param_vector()))
    return out


def _variants_core(a):
    """All variants in parallel on the C++ core (one env per variant)."""
    from .core import CoreBatch
    b = CoreBatch(a.scene, a.n, a.seed, sensors=False, rebuild=a.rebuild)
    if not b.meta0.get("uncertainty"):
        raise SystemExit(f"{a.scene}: no [uncertainty] spec (use a designed vehicle: forest_grounded, mars_yard)")
    b.reset(variants=list(range(a.n)))
    chunks = int(a.seconds / b.timestep) // 25
    errs, pw, tilt = [], [], np.zeros(a.n)
    for _ in range(chunks):
        b.step(25)
        if b.time[0] > 4.0:
            errs.append(b.err.copy())
            pw.append(b.plant[:, 3].copy())
            tilt = np.maximum(tilt, b.tilt_deg)
    E, P = np.asarray(errs), np.asarray(pw)
    return [(E[:, k], P[:, k], float(tilt[k]), b.rejected[k], b.param_vector(k)) for k in range(a.n)]


def cmd_randomize(a):
    from .uncertainty import PARAMS
    print(f"[sim] {a.scene}: {a.n} variants x {a.seconds:.0f} s, ranges from the mission's [uncertainty] "
          f"(engine: {a.engine})")
    t_start = time.time()
    if a.engine == "core":
        try:
            results = _variants_core(a)
        except ImportError as e:
            print(f"[sim] C++ core unavailable ({e}); using the Python reference")
            results = _variants_python(a)
    else:
        results = _variants_python(a)
    rows, vecs, names = [], [], None
    for k, (e, pw, tilt, rej, (names, vec)) in enumerate(results):
        rms, mx = float(np.sqrt(np.mean(e * e))), float(e.max())
        status = "LOST" if (mx > 5.0 or tilt > 75) else "ok"
        rows.append(dict(k=k, rej=rej, rms=rms, max=mx, tilt=tilt, power=float(np.nanmean(pw)), status=status))
        vecs.append(vec)
        print(f"  variant {k:3d}  track rms {rms:5.2f} m  max {mx:5.2f} m  tilt max {tilt:4.0f} deg  "
              f"power {np.nanmean(pw):6.1f} W  [{status}]  (rejected draws: {rej})", flush=True)
    rms = np.array([r["rms"] for r in rows])
    lost = sum(r["status"] != "ok" for r in rows)
    rej = sum(r["rej"] for r in rows)
    print(f"\nrobustness: {a.n - lost}/{a.n} ok; tracking rms median {np.median(rms):.2f} m, "
          f"worst {rms.max():.2f} m; infeasible draws rejected: {rej} "
          f"({rej / (rej + a.n) * 100:.0f}%)   [{time.time() - t_start:.0f} s]")
    # which uncertainties drive the error? Spearman rank correlation per parameter
    V = np.asarray(vecs)
    if a.n >= 5:
        rk = lambda x: np.argsort(np.argsort(x)).astype(float)
        cors = []
        for j, nm in enumerate(names):
            if np.ptp(V[:, j]) > 0:
                cors.append((float(np.corrcoef(rk(V[:, j]), rk(rms))[0, 1]), nm))
        cors.sort(key=lambda c: -abs(c[0]))
        print("most influential on tracking error (Spearman rho, needs ~30+ variants to trust):")
        for c, nm in cors[:6]:
            base = nm.rsplit("_", 1)[0] if nm not in PARAMS else nm
            print(f"  {nm:22s} {c:+.2f}   {PARAMS.get(base, ('', '', ''))[2]}")


def cmd_sensors(a):
    from .core import make_sim
    from .sensing.analysis import report
    sim = make_sim(a.scene, a.seed, rebuild=a.rebuild, variant=a.variant, record=True, engine=a.engine)
    print(f"[sim] engine: {sim.engine}", flush=True)
    if sim.sensors is None:
        raise SystemExit(f"{a.scene}: robot has no sensor spec (use a designed vehicle, e.g. forest_grounded, mars_yard)")
    t0 = time.time()
    sim.step(int(a.seconds / sim.m.opt.timestep))
    print(f"[sim] {a.scene}: {a.seconds:.0f} s simulated in {time.time() - t0:.1f} s\n")
    print(report(sim.sensors))
    if a.log:
        np.savez_compressed(a.log, **{k: np.asarray(v) for k, v in sim.sensors.log.items() if len(v)},
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
    for name in ("build", "view", "render", "sensors", "randomize"):
        s = sub.add_parser(name)
        s.add_argument("scene")
        s.add_argument("--seed", type=int, default=0)
        if name != "build":
            s.add_argument("--rebuild", action="store_true", help="regenerate assets first")
            s.add_argument("--wind", action="store_true", help="draw wind-field arrows")
        if name == "view":
            s.add_argument("--hq", action="store_true", help="full 8k shadows + 8x MSAA (slower)")
        if name in ("view", "render", "sensors", "randomize"):
            s.add_argument("--engine", choices=["core", "python"], default="core",
                           help="core: C++ (default, falls back when unsupported); python: readable reference")
        if name in ("view", "render", "sensors"):
            s.add_argument("--variant", type=int, default=None,
                           help="draw vehicle K from the [uncertainty] ranges (default: nominal)")
        if name == "randomize":
            s.add_argument("--n", type=int, default=20)
            s.add_argument("--seconds", type=float, default=20)
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
     "design": cmd_design, "sensors": cmd_sensors, "randomize": cmd_randomize}[a.cmd](a)


if __name__ == "__main__":
    main()
