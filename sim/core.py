"""Front end of the C++ simulation core (`razzy_core`, sources in sim/cpp/core/).

The C++ core runs physics, controller, power-train plant, sensors and latency for many
environments in parallel. Python only builds one config dict per environment at reset
(from the same scene meta + [uncertainty] draw the Python reference uses) and reads the
results as numpy arrays. The readable Python reference (sim/runtime.py Sim, sim/control.py,
sim/design/plant.py, sim/sensing/) stays the specification; tests/sim/test_core.py checks
the C++ core against it.

    CoreBatch  N envs, for training / batch evaluation:   reset, act, step, arrays
    CoreSim    one env + a mirror MjData of the full visual scene, for view / render
    make_sim   CoreSim if the scene's controller is in C++, else the Python Sim
"""
from __future__ import annotations

import json
import os

import mujoco
import numpy as np

from . import uncertainty as U
from .physics.battery import CHEMISTRY
from .runtime import Sim, load_terrain, scene_dir
from .sensing.models import DelayLine, latency_sampler
from .sensing.suite import FOLIAGE_WORDS
from .physics.aero import WindField

MODES = {"mission": 0, "setpoint": 1, "ctrl": 2}
CORE_CONTROLLERS = {"quad_waypoints", "rover_ackermann", None}


class Unsupported(Exception):
    """Scene uses something only the Python reference implements (e.g. a custom controller)."""


def physics_model(folder) -> str:
    """Compile scene.xml without visual-only geoms/textures into a small .mjb (cached).
    Same bodies, joints, actuators, sensors, masses, contacts -> identical physics."""
    xml, mjb = folder / "scene.xml", folder / "scene_physics.mjb"
    if not mjb.exists() or mjb.stat().st_mtime < xml.stat().st_mtime:
        spec = mujoco.MjSpec.from_file(str(xml))
        spec.compiler.discardvisual = True
        for t in list(spec.textures):
            spec.delete(t)
        for mt in list(spec.materials):
            mt.textures = [""] * len(mt.textures)
        mujoco.mj_saveModel(spec.compile(), str(mjb), None)
    return str(mjb)


def ray_opacity(folder, full: mujoco.MjModel | None, robot_body: str) -> np.ndarray:
    """Per geom of the full scene: 1 solid, -1 foliage (loss set per env), 0 the robot itself."""
    cache = folder / "ray_opacity.npy"
    xml = folder / "scene.xml"
    if cache.exists() and cache.stat().st_mtime >= xml.stat().st_mtime:
        return np.load(cache)
    full = full or mujoco.MjModel.from_xml_path(str(xml))
    op = np.ones(full.ngeom)
    root = full.body(robot_body).id
    for g in range(full.ngeom):
        mid = full.geom_matid[g]
        mat = full.material(mid).name if mid >= 0 else ""
        if any(w in mat for w in FOLIAGE_WORDS):
            op[g] = -1.0
        if full.body_rootid[full.geom_bodyid[g]] == root:
            op[g] = 0.0
    np.save(cache, op)
    return op


def _lat(spec_ms) -> dict:
    if isinstance(spec_ms, (list, tuple)):
        return {"lat_lo": spec_ms[0] / 1000, "lat_hi": spec_ms[1] / 1000}
    return {"lat_lo": float(spec_ms) / 1000, "lat_hi": float(spec_ms) / 1000}


def _pack(b: dict) -> dict:
    c = CHEMISTRY[b["chemistry"]]
    return {"soc_tab": c["soc"], "ocv_tab": c["ocv"], "cells": b["cells"], "cap_ah": b["capacity_mah"] / 1000,
            "r": b["r_cell"] * b["cells"], "v_nom": c["nominal"] * b["cells"]}


class CoreBatch:
    """N environments of one scene stepped in parallel by the C++ core.

    mode "mission"  : built-in reference controller flies/drives the mission
         "setpoint" : act() rows are drone [vx, vy, vz, yaw_rate] or rover [speed, curvature],
                      tracked by the C++ low-level controller (the recommended RL interface)
         "ctrl"     : act() rows are raw actuator commands
    Actions pass through the policy-compute latency model when the vehicle has one.
    """

    def __init__(self, name: str, n_envs: int = 1, seed: int = 0, threads: int | None = None,
                 mode: str = "mission", sensors: bool = True, record: bool = False, rebuild: bool = False) -> None:
        import razzy_core
        folder = scene_dir(name, seed)
        if rebuild or not (folder / "scene.xml").exists():
            from .scenes import build
            print(f"[sim] generating {name} (seed {seed}) ...", flush=True)
            build(name, seed)
        self.name, self.folder, self.seed, self.mode = name, folder, seed, mode
        self.use_sensors, self.record = sensors, record
        self.meta0 = json.loads((folder / "meta.json").read_text())
        if self.meta0.get("controller") not in CORE_CONTROLLERS:
            raise Unsupported(f"controller '{self.meta0.get('controller')}' only exists in the Python reference")
        self.terrain = load_terrain(self.meta0, folder)
        phys = physics_model(folder)
        self.m = mujoco.MjModel.from_binary_path(phys)        # ids / names (same as the full model)
        threads = threads or min(n_envs, os.cpu_count() or 1)
        self.core = razzy_core.Batch(phys, n_envs, threads)
        if self.terrain is not None:
            t = self.terrain
            self.core.set_terrain(np.ascontiguousarray(t.z, float), t.half_x, t.half_y)
        robot = self.meta0["robots"][0]
        if sensors and "gnss" in robot.get("sensors", {}):
            self.core.load_rays(str(folder / "scene.xml"), ray_opacity(folder, None, robot["body"]))
        self.n = n_envs
        self.variants: list[int | None] = [None] * n_envs
        self.params: list[dict] = [{} for _ in range(n_envs)]
        self.metas: list[dict] = [self.meta0] * n_envs
        self.rejected = [0] * n_envs
        self.reset()

    # -------------------------------------------------------------------- config ---
    def _config(self, meta: dict, params: dict, noise_seed: int) -> dict:
        m = self.m
        r = meta["robots"][0]
        name = r["body"]
        bid = m.body(name).id
        jnt = m.body_jntadr[bid]
        free_dof = int(m.jnt_dofadr[jnt]) if jnt >= 0 and m.jnt_type[jnt] == 0 else -1
        cfg: dict = {"mode": MODES[self.mode], "seed": int(noise_seed), "body": bid}
        g = params.get
        cfg["model"] = {"mass_frac": g("mass_frac", 1.0), "inertia_frac": list(np.asarray(g("inertia_frac", np.ones(3)), float)),
                        "com_offset": list(np.asarray(g("com_offset_mm", np.zeros(3)), float)),
                        "friction": g("friction", 1.0), "thrust_coef": g("thrust_coef", 1.0),
                        "motor_gain": list(np.atleast_1d(g("motor_gain", []))), "motor_lag": g("motor_lag", 1.0),
                        "yaw_coef": g("yaw_coef", 1.0), "drive_torque": g("drive_torque", 1.0),
                        "drive_gain": list(np.atleast_1d(g("drive_gain", []))), "steer_kp": g("steer_kp", 1.0)}
        ctrl = meta.get("controller")
        if r.get("type") == "quadrotor":
            agl = "waypoints_agl" in meta
            wps = np.asarray(meta["waypoints_agl"] if agl else meta["waypoints_abs"], float)
            cfg["quad"] = {"body": bid, "act": [m.actuator(a).id for a in r["actuators"]],
                           "waypoints": wps.ravel().tolist(), "agl": int(agl), "cruise": meta.get("cruise", 3.0),
                           "CdA": r.get("drag_CdA", 0.04)}
        if ctrl == "rover_ackermann" or r.get("type") == "rocker_bogie":
            cfg["rover"] = {"body": bid, "drive": [m.actuator(a).id for a in r["drive"]],
                            "steer": [m.actuator(f"{s}_p").id for s in r["steer_xy"]],
                            "jdof": [int(m.jnt_dofadr[m.joint(f"{w}_j").id]) for w in r["wheels"]],
                            "wxy": np.asarray(r["wheel_xy"], float).ravel().tolist(),
                            "sxy": np.asarray(list(r["steer_xy"].values()), float).ravel().tolist(),
                            "rw": r["wheel_radius"], "w0": r["omega0"], "v": meta.get("speed", 0.5),
                            "waypoints": np.asarray(meta.get("waypoints", [[0, 0], [1, 0]]), float).ravel().tolist()}
        if "quad" not in cfg and "rover" not in cfg:
            raise Unsupported(f"robot type '{r.get('type')}' has no C++ controller yet")
        if "rotor_speed" in meta:
            cfg["turbine"] = {"act": m.actuator("rotor_drive").id, "speed": meta["rotor_speed"]}
        if "wind" in meta:
            w = WindField(meta["wind"], self.terrain, self.seed)          # same draws as the reference
            wc = {"U": w.U, "heading": float(w.heading), "gust": w.gust, "z_ref": w.z_ref, "z0": w.z0,
                  "cyl": w.cyl.ravel().tolist(), "kx": w.k[:, 0].tolist(), "ky": w.k[:, 1].tolist(),
                  "phase": w.phase.tolist(), "omega": w.omega.tolist(), "amp": w.amp.tolist(), "wphase": w.wphase.tolist()}
            if w.turbine:
                wc.update(turbine_hub=list(w.turbine["hub"]), turbine_R=w.turbine["R"], turbine_Ct=w.turbine["Ct"],
                          turbine_k=w.turbine["k"])
            cfg["wind"] = wc
        p = r.get("plant")
        if p and p["kind"] == "drone":
            cfg["dplant"] = {"bid": bid, "dof": free_dof, "act": [m.actuator(a).id for a in r["actuators"]],
                             "pack": _pack(p["battery"]), "kv": p["motor"]["kv"], "r_ohm": p["motor"]["r_ohm"],
                             "i0": p["motor"]["i0"], "D": p["prop"]["diameter_in"] * 0.0254, "ct": p["prop"]["ct"],
                             "cp": p["prop"]["cp"], "rho": p["rho"], "esc_eff": p["esc_eff"],
                             "avionics_w": p["avionics_w"], "t_max_ref": p["t_max_ref"], "rotor_drag": p["rotor_drag"]}
        if p and p["kind"] == "rover":
            n = len(r["drive"])
            cfg["rplant"] = {"act": [m.actuator(a).id for a in r["drive"]], "pack": _pack(p["battery"]),
                             "tau_s": np.broadcast_to(np.asarray(p["tau_stall"], float), (n,)).tolist(),
                             "i0": p["motor"]["no_load_current"], "i_s": p["motor"]["stall_current"] * p["v_ratio"],
                             "avionics_w": p["avionics_w"], "servo_w": p["servo_w"], "n_servo": len(r["steer"])}
        sp = r.get("sensors")
        if self.use_sensors and sp:
            s = {"gyro_adr": int(m.sensor_adr[m.sensor(f"{name}_gyro").id]),
                 "acc_adr": int(m.sensor_adr[m.sensor(f"{name}_acc").id]),
                 "site": m.site(f"{name}_imu").id, "bid": bid, "dof": free_dof, "record": int(self.record)}
            if "imu" in sp:
                s["imu"] = {**{k: v for k, v in sp["imu"].items() if isinstance(v, (int, float))}, **_lat(sp["imu"].get("latency_ms", 0))}
            v = r.get("vibration")
            if v and "imu" in sp and p:
                D = p["prop"]["diameter_in"] * 0.0254
                s["vib"] = {"rotor_pos": np.asarray(v["rotor_pos"], float).ravel().tolist(), "spin": list(map(float, v["spin"])),
                            "mass": v["mass"], "inertia_inv": np.linalg.inv(np.asarray(v["inertia"], float)).ravel().tolist(),
                            "imbalance_gmm": v["imbalance_gmm"], "thrust_ripple": v["thrust_ripple"], "blades": v["blades"],
                            "arm_hz": v["arm_hz"], "arm_zeta": v.get("arm_zeta", 0.02), "mount_hz": v["mount_hz"],
                            "mount_zeta": v["mount_zeta"], "rotor_act": [m.actuator(a).id for a in r["actuators"]],
                            "kn": 1.0 / (p["prop"]["ct"] * p["rho"] * D ** 4)}
            if "baro" in sp:
                s["baro"] = {**{k: v for k, v in sp["baro"].items() if isinstance(v, (int, float))}, **_lat(sp["baro"].get("latency_ms", 0))}
            if "mag" in sp:
                s["mag"] = {**{k: v for k, v in sp["mag"].items() if isinstance(v, (int, float))},
                            **{k: float(v) for k, v in sp["environment"].items()}, **_lat(sp["mag"].get("latency_ms", 0))}
            if "gnss" in sp:
                s["gnss"] = {**{k: v for k, v in sp["gnss"].items() if isinstance(v, (int, float))}, **_lat(sp["gnss"]["latency_ms"])}
                s["antenna"] = [0.0, 0.0, sp.get("installation", {}).get("gnss_antenna_height_m", 0.08)]
            if "depth_camera" in sp:
                s["depth"] = {k: (int(v) if isinstance(v, bool) else v) for k, v in sp["depth_camera"].items()
                              if isinstance(v, (int, float))}
            if "rgb_camera" in sp:
                q = sp["rgb_camera"]
                s["rgb"] = {**{k: v for k, v in q.items() if isinstance(v, (int, float))},
                            "exp_lo": q["exposure_ms"][0], "exp_hi": q["exposure_ms"][1], "rolling": int(q.get("shutter") == "rolling")}
            cfg["sensors"] = s
            if self.mode != "mission" and "compute" in sp:
                cfg["compute"] = {"rate_hz": sp["compute"]["rate_hz"], **_lat(sp["compute"]["inference_ms"])}
        return cfg

    # ------------------------------------------------------------------- control ---
    def reset(self, envs=None, variants=None) -> None:
        """Reset envs (default all) to the given variants (None = nominal vehicle)."""
        envs = range(self.n) if envs is None else envs
        variants = [None] * len(envs) if variants is None else variants
        unc = self.meta0.get("uncertainty", {})
        robot0 = self.meta0["robots"][0]
        for i, v in zip(envs, variants):
            params, rej = {}, 0
            if v is not None:
                if not unc:
                    raise SystemExit("this scene has no [uncertainty] spec (designed vehicles only)")
                params, rej = U.sample(unc, robot0, np.random.default_rng([self.seed, 0xD0A1, v]))
            meta = U.apply_meta(self.meta0, params) if params else self.meta0
            noise_seed = self.seed if v is None else self.seed + 1_000_003 * (v + 1)
            self.core.reset(int(i), self._config(meta, params, noise_seed))
            self.variants[i], self.params[i], self.metas[i], self.rejected[i] = v, params, meta, rej

    def act(self, actions) -> None:
        self.core.act(np.ascontiguousarray(actions, float))

    def step(self, n: int = 1) -> None:
        self.core.step(int(n))

    def param_vector(self, i: int = 0):
        return U.flatten(self.params[i])

    def __getattr__(self, k):              # time, qpos, qvel, ctrl, err, plant, imu, gnss, ...
        return getattr(self.core, k)


# ============================================================================ CoreSim ===
class _Plant:
    def __init__(self, b: CoreBatch, i: int) -> None:
        self.b, self.i = b, i

    soc = property(lambda s: float(s.b.plant[s.i, 0]))
    v = property(lambda s: float(s.b.plant[s.i, 1]))
    i_a = property(lambda s: float(s.b.plant[s.i, 2]))
    power = property(lambda s: float(s.b.plant[s.i, 3]))
    energy_wh = property(lambda s: float(s.b.plant[s.i, 4]))

    def status(self) -> str:
        return f"battery {self.soc*100:5.1f}%  {self.v:5.2f} V  {self.power:6.1f} W"


class _Spec:
    def __init__(self, **kw) -> None:
        self.__dict__.update(kw)


class _Sensors:
    """Same surface as sensing.suite.SensorSuite (status, report, cameras) on the C++ core."""

    def __init__(self, sim: "CoreSim") -> None:
        self.sim, self.b = sim, sim.batch
        self.spec = sim.robot["sensors"]
        self.lines = {k: DelayLine() for k in ("depth_camera", "rgb_camera")}
        rng = np.random.default_rng([sim.batch.seed, 0xCA3])
        self.lat = {k: latency_sampler(self.spec[k].get("latency_ms", 0), rng) for k in self.lines if k in self.spec}
        self.next_t = {k: 0.0 for k in self.lines}
        m = sim.m
        name = sim.robot["body"]
        self.bid, self.imu_site = m.body(name).id, m.site(f"{name}_imu").id
        self.gyro_adr = m.sensor_adr[m.sensor(f"{name}_gyro").id]
        rate = self.spec.get("imu", {}).get("rate_hz", 0)
        self.imu = _Spec(rate=rate, saturated=0)
        v = sim.robot.get("vibration")
        self.vib = _Spec(n=np.zeros(4), blades=v["blades"], arm_hz=v["arm_hz"], mount_hz=v["mount_hz"]) if v else None

    @property
    def log(self) -> dict:
        self.imu.saturated = self.b.core.imu_saturated(0) if "imu" in self.spec else 0
        if self.vib is not None:
            self.vib.n = np.asarray(self.b.core.rotor_speeds(0))
        return {k: self.b.core.log(0, k) for k in ("imu", "baro", "mag", "gnss") if k in self.spec}

    def status(self) -> str:
        b, d = self.b, self.sim.d
        parts = []
        if "gnss" in self.spec and np.isfinite(b.gnss[0, 0]):
            g = b.gnss[0]
            if g[1] > 0:
                err = float(np.linalg.norm(g[5:7] - d.xpos[self.bid][:2]))
                parts.append(f"gnss {int(g[2]):2d} sats hdop {g[3]:.1f} err {err:4.1f} m sky-blocked {g[4]*100:3.0f}%")
            else:
                parts.append(f"gnss NO FIX ({int(g[2])} sats, sky-blocked {g[4]*100:3.0f}%)")
        if self.vib is not None:
            parts.append(f"vib {b.vib[0]:4.2f} m/s2")
        if "baro" in self.spec and np.isfinite(b.baro[0, 1]):
            parts.append(f"baro err {b.baro[0, 1] - float(d.site_xpos[self.imu_site][2]):+5.2f} m")
        return " | ".join(parts)

    # cameras: rendering stays in Python (mirror scene); the image models run in C++
    def camera_due(self, kind: str, t: float) -> bool:
        spec = self.spec.get(kind)
        if spec is None:
            return False
        if t + 1e-9 >= self.next_t[kind]:
            self.next_t[kind] = max(self.next_t[kind] + 1 / spec["rate_hz"], t)
            return True
        return False

    def _omega_cam(self, cam: str) -> np.ndarray:
        d, m = self.sim.d, self.sim.m
        w_body = d.sensordata[self.gyro_adr:self.gyro_adr + 3]
        Rs = d.site_xmat[self.imu_site].reshape(3, 3)
        Rc = d.cam_xmat[m.camera(cam).id].reshape(3, 3)
        return Rc.T @ (Rs @ w_body)

    def capture(self, kind: str, t: float, img: np.ndarray, gray: np.ndarray | None = None) -> np.ndarray:
        if kind == "depth_camera":
            g = None if gray is None else np.ascontiguousarray(gray, np.float32)
            out = self.b.core.depth_apply(0, np.ascontiguousarray(img, np.float32), g)
        else:
            out = self.b.core.rgb_apply(0, np.ascontiguousarray(img, np.uint8),
                                        list(map(float, self._omega_cam(self.spec[kind]["camera"]))))
        self.lines[kind].push(t, out, self.lat[kind]())
        return out

    def frame(self, kind: str, t: float):
        return self.lines[kind].latest(t)


class CoreSim:
    """One environment on the C++ core, with the same surface the CLI uses from Sim:
    .m/.d (full visual scene, mirrored every step for rendering), .robot, .meta, .plant,
    .sensors, .wind (for drawing arrows), step(), reset(variant), draw_wind()."""

    engine = "c++"

    def __init__(self, name: str, seed: int = 0, rebuild: bool = False, variant: int | None = None,
                 record: bool = False, sensors: bool = True) -> None:
        self.batch = CoreBatch(name, 1, seed, threads=1, sensors=sensors, record=record, rebuild=rebuild)
        self.folder, self.seed = self.batch.folder, seed
        self.m = mujoco.MjModel.from_xml_path(str(self.folder / "scene.xml"))
        self.d = mujoco.MjData(self.m)
        self.terrain = self.batch.terrain
        self.reset(variant)

    def reset(self, variant: int | None = None) -> None:
        self.batch.reset([0], [variant])
        self.variant, self.params, self.rejected = variant, self.batch.params[0], self.batch.rejected[0]
        self.meta = self.batch.metas[0]
        self.wind = WindField(self.meta["wind"], self.terrain, self.seed) if "wind" in self.meta else None
        self.plant = _Plant(self.batch, 0) if "plant" in self.robot else None
        self.sensors = _Sensors(self) if (self.batch.use_sensors and "sensors" in self.robot) else None
        self.ctrls = []
        self._sync()

    @property
    def robot(self) -> dict:
        return self.meta["robots"][0]

    def _sync(self) -> None:
        b, d = self.batch, self.d
        d.qpos[:] = b.qpos[0]
        d.qvel[:] = b.qvel[0]
        d.ctrl[:] = b.ctrl[0]
        d.time = float(b.time[0])
        mujoco.mj_forward(self.m, d)

    def step(self, n: int = 1) -> None:
        self.batch.step(n)
        self._sync()

    def param_vector(self):
        return self.batch.param_vector(0)

    draw_wind = Sim.draw_wind


def make_sim(name: str, seed: int = 0, rebuild: bool = False, variant: int | None = None, record: bool = False,
             engine: str = "core"):
    """C++ core when the scene is supported, else (or with engine="python") the reference."""
    if engine == "core":
        try:
            return CoreSim(name, seed, rebuild=rebuild, variant=variant, record=record)
        except Unsupported as e:
            print(f"[sim] {name}: {e}; using the Python reference engine", flush=True)
        except ImportError as e:
            print(f"[sim] C++ core unavailable ({e}); using the Python reference engine", flush=True)
        rebuild = False
    s = Sim(name, seed, rebuild=rebuild, variant=variant, record=record)
    s.engine = "python"
    return s
