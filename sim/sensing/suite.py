"""Glue between the sensor models and a running MuJoCo scene.

Spec resolution (build time): the mission file names parts (`[sensors]`), the part
database holds their datasheet numbers, `[installation]` says how they're mounted and
`[environment]` the site (magnetic field). `resolve()` merges them into plain dicts that go
into the robot meta, so a generated scene carries its complete sensor spec.

SensorSuite (run time) samples every sensor at its own rate from the MuJoCo state, pushes
measurements through per-sensor latency lines and serves what a real flight computer would
have at time t:  suite.read(t), suite.drain_imu(t), suite.frame("depth", t).
"""
from __future__ import annotations


import mujoco
import numpy as np

from . import models as M

FOLIAGE_WORDS = ("leaf", "conifer", "grass", "weed", "lettuce", "crown", "needle")
SENSOR_KINDS = ("imu", "baro", "mag", "gnss", "depth_camera", "rgb_camera")
STREAM_IDS = {"imu": 1, "baro": 2, "mag": 3, "gnss": 4, "depth_camera": 5, "rgb_camera": 6,
              "compute": 7, "vibration": 8}


def resolve(cfg: dict, db: dict) -> dict:
    """Mission [sensors]/[installation]/[environment]/[compute] + part database -> spec."""
    chosen = cfg.get("sensors", {})
    out = {"installation": dict(cfg.get("installation", {})),
           "environment": dict(cfg.get("environment", {}))}
    for kind in SENSOR_KINDS:
        name = chosen.get(kind)
        if not name:
            continue
        if name not in db.get(kind, {}):
            raise SystemExit(f"unknown {kind} '{name}'; known: {list(db.get(kind, {}))}")
        out[kind] = dict(db[kind][name], part=name)
    inst = out["installation"]
    if "baro" in out:
        out["baro"]["static_cp"] = inst.get("baro_static_cp", 0.3)
    if "mag" in out:
        out["mag"]["current_ut_per_a"] = inst.get("mag_current_ut_per_a", 0.0)
    comp = cfg.get("compute", {})
    if comp.get("part"):
        out["compute"] = dict(db["compute"][comp["part"]], part=comp["part"], **{k: v for k, v in comp.items() if k != "part"})
    return out


def _rng(seed: int, stream: str) -> np.random.Generator:
    # one independent stream per sensor: adding a sensor never changes another one's noise
    return np.random.default_rng([seed, 0x5E45, STREAM_IDS[stream]])


class SensorSuite:
    def __init__(self, sim, robot: dict, seed: int = 0, record: bool = False) -> None:
        self.sim, self.robot = sim, robot
        spec = robot["sensors"]
        self.spec = spec
        m = sim.m
        self.bid = m.body(robot["body"]).id
        name = robot["body"]
        self.gyro_adr = m.sensor_adr[m.sensor(f"{name}_gyro").id]
        self.acc_adr = m.sensor_adr[m.sensor(f"{name}_acc").id]
        self.imu_site = m.site(f"{name}_imu").id
        inst = spec.get("installation", {})
        self.record = record
        self.log: dict[str, list] = {k: [] for k in ("imu", "baro", "mag", "gnss")}
        self.lines = {k: M.DelayLine() for k in SENSOR_KINDS}
        self.lat = {k: M.latency_sampler(spec[k].get("latency_ms", 0), _rng(seed, k)) for k in SENSOR_KINDS if k in spec}
        self.next_t = {k: 0.0 for k in SENSOR_KINDS}
        self.last = {}
        self.imu = M.Imu(spec["imu"], _rng(seed, "imu")) if "imu" in spec else None
        self.baro = M.Baro(spec["baro"], _rng(seed, "baro")) if "baro" in spec else None
        self.mag = M.Mag(spec["mag"], spec["environment"], _rng(seed, "mag")) if "mag" in spec else None
        self.gnss = M.Gnss(spec["gnss"], _rng(seed, "gnss")) if "gnss" in spec else None
        self.depth = M.StereoDepth(spec["depth_camera"], _rng(seed, "depth_camera")) if "depth_camera" in spec else None
        self.rgb = M.RgbCamera(spec["rgb_camera"], _rng(seed, "rgb_camera")) if "rgb_camera" in spec else None
        self.compute = M.ComputeLatency(spec["compute"], _rng(seed, "compute")) if "compute" in spec else None
        self.vib = None
        if "vibration" in robot and self.imu is not None:
            self.vib = M.RotorVibration(robot["vibration"], spec["imu"].get("aaf_hz", 1e9), _rng(seed, "vibration"))
            self.act = np.array([m.actuator(a).id for a in robot["actuators"]])
            p = robot["plant"]
            D = p["prop"]["diameter_in"] * 0.0254
            self._kn = 1.0 / (p["prop"]["ct"] * p["rho"] * D ** 4)
        if self.gnss is not None:
            self._gnss_setup(inst)

    # ------------------------------------------------------------------ GNSS sky view ---
    def _gnss_setup(self, inst: dict) -> None:
        m = self.sim.m
        foliage = float(self.spec["gnss"].get("foliage_loss", 0.6))
        self.opacity = np.ones(m.ngeom)
        for g in range(m.ngeom):
            mid = m.geom_matid[g]
            mat = m.material(mid).name if mid >= 0 else ""
            if any(w in mat for w in FOLIAGE_WORDS):
                self.opacity[g] = foliage
        # trace against what is really there (visual geometry), not the coarse colliders
        self.ray_groups = np.array([1, 1, 1, 0, 0, 0], np.uint8)
        self.antenna = np.array([0.0, 0.0, inst.get("gnss_antenna_height_m", 0.08)], float)
        self._gid = np.array([-1], np.int32)

    def sky_survival(self, p: np.ndarray, dirs: np.ndarray, max_layers: int = 4) -> np.ndarray:
        """Per direction: probability a signal from the sky reaches point p."""
        m, d = self.sim.m, self.sim.d
        out = np.ones(len(dirs))
        for i, u in enumerate(dirs):
            q = p.copy()
            for _ in range(max_layers):
                dist = mujoco.mj_ray(m, d, q, u, self.ray_groups, 1, self.bid, self._gid)
                if dist < 0:
                    break
                out[i] *= 1.0 - self.opacity[self._gid[0]]
                if out[i] < 0.02:
                    out[i] = 0.0
                    break
                q = q + u * (dist + 0.05)
        return out

    # ------------------------------------------------------------------------ stepping ---
    def _due(self, kind: str, t0: float, t1: float, rate: float):
        """Sample times of `kind` falling in [t0, t1)."""
        if self.next_t[kind] < t0 - 1.0 / rate:
            self.next_t[kind] = t0
        ts = []
        while self.next_t[kind] < t1:
            ts.append(self.next_t[kind])
            self.next_t[kind] += 1.0 / rate
        return ts

    def update(self, t0: float, dt: float) -> None:
        """Call after mj_step; sensordata/actuator_force describe the state at t0."""
        sim = self.sim
        d, m = sim.d, sim.m
        t1 = t0 + dt
        g_true = d.sensordata[self.gyro_adr:self.gyro_adr + 3].copy()
        a_true = d.sensordata[self.acc_adr:self.acc_adr + 3].copy()
        if self.vib is not None:
            T = np.maximum(d.actuator_force[self.act], 0.0)
            self.vib.set_rotors(t0, np.sqrt(T * self._kn), T)
        if self.imu is not None:
            for ts in self._due("imu", t0, t1, self.imu.rate):
                gv, av = self.vib.at(ts) if self.vib is not None else (0.0, 0.0)
                g, a = self.imu.sample(g_true, a_true, gv, av)
                self.lines["imu"].push(ts, (g, a), self.lat["imu"]())
                if self.record:
                    self.log["imu"].append((ts, *g, *a, *g_true, *a_true))
        need_pose = any(x is not None for x in (self.baro, self.mag, self.gnss))
        if not need_pose:
            return
        p_site = d.site_xpos[self.imu_site]
        R = d.xmat[self.bid].reshape(3, 3)
        j = m.body_jntadr[self.bid]
        v_lin = d.qvel[m.jnt_dofadr[j]:m.jnt_dofadr[j] + 3].copy() if m.jnt_type[j] == 0 else np.zeros(3)
        if self.baro is not None:
            for ts in self._due("baro", t0, t1, self.baro.rate):
                v_air = v_lin - (sim.wind.velocity(p_site, ts)[0] if sim.wind is not None else 0.0)
                h = self.baro.sample(float(p_site[2]), float(np.linalg.norm(v_air)))
                self.lines["baro"].push(ts, h, self.lat["baro"]())
                if self.record:
                    self.log["baro"].append((ts, h, float(p_site[2])))
        if self.mag is not None:
            cur = float(sim.plant.i) if sim.plant is not None else 0.0
            for ts in self._due("mag", t0, t1, self.mag.rate):
                b = self.mag.sample(R, cur)
                self.lines["mag"].push(ts, b, self.lat["mag"]())
                if self.record:
                    self.log["mag"].append((ts, *b, *(R.T @ self.mag.b_world)))
        if self.gnss is not None:
            for ts in self._due("gnss", t0, t1, self.gnss.rate):
                ant = d.xpos[self.bid] + R @ self.antenna
                surv = self.sky_survival(ant, self.gnss.sats_at(ts))
                fix = self.gnss.sample(ts, ant, v_lin, surv)
                self.lines["gnss"].push(ts, fix, self.lat["gnss"]())
                if self.record:
                    pos = fix.get("pos", np.full(3, np.nan))
                    self.log["gnss"].append((ts, float(fix["fix"]), fix["n_sats"], fix["hdop"], fix["blocked"], *pos, *ant))

    # ------------------------------------------------------------------------ cameras ---
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
        """Turn a perfect render of `kind` (rgb uint8 or metric depth) into a sensor frame,
        queue it behind the sensor latency, and return the processed frame."""
        spec = self.spec[kind]
        if kind == "depth_camera":
            out = self.depth.apply(img, gray)
        else:
            out = self.rgb.apply(img, self._omega_cam(spec["camera"]))
        self.lines[kind].push(t, out, self.lat[kind]())
        return out

    def frame(self, kind: str, t: float):
        """(t_capture, frame) of the newest frame the flight computer has received."""
        return self.lines[kind].latest(t)

    # ------------------------------------------------------------------------ outputs ---
    def drain_imu(self, t: float) -> list:
        return self.lines["imu"].drain(t)

    def read(self, t: float) -> dict:
        out = {}
        for k in ("imu", "baro", "mag", "gnss"):
            if k in self.spec:
                last = self.lines[k].latest(t)
                if last is not None:
                    out[k] = last
        return out

    def status(self) -> str:
        t = self.sim.d.time
        r = self.read(t)
        parts = []
        if "gnss" in r:
            g = r["gnss"][1]
            if g["fix"]:
                truth = self.sim.d.xpos[self.bid]
                err = float(np.linalg.norm(g["pos"][:2] - truth[:2]))
                parts.append(f"gnss {g['n_sats']:2d} sats hdop {g['hdop']:.1f} err {err:4.1f} m "
                             f"sky-blocked {g['blocked']*100:3.0f}%")
            else:
                parts.append(f"gnss NO FIX ({g['n_sats']} sats, sky-blocked {g['blocked']*100:3.0f}%)")
        if self.vib is not None:
            _, av = self.vib.at(t)
            parts.append(f"vib {np.linalg.norm(av):4.2f} m/s2")
        if "baro" in r:
            parts.append(f"baro err {r['baro'][1] - float(self.sim.d.site_xpos[self.imu_site][2]):+5.2f} m")
        return " | ".join(parts)
