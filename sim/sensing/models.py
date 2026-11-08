"""Sensor, latency and compute models. Pure numpy, no MuJoCo: unit-testable, and the same
functions can be fitted against real logs.

Every model takes a *spec* dict whose fields are datasheet quantities (see the `imu.*`,
`gnss.*`, ... sections of sim/parts/sensors.toml) and a seeded
numpy Generator, so a (scene, seed) pair reproduces every noise sample bit-for-bit.

Noise conventions (match how datasheets and Allan-variance plots state them):
  * white noise density ND [unit/sqrt(Hz)] sampled at rate f  ->  per-sample sigma = ND*sqrt(f)
    (Allan deviation of the result at tau = 1 s equals ND)
  * slowly wandering biases are first-order Gauss-Markov processes (sigma, tau)
"""
from __future__ import annotations

import math
import numpy as np

G0 = 9.80665
DEG = math.pi / 180


# ============================================================================ helpers ===
# building blocks live in sim/physics (re-exported here for the sensor models below)
from ..physics.noise import DelayLine, GaussMarkov, latency_sampler  # noqa: E402,F401
from ..physics.vibration import lowpass_gain, transmissibility  # noqa: E402,F401


def _small_rotation(rng, sigma_rad: float) -> np.ndarray:
    a = rng.normal(0, sigma_rad, 3)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + K


# ================================================================================ IMU ===
class Imu:
    """MEMS gyro + accelerometer.

    measured = M (truth + vibration) + bias(t) + white noise, then clipped to the full-scale
    range and quantised to the ADC LSB. M holds per-axis scale error and cross-axis
    misalignment (after the user's calibration, so use *residual* values in the spec).
    Vibration is added before the clip on purpose: real IMUs saturate on vibration peaks.
    """

    def __init__(self, spec: dict, rng: np.random.Generator) -> None:
        s = spec
        self.spec, self.rng = s, rng
        self.rate = float(s["rate_hz"])
        sq = math.sqrt(self.rate)
        self.g_sig = s["gyro_noise_dps_rthz"] * DEG * sq
        self.a_sig = s["accel_noise_ug_rthz"] * 1e-6 * G0 * sq
        self.g_bias0 = rng.normal(0, s["gyro_bias_dps"] * DEG, 3)
        self.a_bias0 = rng.normal(0, s["accel_bias_mg"] * 1e-3 * G0, 3)
        tau = s.get("bias_tau_s", 200.0)
        self.g_walk = GaussMarkov(s["gyro_bias_instability_dph"] * DEG / 3600, tau, rng, 3)
        self.a_walk = GaussMarkov(s["accel_bias_instability_ug"] * 1e-6 * G0, tau, rng, 3)
        se = s.get("scale_error_pct", 0.0) / 100
        mis = s.get("misalign_deg", 0.0) * DEG
        self.Mg = np.diag(1 + rng.normal(0, se, 3)) @ _small_rotation(rng, mis)
        self.Ma = np.diag(1 + rng.normal(0, se, 3)) @ _small_rotation(rng, mis)
        self.g_fs = s["gyro_range_dps"] * DEG
        self.a_fs = s["accel_range_g"] * G0
        bits = s.get("bits", 16)
        self.g_lsb = 2 * self.g_fs / 2 ** bits
        self.a_lsb = 2 * self.a_fs / 2 ** bits
        self.next_t = 0.0
        self.saturated = 0

    def sample(self, gyro_true, acc_true, gyro_vib=0.0, acc_vib=0.0) -> tuple[np.ndarray, np.ndarray]:
        dt = 1.0 / self.rate
        bg = self.g_bias0 + self.g_walk.step(dt)
        ba = self.a_bias0 + self.a_walk.step(dt)
        g = self.Mg @ (np.asarray(gyro_true) + gyro_vib) + bg + self.g_sig * self.rng.standard_normal(3)
        a = self.Ma @ (np.asarray(acc_true) + acc_vib) + ba + self.a_sig * self.rng.standard_normal(3)
        if np.any(np.abs(g) > self.g_fs) or np.any(np.abs(a) > self.a_fs):
            self.saturated += 1
        g = np.round(np.clip(g, -self.g_fs, self.g_fs) / self.g_lsb) * self.g_lsb
        a = np.round(np.clip(a, -self.a_fs, self.a_fs) / self.a_lsb) * self.a_lsb
        return g, a


class RotorVibration:
    """Vibration at the IMU from the rotors, the dominant real-world IMU disturbance.

    Per rotor i spinning at n_i [rev/s]:
      * residual prop/motor imbalance U [g mm] -> rotating in-plane force U (2 pi n)^2 at 1x n
      * blade-pass thrust ripple (fraction of thrust) -> vertical force at blades x n
    Each force reaches the IMU through: the arm (resonant at the arm's first bending mode,
    lightly damped carbon), the flight-controller soft mount (isolator), and the IMU's
    anti-alias filter. Linear acceleration = F/m, angular rate = I^-1 (r x integral F dt).
    The arm mode comes from the structural sizing, so a floppy arm *shows up* in the data.
    """

    def __init__(self, vib: dict, aaf_hz: float, rng: np.random.Generator) -> None:
        self.r = np.asarray(vib["rotor_pos"], float)            # rotor hub rel. CoM, body frame
        self.spin = np.asarray(vib["spin"], float)
        self.m = float(vib["mass"])
        self.Iinv = np.linalg.inv(np.asarray(vib["inertia"], float))
        self.U = vib["imbalance_gmm"] * 1e-6                   # kg m
        self.ripple = vib["thrust_ripple"]
        self.blades = vib["blades"]
        self.arm_hz, self.arm_zeta = vib["arm_hz"], vib.get("arm_zeta", 0.02)
        self.mount_hz, self.mount_zeta = vib["mount_hz"], vib["mount_zeta"]
        self.aaf_hz = aaf_hz
        n = len(self.r)
        self.phase = rng.uniform(0, 2 * math.pi, n)             # 1x phase per rotor
        self.rip_off = rng.uniform(0, 2 * math.pi, n)
        self.n = np.zeros(n)
        self.T = np.zeros(n)
        self.t0 = 0.0

    def set_rotors(self, t: float, n_revs, thrust) -> None:
        """Advance phases to time t at the previous speeds, then latch new speeds/thrusts.
        Called once per physics step; the per-sample evaluation in at() is then cheap."""
        self.phase = self.phase + 2 * math.pi * self.n * (t - self.t0)
        self.t0, self.n, self.T = t, np.asarray(n_revs, float), np.asarray(thrust, float)
        self.w = 2 * math.pi * np.maximum(self.n, 1e-3)
        self.A = self.U * self.w ** 2 * self.gain(self.n)                 # 1x imbalance force
        self.Ar = self.ripple * self.T * self.gain(self.blades * self.n)  # blade-pass ripple

    def gain(self, f):
        return (transmissibility(f, self.arm_hz, self.arm_zeta)
                * transmissibility(f, self.mount_hz, self.mount_zeta) * lowpass_gain(f, self.aaf_hz))

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """(gyro_vib [rad/s], acc_vib [m/s^2]) in body frame at time t."""
        if not hasattr(self, "A"):
            self.set_rotors(t, self.n, self.T)
        ph = self.phase + self.w * (t - self.t0)
        c, s = np.cos(ph), np.sin(ph)
        pb = self.blades * ph + self.rip_off
        # forces (rotating in-plane imbalance + vertical ripple) and their time integrals
        Fx, Fy, Fz = self.A * c, self.A * s * self.spin, self.Ar * np.cos(pb)
        Ix, Iy = self.A / self.w * s, -self.A / self.w * self.spin * c
        Iz = self.Ar * np.sin(pb) / (self.blades * self.w)
        acc = np.array([Fx.sum(), Fy.sum(), Fz.sum()]) / self.m
        rx, ry, rz = self.r[:, 0], self.r[:, 1], self.r[:, 2]
        tau_int = np.array([(ry * Iz - rz * Iy).sum(), (rz * Ix - rx * Iz).sum(), (rx * Iy - ry * Ix).sum()])
        return self.Iinv @ tau_int, acc


# ============================================================================== baro ===
class Baro:
    """Barometric altimeter: white noise + slow drift (temperature/weather) + static-port
    error from airspeed (dynamic pressure on the port: dh = -cp v^2 / 2g)."""

    def __init__(self, spec: dict, rng: np.random.Generator, rho: float = 1.2) -> None:
        self.spec, self.rng = spec, rng
        self.rate = float(spec["rate_hz"])
        k = 1.0 / (rho * G0)                                     # m per Pa
        self.sig = spec["noise_pa"] * k
        self.lsb = spec.get("lsb_pa", 0.016) * k
        self.drift = GaussMarkov(spec["drift_pa"] * k, spec.get("drift_tau_s", 300.0), rng)
        self.offset = rng.normal(0, spec["offset_pa"] * k)
        self.cp = spec.get("static_cp", 0.3)

    def sample(self, h_true: float, airspeed: float) -> float:
        h = (h_true + self.offset + self.drift.step(1 / self.rate)
             - self.cp * airspeed ** 2 / (2 * G0) + self.sig * self.rng.standard_normal())
        return float(np.round(h / self.lsb) * self.lsb)


# ======================================================================= magnetometer ===
class Mag:
    """Magnetometer: earth field rotated into body + residual hard/soft iron + field from the
    battery/motor current (a classic real-world compass error) + white noise."""

    def __init__(self, spec: dict, env: dict, rng: np.random.Generator) -> None:
        self.spec, self.rng = spec, rng
        self.rate = float(spec["rate_hz"])
        B, inc, dec = env["field_ut"], env["inclination_deg"] * DEG, env["declination_deg"] * DEG
        # world frame: x east, y north, z up
        self.b_world = B * np.array([math.cos(inc) * math.sin(dec), math.cos(inc) * math.cos(dec), -math.sin(inc)])
        self.hard = rng.normal(0, spec["hard_iron_ut"], 3)
        self.soft = np.eye(3) + rng.normal(0, spec["soft_iron_pct"] / 100, (3, 3))
        d = rng.normal(size=3)
        self.k_current = spec.get("current_ut_per_a", 0.0) * d / np.linalg.norm(d)
        self.sig = spec["noise_ut"]

    def sample(self, R_world_body: np.ndarray, current_a: float) -> np.ndarray:
        b = R_world_body.T @ self.b_world
        return self.soft @ b + self.hard + self.k_current * current_a + self.sig * self.rng.standard_normal(3)


# ============================================================================== GNSS ===
def sky_constellation(n: int, rng: np.random.Generator, min_el_deg: float = 10.0) -> np.ndarray:
    """n satellite line-of-sight unit vectors (world, z up), elevation >= min_el."""
    az = rng.uniform(0, 2 * math.pi, n)
    # uniform over the sky cap above min_el (equal solid angle)
    s0 = math.sin(min_el_deg * DEG)
    el = np.arcsin(rng.uniform(s0, 1.0, n))
    return np.stack([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], 1)


def dops(u: np.ndarray) -> tuple[float, float]:
    """(HDOP, VDOP) from visible line-of-sight unit vectors (rows)."""
    if len(u) < 4:
        return math.inf, math.inf
    H = np.hstack([-u, np.ones((len(u), 1))])
    try:
        Q = np.linalg.inv(H.T @ H)
    except np.linalg.LinAlgError:
        return math.inf, math.inf
    return float(math.sqrt(max(Q[0, 0] + Q[1, 1], 0))), float(math.sqrt(max(Q[2, 2], 0)))


class Gnss:
    """GNSS receiver whose accuracy follows what the antenna can actually see.

    `visibility(u)` (supplied by the runtime: ray casts into the scene) returns, per
    satellite, the probability its signal survives (1 open sky, ~0.3 through a crown,
    0 behind a trunk or hill). Per epoch: draw which satellites are tracked, compute real
    HDOP/VDOP from their geometry; no fix with < min_sats. Error = slow atmospheric bias
    (Gauss-Markov, scaled by DOP) + multipath (grows with blocked sky) + white noise.
    Open-sky CEP50 from the datasheet calibrates the range-error scale.
    """

    def __init__(self, spec: dict, rng: np.random.Generator) -> None:
        self.spec, self.rng = spec, rng
        self.rate = float(spec["rate_hz"])
        self.sats = sky_constellation(spec["sats_open_sky"], rng, spec.get("min_elevation_deg", 10))
        self.h_open, self.v_open = dops(self.sats)
        sig_axis = spec["cep_m"] / 1.1774                          # CEP50 -> per-axis sigma
        self.bias = GaussMarkov(sig_axis * 0.9, spec.get("bias_tau_s", 60.0), rng, 3)
        self.white = sig_axis * 0.4
        self.mp = GaussMarkov(spec["multipath_m"], spec.get("multipath_tau_s", 8.0), rng, 3)
        self.vel_sig = spec["vel_acc_mps"]
        self.min_sats = spec.get("min_sats", 5)
        self.az_rate = 2 * math.pi / (12 * 3600)                 # constellations drift ~half a turn / 12 h

    def sats_at(self, t: float) -> np.ndarray:
        a = self.az_rate * t
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return self.sats @ R.T

    def sample(self, t: float, pos_true, vel_true, survive: np.ndarray) -> dict:
        u = self.sats_at(t)
        tracked = self.rng.random(len(u)) < survive
        n = int(tracked.sum())
        blocked = 1.0 - float(np.mean(survive))
        hd, vd = dops(u[tracked])
        dt = 1 / self.rate
        b, mp = self.bias.step(dt), self.mp.step(dt)
        if n < self.min_sats or not math.isfinite(hd) or hd > 20:
            return dict(fix=False, n_sats=n, hdop=hd, blocked=blocked)
        kh, kv = hd / self.h_open, vd / self.v_open
        k = np.array([kh, kh, kv])
        err = k * (b + self.white * self.rng.standard_normal(3)) + blocked * mp
        vel = np.asarray(vel_true) + k * self.vel_sig * self.rng.standard_normal(3)
        return dict(fix=True, n_sats=n, hdop=hd, blocked=blocked, pos=np.asarray(pos_true) + err, vel=vel)


# =========================================================================== cameras ===
def focal_px(fov_deg: float, pixels: int) -> float:
    return pixels / 2 / math.tan(fov_deg * DEG / 2)


def _box_mean(x: np.ndarray, k: int) -> np.ndarray:
    p = k // 2
    xp = np.pad(x, p, mode="edge")
    c = np.cumsum(np.cumsum(xp, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


class StereoDepth:
    """Active/passive stereo depth camera error model applied to a perfect depth image.

    disparity d = f B / z (in native pixels) is perturbed by sub-pixel matching noise and
    quantised to the sub-pixel step, so the depth error grows as z^2 / (f B), exactly like
    real stereo. Pixels are invalid (0) outside [min, max], where the left/right check
    fails near depth discontinuities (occlusion band ~ disparity jump wide), and where the
    image has too little texture for block matching (optional, needs the grey image).
    """

    def __init__(self, spec: dict, rng: np.random.Generator) -> None:
        self.spec, self.rng = spec, rng
        self.f_native = focal_px(spec["hfov_deg"], spec["width"])
        self.B = spec["baseline_m"]

    def apply(self, z: np.ndarray, gray: np.ndarray | None = None) -> np.ndarray:
        s = self.spec
        w = z.shape[1]
        k = w / s["width"]                                       # render px per native px
        f = self.f_native * k
        zc = np.where(z > 1e-6, z, np.inf)
        disp = f * self.B / zc                                    # render pixels
        sig = s["subpixel_sigma_px"] * k
        step = k / 2 ** s.get("subpixel_bits", 3)
        dn = disp + sig * self.rng.standard_normal(disp.shape)
        dn = np.round(dn / step) * step
        out = np.where(dn > 0, f * self.B / np.maximum(dn, 1e-9), 0.0)
        bad = (z < s["min_m"]) | (z > s["max_m"]) | ~np.isfinite(zc) | (dn <= 0)
        if s.get("lr_check", True):
            # occlusion: pixels just right of a near->far edge are unseen by the right camera
            jump = np.zeros_like(disp)
            jump[:, 1:] = disp[:, :-1] - disp[:, 1:]
            jump = np.where(np.isfinite(jump), jump, 0.0)
            occ = np.zeros_like(bad)
            width = np.minimum(np.ceil(jump), 64).astype(int)
            rows, cols = np.nonzero(width > 1)
            for r, c0, wd in zip(rows, cols, width[rows, cols]):
                occ[r, c0:c0 + wd] = True
            bad |= occ
        if gray is not None and s.get("min_texture", 0) > 0:
            g = gray.astype(float) / 255.0
            win = max(3, int(round(s.get("match_window", 7) * k)) | 1)
            var = _box_mean(g * g, win) - _box_mean(g, win) ** 2
            bad |= np.sqrt(np.maximum(var, 0)) < s["min_texture"]
        out[bad] = 0.0
        return out


class RgbCamera:
    """Photometric + motion model applied to a rendered (ideal pinhole, instant) image.

    * auto-exposure: exposure set to reach target brightness, clipped to the sensor range;
      when it clips (dark forest), analogue gain rises -> more noise
    * motion blur from body rotation during the exposure (line kernel, f * omega * t_exp)
    * rolling shutter: each row is read `readout_ms / H` later -> rows shift with rotation
    * shot + read noise, vignetting, 8-bit quantisation
    `omega_cam` is the angular velocity in the camera frame (x right, y up, z backward,
    MuJoCo convention).
    """

    def __init__(self, spec: dict, rng: np.random.Generator) -> None:
        self.spec, self.rng = spec, rng
        self.exposure = spec["exposure_ms"][1] / 1000 / 4
        self.gain = 1.0

    def apply(self, img: np.ndarray, omega_cam) -> np.ndarray:
        s = self.spec
        h, w = img.shape[:2]
        f = focal_px(s["hfov_deg"], w)
        x = img.astype(float) / 255.0
        # --- auto exposure (renderer output ~= scene radiance at reference exposure)
        e_ref = s.get("exposure_ref_ms", 4.0) / 1000
        lum = max(float(x.mean()), 1e-3)
        want = e_ref * s.get("ae_target", 0.45) / lum
        lo, hi = s["exposure_ms"][0] / 1000, s["exposure_ms"][1] / 1000
        self.exposure = float(np.clip(want, lo, hi))
        self.gain = float(np.clip(want / self.exposure, 1.0, s.get("max_gain", 16.0)))
        x = x * (self.exposure / e_ref) * self.gain
        # --- motion blur: image-plane velocity from rotation (px/s)
        wx, wy = float(omega_cam[0]), float(omega_cam[1])
        vx, vy = -f * wy, f * wx                                  # yaw -> horizontal, pitch -> vertical
        L = math.hypot(vx, vy) * self.exposure
        if L >= 1.0:
            nk = int(min(math.ceil(L), 40))
            acc = np.zeros_like(x)
            for i in range(nk):
                fr = (i / max(nk - 1, 1)) - 0.5
                acc += np.roll(x, (int(round(fr * vy * self.exposure)), int(round(fr * vx * self.exposure))), (0, 1))
            x = acc / nk
        # --- rolling shutter: row r captured r*readout/h later
        if s.get("shutter", "rolling") == "rolling":
            tr = np.arange(h) * s["readout_ms"] / 1000 / h
            dx = np.round(vx * (tr - tr[h // 2])).astype(int)
            dy = np.round(vy * (tr - tr[h // 2])).astype(int)
            cols = (np.arange(w)[None, :] - dx[:, None]) % w
            rows = np.clip(np.arange(h) - dy, 0, h - 1)
            x = x[rows[:, None], cols]
        # --- vignetting (cos^4 law-ish)
        v = s.get("vignetting", 0.0)
        if v:
            yy, xx = np.mgrid[0:h, 0:w]
            r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (f * f)
            x = x * (1 - v * (1 - 1 / (1 + r2) ** 2))[..., None]
        # --- sensor noise in DN (0..255): shot noise ~ sqrt(signal), read noise x gain
        dn = np.clip(x, 0, 1) * 255.0
        shot = np.sqrt(dn * s.get("shot_noise_dn", 0.35) * self.gain)
        sig = np.sqrt(shot ** 2 + (s.get("read_noise_dn", 1.0) * self.gain) ** 2)
        dn = dn + sig * self.rng.standard_normal(dn.shape)
        return np.clip(np.round(dn), 0, 255).astype(np.uint8)


# =========================================================================== compute ===
class ComputeLatency:
    """Policy compute: observations taken at t produce an action usable at t + inference.
    The policy runs at `rate_hz`; between outputs the last action is held (zero-order hold).
    For RL: env.step() submits the obs time + action, and the plant only ever sees
    `current(t)` — the policy never acts on the instant it observed."""

    def __init__(self, spec: dict, rng: np.random.Generator) -> None:
        self.rate = float(spec["rate_hz"])
        self.lat = latency_sampler(spec["inference_ms"], rng)
        self.line = DelayLine(64)
        self.held = None

    def submit(self, t_obs: float, action) -> None:
        self.line.push(t_obs, np.asarray(action, float), self.lat())

    def current(self, t: float):
        last = self.line.latest(t)
        if last is not None:
            self.held = last[1]
        return self.held
