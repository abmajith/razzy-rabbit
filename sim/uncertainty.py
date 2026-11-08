"""Uncertainty model: how well do we know each number of the vehicle spec?

The mission file's `[uncertainty]` table gives a range per parameter. Every episode
("variant") draws one plausible vehicle from those ranges: slightly different mass, centre
of mass, thrust, motor lag, battery, sensor noise, latency, wind... Training on many
variants makes a policy robust to what we don't know (domain randomisation). Measuring
the real vehicle shrinks the ranges.

Ranges (uniform distributions):
  kind  "frac"   a -> factor in [1-a, 1+a]     [lo, hi] -> factor in [lo, hi]
        "frac3"  as frac, independently per axis
        "fracN"  as frac, independently per motor/actuator
        "add"    a -> offset in [-a, a]        [lo, hi] -> offset in [lo, hi]
        "abs"    [lo, hi] -> value             number -> fixed value
        "ball"   r -> vector uniform in a ball of radius r     [r_xy, r_z] -> disk x slab
Keys starting with `require_` are feasibility limits: a draw that violates them is
rejected and redrawn (a real build would not fly / drive), the rejection rate is reported.

Who knows what: the *plant* (MuJoCo model, battery, sensors, wind) gets the drawn values;
the *controllers* are built against the nominal model, like a real flight controller that
only knows the designed mass and inertia.
"""
from __future__ import annotations

import copy

import mujoco
import numpy as np

# key: (kind, applies to, description)
PARAMS: dict[str, tuple[str, str, str]] = {
    # --- rigid body ------------------------------------------------------------------
    "mass_frac":          ("frac",  "any",   "main body mass (inertia scales with it)"),
    "inertia_frac":       ("frac3", "any",   "extra per-axis inertia factor"),
    "com_offset_mm":      ("ball",  "any",   "centre-of-mass offset [mm]"),
    "friction":           ("frac",  "any",   "sliding friction of every contact"),
    # --- energy ----------------------------------------------------------------------
    "battery_capacity":   ("frac",  "any",   "usable capacity (cell ageing, temperature)"),
    "battery_resistance": ("frac",  "any",   "pack internal resistance"),
    # --- drone propulsion / aero -------------------------------------------------------
    "thrust_coef":        ("frac",  "drone", "thrust per command, all rotors (prop ct, air density)"),
    "motor_gain":         ("fracN", "drone", "per-rotor thrust mismatch"),
    "yaw_coef":           ("frac",  "drone", "rotor reaction torque per thrust"),
    "motor_lag":          ("frac",  "drone", "motor time constant"),
    "rotor_drag":         ("frac",  "drone", "rotor (blade-flapping) drag"),
    "body_drag":          ("frac",  "drone", "body drag area CdA"),
    "prop_imbalance":     ("frac",  "drone", "prop/motor imbalance -> vibration"),
    "imu_mount_hz":       ("frac",  "drone", "FC soft-mount natural frequency"),
    "arm_hz":             ("frac",  "drone", "arm first bending mode (beam model vs real)"),
    # --- rover drive -------------------------------------------------------------------
    "drive_torque":       ("frac",  "rover", "stall torque, all drive motors"),
    "drive_gain":         ("fracN", "rover", "per-motor torque mismatch"),
    "steer_kp":           ("frac",  "rover", "steering servo stiffness"),
    # --- sensors / compute ----------------------------------------------------------------
    "imu_noise":          ("frac",  "any",   "IMU noise densities"),
    "imu_bias":           ("frac",  "any",   "IMU residual biases and bias instability"),
    "baro_noise":         ("frac",  "any",   "barometer noise"),
    "baro_drift":         ("frac",  "any",   "barometer drift"),
    "mag_current":        ("frac",  "any",   "compass field per amp of battery current"),
    "gnss_foliage":       ("abs",   "any",   "GNSS loss probability per leaf layer"),
    "gnss_multipath":     ("frac",  "any",   "GNSS multipath error under blocked sky"),
    "latency_ms_add":     ("add",   "any",   "added to every sensor latency [ms]"),
    "compute_ms_add":     ("add",   "any",   "added to policy inference time [ms]"),
    # --- environment -------------------------------------------------------------------
    "wind_speed":         ("abs",   "any",   "mean wind speed [m/s]"),
    "wind_heading_deg":   ("abs",   "any",   "wind direction [deg]"),
    "gust":               ("frac",  "any",   "gust / turbulence intensity"),
}
REQUIRE = {
    "require_twr_end":           "drone: thrust/weight at end of discharge must stay >= this",
    "require_hover_throttle":    "drone: hover thrust / max thrust must stay <= this",
    "require_slope_torque_frac": "rover: slope torque / stall torque must stay <= this",
    "require_tip_deg":           "rover: lateral tip-over angle must stay >= this",
}


def vehicle_kind(robot: dict) -> str:
    return {"drone": "drone", "rover": "rover"}.get(robot.get("plant", {}).get("kind"), "other")


def check_config(unc: dict, kind: str) -> None:
    for k in unc:
        if k == "max_attempts" or k in REQUIRE:
            continue
        if k not in PARAMS:
            raise SystemExit(f"[uncertainty] unknown key '{k}'. Known: {sorted(PARAMS)} + {sorted(REQUIRE)}")
        target = PARAMS[k][1]
        if target not in ("any", kind):
            raise SystemExit(f"[uncertainty] '{k}' applies to {target} vehicles, not {kind}")


def _n_actuators(robot: dict, kind: str) -> int:
    return len(robot["actuators"]) if kind == "drone" else len(robot.get("drive", []))


def _draw(kind: str, spec, rng: np.random.Generator, n: int = 1):
    if kind in ("frac", "frac3", "fracN"):
        lo, hi = (spec if isinstance(spec, list) else (1 - spec, 1 + spec))
        size = {"frac": None, "frac3": 3, "fracN": n}[kind]
        return rng.uniform(lo, hi, size)
    if kind == "add":
        lo, hi = (spec if isinstance(spec, list) else (-spec, spec))
        return float(rng.uniform(lo, hi))
    if kind == "abs":
        return float(rng.uniform(*spec)) if isinstance(spec, list) else float(spec)
    if kind == "ball":
        if isinstance(spec, list):                         # disk in xy x uniform z
            r = spec[0] * np.sqrt(rng.random())
            a = rng.uniform(0, 2 * np.pi)
            return np.array([r * np.cos(a), r * np.sin(a), rng.uniform(-spec[1], spec[1])]) / 1000
        v = rng.normal(size=3)
        return v / np.linalg.norm(v) * spec * rng.random() ** (1 / 3) / 1000
    raise ValueError(kind)


def feasibility(params: dict, robot: dict, unc: dict, kind: str) -> list[str]:
    """First-order checks of a drawn vehicle against the `require_*` limits."""
    met = robot.get("design", {}).get("metrics", {})
    bad = []
    mass = params.get("mass_frac", 1.0)
    if kind == "drone" and met:
        gains = params.get("thrust_coef", 1.0) * np.asarray(params.get("motor_gain", [1.0]))
        twr = met["twr_end"] * float(gains.min()) / mass          # weakest rotor limits authority
        thr = met["hover_throttle"] * mass / float(gains.mean())
        if twr < unc.get("require_twr_end", 0):
            bad.append(f"twr_end {twr:.2f} < {unc['require_twr_end']}")
        if thr > unc.get("require_hover_throttle", 1e9):
            bad.append(f"hover throttle {thr:.2f} > {unc['require_hover_throttle']}")
    if kind == "rover" and met:
        tq = params.get("drive_torque", 1.0) * float(np.min(params.get("drive_gain", [1.0])))
        frac = met["tau_req_slope"] / (met["tau_stall"] * tq) * mass
        if frac > unc.get("require_slope_torque_frac", 1e9):
            bad.append(f"slope torque {frac:.2f} of stall > {unc['require_slope_torque_frac']}")
        dy = abs(params.get("com_offset_mm", np.zeros(3))[1])
        dz = params.get("com_offset_mm", np.zeros(3))[2]
        tip = np.degrees(np.arctan2(met["track_m"] / 2 - dy, met["h_cg"] + dz))
        if tip < unc.get("require_tip_deg", -1e9):
            bad.append(f"tip-over {tip:.1f} deg < {unc['require_tip_deg']}")
    return bad


def sample(unc: dict, robot: dict, rng: np.random.Generator) -> tuple[dict, int]:
    """Draw one feasible variant -> (params, number of rejected draws)."""
    kind = vehicle_kind(robot)
    check_config(unc, kind)
    n = _n_actuators(robot, kind)
    rejected = 0
    for _ in range(int(unc.get("max_attempts", 50))):
        p = {k: _draw(PARAMS[k][0], v, rng, n) for k, v in unc.items() if k in PARAMS}
        if not feasibility(p, robot, unc, kind):
            return p, rejected
        rejected += 1
    raise SystemExit(f"[uncertainty] no feasible draw in {rejected} attempts: the ranges make this "
                     f"design infeasible too often (last: {feasibility(p, robot, unc, kind)})")


def flatten(params: dict) -> tuple[list[str], np.ndarray]:
    """Fixed-order flat vector (what a C++ core / policy conditioning would consume)."""
    names, vals = [], []
    for k in sorted(params):
        v = np.atleast_1d(np.asarray(params[k], float))
        if v.size == 1:
            names.append(k)
        else:
            sfx = "xyz" if PARAMS[k][0] in ("ball", "frac3") else range(v.size)
            names += [f"{k}_{s}" for s in sfx]
        vals += list(v)
    return names, np.array(vals)


# ------------------------------------------------------------------------- applying ---
def apply_meta(meta: dict, params: dict) -> dict:
    """Randomise the scene meta (plant, sensors, wind, drag). Returns a modified deep copy."""
    meta = copy.deepcopy(meta)
    r = meta["robots"][0]
    g = params.get
    p = r.get("plant")
    if p is not None:
        b = p["battery"]
        b["capacity_mah"] *= g("battery_capacity", 1.0)
        b["r_cell"] *= g("battery_resistance", 1.0)
        if p["kind"] == "drone":
            p["rotor_drag"] *= g("rotor_drag", 1.0)
        if p["kind"] == "rover":
            p["tau_stall"] = (p["tau_stall"] * g("drive_torque", 1.0)
                              * np.asarray(g("drive_gain", np.ones(len(r["drive"]))))).tolist()
    if "drag_CdA" in r:
        r["drag_CdA"] *= g("body_drag", 1.0)
    v = r.get("vibration")
    if v is not None:
        v["imbalance_gmm"] *= g("prop_imbalance", 1.0)
        v["mount_hz"] *= g("imu_mount_hz", 1.0)
        v["arm_hz"] *= g("arm_hz", 1.0)
        v["mass"] *= g("mass_frac", 1.0)
    s = r.get("sensors")
    if s is not None:
        if "imu" in s:
            for k in ("gyro_noise_dps_rthz", "accel_noise_ug_rthz"):
                s["imu"][k] *= g("imu_noise", 1.0)
            for k in ("gyro_bias_dps", "accel_bias_mg", "gyro_bias_instability_dph", "accel_bias_instability_ug"):
                s["imu"][k] *= g("imu_bias", 1.0)
        if "baro" in s:
            s["baro"]["noise_pa"] *= g("baro_noise", 1.0)
            s["baro"]["drift_pa"] *= g("baro_drift", 1.0)
        if "mag" in s:
            s["mag"]["current_ut_per_a"] = s["mag"].get("current_ut_per_a", 0.0) * g("mag_current", 1.0)
        if "gnss" in s:
            if "gnss_foliage" in params:
                s["gnss"]["foliage_loss"] = params["gnss_foliage"]
            s["gnss"]["multipath_m"] *= g("gnss_multipath", 1.0)
        add = g("latency_ms_add", 0.0)
        for spec in s.values():
            if isinstance(spec, dict) and "latency_ms" in spec:
                spec["latency_ms"] = _shift(spec["latency_ms"], add)
        if "compute" in s:
            s["compute"]["inference_ms"] = _shift(s["compute"]["inference_ms"], g("compute_ms_add", 0.0))
    w = meta.get("wind")
    if w is not None:
        if "wind_speed" in params:
            w["U"] = params["wind_speed"]
        if "wind_heading_deg" in params:
            w["heading_deg"] = params["wind_heading_deg"]
        w["gust"] *= g("gust", 1.0)
    return meta


def _shift(lat, add: float):
    if isinstance(lat, list):
        return [max(0.0, x + add) for x in lat]
    return max(0.0, lat + add)


# everything the applier randomises + what the plants change live (voltage sag rescales
# ctrlrange / forcerange), so a reset starts from exactly the nominal model
MODEL_FIELDS = ("body_mass", "body_inertia", "body_ipos", "geom_friction", "actuator_gainprm",
                "actuator_biasprm", "actuator_dynprm", "actuator_gear", "actuator_ctrlrange",
                "actuator_forcerange")


class ModelSnapshot:
    """Nominal copies of every model field the applier touches, so variants never stack."""

    def __init__(self, m: mujoco.MjModel) -> None:
        self.v = {f: getattr(m, f).copy() for f in MODEL_FIELDS}

    def restore(self, m: mujoco.MjModel) -> None:
        for f, v in self.v.items():
            getattr(m, f)[:] = v


def apply_model(m: mujoco.MjModel, d: mujoco.MjData, robot: dict, params: dict) -> None:
    """Randomise the MuJoCo model in place (call on a restored nominal model)."""
    g = params.get
    kind = vehicle_kind(robot)
    bid = m.body(robot["body"]).id
    s = g("mass_frac", 1.0)
    m.body_mass[bid] *= s
    m.body_inertia[bid] *= s * np.asarray(g("inertia_frac", np.ones(3)))
    m.body_ipos[bid] += np.asarray(g("com_offset_mm", np.zeros(3)))
    m.geom_friction[:, 0] *= g("friction", 1.0)
    if kind == "drone":
        act = [m.actuator(a).id for a in robot["actuators"]]
        gain = g("thrust_coef", 1.0) * np.asarray(g("motor_gain", np.ones(len(act))))
        m.actuator_gainprm[act, 0] *= gain
        m.actuator_dynprm[act, 0] *= g("motor_lag", 1.0)
        m.actuator_gear[act, 5] *= g("yaw_coef", 1.0)
    if kind == "rover":
        act = [m.actuator(a).id for a in robot["drive"]]
        tq = g("drive_torque", 1.0) * np.asarray(g("drive_gain", np.ones(len(act))))
        m.actuator_gainprm[act, 0] *= tq                   # (the plant rescales this live)
        m.actuator_biasprm[act, 2] *= tq                   # back-EMF slope follows the motor constant
        st = [m.actuator(a).id for a in robot["steer"]]
        m.actuator_gainprm[st, 0] *= g("steer_kp", 1.0)
        m.actuator_biasprm[st, 1] *= g("steer_kp", 1.0)
    mujoco.mj_setConst(m, d)
