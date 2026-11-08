"""Quadrotor sizing (physics + exhaustive search) and MJCF generation.

Model chain for one motor at thrust T (static, per rotor):
    n   = sqrt(T / (ct rho D^4))                 prop speed [rev/s]
    P_m = cp rho n^3 D^5,  Q = P_m / (2 pi n)    shaft power, torque
    I   = Q / Kt + I0,     Kt = 60 / (2 pi KV)   motor current
    V   = omega Kt + I R                         motor voltage (back-EMF + copper drop)
    P_e = V I / eta_esc                          drawn from the bus
Bus voltage comes from the pack model (OCV(soc) - I R). Max thrust is where the motor
needs more voltage than the ESC can give, or exceeds its current limit.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .common import Pack, components, fmt_table, load_toml, select

IN = 0.0254


# ------------------------------------------------------------------------- physics ---
def rotor_point(T, motor: dict, prop: dict, rho: float) -> dict:
    """Operating point of one motor+prop producing thrust T [N] (vectorised in T)."""
    T = np.maximum(np.asarray(T, float), 0.0)
    D = prop["diameter_in"] * IN
    n = np.sqrt(T / (prop["ct"] * rho * D ** 4))
    w = 2 * np.pi * n
    Pm = prop["cp"] * rho * n ** 3 * D ** 5
    Kt = 60.0 / (2 * np.pi * motor["kv"])
    Q = np.where(w > 0, Pm / np.maximum(w, 1e-9), 0.0)
    I = Q / Kt + motor["i0"]
    V = w * Kt + I * motor["r_ohm"]
    return dict(n=n, w=w, Q=Q, I=I, V=V, Pe=V * I, Pm=Pm)


def bus_state(T_each, n_rot, motor, prop, pack: Pack, soc, rho, esc_eff, p_avionics):
    op = rotor_point(T_each, motor, prop, rho)
    p_bus = n_rot * float(op["Pe"]) / esc_eff + p_avionics
    v_bus, i_bus = pack.terminal(p_bus, soc)
    return op, p_bus, v_bus, i_bus


def max_thrust(motor, prop, pack: Pack, soc, rho, esc_eff, duty, p_avionics, n_rot=4) -> float:
    def ok(T):
        op, _, v_bus, i_bus = bus_state(T, n_rot, motor, prop, pack, soc, rho, esc_eff, p_avionics)
        return op["V"] <= duty * v_bus and op["I"] <= motor["i_max"] and i_bus <= pack.i_max
    lo, hi = 0.0, 400.0
    if not ok(1e-3):
        return 0.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def soc_at_cutoff(pack: Pack) -> float:
    """State of charge where OCV hits the end-of-discharge level used for min TWR."""
    socs = np.linspace(0, 1, 201)
    v = np.array([pack.ocv(s) for s in socs])
    return float(socs[np.argmax(v >= pack.v_cut + 0.1 * pack.cells)])


# -------------------------------------------------------------------------- design ---
@dataclass
class DroneDesign:
    motor: str
    prop: str
    battery: str
    arm_tube: str
    feasible: bool = False
    violations: list[str] = field(default_factory=list)
    m: dict = field(default_factory=dict)          # metrics
    geo: dict = field(default_factory=dict)        # geometry for the generator

    @property
    def label(self) -> str:
        return f"{self.motor} + {self.prop} + {self.battery} + {self.arm_tube}"


def evaluate(cfg: dict, db: dict, motor_name, prop_name, batt_name, tube_name) -> DroneDesign | None:
    mis, st, ds = cfg["mission"], cfg["structure"], cfg["design_space"]
    motor, prop = db["motor"][motor_name], db["prop"][prop_name]
    bspec, tube, fr = db["battery"][batt_name], db["arm_tube"][tube_name], db["frame"]
    # compatibility (hard, not a "violation")
    if not motor["prop_in"][0] <= prop["diameter_in"] <= motor["prop_in"][1]:
        return None
    if not motor["cells"][0] <= bspec["cells"] <= motor["cells"][1]:
        return None
    d = DroneDesign(motor_name, prop_name, batt_name, tube_name)
    pack = Pack(bspec)
    rho, g = mis["air_density"], mis["gravity"]
    esc, duty = fr["esc_efficiency"], fr["esc_max_duty"]
    payload = [db["avionics"][k] for k in cfg["payload"]["items"]]
    p_av = sum(p["power_w"] for p in payload)

    # geometry: adjacent prop tips separated by clearance * D
    D = prop["diameter_in"] * IN
    L = max(D * (1 + ds["prop_clearance"]) / math.sqrt(2), ds["min_arm_m"])
    cage_gap = 0.012
    span = 2 * L + D + (2 * (cage_gap + 0.004) if mis["cage"] else 0.0)

    # structure: thin-walled tube cantilever
    od, idd = tube["od_mm"] / 1000, tube["id_mm"] / 1000
    E = tube["e_gpa"] * 1e9
    Ia = math.pi / 64 * (od ** 4 - idd ** 4)
    A = math.pi / 4 * (od ** 2 - idd ** 2)
    m_arm = tube["density"] * A * L
    m_motor, m_prop = motor["mass_g"] / 1000, prop["mass_g"] / 1000
    m_tip = m_motor + m_prop + fr["motor_mount_g"] / 1000
    f_arm = 1 / (2 * math.pi) * math.sqrt(3 * E * Ia / (L ** 3 * (m_tip + 0.236 * m_arm)))

    # mass budget
    m_cage = 0.0
    if mis["cage"]:
        rod = fr["cage_rod_g_per_m"] / 1000
        ring = 2 * math.pi * (D / 2 + cage_gap)
        m_cage = rod * (4 * 2 * ring + 8 * 0.05 + 4 * 2 * L)
    m_payload = sum(p["mass_g"] for p in payload) / 1000 + cfg["payload"].get("extra_mass_g", 0) / 1000
    m_frame = fr["center_plate_g"] / 1000 + 4 * (m_arm + fr["motor_mount_g"] / 1000) + m_cage
    m_sub = m_frame + 4 * (m_motor + m_prop) + pack.mass + m_payload
    mass = m_sub * (1 + fr["wiring_frac"])

    # performance
    soc_nom, soc_end = 0.5, soc_at_cutoff(pack)
    T_hover = mass * g / 4
    T_max_nom = max_thrust(motor, prop, pack, soc_nom, rho, esc, duty, p_av)
    T_max_end = max_thrust(motor, prop, pack, soc_end, rho, esc, duty, p_av)
    op_h, p_hover, v_h, i_h = bus_state(T_hover, 4, motor, prop, pack, soc_nom, rho, esc, p_av)
    e_use = pack.energy_wh * (1 - mis["reserve_frac"])
    endurance = e_use / (p_hover * mis["maneuver_factor"]) * 60
    deflect = (T_max_nom * L ** 3) / (3 * E * Ia)

    # motor/prop speed loop time constant (linearised):  J dw/dt = Kt^2/R (V/Kt - w) - kq w^2
    Kt = 60.0 / (2 * math.pi * motor["kv"])
    r_rotor = 0.5 * motor["stator_mm"] / 1000 + 0.003
    J = 0.5 * m_motor * r_rotor ** 2 + m_prop * D ** 2 / 24
    kq = prop["cp"] * rho * D ** 5 / (8 * math.pi ** 3)
    w_h = float(op_h["w"])
    tau_motor = J / (2 * kq * w_h + Kt * Kt / motor["r_ohm"]) if w_h > 0 else 0.05

    d.m = dict(mass_kg=mass, span_m=span, arm_m=L, hover_w=p_hover, hover_a=i_h,
               endurance_min=endurance, twr_nom=4 * T_max_nom / (mass * g),
               twr_end=4 * T_max_end / (mass * g), t_max_n=T_max_nom, t_hover_n=T_hover,
               hover_throttle=T_hover / max(T_max_nom, 1e-9), hover_rpm=float(op_h["n"]) * 60,
               arm_freq_hz=f_arm, arm_deflect_frac=deflect / L, max_motor_a=float(rotor_point(T_max_nom, motor, prop, rho)["I"]),
               tau_motor_s=tau_motor, k_yaw=prop["cp"] * D / (2 * math.pi * prop["ct"]),
               g_per_w=T_hover / g * 1000 / (float(op_h["Pe"]) / esc), cage_kg=m_cage, frame_kg=m_frame,
               payload_kg=m_payload, battery_kg=pack.mass, avionics_w=p_av)
    d.geo = dict(D=D, L=L, tube_od=od, cage=mis["cage"], cage_gap=cage_gap, m_arm=m_arm,
                 m_motor=m_motor, m_prop=m_prop, stator=motor["stator_mm"] / 1000,
                 battery_size=[s / 1000 for s in bspec["size_mm"]], payload=cfg["payload"]["items"])

    v = d.violations
    if span > mis["max_span_m"]:
        v.append(f"span {span:.2f} > {mis['max_span_m']} m")
    if d.m["twr_end"] < mis["min_twr"]:
        v.append(f"TWR(end) {d.m['twr_end']:.2f} < {mis['min_twr']}")
    if endurance < mis["min_endurance_min"]:
        v.append(f"endurance {endurance:.1f} < {mis['min_endurance_min']} min")
    if d.m["hover_throttle"] > mis["max_hover_throttle"]:
        v.append(f"hover throttle {d.m['hover_throttle']:.2f} > {mis['max_hover_throttle']}")
    if d.m["arm_deflect_frac"] > st["max_arm_deflection_frac"]:
        v.append(f"arm deflection {d.m['arm_deflect_frac']*100:.2f}% > {st['max_arm_deflection_frac']*100}%")
    if f_arm < st["min_arm_freq_hz"]:
        v.append(f"arm mode {f_arm:.0f} Hz < {st['min_arm_freq_hz']} Hz")
    if T_max_nom <= T_hover:
        v.append("cannot hover")
    d.feasible = not v
    return d


def optimize(config="drone_forest.toml") -> tuple[dict, list[DroneDesign]]:
    cfg = load_toml(config)
    db = components()
    ds = cfg["design_space"]
    combos = itertools.product(select(db, "motor", ds["motors"]), select(db, "prop", ds["props"]),
                               select(db, "battery", ds["batteries"]), select(db, "arm_tube", ds["arm_tubes"]))
    designs = [d for c in combos if (d := evaluate(cfg, db, *c)) is not None]
    key = {"endurance": lambda d: -d.m["endurance_min"], "twr": lambda d: -d.m["twr_end"],
           "min_mass": lambda d: d.m["mass_kg"]}[cfg["objective"]["maximize"]]
    designs.sort(key=lambda d: (not d.feasible, len(d.violations), key(d), d.m["mass_kg"]))
    return cfg, designs


def best(config) -> tuple[dict, DroneDesign]:
    """Best feasible design for a mission file (raises with the reasons if none)."""
    cfg, designs = optimize(config)
    feas = [d for d in designs if d.feasible]
    if not feas:
        raise SystemExit(report(cfg, designs))
    return cfg, feas[0]


def report(cfg, designs, top=8) -> str:
    feas = [d for d in designs if d.feasible]
    cols = [("label", "design (motor + prop + battery + arm)", ""), ("mass_kg", "mass kg", ".3f"),
            ("span_m", "span m", ".2f"), ("endurance_min", "endur min", ".1f"), ("twr_end", "TWR end", ".2f"),
            ("hover_w", "hover W", ".0f"), ("g_per_w", "g/W", ".1f"), ("arm_freq_hz", "arm Hz", ".0f")]
    rows = [dict(label=d.label, **d.m) for d in (feas[:top] if feas else designs[:top])]
    out = [f"== {cfg['mission']['name']}: {len(designs)} compatible combinations, {len(feas)} feasible "
           f"(objective: {cfg['objective']['maximize']})", fmt_table(rows, cols)]
    if not feas:
        out.append("\nNO FEASIBLE DESIGN. Closest candidates violate:")
        for d in designs[:5]:
            out.append(f"  {d.label}: " + "; ".join(d.violations))
    else:
        b = feas[0]
        out.append(f"\nselected: {b.label}")
        out.append(f"  mass {b.m['mass_kg']*1000:.0f} g (battery {b.m['battery_kg']*1000:.0f}, frame "
                   f"{b.m['frame_kg']*1000:.0f} incl. cage {b.m['cage_kg']*1000:.0f}, payload {b.m['payload_kg']*1000:.0f})")
        out.append(f"  hover {b.m['hover_w']:.0f} W @ {b.m['hover_rpm']:.0f} rpm, throttle {b.m['hover_throttle']:.2f}; "
                   f"max thrust {b.m['t_max_n']:.1f} N/motor; motor lag {b.m['tau_motor_s']*1000:.0f} ms")
        out.append(f"  arm {b.m['arm_m']*1000:.0f} mm, deflection {b.m['arm_deflect_frac']*100:.2f}%, "
                   f"first mode {b.m['arm_freq_hz']:.0f} Hz")
    return "\n".join(out)


# ----------------------------------------------------------------------- generator ---
def _box_inertia(m, sx, sy, sz):
    return m / 12 * np.diag([sy * sy + sz * sz, sx * sx + sz * sz, sx * sx + sy * sy])


def to_mjcf(b, name: str, pos, design: DroneDesign, cfg: dict, chase_offset=(-3.0, -1.8, 1.4)) -> dict:
    """Write the designed quadrotor into a Builder; returns robot meta (incl. plant params)."""
    from ..builder import fmt, lookat_xyaxes
    from ..robots import _robot_materials

    _robot_materials(b)
    b.material("cage", [0.15, 0.15, 0.16, 1], specular=0.3)
    db = components()
    g, m = design.geo, design.m
    D, L = g["D"], g["L"]
    parts = []   # (mass, pos, inertia_about_own_com, geom_xml or None)
    geoms = []

    def add(mass, p, I_own, xml):
        parts.append((mass, np.asarray(p, float), I_own))
        if xml:
            geoms.append(xml)

    plate = db["frame"]["center_plate_g"] / 1000
    add(plate, [0, 0, 0], _box_inertia(plate, 0.11, 0.08, 0.02),
        '<geom type="box" size=".055 .04 .01" material="carbon"/>')
    bx, by, bz = g["battery_size"]
    add(m["battery_kg"], [0, 0, 0.012 + bz / 2], _box_inertia(m["battery_kg"], bx, by, bz),
        f'<geom type="box" size="{bx/2:.4f} {by/2:.4f} {bz/2:.4f}" pos="0 0 {0.012 + bz/2:.4f}" material="accent" contype="0" conaffinity="0"/>')
    mounts = {"center": [0, 0, -0.012], "top": [-0.02, 0, 0.014 + bz], "front": [0.06, 0, -0.01],
              "bottom": [-0.03, 0, -0.03]}
    cam_pos = None
    for item in g["payload"]:
        a = db["avionics"][item]
        sx, sy, sz = (s / 1000 for s in a["size_mm"])
        p = np.array(mounts[a["mount"]], float)
        if a["mount"] == "front":
            sx, sy = sy, sx            # long side across the nose
            p[0] = 0.055 + sx / 2
            cam_pos = p + [sx / 2 + 0.002, 0, 0]
        if a["mount"] == "top":
            p[2] = 0.012 + bz + sz / 2
        if a["mount"] in ("center", "bottom"):
            p[2] -= sz / 2
        mat = "lens" if a["mount"] == "front" else "carbon"
        add(a["mass_g"] / 1000, p, _box_inertia(a["mass_g"] / 1000, sx, sy, sz),
            f'<geom type="box" size="{sx/2:.4f} {sy/2:.4f} {sz/2:.4f}" pos="{fmt(p)}" material="{mat}" contype="0" conaffinity="0"/>')
    rotors = [(1, -1, -1), (1, 1, 1), (-1, 1, -1), (-1, -1, 1)]   # FR, FL, RL, RR, spin sign
    r_mot = g["stator"] / 2 + 0.003
    sites = []
    for i, (sx, sy, s) in enumerate(rotors):
        u = np.array([sx, sy, 0]) / math.sqrt(2)
        tip = u * L
        # arm as a thin rod along u
        Irod = g["m_arm"] / 12 * L * L * (np.eye(3) - np.outer(u, u))
        add(g["m_arm"], tip / 2, Irod,
            f'<geom type="capsule" fromto="0 0 0 {tip[0]:.4f} {tip[1]:.4f} 0" size="{g["tube_od"]/2:.4f}" material="carbon" contype="0" conaffinity="0"/>')
        m_mm = g["m_motor"] + db["frame"]["motor_mount_g"] / 1000          # motor + its mount
        add(m_mm, tip + [0, 0, 0.012], _box_inertia(m_mm, 2 * r_mot, 2 * r_mot, 0.02),
            f'<geom type="cylinder" size="{r_mot:.4f} .011" pos="{tip[0]:.4f} {tip[1]:.4f} .012" material="alu" contype="0" conaffinity="0"/>')
        zp = 0.03
        add(g["m_prop"], tip + [0, 0, zp], g["m_prop"] * D * D / 24 * np.diag([0.5, 0.5, 1.0]),
            f'<geom name="{name}_prop{i}" type="cylinder" size="{D/2:.4f} .003" pos="{tip[0]:.4f} {tip[1]:.4f} {zp}" material="prop" contype="0" conaffinity="0"/>')
        sites.append((tip[0], tip[1], zp, s))
        if g["cage"]:
            rr = D / 2 + g["cage_gap"]
            for zc in (zp - 0.02, zp + 0.02):
                n = 10
                for k in range(n):
                    a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
                    p0 = [tip[0] + rr * math.cos(a0), tip[1] + rr * math.sin(a0), zc]
                    p1 = [tip[0] + rr * math.cos(a1), tip[1] + rr * math.sin(a1), zc]
                    geoms.append(f'<geom type="capsule" fromto="{fmt(p0)} {fmt(p1)}" size=".0018" material="cage" contype="0" conaffinity="0"/>')
    if g["cage"]:
        m_cage = m["cage_kg"]
        add(m_cage, [0, 0, 0.03], m_cage * (L + D / 2) ** 2 * np.diag([0.5, 0.5, 1.0]), None)

    # total mass properties (wiring spread like the frame)
    mass = sum(p[0] for p in parts) * (1 + db["frame"]["wiring_frac"])
    scale = mass / sum(p[0] for p in parts)
    com = sum(p[0] * p[1] for p in parts) / sum(p[0] for p in parts)
    I = np.zeros((3, 3))
    for mm, p, Io in parts:
        r = p - com
        I += scale * (Io + mm * (r @ r * np.eye(3) - np.outer(r, r)))
    full = f"{I[0,0]:.6g} {I[1,1]:.6g} {I[2,2]:.6g} {I[0,1]:.6g} {I[0,2]:.6g} {I[1,2]:.6g}"

    # collision: cage envelope (or hull) + landing feet
    half = L / math.sqrt(2) + D / 2 + (g["cage_gap"] if g["cage"] else 0)
    coll = [f'<geom name="{name}_hull" type="box" size="{half:.4f} {half:.4f} .035" pos="0 0 .02" rgba="0 0 0 0" group="3"/>']
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        coll.append(f'<geom type="capsule" fromto="{sx*.04} {sy*.035} -.01 {sx*.05} {sy*.045} -.065" size=".004" material="carbon"/>')

    cam = cam_pos if cam_pos is not None else np.array([0.07, 0, -0.01])
    from ..sensing.suite import resolve
    sensors = resolve(cfg, db)
    inst = sensors["installation"]
    # onboard cameras take their field of view from the chosen parts (MuJoCo fovy = vertical)
    rgb_fovy = sensors.get("rgb_camera", {}).get("vfov_deg", 80)
    if "rgb_camera" in sensors:
        sensors["rgb_camera"]["camera"] = f"{name}_fpv"
    if "depth_camera" in sensors:
        sensors["depth_camera"]["camera"] = f"{name}_depth"
    body = [f'<body name="{name}" pos="{fmt(pos)}">',
            f'  <freejoint name="{name}_root"/>',
            f'  <inertial pos="{fmt(com)}" mass="{mass:.5f}" fullinertia="{full}"/>',
            *["  " + x for x in geoms + coll]]
    tau = max(m["tau_motor_s"], 0.005)
    for i, (x, y, z, s) in enumerate(sites):
        body.append(f'  <site name="{name}_rotor{i}" pos="{x:.4f} {y:.4f} {z}" size=".008"/>')
        b.actuators.append(
            f'<general name="{name}_m{i}" site="{name}_rotor{i}" gear="0 0 1 0 0 {s * m["k_yaw"]:.5f}" '
            f'dyntype="filter" dynprm="{tau:.4f}" gaintype="fixed" gainprm="1" ctrlrange="0 {m["t_max_n"]:.3f}"/>')
    body += [f'  <site name="{name}_imu" pos="{fmt(com)}" size=".008"/>',
             f'  <camera name="{name}_fpv" pos="{fmt(cam)}" xyaxes="0 -1 0 .259 0 .966" fovy="{rgb_fovy}"/>',
             f'  <camera name="{name}_depth" pos="{fmt(cam)}" xyaxes="0 -1 0 .259 0 .966" '
             f'fovy="{sensors.get("depth_camera", {}).get("vfov_deg", 58)}"/>',
             f'  <camera name="{name}_nadir" pos="0 0 -.04" xyaxes="0 -1 0 1 0 0" fovy="70"/>',
             f'  <camera name="{name}_chase" mode="trackcom" pos="{fmt(chase_offset)}" xyaxes="{lookat_xyaxes(chase_offset)}"/>',
             '</body>']
    b.world.append("\n    ".join(body))
    b.sensors += [f'<gyro name="{name}_gyro" site="{name}_imu"/>',
                  f'<accelerometer name="{name}_acc" site="{name}_imu"/>',
                  f'<framepos name="{name}_pos" objtype="body" objname="{name}"/>',
                  f'<framequat name="{name}_quat" objtype="body" objname="{name}"/>']

    mis = cfg["mission"]
    span = m["span_m"]
    meta = dict(type="quadrotor", body=name, mass=mass, actuators=[f"{name}_m{i}" for i in range(4)],
                cameras=[f"{name}_chase", f"{name}_fpv", f"{name}_nadir"],
                drag_CdA=(0.08 * span + 0.008) * (1.3 if g["cage"] else 1.0),
                design=dict(label=design.label, metrics=m),
                plant=dict(kind="drone", motor=db["motor"][design.motor], prop=db["prop"][design.prop],
                           battery=db["battery"][design.battery], rho=mis["air_density"],
                           esc_eff=db["frame"]["esc_efficiency"], esc_duty=db["frame"]["esc_max_duty"],
                           avionics_w=m["avionics_w"], t_max_ref=m["t_max_n"], rotor_drag=0.012),
                sensors=sensors,
                vibration=dict(rotor_pos=[[x - com[0], y - com[1], z - com[2]] for x, y, z, _ in sites],
                               spin=[s for *_, s in sites], mass=mass, inertia=I.tolist(),
                               blades=db["prop"][design.prop].get("blades", 2),
                               imbalance_gmm=inst.get("prop_imbalance_gmm", 0.5),
                               thrust_ripple=inst.get("thrust_ripple", 0.03),
                               arm_hz=m["arm_freq_hz"], arm_zeta=inst.get("arm_zeta", 0.02),
                               mount_hz=inst.get("imu_mount_hz", 1e4), mount_zeta=inst.get("imu_mount_zeta", 0.15)))
    b.meta.setdefault("robots", []).append(meta)
    b.meta["uncertainty"] = dict(cfg.get("uncertainty", {}))   # drawn per episode by Sim(variant=k)
    return meta
