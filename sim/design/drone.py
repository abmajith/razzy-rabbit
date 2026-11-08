"""Quadrotor sizing (exhaustive search over parts) and MJCF generation.

The equations live in sim/physics (propeller, dc_motor, battery, beam); this file is the
sizing logic: per rotor at thrust T the motor + prop operating point gives current and
voltage (physics.propeller.rotor_point), the bus draws 4 V I / eta_esc + avionics from the
pack (physics.battery), and the max thrust is where the motor needs more voltage than the
ESC can give, exceeds its current limit, or the pack its C-rating.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field


from ..physics import beam, dc_motor, propeller
from ..physics.battery import Pack, soc_at_cutoff
from ..physics.propeller import rotor_point
from .common import components, fmt_table, load_toml, select

IN = propeller.INCH


# ------------------------------------------------------------------------- physics ---
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
    D = propeller.diameter_m(prop)
    L = max(D * (1 + ds["prop_clearance"]) / math.sqrt(2), ds["min_arm_m"])
    cage_gap = 0.012
    span = 2 * L + D + (2 * (cage_gap + 0.004) if mis["cage"] else 0.0)

    # structure: thin-walled tube cantilever
    od, idd = tube["od_mm"] / 1000, tube["id_mm"] / 1000
    E = tube["e_gpa"] * 1e9
    A, Ia = beam.round_tube(od, idd)
    m_arm = tube["density"] * A * L
    m_motor, m_prop = motor["mass_g"] / 1000, prop["mass_g"] / 1000
    m_tip = m_motor + m_prop + fr["motor_mount_g"] / 1000
    f_arm = beam.cantilever_first_mode_hz(E, Ia, L, m_tip, m_arm)

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
    deflect = beam.cantilever_tip_deflection(T_max_nom, L, E, Ia)

    # motor/prop speed loop time constant (physics.dc_motor docstring)
    Kt = dc_motor.torque_constant(motor["kv"])
    r_rotor = 0.5 * motor["stator_mm"] / 1000 + 0.003
    J = dc_motor.rotor_inertia(m_motor, r_rotor) + propeller.inertia(m_prop, D)
    kq = propeller.drag_torque_coef(D, prop["cp"], rho)
    w_h = float(op_h["w"])
    tau_motor = dc_motor.speed_loop_time_constant(J, kq, w_h, Kt, motor["r_ohm"])

    d.m = dict(mass_kg=mass, span_m=span, arm_m=L, hover_w=p_hover, hover_a=i_h,
               endurance_min=endurance, twr_nom=4 * T_max_nom / (mass * g),
               twr_end=4 * T_max_end / (mass * g), t_max_n=T_max_nom, t_hover_n=T_hover,
               hover_throttle=T_hover / max(T_max_nom, 1e-9), hover_rpm=float(op_h["n"]) * 60,
               arm_freq_hz=f_arm, arm_deflect_frac=deflect / L, max_motor_a=float(rotor_point(T_max_nom, motor, prop, rho)["I"]),
               tau_motor_s=tau_motor, k_yaw=propeller.yaw_moment_coef(D, prop["ct"], prop["cp"]),
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
def to_mjcf(b, name: str, pos, design: DroneDesign, cfg: dict, chase_offset=(-3.0, -1.8, 1.4)) -> dict:
    """Write the designed quadrotor into a scene Builder (assembled from components by
    sim/robots/templates.py); returns the robot meta (incl. plant and sensor specs)."""
    from ..robots.templates import quadrotor
    robot, meta = quadrotor(name, cfg, design, chase_offset=chase_offset)
    robot.write(b, pos)
    b.meta.setdefault("robots", []).append(meta)
    b.meta["uncertainty"] = dict(cfg.get("uncertainty", {}))   # drawn per episode by Sim(variant=k)
    return meta
