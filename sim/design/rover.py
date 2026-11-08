"""Six-wheel rocker-bogie rover sizing (physics + exhaustive search) and MJCF generation.

Suspension: on each side a *rocker* (pivoting on the body) carries the front wheel and a
*bogie* pivot; the bogie carries the middle and rear wheels. A differential makes the
left and right rockers rotate in opposite directions, so the body pitches at the mean of
the two. All six wheels are driven, and the four corner wheels steer (Curiosity-style).

Drive motors use the linear brushed-DC gearmotor model (physics.dc_motor); wheel torques,
tip-over and steering clearance come from physics.vehicle, rocker stress from physics.beam.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field


from ..physics import beam, dc_motor, vehicle
from ..physics.battery import Pack
from .common import components, fmt_table, load_toml, select


STEER_LOCK_DEG = 40.0   # corner steering range (joint limit in the generated model)


def tube_props(t: dict) -> tuple[float, float, float]:
    """Square tube part: (linear density kg/m, second moment m^4, half-depth m)."""
    A, I, c = beam.square_tube(t["side_mm"] / 1000, t["wall_mm"] / 1000)
    return t["density"] * A, I, c


@dataclass
class RoverDesign:
    wheel: str
    motor: str
    battery: str
    tube: str
    feasible: bool = False
    violations: list[str] = field(default_factory=list)
    m: dict = field(default_factory=dict)
    geo: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.wheel} + {self.motor} + {self.battery} + {self.tube}"


def evaluate(cfg: dict, db: dict, wheel_n, motor_n, batt_n, tube_n) -> RoverDesign | None:
    mis, body, st = cfg["mission"], cfg["body"], cfg["structure"]
    wheel, motor, bspec, tube = db["wheel"][wheel_n], db["drive_motor"][motor_n], db["battery"][batt_n], db["rover_tube"][tube_n]
    pack = Pack(bspec)
    ratio = pack.v_nom / motor["voltage"]
    if not 0.8 <= ratio <= 1.25:          # electrically incompatible: not a candidate
        return None
    d = RoverDesign(wheel_n, motor_n, batt_n, tube_n)
    g = mis["gravity"]
    r, w = wheel["diameter_mm"] / 2000, wheel["width_mm"] / 1000
    servo = db["steer_servo"][body["steer_servo"]]

    # ---- geometry (scales with wheel radius) ----
    s = 2 * r + 0.04                                   # front-middle and middle-rear spacing
    H = max(1.9 * r, mis["min_clearance_m"] + body["height_m"] / 2)   # rocker pivot / body centre height
    zb = r + 0.55 * r                                  # bogie pivot height
    lin, I_t, c_t = tube_props(tube)
    leg_front = math.hypot(s, H - r)
    leg_rear = math.hypot(s / 2, H - zb)
    leg_bogie = math.hypot(s / 2, zb - r)

    # ---- mass budget ----
    m_wheel_unit = wheel["mass_g"] / 1000 + motor["mass_g"] / 1000
    tube_len_side = leg_front + leg_rear + 2 * leg_bogie + 4 * 0.6 * r      # + steering/wheel forks
    m_body_frame = lin * 2 * (body["length_m"] + body["width_m"])
    m_fixed = (pack.mass + body["electronics_g"] / 1000 + mis["payload_kg"] + m_body_frame
               + 4 * servo["mass_g"] / 1000)
    # track depends on CG height which depends on track (diff bar mass) -> iterate twice
    # steered corner wheels swing inward by r*sin(lock) at full lock: keep them off the body
    track_geom = body["width_m"] + w + 2 * vehicle.steer_swing(r, STEER_LOCK_DEG) + 0.06
    track = track_geom
    for _ in range(3):
        m_tubes = lin * (2 * tube_len_side + track)
        mass = (m_fixed + 6 * m_wheel_unit + m_tubes) * 1.08               # +8% fasteners, cabling
        h_cg = ((m_fixed * H + 6 * m_wheel_unit * r + m_tubes * (H + r) / 2) * 1.08) / mass
        track_tip = vehicle.track_for_tip_angle(h_cg, mis["max_slope_deg"] + mis["tip_margin_deg"])
        track = max(track_geom, track_tip)
    width, length = track + w, 2 * s + 2 * r

    # ---- drive ----
    gm = dc_motor.gearmotor_at_voltage(motor, ratio)
    tau_s, w0, i0, i_s = gm["tau_s"], gm["omega_0"], gm["i0"], gm["i_s"]
    tau_flat = vehicle.wheel_torque_rolling(mass, g, mis["terrain_crr"], r, 6)
    tau_slope = vehicle.wheel_torque_slope(mass, g, mis["max_slope_deg"], mis["terrain_crr"], r, 6)
    tau_trac = vehicle.wheel_torque_traction(mis["terrain_mu"], mass, g, r, 6)
    v_flat = dc_motor.gearmotor_speed(tau_flat, tau_s, w0) * r
    v_slope = dc_motor.gearmotor_speed(tau_slope, tau_s, w0) * r

    # energy at cruise on flat ground
    v = mis["cruise_speed_mps"]
    u = min(dc_motor.gearmotor_duty(v / r, tau_flat, tau_s, w0), 1.0)
    i_mot = dc_motor.gearmotor_current(tau_flat, i0, i_s, tau_s)
    p_drive = 6 * pack.v_nom * u * i_mot
    p_total = p_drive + body["avionics_w"] + 4 * servo["power_w"]
    e_use = pack.energy_wh * (1 - mis["reserve_frac"])
    range_km = e_use / p_total * v * 3.6
    i_peak = 6 * i_s * 0.8 + p_total / pack.v_nom

    # structure: rocker front leg as cantilever, 2 g vertical load on the front wheel
    F = 2 * mass * g / 6
    stress = beam.bending_stress_mpa(F * s, c_t, I_t)
    tip_lat = vehicle.tip_over_angle_deg(track / 2, h_cg)
    tip_lon = vehicle.tip_over_angle_deg(s, h_cg)
    contact = 0.35 * r
    pressure = vehicle.contact_pressure_kpa(mass * g / 6, w, contact)

    d.m = dict(mass_kg=mass, width_m=width, length_m=length, track_m=track, wheel_r=r, h_cg=h_cg,
               range_km=range_km, cruise_w=p_total, v_flat=v_flat, v_slope=v_slope,
               tau_req_slope=tau_slope, tau_req_trac=tau_trac, tau_stall=tau_s,
               climb_m=mis["climb_factor"] * r, tip_lat_deg=tip_lat, tip_lon_deg=tip_lon,
               stress_mpa=stress, pressure_kpa=pressure, peak_a=i_peak, v_pack=pack.v_nom, clearance_m=H - body["height_m"] / 2)
    d.geo = dict(r=r, w=w, s=s, H=H, zb=zb, track=track, tube=tube["side_mm"] / 1000, tube_lin=lin,
                 m_wheel=m_wheel_unit, m_fixed=m_fixed * 1.08, body=body, servo=servo)

    viol = d.violations
    if width > mis["max_width_m"]:
        viol.append(f"width {width:.2f} > {mis['max_width_m']} m")
    if length > mis["max_length_m"]:
        viol.append(f"length {length:.2f} > {mis['max_length_m']} m")
    if d.m["climb_m"] < mis["obstacle_height_m"]:
        viol.append(f"climbs {d.m['climb_m']:.2f} < {mis['obstacle_height_m']} m obstacle")
    if 0.5 * tau_s < tau_slope:
        viol.append(f"slope torque {tau_slope:.2f} > 50% stall {0.5*tau_s:.2f} Nm")
    if 0.8 * tau_s < tau_trac:
        viol.append(f"can't use full traction ({tau_trac:.2f} > 80% stall {0.8*tau_s:.2f} Nm)")
    if v_flat < v:
        viol.append(f"top speed {v_flat:.2f} < cruise {v} m/s")
    if v_slope < 0.4 * v:
        viol.append(f"slope speed {v_slope:.2f} < 40% cruise")
    if range_km < mis["min_range_km"]:
        viol.append(f"range {range_km:.1f} < {mis['min_range_km']} km")
    if tip_lon < mis["max_slope_deg"] + mis["tip_margin_deg"]:
        viol.append(f"pitch tip {tip_lon:.0f} deg too low")
    if stress > st["max_tube_stress_mpa"]:
        viol.append(f"rocker stress {stress:.0f} > {st['max_tube_stress_mpa']} MPa")
    if i_peak > pack.i_max:
        viol.append(f"peak current {i_peak:.0f} > pack {pack.i_max:.0f} A")
    d.feasible = not viol
    return d


def optimize(config="rover_yard.toml") -> tuple[dict, list[RoverDesign]]:
    cfg = load_toml(config)
    db = components()
    ds = cfg["design_space"]
    combos = itertools.product(select(db, "wheel", ds["wheels"]), select(db, "drive_motor", ds["drive_motors"]),
                               select(db, "battery", ds["batteries"]), select(db, "rover_tube", ds["tubes"]))
    designs = [d for c in combos if (d := evaluate(cfg, db, *c)) is not None]
    key = {"range": lambda d: -d.m["range_km"], "min_mass": lambda d: d.m["mass_kg"]}[cfg["objective"]["maximize"]]
    designs.sort(key=lambda d: (not d.feasible, len(d.violations), key(d), d.m["mass_kg"]))
    return cfg, designs


def best(config) -> tuple[dict, RoverDesign]:
    """Best feasible design for a mission file (raises with the reasons if none)."""
    cfg, designs = optimize(config)
    feas = [d for d in designs if d.feasible]
    if not feas:
        raise SystemExit(report(cfg, designs))
    return cfg, feas[0]


def report(cfg, designs, top=8) -> str:
    feas = [d for d in designs if d.feasible]
    cols = [("label", "design (wheel + motor + battery + tube)", ""), ("mass_kg", "mass kg", ".1f"),
            ("width_m", "width m", ".2f"), ("range_km", "range km", ".1f"), ("v_flat", "vmax m/s", ".2f"),
            ("cruise_w", "cruise W", ".0f"), ("tip_lat_deg", "tip deg", ".0f"), ("stress_mpa", "MPa", ".0f")]
    rows = [dict(label=d.label, **d.m) for d in (feas[:top] if feas else designs[:top])]
    out = [f"== {cfg['mission']['name']}: {len(designs)} compatible combinations, {len(feas)} feasible "
           f"(objective: {cfg['objective']['maximize']})", fmt_table(rows, cols)]
    if not feas:
        out.append("\nNO FEASIBLE DESIGN. Closest candidates violate:")
        for d in designs[:5]:
            out.append(f"  {d.label}: " + "; ".join(d.violations))
    else:
        b = feas[0].m
        out.append(f"\nselected: {feas[0].label}")
        out.append(f"  mass {b['mass_kg']:.1f} kg, {b['length_m']:.2f} x {b['width_m']:.2f} m, wheel r {b['wheel_r']*1000:.0f} mm, "
                   f"clearance {b['clearance_m']*1000:.0f} mm, CG {b['h_cg']*1000:.0f} mm")
        out.append(f"  torque/wheel: slope {b['tau_req_slope']:.2f}, traction {b['tau_req_trac']:.2f}, stall {b['tau_stall']:.2f} Nm; "
                   f"climbs {b['climb_m']*1000:.0f} mm; tip-over {b['tip_lat_deg']:.0f} deg sideways")
        out.append(f"  cruise {b['cruise_w']:.0f} W -> {b['range_km']:.1f} km; ground pressure {b['pressure_kpa']:.0f} kPa; "
                   f"rocker stress {b['stress_mpa']:.0f} MPa")
    return "\n".join(out)


# ----------------------------------------------------------------------- generator ---
def to_mjcf(b, name: str, pos, design: RoverDesign, cfg: dict, yaw: float = 0.0) -> dict:
    """Write the designed rover into a scene Builder (assembled from components by
    sim/robots/templates.py); returns the robot meta."""
    from ..robots.templates import rocker_bogie
    robot, meta = rocker_bogie(name, cfg, design)
    robot.write(b, pos, yaw=yaw)
    b.meta.setdefault("robots", []).append(meta)
    b.meta["uncertainty"] = dict(cfg.get("uncertainty", {}))   # drawn per episode by Sim(variant=k)
    return meta
