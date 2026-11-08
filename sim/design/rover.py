"""Six-wheel rocker-bogie rover sizing (physics + exhaustive search) and MJCF generation.

Suspension: on each side a *rocker* (pivoting on the body) carries the front wheel and a
*bogie* pivot; the bogie carries the middle and rear wheels. A differential makes the
left and right rockers rotate in opposite directions, so the body pitches at the mean of
the two. All six wheels are driven, and the four corner wheels steer (Curiosity-style).

Drive motors use the linear brushed-DC gearmotor model at the output shaft:
    torque  tau = tau_s * (u - omega / omega_0)        u in [-1, 1] = PWM duty
    current I   = I_0 + (I_s - I_0) * tau / tau_s
with tau_s and omega_0 scaled by (pack voltage / motor rated voltage).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .common import Pack, components, fmt_table, load_toml, select


STEER_LOCK_DEG = 40.0   # corner steering range (joint limit in the generated model)


def tube_props(t: dict) -> tuple[float, float, float]:
    """Square tube: (linear density kg/m, second moment m^4, half-depth m)."""
    a, w = t["side_mm"] / 1000, t["wall_mm"] / 1000
    A = a * a - (a - 2 * w) ** 2
    I = (a ** 4 - (a - 2 * w) ** 4) / 12
    return t["density"] * A, I, a / 2


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
    track_geom = body["width_m"] + w + 2 * r * math.sin(math.radians(STEER_LOCK_DEG)) + 0.06
    track = track_geom
    for _ in range(3):
        m_tubes = lin * (2 * tube_len_side + track)
        mass = (m_fixed + 6 * m_wheel_unit + m_tubes) * 1.08               # +8% fasteners, cabling
        h_cg = ((m_fixed * H + 6 * m_wheel_unit * r + m_tubes * (H + r) / 2) * 1.08) / mass
        track_tip = 2 * h_cg * math.tan(math.radians(mis["max_slope_deg"] + mis["tip_margin_deg"]))
        track = max(track_geom, track_tip)
    width, length = track + w, 2 * s + 2 * r

    # ---- drive ----
    th = math.radians(mis["max_slope_deg"])
    tau_s, w0 = motor["stall_torque"] * ratio, motor["no_load_rpm"] * 2 * math.pi / 60 * ratio
    i0, i_s = motor["no_load_current"], motor["stall_current"] * ratio
    tau_flat = mass * g * mis["terrain_crr"] * r / 6
    tau_slope = mass * g * (math.sin(th) + mis["terrain_crr"] * math.cos(th)) * r / 6
    tau_trac = mis["terrain_mu"] * mass * g * r / 6
    v_flat = w0 * (1 - tau_flat / tau_s) * r
    v_slope = w0 * (1 - tau_slope / tau_s) * r

    # energy at cruise on flat ground
    v = mis["cruise_speed_mps"]
    u = min(v / r / w0 + tau_flat / tau_s, 1.0)
    i_mot = i0 + (i_s - i0) * tau_flat / tau_s
    p_drive = 6 * pack.v_nom * u * i_mot
    p_total = p_drive + body["avionics_w"] + 4 * servo["power_w"]
    e_use = pack.energy_wh * (1 - mis["reserve_frac"])
    range_km = e_use / p_total * v * 3.6
    i_peak = 6 * i_s * 0.8 + p_total / pack.v_nom

    # structure: rocker front leg as cantilever, 2 g vertical load on the front wheel
    F = 2 * mass * g / 6
    stress = F * s * c_t / I_t / 1e6
    tip_lat = math.degrees(math.atan2(track / 2, h_cg))
    tip_lon = math.degrees(math.atan2(s, h_cg))
    contact = 0.35 * r
    pressure = mass * g / 6 / (w * contact) / 1000

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
    from ..builder import fmt, lookat_xyaxes
    from ..builder import robot_materials

    robot_materials(b)
    b.material("rover_body", [0.85, 0.85, 0.82, 1], specular=0.3, shininess=0.3)
    b.material("gold_foil", [0.85, 0.65, 0.25, 1], specular=0.8, shininess=0.8, reflectance=0.1)
    db = components()
    g, m = design.geo, design.m
    r, w, s, H, zb, track = g["r"], g["w"], g["s"], g["H"], g["zb"], g["track"]
    bd = g["body"]
    a = g["tube"]
    lin = g["tube_lin"] * 1.08                  # same fastener/cabling margin as the sizing
    motor = db["drive_motor"][design.motor]
    pack = Pack(db["battery"][design.battery])
    ratio = pack.v_nom / motor["voltage"]
    tau_s = motor["stall_torque"] * ratio
    w0 = motor["no_load_rpm"] * 2 * math.pi / 60 * ratio
    servo = g["servo"]

    bl, bw, bh = bd["length_m"], bd["width_m"], bd["height_m"]
    m_servo = servo["mass_g"] / 1000
    mb = g["m_fixed"] - 4 * m_servo * 1.08 + lin * track     # servos live on the steering knuckles
    Ib = mb / 12 * np.array([bw * bw + bh * bh, bl * bl + bh * bh, bl * bl + bw * bw])
    q = f'{math.cos(yaw/2):.5f} 0 0 {math.sin(yaw/2):.5f}'
    chase = (-3.2, -2.2, 1.6)
    x = [f'<body name="{name}" pos="{fmt(pos)}" quat="{q}">',
         f'  <freejoint name="{name}_root"/>',
         f'  <inertial pos="0 0 0" mass="{mb:.4f}" diaginertia="{fmt(Ib)}"/>',
         f'  <geom name="{name}_body" type="box" size="{bl/2} {bw/2} {bh/2}" material="rover_body"/>',
         f'  <geom type="box" size="{bl/2+.02} {bw/2+.02} .006" pos="0 0 {bh/2+.006}" material="solar" contype="0" conaffinity="0"/>',
         f'  <geom type="box" size="{bl/2-.03} {bw/2+.002} {bh/2-.03}" material="gold_foil" contype="0" conaffinity="0"/>',
         # differential bar across the top + mast with navcam
         f'  <geom type="capsule" fromto="-.05 {-track/2+w/2+a:.3f} {bh/2+.03} -.05 {track/2-w/2-a:.3f} {bh/2+.03}" size="{a/2:.3f}" material="alu" contype="0" conaffinity="0"/>',
         f'  <geom type="cylinder" fromto="{bl/2-.06} 0 {bh/2} {bl/2-.06} 0 {bh/2+.45}" size=".015" material="alu" contype="0" conaffinity="0"/>',
         f'  <geom type="box" size=".03 .07 .03" pos="{bl/2-.04} 0 {bh/2+.48}" material="rover_body" contype="0" conaffinity="0"/>',
         f'  <geom type="sphere" size=".012" pos="{bl/2-.005} .035 {bh/2+.48}" material="lens" contype="0" conaffinity="0"/>',
         f'  <geom type="sphere" size=".012" pos="{bl/2-.005} -.035 {bh/2+.48}" material="lens" contype="0" conaffinity="0"/>',
         f'  <camera name="{name}_navcam" pos="{bl/2} 0 {bh/2+.48}" xyaxes="0 -1 0 .34 0 .94" fovy="70"/>',
         f'  <camera name="{name}_hazcam" pos="{bl/2+.005} 0 {-bh/2+.02}" xyaxes="0 -1 0 .707 0 .707" fovy="100"/>',
         f'  <camera name="{name}_chase" mode="trackcom" pos="{fmt(chase)}" xyaxes="{lookat_xyaxes(chase)}"/>',
         f'  <site name="{name}_imu" size=".01"/>']

    def tube(p0, p1):
        L = float(np.linalg.norm(np.subtract(p1, p0)))
        return (f'<geom type="capsule" fromto="{fmt(p0)} {fmt(p1)}" size="{a/2:.4f}" material="alu" '
                f'mass="{lin*L:.4f}" contype="0" conaffinity="0"/>')

    def wheel_xml(wn, ind):
        return [f'{ind}<body name="{wn}">',
                f'{ind}  <joint name="{wn}_j" type="hinge" axis="0 1 0" damping="0.05" armature="0.01"/>',
                f'{ind}  <geom name="{wn}_tire" type="cylinder" size="{r:.4f} {w/2:.4f}" euler="90 0 0" material="rubber" '
                f'mass="{g["m_wheel"] * 1.08:.4f}" friction="1.0 0.02 0.002" condim="3"/>',
                f'{ind}  <geom type="cylinder" size="{r*0.6:.4f} {w/2+.002:.4f}" euler="90 0 0" material="alu" mass="0" contype="0" conaffinity="0"/>',
                *[f'{ind}  <geom type="box" size=".006 {w/2+.003:.4f} {r*0.95:.4f}" euler="0 {k*30} 0" material="rubber" mass="0" contype="0" conaffinity="0"/>'
                  for k in range(6)],                                   # grousers (visual)
                f'{ind}</body>']

    wheels, steers, wheel_xy = [], [], []
    for sgn, side in ((1, "l"), (-1, "r")):
        yr = sgn * (bw / 2 + a)                               # rocker plane just outside the body
        dy = sgn * track / 2 - yr                             # wheel offset from rocker plane
        x += [f'  <body name="{name}_rocker_{side}" pos="0 {yr:.4f} 0">',
              f'    <joint name="{name}_rocker_{side}" type="hinge" axis="0 1 0" range="-35 35" damping="2"/>',
              '    ' + tube([0, 0, 0], [s, 0, r - H + 0.6 * r]),
              '    ' + tube([s, 0, r - H + 0.6 * r], [s, dy, r - H + 0.6 * r]),
              '    ' + tube([0, 0, 0], [-s / 2, 0, zb - H]),
              # front corner: steering + wheel
              f'    <body name="{name}_steer_f{side}" pos="{s:.4f} {dy:.4f} {r - H:.4f}">',
              f'      <joint name="{name}_steer_f{side}" type="hinge" axis="0 0 1" range="-{STEER_LOCK_DEG:.0f} {STEER_LOCK_DEG:.0f}" damping="0.5" armature="0.005"/>',
              f'      <geom type="box" size=".022 .016 .02" pos="0 0 {0.6 * r + 0.02:.4f}" material="carbon" mass="{m_servo:.4f}" contype="0" conaffinity="0"/>',
              '      ' + tube([0, 0, 0.6 * r], [0, 0, 0.02]),
              *wheel_xml(f"{name}_w_f{side}", "      "),
              '    </body>',
              # bogie: middle + rear wheels
              f'    <body name="{name}_bogie_{side}" pos="{-s/2:.4f} 0 {zb - H:.4f}">',
              f'      <joint name="{name}_bogie_{side}" type="hinge" axis="0 1 0" range="-40 40" damping="1"/>',
              '      ' + tube([0, 0, 0], [s / 2, 0, r - zb + 0.6 * r]),
              '      ' + tube([0, 0, 0], [-s / 2, 0, r - zb + 0.6 * r]),
              '      ' + tube([s / 2, 0, r - zb + 0.6 * r], [s / 2, dy, r - zb + 0.6 * r]),
              '      ' + tube([-s / 2, 0, r - zb + 0.6 * r], [-s / 2, dy, r - zb + 0.6 * r]),
              f'      <body name="{name}_mid_{side}" pos="{s/2:.4f} {dy:.4f} {r - zb:.4f}">',
              *wheel_xml(f"{name}_w_m{side}", "        "),
              '      </body>',
              f'      <body name="{name}_steer_r{side}" pos="{-s/2:.4f} {dy:.4f} {r - zb:.4f}">',
              f'        <joint name="{name}_steer_r{side}" type="hinge" axis="0 0 1" range="-{STEER_LOCK_DEG:.0f} {STEER_LOCK_DEG:.0f}" damping="0.5" armature="0.005"/>',
              f'        <geom type="box" size=".022 .016 .02" pos="0 0 {0.6 * r + 0.02:.4f}" material="carbon" mass="{m_servo:.4f}" contype="0" conaffinity="0"/>',
              *wheel_xml(f"{name}_w_r{side}", "        "),
              '      </body>',
              '    </body>',
              '  </body>']
        for pos_tag, xw in (("f", s), ("m", 0.0), ("r", -s)):
            wn = f"{name}_w_{pos_tag}{side}"
            wheels.append(wn)
            wheel_xy.append([xw, sgn * track / 2])
            b.actuators.append(
                f'<general name="{wn}_drive" joint="{wn}_j" gaintype="fixed" gainprm="{tau_s:.4f}" '
                f'biastype="affine" biasprm="0 0 {-tau_s / w0:.5f}" ctrlrange="-1 1" forcerange="{-tau_s:.4f} {tau_s:.4f}"/>')
            b.sensors.append(f'<jointvel name="{wn}_qd" joint="{wn}_j"/>')
        for pos_tag in ("f", "r"):
            sn = f"{name}_steer_{pos_tag}{side}"
            steers.append(sn)
            b.actuators.append(f'<position name="{sn}_p" joint="{sn}" kp="{servo["stall_torque"] / 0.25:.2f}" '
                               f'kv="0.3" ctrlrange="-0.7 0.7" forcerange="{-servo["stall_torque"]} {servo["stall_torque"]}"/>')
    x.append('</body>')
    b.world.append("\n    ".join(x))
    # differential: left rocker angle = - right rocker angle
    b.contacts.append(f'<exclude body1="{name}" body2="{name}_rocker_l"/>')
    b.contacts.append(f'<exclude body1="{name}" body2="{name}_rocker_r"/>')
    b.equalities.append(
        f'<joint name="{name}_differential" joint1="{name}_rocker_l" joint2="{name}_rocker_r" polycoef="0 -1 0 0 0"/>')
    b.sensors += [f'<gyro name="{name}_gyro" site="{name}_imu"/>',
                  f'<accelerometer name="{name}_acc" site="{name}_imu"/>',
                  f'<framepos name="{name}_pos" objtype="body" objname="{name}"/>',
                  f'<framequat name="{name}_quat" objtype="body" objname="{name}"/>',
                  f'<jointpos name="{name}_rocker_l_q" joint="{name}_rocker_l"/>',
                  f'<jointpos name="{name}_bogie_l_q" joint="{name}_bogie_l"/>',
                  f'<jointpos name="{name}_bogie_r_q" joint="{name}_bogie_r"/>']
    meta = dict(type="rocker_bogie", body=name, wheels=wheels, wheel_xy=wheel_xy, wheel_radius=r,
                drive=[f"{wn}_drive" for wn in wheels], steer=[f"{sn}_p" for sn in steers],
                steer_xy={sn: ([s if "_f" in sn[-3:] else -s, (1 if sn.endswith("l") else -1) * track / 2])
                          for sn in steers},
                omega0=w0, tau_stall=tau_s,
                cameras=[f"{name}_chase", f"{name}_navcam", f"{name}_hazcam"],
                design=dict(label=design.label, metrics=m),
                plant=dict(kind="rover", motor=motor, battery=db["battery"][design.battery], v_ratio=ratio,
                           tau_stall=tau_s, avionics_w=bd["avionics_w"], servo_w=servo["power_w"]))
    from ..sensing.suite import resolve
    meta["sensors"] = resolve(cfg, db)
    b.meta.setdefault("robots", []).append(meta)
    b.meta["uncertainty"] = dict(cfg.get("uncertainty", {}))   # drawn per episode by Sim(variant=k)
    return meta
