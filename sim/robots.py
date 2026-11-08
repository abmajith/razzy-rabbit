"""Robot MJCF fragments: quadrotor UAV, high-clearance field rover, greenhouse
pipe-rail harvesting trolley with arm. Each function registers materials/actuators/
sensors on the Builder and returns robot metadata used by the controllers.
"""
from __future__ import annotations

import numpy as np

from .builder import Builder, fmt, lookat_xyaxes


def _robot_materials(b: Builder) -> None:
    b.material("carbon", [0.10, 0.10, 0.11, 1], specular=0.5, shininess=0.6)
    b.material("accent", [0.95, 0.45, 0.08, 1], specular=0.3, shininess=0.4)
    b.material("alu", [0.72, 0.74, 0.77, 1], specular=0.6, shininess=0.7, reflectance=0.05)
    b.material("rubber", [0.06, 0.06, 0.06, 1], specular=0.05)
    b.material("prop", [0.15, 0.15, 0.15, 0.35])
    b.material("solar", [0.08, 0.12, 0.30, 1], specular=0.9, shininess=0.9, reflectance=0.15)
    b.material("white_paint", [0.92, 0.92, 0.90, 1], specular=0.3, shininess=0.4)
    b.material("lens", [0.02, 0.02, 0.05, 1], specular=1.0, shininess=1.0)


# ======================================================================================
# Quadrotor (X configuration, ~1.2 kg, 4 x 8 N thrust)
# ======================================================================================
def quadrotor(b: Builder, name: str, pos, arm: float = 0.17, mass: float = 1.2,
              chase_offset=(-3.5, -2.0, 1.6)) -> dict:
    _robot_materials(b)
    a = arm / np.sqrt(2)
    k_yaw = 0.016  # rotor drag torque / thrust [m]
    rotors = [(a, -a, -1), (a, a, 1), (-a, a, -1), (-a, -a, 1)]  # FR, FL, RL, RR
    g = [f'<body name="{name}" pos="{fmt(pos)}">',
         f'  <freejoint name="{name}_root"/>',
         f'  <inertial pos="0 0 0" mass="{mass}" diaginertia="0.0125 0.0125 0.0220"/>',
         f'  <geom name="{name}_hull" type="box" size=".09 .055 .028" material="carbon"/>',
         f'  <geom type="box" size=".06 .042 .016" pos="-.01 0 .042" material="accent" contype="0" conaffinity="0"/>',
         f'  <geom type="capsule" fromto=".09 0 -.01 .07 0 -.045" size=".018" material="carbon" contype="0" conaffinity="0"/>',
         f'  <geom type="sphere" size=".016" pos=".08 0 -.05" material="lens" contype="0" conaffinity="0"/>']
    for i, (x, y, s) in enumerate(rotors):
        g += [f'  <geom type="capsule" fromto="0 0 0 {x:.4f} {y:.4f} 0" size=".011" material="carbon" contype="0" conaffinity="0"/>',
              f'  <geom type="cylinder" size=".022 .02" pos="{x:.4f} {y:.4f} .015" material="alu" contype="0" conaffinity="0"/>',
              f'  <geom name="{name}_prop{i}" type="cylinder" size=".115 .004" pos="{x:.4f} {y:.4f} .04" material="prop"/>',
              f'  <site name="{name}_rotor{i}" pos="{x:.4f} {y:.4f} .04" size=".01"/>',
              f'  <geom type="capsule" fromto="{x*.6:.4f} {y*.6:.4f} -.02 {x*.75:.4f} {y*.75:.4f} -.1" size=".007" material="carbon"/>']
        b.actuators.append(f'<general name="{name}_m{i}" site="{name}_rotor{i}" gear="0 0 1 0 0 {s * k_yaw}" '
                           f'ctrlrange="0 8"/>')
    g += [f'  <site name="{name}_imu" pos="0 0 0" size=".01"/>',
          f'  <camera name="{name}_fpv" pos=".1 0 -.04" xyaxes="0 -1 0 .342 0 .940" fovy="85"/>',
          f'  <camera name="{name}_nadir" pos="0 0 -.05" xyaxes="0 -1 0 1 0 0" fovy="70"/>',
          f'  <camera name="{name}_chase" mode="trackcom" pos="{fmt(chase_offset)}" xyaxes="{lookat_xyaxes(chase_offset)}"/>',
          '</body>']
    b.world.append("\n    ".join(g))
    b.sensors += [f'<gyro name="{name}_gyro" site="{name}_imu"/>',
                  f'<accelerometer name="{name}_acc" site="{name}_imu"/>',
                  f'<framepos name="{name}_pos" objtype="body" objname="{name}"/>',
                  f'<framequat name="{name}_quat" objtype="body" objname="{name}"/>',
                  f'<framelinvel name="{name}_vel" objtype="body" objname="{name}"/>']
    meta = dict(type="quadrotor", body=name, mass=mass, k_yaw=k_yaw,
                actuators=[f"{name}_m{i}" for i in range(4)],
                cameras=[f"{name}_chase", f"{name}_fpv", f"{name}_nadir"],
                drag_CdA=0.045)
    b.meta.setdefault("robots", []).append(meta)
    return meta


# ======================================================================================
# High-clearance gantry rover straddling a crop bed (skid steer, 4 driven wheels)
# ======================================================================================
def field_rover(b: Builder, name: str, pos, yaw: float = 0.0, track: float = 1.5,
                wheelbase: float = 1.3, r: float = 0.3, height: float = 1.05) -> dict:
    _robot_materials(b)
    hz = height - r      # wheel centre below body origin
    q = f'{np.cos(yaw/2):.5f} 0 0 {np.sin(yaw/2):.5f}'
    chase = (-4.5, -3.0, 2.6)
    g = [f'<body name="{name}" pos="{fmt(pos)}" quat="{q}">',
         f'  <freejoint name="{name}_root"/>',
         f'  <inertial pos="0 0 -.25" mass="90" diaginertia="28 26 40"/>',
         # bridge + side beams
         f'  <geom name="{name}_bridge" type="box" size=".35 {track/2+.05:.3f} .06" material="white_paint"/>',
         f'  <geom type="box" size="{wheelbase/2+.08:.3f} .05 .06" pos="0 {track/2:.3f} 0" material="white_paint"/>',
         f'  <geom type="box" size="{wheelbase/2+.08:.3f} .05 .06" pos="0 {-track/2:.3f} 0" material="white_paint"/>',
         f'  <geom type="box" size=".25 .3 .12" pos="-.05 0 .18" material="accent"/>',
         f'  <geom type="box" size=".45 .55 .012" pos="-.05 0 .34" euler="0 -8 0" material="solar" contype="0" conaffinity="0"/>',
         f'  <geom type="cylinder" fromto="-.25 .3 .3 -.25 .3 .8" size=".015" material="alu" contype="0" conaffinity="0"/>',
         f'  <geom type="cylinder" size=".07 .02" pos="-.25 .3 .82" material="white_paint" contype="0" conaffinity="0"/>',
         # weeding tool bar under the bridge
         f'  <geom type="box" size=".03 {track/2-.15:.3f} .03" pos=".25 0 -.45" material="alu" contype="0" conaffinity="0"/>']
    for k in range(-3, 4):
        g.append(f'  <geom type="box" size=".012 .012 .2" pos=".25 {k*0.16:.3f} -.25" material="alu" contype="0" conaffinity="0"/>')
        g.append(f'  <geom type="box" size=".06 .02 .005" pos=".29 {k*0.16:.3f} -.65" euler="0 15 0" material="alu" contype="0" conaffinity="0"/>')
    g += [f'  <camera name="{name}_bedcam" pos=".38 0 -.05" xyaxes="0 -1 0 .707 0 .707" fovy="75"/>',
          f'  <camera name="{name}_front" pos=".45 0 .2" xyaxes="0 -1 0 .26 0 .97" fovy="80"/>',
          f'  <camera name="{name}_chase" mode="trackcom" pos="{fmt(chase)}" xyaxes="{lookat_xyaxes(chase)}"/>',
          f'  <site name="{name}_imu" size=".01"/>']
    wheels = []
    for sx, tag_x in ((1, "f"), (-1, "r")):
        for sy, tag_y in ((1, "l"), (-1, "r")):
            wn = f"{name}_w{tag_x}{tag_y}"
            x, y = sx * wheelbase / 2, sy * track / 2
            g += [f'  <geom type="box" size=".05 .04 {hz/2:.3f}" pos="{x:.3f} {y:.3f} {-hz/2:.3f}" material="white_paint" contype="0" conaffinity="0"/>',
                  f'  <body name="{wn}" pos="{x:.3f} {y + sy*0.1:.3f} {-hz:.3f}">',
                  f'    <joint name="{wn}_j" type="hinge" axis="0 1 0" damping="0.5"/>',
                  f'    <geom name="{wn}_tire" type="cylinder" size="{r} .07" euler="90 0 0" material="rubber" mass="6" friction="1.2 0.02 0.001" condim="3"/>',
                  f'    <geom type="cylinder" size="{r*0.55:.3f} .072" euler="90 0 0" material="alu" contype="0" conaffinity="0" mass="0"/>',
                  '  </body>']
            wheels.append(wn)
            b.actuators.append(f'<velocity name="{wn}_v" joint="{wn}_j" kv="120" ctrlrange="-8 8" forcerange="-160 160"/>')
            b.sensors.append(f'<jointvel name="{wn}_qd" joint="{wn}_j"/>')
    g.append('</body>')
    b.world.append("\n    ".join(g))
    b.sensors += [f'<gyro name="{name}_gyro" site="{name}_imu"/>',
                  f'<accelerometer name="{name}_acc" site="{name}_imu"/>',
                  f'<framepos name="{name}_pos" objtype="body" objname="{name}"/>',
                  f'<framequat name="{name}_quat" objtype="body" objname="{name}"/>']
    meta = dict(type="rover", body=name, track=track + 0.2, wheel_radius=r,
                actuators=[f"{w}_v" for w in wheels], wheels=wheels,
                cameras=[f"{name}_chase", f"{name}_bedcam", f"{name}_front"])
    b.meta.setdefault("robots", []).append(meta)
    return meta


# ======================================================================================
# Greenhouse pipe-rail trolley + scissor lift + 4-DoF arm + parallel gripper
# ======================================================================================
def pipe_rail_harvester(b: Builder, name: str, x0: float, y_aisle: float, rail_z: float,
                        n_basket: int = 12) -> dict:
    _robot_materials(b)
    b.material("crate", [0.15, 0.35, 0.65, 1], specular=0.2)
    b.material("tomato_red", [0.78, 0.07, 0.04, 1], specular=0.6, shininess=0.7)
    hide = "0 0 0 0"
    chase = (-2.3, 0.15, 1.75)
    g = [f'<body name="{name}" pos="{x0:.3f} {y_aisle:.3f} {rail_z + 0.06:.3f}">',
         f'  <joint name="{name}_rail" type="slide" axis="1 0 0" range="-30 30" damping="200"/>',
         f'  <inertial pos="0 0 .1" mass="80" diaginertia="5 8 10"/>',
         f'  <geom type="box" size=".65 .32 .04" pos="0 0 .06" material="white_paint" contype="0" conaffinity="0"/>',
         f'  <geom type="box" size=".2 .2 .1" pos=".35 0 .2" material="accent" contype="0" conaffinity="0"/>']
    for sx in (-0.5, 0.5):
        for sy in (-0.275, 0.275):
            g.append(f'  <geom type="cylinder" size=".055 .025" pos="{sx} {sy} 0" euler="90 0 0" material="rubber" contype="0" conaffinity="0"/>')
    # crate on the rear deck (5 thin walls) + hidden fruit slots shown as harvest progresses
    cx = -0.38
    g += [f'  <geom type="box" size=".22 .17 .01" pos="{cx} 0 .11" material="crate" contype="0" conaffinity="0"/>',
          f'  <geom type="box" size=".22 .01 .07" pos="{cx} .17 .18" material="crate" contype="0" conaffinity="0"/>',
          f'  <geom type="box" size=".22 .01 .07" pos="{cx} -.17 .18" material="crate" contype="0" conaffinity="0"/>',
          f'  <geom type="box" size=".01 .17 .07" pos="{cx+.22} 0 .18" material="crate" contype="0" conaffinity="0"/>',
          f'  <geom type="box" size=".01 .17 .07" pos="{cx-.22} 0 .18" material="crate" contype="0" conaffinity="0"/>']
    for i in range(n_basket):
        px = cx - 0.15 + 0.1 * (i % 4)
        py = -0.1 + 0.1 * ((i // 4) % 3)
        g.append(f'  <geom name="{name}_crate{i}" type="sphere" size=".038" pos="{px:.3f} {py:.3f} .16" rgba="{hide}" material="tomato_red" contype="0" conaffinity="0"/>')
    # mast for the lift
    for sy in (-0.12, 0.12):
        g.append(f'  <geom type="box" size=".02 .02 .9" pos=".1 {sy} 1.0" material="alu" contype="0" conaffinity="0"/>')
    g += [f'  <camera name="{name}_chase" mode="trackcom" pos="{fmt(chase)}" xyaxes="{lookat_xyaxes(chase)}"/>',
          f'  <body name="{name}_lift" pos=".1 0 .25" gravcomp="1">',
          f'    <joint name="{name}_lift_z" type="slide" axis="0 0 1" range="0 1.5" damping="300"/>',
          f'    <geom type="box" size=".14 .18 .025" material="alu" mass="8" contype="0" conaffinity="0"/>',
          f'    <body name="{name}_base" pos="0 0 .05" gravcomp="1">',
          f'      <joint name="{name}_yaw" type="hinge" axis="0 0 1" range="-180 180" damping="8" armature=".05"/>',
          f'      <geom type="cylinder" size=".07 .05" pos="0 0 .05" material="carbon" mass="3" contype="0" conaffinity="0"/>',
          f'      <body name="{name}_upper" pos="0 0 .12" gravcomp="1">',
          f'        <joint name="{name}_shoulder" type="hinge" axis="0 1 0" range="-110 110" damping="6" armature=".05"/>',
          f'        <geom type="capsule" fromto="0 0 0 .36 0 0" size=".035" material="white_paint" mass="2" contype="0" conaffinity="0"/>',
          f'        <geom type="cylinder" size=".045 .05" euler="90 0 0" material="accent" mass="0" contype="0" conaffinity="0"/>',
          f'        <body name="{name}_fore" pos=".36 0 0" gravcomp="1">',
          f'          <joint name="{name}_elbow" type="hinge" axis="0 1 0" range="-150 150" damping="4" armature=".03"/>',
          f'          <geom type="capsule" fromto="0 0 0 .3 0 0" size=".028" material="white_paint" mass="1.2" contype="0" conaffinity="0"/>',
          f'          <geom type="cylinder" size=".038 .045" euler="90 0 0" material="accent" mass="0" contype="0" conaffinity="0"/>',
          f'          <body name="{name}_wrist" pos=".3 0 0" gravcomp="1">',
          f'            <joint name="{name}_pitch" type="hinge" axis="0 1 0" range="-120 120" damping="2" armature=".02"/>',
          f'            <geom type="box" size=".05 .045 .03" pos=".04 0 0" material="carbon" mass=".5" contype="0" conaffinity="0"/>',
          f'            <geom type="sphere" size=".012" pos=".09 0 .04" material="lens" contype="0" conaffinity="0"/>',
          f'            <camera name="{name}_wristcam" pos=".09 0 .045" xyaxes="0 -1 0 0 0 1" fovy="70"/>',
          f'            <site name="{name}_ee" pos=".15 0 0" size=".01"/>',
          f'            <geom name="{name}_held" type="sphere" size=".038" pos=".15 0 0" rgba="{hide}" material="tomato_red" contype="0" conaffinity="0"/>']
    for side, sgn in (("l", 1), ("r", -1)):
        g += [f'            <body name="{name}_finger_{side}" pos=".1 {sgn*0.03} 0" gravcomp="1">',
              f'              <joint name="{name}_finger_{side}_j" type="slide" axis="0 {sgn} 0" range="0 .035" damping="1"/>',
              f'              <geom type="box" size=".045 .006 .018" pos=".04 0 0" material="alu" mass=".05" contype="0" conaffinity="0"/>',
              '            </body>']
    g += ['          </body>', '        </body>', '      </body>', '    </body>', '  </body>', '</body>']
    b.world.append("\n    ".join(g))
    acts = [("rail", 6000, 3000, "-30 30"), ("lift_z", 8000, 2000, "0 1.5"), ("yaw", 300, 150, "-3.14 3.14"),
            ("shoulder", 300, 150, "-1.9 1.9"), ("elbow", 200, 100, "-2.6 2.6"), ("pitch", 80, 40, "-2.1 2.1")]
    for j, kp, fr, cr in acts:
        b.actuators.append(f'<position name="{name}_{j}_p" joint="{name}_{j}" kp="{kp}" kv="{kp/10:.0f}" '
                           f'ctrlrange="{cr}" forcerange="-{fr} {fr}"/>')
    for side in ("l", "r"):
        b.actuators.append(f'<position name="{name}_finger_{side}_p" joint="{name}_finger_{side}_j" kp="200" ctrlrange="0 .035"/>')
    b.sensors += [f'<framepos name="{name}_ee_pos" objtype="site" objname="{name}_ee"/>']
    meta = dict(type="harvester", body=name, cameras=[f"{name}_chase", f"{name}_wristcam"],
                joints=[f"{name}_{j}" for j, *_ in acts], ee=f"{name}_ee",
                n_basket=n_basket)
    b.meta.setdefault("robots", []).append(meta)
    return meta
