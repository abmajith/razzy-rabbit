"""Multirotor frame: centre plate + carbon arms, and an optional prop guard (cage).

QuadFrame creates the vehicle's root body. Its mass properties are summed from every part
attached to it (plate, arms, motors, props, battery, boards, cage) with the parallel-axis
theorem, then scaled by (1 + wiring fraction) for cables, screws and glue (`frame` part).

Mounts (body frame: x forward, z up):
    deck     top of the plate, parts sit on it (+z)      -> battery
    center   under the plate, parts hang below (-z)      -> flight controller stack
    bottom   aft, under the plate (-z)                   -> radio
    nose     front edge of the plate (+x)                -> camera module
    belly    under the plate (-z)                        -> downward camera
    cg       the body's centre of mass (resolved when written) -> IMU
    origin   body origin                                  -> chase camera
    arms[i].tip   end of each arm, where a rotor goes (with its spin direction)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import parts
from ..physics import beam
from .base import Component, MassPart, Mount, box_inertia, rod_inertia

# X layout: (x sign, y sign, spin) for FR, FL, RL, RR; spin +1 = counter-clockwise from above
QUAD_X = [("FR", 1, -1, -1), ("FL", 1, 1, 1), ("RL", -1, 1, -1), ("RR", -1, -1, 1)]


@dataclass
class Arm:
    name: str
    tip: Mount
    spin: int
    direction: np.ndarray


class QuadFrame(Component):
    kind = "frame"

    def __init__(self, arm_tube: str, arm_length: float, prop_d: float, guard_gap: float = 0.0,
                 frame: str = "frame") -> None:
        super().__init__()
        self.tube = parts.get("arm_tube", arm_tube)
        self.spec = parts.load()[frame]                    # plate, motor mount, wiring fraction
        self.arm_length, self.prop_d, self.guard_gap = arm_length, prop_d, guard_gap
        self.arms: list[Arm] = []

    def attach(self, robot, at) -> None:
        sp, L = self.spec, self.arm_length
        body = robot.new_body(robot.name, None, (0, 0, 0), inertial="parts", mass_scale=1 + sp["wiring_frac"])
        self.body = body
        plate = sp["center_plate_g"] / 1000
        body.mass_parts.append(MassPart(plate, np.zeros(3), box_inertia(plate, 0.11, 0.08, 0.02), "plate"))
        body.geoms.append('<geom type="box" size=".055 .04 .01" material="carbon"/>')
        # arms: thin-walled carbon tubes from the centre to each motor
        od, idd = self.tube["od_mm"] / 1000, self.tube["id_mm"] / 1000
        A, _ = beam.round_tube(od, idd)
        self.m_arm = self.tube["density"] * A * L
        for arm_name, sx, sy, spin in QUAD_X:
            u = np.array([sx, sy, 0]) / math.sqrt(2)
            tip = u * L
            body.mass_parts.append(MassPart(self.m_arm, tip / 2, rod_inertia(self.m_arm, L, u), f"arm {arm_name}"))
            body.geoms.append(f'<geom type="capsule" fromto="0 0 0 {tip[0]:.4f} {tip[1]:.4f} 0" size="{od/2:.4f}" '
                              'material="carbon" contype="0" conaffinity="0"/>')
            self.arms.append(Arm(arm_name, self.publish(f"tip_{arm_name}", body, tip), spin, u))
        # collision: invisible envelope around the props (+ guard) and four landing feet
        half = L / math.sqrt(2) + self.prop_d / 2 + self.guard_gap
        body.geoms.append(f'<geom name="{robot.name}_hull" type="box" size="{half:.4f} {half:.4f} .035" pos="0 0 .02" '
                          'rgba="0 0 0 0" group="3"/>')
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            body.geoms.append(f'<geom type="capsule" fromto="{sx*.04} {sy*.035} -.01 {sx*.05} {sy*.045} -.065" '
                              'size=".004" material="carbon"/>')
        self.publish("deck", body, [0, 0, 0.012], (0, 0, 1))
        self.publish("center", body, [0, 0, -0.012], (0, 0, -1))
        self.publish("bottom", body, [-0.03, 0, -0.03], (0, 0, -1))
        self.publish("nose", body, [0.055, 0, -0.01], (1, 0, 0))
        self.publish("belly", body, [0, 0, -0.04], (0, 0, -1))
        self.publish("cg", body, None)
        self.publish("origin", body, [0, 0, 0])

    def summary(self) -> str:
        return (f"QuadFrame: arms {self.arm_length*1000:.0f} mm x4 ({self.m_arm*1000:.1f} g each), "
                f"plate {self.spec['center_plate_g']:.0f} g")


class PropGuard(Component):
    """Ring guards around every rotor of the frame it is attached to (collision tolerance),
    two rings per rotor; its mass is lumped near the centre (spread over the arm span)."""

    kind = "guard"

    def __init__(self, prop_d: float, gap: float, mass_kg: float, hub_height: float = 0.03) -> None:
        super().__init__()
        self.prop_d, self.gap, self.mass, self.zp = prop_d, gap, mass_kg, hub_height

    def attach(self, robot, at) -> None:
        frame = at.owner
        body = at.body
        robot.material("cage", [0.15, 0.15, 0.16, 1], specular=0.3)
        rr = self.prop_d / 2 + self.gap
        for arm in frame.arms:
            tip = arm.tip.pos
            for zc in (self.zp - 0.02, self.zp + 0.02):
                n = 10
                for k in range(n):
                    a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
                    p0 = [tip[0] + rr * math.cos(a0), tip[1] + rr * math.sin(a0), zc]
                    p1 = [tip[0] + rr * math.cos(a1), tip[1] + rr * math.sin(a1), zc]
                    body.geoms.append(f'<geom type="capsule" fromto="{_f(p0)} {_f(p1)}" size=".0018" material="cage" '
                                      'contype="0" conaffinity="0"/>')
        span = frame.arm_length + self.prop_d / 2
        body.mass_parts.append(MassPart(self.mass, np.array([0, 0, 0.03]),
                                        self.mass * span ** 2 * np.diag([0.5, 0.5, 1.0]), "prop guard"))

    def summary(self) -> str:
        return f"PropGuard: {self.mass*1000:.0f} g, ring radius {(self.prop_d/2+self.gap)*1000:.0f} mm"


def _f(v) -> str:
    return " ".join(f"{float(x):.4g}" for x in v)
