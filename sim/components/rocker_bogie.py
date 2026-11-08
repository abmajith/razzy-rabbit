"""Rocker-bogie suspension (Mars-rover style), attached to a chassis.

On each side a *rocker* pivots on the chassis and carries the front corner; at its rear it
carries the pivot of a *bogie*, which carries the middle and rear corners. A differential
(an equality constraint here) makes the left and right rockers turn in opposite directions,
so the chassis pitches at their mean. Over a rock one wheel lifts while the others keep
contact: the rover climbs obstacles up to ~1.5 wheel radii.

Geometry (all from the sizing, see design/rover.py): wheel radius r, wheel spacing s, rocker
pivot height H, bogie pivot height zb, track (wheel centre to wheel centre). The legs are
square aluminium tubes (`rover_tube.*`) whose geom masses carry the fastener margin.

Mounts, one per corner, where a SteeringServo or a WheelDrive goes:
    fl, ml, rl   front / middle / rear, left        fr, mr, rr   same, right
"""
from __future__ import annotations

import numpy as np

from .. import parts
from ..physics import beam
from .base import Component, fmt


class RockerBogie(Component):
    kind = "suspension"

    def __init__(self, wheel_r: float, wheel_w: float, spacing: float, pivot_h: float, bogie_h: float,
                 track: float, body_width: float, body_height: float, tube: str, margin: float = 1.08) -> None:
        super().__init__()
        self.r, self.w, self.s, self.H, self.zb, self.track = wheel_r, wheel_w, spacing, pivot_h, bogie_h, track
        self.bw, self.bh = body_width, body_height
        self.tube_name, self.tube = tube, parts.get("rover_tube", tube)
        A, _, _ = beam.square_tube(self.tube["side_mm"] / 1000, self.tube["wall_mm"] / 1000)
        self.a = self.tube["side_mm"] / 1000
        self.lin = self.tube["density"] * A * margin          # kg/m including the margin

    def tube_geom(self, p0, p1) -> str:
        L = float(np.linalg.norm(np.subtract(p1, p0)))
        return (f'<geom type="capsule" fromto="{fmt(p0)} {fmt(p1)}" size="{self.a/2:.4f}" material="alu" '
                f'mass="{self.lin*L:.4f}" contype="0" conaffinity="0"/>')

    def attach(self, robot, at) -> None:
        chassis, rn = at.body, robot.name
        r, s, H, zb, track, a, w = self.r, self.s, self.H, self.zb, self.track, self.a, self.w
        bh = self.bh
        # differential bar across the top of the chassis (visual; its mass is a chassis part)
        chassis.geoms.append(f'<geom type="capsule" fromto="-.05 {-track/2+w/2+a:.3f} {bh/2+.03} -.05 '
                             f'{track/2-w/2-a:.3f} {bh/2+.03}" size="{a/2:.3f}" material="alu" contype="0" conaffinity="0"/>')
        from .base import MassPart
        chassis.mass_parts.append(MassPart(self.lin / chassis.mass_scale * track, np.zeros(3), np.zeros((3, 3)),
                                           "differential bar"))
        for sgn, side in ((1, "l"), (-1, "r")):
            yr = sgn * (self.bw / 2 + a)                       # rocker plane just outside the body
            dy = sgn * track / 2 - yr                          # wheel offset from the rocker plane
            rocker = robot.new_body(f"{rn}_rocker_{side}", chassis, (0, yr, 0), pos_str=f"0 {yr:.4f} 0",
                                    joint=f'<joint name="{rn}_rocker_{side}" type="hinge" axis="0 1 0" range="-35 35" damping="2"/>')
            rocker.geoms += [self.tube_geom([0, 0, 0], [s, 0, r - H + 0.6 * r]),
                             self.tube_geom([s, 0, r - H + 0.6 * r], [s, dy, r - H + 0.6 * r]),
                             self.tube_geom([0, 0, 0], [-s / 2, 0, zb - H])]
            self.publish(f"f{side}", rocker, [s, dy, r - H], pos_str=f"{s:.4f} {dy:.4f} {r - H:.4f}")
            # order=1: listed after the front corner (added later by a SteeringServo), as in the
            # original model, so body/joint numbering is unchanged
            bogie = robot.new_body(f"{rn}_bogie_{side}", rocker, (-s / 2, 0, zb - H), pos_str=f"{-s/2:.4f} 0 {zb - H:.4f}",
                                   order=1,
                                   joint=f'<joint name="{rn}_bogie_{side}" type="hinge" axis="0 1 0" range="-40 40" damping="1"/>')
            bogie.geoms += [self.tube_geom([0, 0, 0], [s / 2, 0, r - zb + 0.6 * r]),
                            self.tube_geom([0, 0, 0], [-s / 2, 0, r - zb + 0.6 * r]),
                            self.tube_geom([s / 2, 0, r - zb + 0.6 * r], [s / 2, dy, r - zb + 0.6 * r]),
                            self.tube_geom([-s / 2, 0, r - zb + 0.6 * r], [-s / 2, dy, r - zb + 0.6 * r])]
            mid = robot.new_body(f"{rn}_mid_{side}", bogie, (s / 2, dy, r - zb), pos_str=f"{s/2:.4f} {dy:.4f} {r - zb:.4f}")
            self.publish(f"m{side}", mid, [0, 0, 0])
            self.publish(f"r{side}", bogie, [-s / 2, dy, r - zb], pos_str=f"{-s/2:.4f} {dy:.4f} {r - zb:.4f}")
        robot.contacts += [f'<exclude body1="{rn}" body2="{rn}_rocker_l"/>',
                           f'<exclude body1="{rn}" body2="{rn}_rocker_r"/>']
        robot.equalities.append(
            f'<joint name="{rn}_differential" joint1="{rn}_rocker_l" joint2="{rn}_rocker_r" polycoef="0 -1 0 0 0"/>')
        robot.sensors += [f'<jointpos name="{rn}_rocker_l_q" joint="{rn}_rocker_l"/>',
                          f'<jointpos name="{rn}_bogie_l_q" joint="{rn}_bogie_l"/>',
                          f'<jointpos name="{rn}_bogie_r_q" joint="{rn}_bogie_r"/>']

    def summary(self) -> str:
        return (f"RockerBogie: track {self.track:.2f} m, wheelbase {2*self.s:.2f} m, "
                f"{self.tube_name} legs")
