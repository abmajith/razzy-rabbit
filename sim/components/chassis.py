"""Ground-vehicle chassis (a box body) and a camera mast.

Chassis creates the root body: a box of the given size whose mass is everything attached to
it (battery, electronics, payload, its own tube frame...), times a fastener/cabling margin,
with the inertia of a uniform box (`Body.inertial = "box"`).

Mounts (x forward, z up):
    center     body centre (IMU, internal battery, electronics)
    deck       top surface (+z)
    mast       front of the deck, for a Mast (+z)
    nose_low   low on the front face, for a hazard camera (+x)
    origin     body origin (chase camera)
"""
from __future__ import annotations

import numpy as np

from .. import parts
from ..physics import beam
from .base import Component, MassPart, fmt


class Chassis(Component):
    kind = "chassis"

    def __init__(self, length: float, width: float, height: float, frame_tube: str | None = None,
                 margin: float = 1.08, solar_deck: bool = True) -> None:
        super().__init__()
        self.size = (length, width, height)
        self.tube = parts.get("rover_tube", frame_tube) if frame_tube else None
        self.margin, self.solar_deck = margin, solar_deck

    def attach(self, robot, at) -> None:
        bl, bw, bh = self.size
        rn = robot.name
        robot.material("rover_body", [0.85, 0.85, 0.82, 1], specular=0.3, shininess=0.3)
        robot.material("gold_foil", [0.85, 0.65, 0.25, 1], specular=0.8, shininess=0.8, reflectance=0.1)
        body = robot.new_body(rn, None, (0, 0, 0), inertial="box", box_size=self.size, mass_scale=self.margin)
        self.body = body
        body.geoms.append(f'<geom name="{rn}_body" type="box" size="{bl/2} {bw/2} {bh/2}" material="rover_body"/>')
        if self.solar_deck:
            body.geoms.append(f'<geom type="box" size="{bl/2+.02} {bw/2+.02} .006" pos="0 0 {bh/2+.006}" material="solar" '
                              'contype="0" conaffinity="0"/>')
            body.geoms.append(f'<geom type="box" size="{bl/2-.03} {bw/2+.002} {bh/2-.03}" material="gold_foil" '
                              'contype="0" conaffinity="0"/>')
        if self.tube:                              # perimeter tube frame of the body
            A, _, _ = beam.square_tube(self.tube["side_mm"] / 1000, self.tube["wall_mm"] / 1000)
            m = self.tube["density"] * A * 2 * (bl + bw)
            body.mass_parts.append(MassPart(m, np.zeros(3), np.zeros((3, 3)), "body tube frame"))
        self.publish("center", body, [0, 0, 0])
        self.publish("deck", body, [0, 0, bh / 2], (0, 0, 1))
        self.publish("mast", body, [bl / 2 - .06, 0, bh / 2], (0, 0, 1))
        self.publish("nose_low", body, [bl / 2 + .005, 0, -bh / 2 + .02], (1, 0, 0))
        self.publish("origin", body, [0, 0, 0])

    def summary(self) -> str:
        return "Chassis: {:.2f} x {:.2f} x {:.2f} m".format(*self.size)


class Mast(Component):
    """Vertical mast with a stereo camera head; the camera is `<robot>_navcam`."""

    kind = "mast"

    def __init__(self, height: float = 0.45, fovy: float = 70, xyaxes: str = "0 -1 0 .34 0 .94") -> None:
        super().__init__()
        self.height, self.fovy, self.xyaxes = height, fovy, xyaxes

    def attach(self, robot, at) -> None:
        b0, h, body = np.asarray(at.pos, float), self.height, at.body
        top = b0 + [0, 0, h]
        head = b0 + [0.02, 0, h + 0.03]
        body.geoms.append(f'<geom type="cylinder" fromto="{fmt(b0)} {fmt(top)}" size=".015" material="alu" '
                          'contype="0" conaffinity="0"/>')
        body.geoms.append(f'<geom type="box" size=".03 .07 .03" pos="{fmt(head)}" material="rover_body" '
                          'contype="0" conaffinity="0"/>')
        for dy in (.035, -.035):
            body.geoms.append(f'<geom type="sphere" size=".012" pos="{fmt(head + [0.035, dy, 0])}" material="lens" '
                              'contype="0" conaffinity="0"/>')
        self.camera = f"{robot.name}_navcam"
        body.cameras.append(f'<camera name="{self.camera}" pos="{fmt(head + [0.04, 0, 0])}" xyaxes="{self.xyaxes}" '
                            f'fovy="{self.fovy:g}"/>')

    def summary(self) -> str:
        return f"Mast: {self.height:.2f} m, camera {self.camera}"
