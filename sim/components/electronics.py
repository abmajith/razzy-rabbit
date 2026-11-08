"""Electronics boards (flight controller, radio, companion computer) and camera modules.

Board: an `avionics.*` part (mass, power, size) as a box on a mount. On a forward-facing
mount (the nose) the long side is turned across the vehicle.

CameraModule: a board that also carries cameras (e.g. an OAK-D: colour + stereo depth).
Its cameras look forward and slightly down from the lens face; their vertical field of view
comes from the chosen `rgb_camera.*` / `depth_camera.*` parts. Camera names:
`<robot>_fpv` (colour) and `<robot>_depth`.
"""
from __future__ import annotations

import numpy as np

from .. import parts
from .base import Component, MassPart, box_inertia, fmt

LOOK_FORWARD_DOWN = "0 -1 0 .259 0 .966"     # 15 deg below the horizon


class Board(Component):
    kind = "board"
    material = "carbon"

    def __init__(self, part: str) -> None:
        super().__init__()
        self.part_name, self.spec = part, parts.get("avionics", part)

    def attach(self, robot, at) -> None:
        body = at.body
        sx, sy, sz = (s / 1000 for s in self.spec["size_mm"])
        forward = abs(at.normal[0]) > 0.5
        if forward:
            sx, sy = sy, sx                       # long side across the nose
        pos = at.place(sx if forward else sz)
        m = self.spec["mass_g"] / 1000
        body.mass_parts.append(MassPart(m, pos, box_inertia(m, sx, sy, sz), f"board {self.part_name}"))
        body.geoms.append(f'<geom type="box" size="{sx/2:.4f} {sy/2:.4f} {sz/2:.4f}" pos="{fmt(pos)}" '
                          f'material="{self.material}" contype="0" conaffinity="0"/>')
        robot.info["avionics_w"] = robot.info.get("avionics_w", 0.0) + self.spec["power_w"]
        self.pos, self.size = pos, (sx, sy, sz)

    def summary(self) -> str:
        return f"Board: {self.part_name} ({self.spec['mass_g']:.0f} g, {self.spec['power_w']} W)"


class CameraModule(Board):
    kind = "camera"
    material = "lens"

    def __init__(self, part: str, rgb_vfov_deg: float | None = 80, depth_vfov_deg: float | None = 58) -> None:
        super().__init__(part)
        self.rgb_vfov, self.depth_vfov = rgb_vfov_deg, depth_vfov_deg

    def attach(self, robot, at) -> None:
        super().attach(robot, at)
        lens = self.pos + np.asarray(at.normal) * (self.size[0] / 2 + 0.002)
        rn = robot.name
        self.rgb_camera, self.depth_camera = f"{rn}_fpv", f"{rn}_depth"
        at.body.cameras.append(f'<camera name="{self.rgb_camera}" pos="{fmt(lens)}" xyaxes="{LOOK_FORWARD_DOWN}" '
                               f'fovy="{self.rgb_vfov}"/>')
        at.body.cameras.append(f'<camera name="{self.depth_camera}" pos="{fmt(lens)}" xyaxes="{LOOK_FORWARD_DOWN}" '
                               f'fovy="{self.depth_vfov}"/>')
        self.publish("lens", at.body, lens, at.normal)

    def summary(self) -> str:
        return f"CameraModule: {self.part_name} ({self.spec['mass_g']:.0f} g), rgb {self.rgb_vfov} deg, depth {self.depth_vfov} deg"
