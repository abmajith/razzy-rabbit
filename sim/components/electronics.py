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

    def contribute(self, robot, meta: dict) -> None:
        meta["power"]["loads"].append({"type": "constant", "group": "avionics", "label": self.part_name,
                                       "watts": self.spec["power_w"]})

    def summary(self) -> str:
        return f"Board: {self.part_name} ({self.spec['mass_g']:.0f} g, {self.spec['power_w']} W)"


class CameraModule(Board):
    kind = "camera"
    material = "lens"

    def __init__(self, part: str, rgb: str | None = None, depth: str | None = None) -> None:
        """rgb / depth: `rgb_camera.*` / `depth_camera.*` sensor parts carried by the module
        (their datasheets set the field of view and drive the camera error models)."""
        super().__init__(part)
        self.rgb = parts.get("rgb_camera", rgb) if rgb else None
        self.depth = parts.get("depth_camera", depth) if depth else None
        self.rgb_name, self.depth_name = rgb, depth
        self.rgb_vfov = self.rgb["vfov_deg"] if self.rgb else 80
        self.depth_vfov = self.depth["vfov_deg"] if self.depth else 58

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

    def contribute(self, robot, meta: dict) -> None:
        super().contribute(robot, meta)
        for kind, spec, part, cam in (("rgb_camera", self.rgb, self.rgb_name, self.rgb_camera),
                                      ("depth_camera", self.depth, self.depth_name, self.depth_camera)):
            if spec:
                meta["sensors"].append(dict(spec, type=kind, name=self.name, part=part, camera=cam))

    def summary(self) -> str:
        return f"CameraModule: {self.part_name} ({self.spec['mass_g']:.0f} g), rgb {self.rgb_vfov} deg, depth {self.depth_vfov} deg"


class Computer(Component):
    """Where the policy runs (`compute.*` part: inference time, rate). Its mass and power are
    those of the board it runs on (e.g. the camera module or a companion computer), so it
    adds none; it tells the runtime how late actions arrive."""

    kind = "computer"

    def __init__(self, part: str) -> None:
        super().__init__()
        self.part_name, self.spec = part, parts.get("compute", part)

    def attach(self, robot, at) -> None:
        pass

    def contribute(self, robot, meta: dict) -> None:
        meta["compute"] = dict(self.spec, part=self.part_name)

    def summary(self) -> str:
        return f"Computer: {self.part_name}, inference {self.spec['inference_ms']} ms at {self.spec['rate_hz']} Hz"
