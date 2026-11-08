"""Sensors with a place on the vehicle, and view cameras.

Imu: a site where the gyro and accelerometer measure (`<robot>_imu`, with MuJoCo `gyro`
and `accelerometer` sensors). Mounted at a named point; `frame.mount("cg")` puts it at the
body's centre of mass. Its noise, bias and vibration models (sim/sensing/) are chosen in
the mission file's [sensors] table; in step 4 they move into this component.

ViewCamera: a camera that is not a sensor part (downward view, chase view for the viewer).
"""
from __future__ import annotations

import numpy as np

from ..components.base import Component, fmt


class Imu(Component):
    kind = "imu"

    def __init__(self, site_size: float = 0.008) -> None:
        super().__init__()
        self.site_size = site_size

    def attach(self, robot, at) -> None:
        rn = robot.name
        self.site = f"{rn}_imu"
        if at.pos is None:                       # centre of mass: known only when the robot is complete
            at.body.sites.append(lambda body: f'<site name="{self.site}" pos="{fmt(body.com())}" size="{self.site_size:g}"/>')
        elif np.allclose(at.pos, 0):
            at.body.sites.append(f'<site name="{self.site}" size="{self.site_size:g}"/>')
        else:
            at.body.sites.append(f'<site name="{self.site}" pos="{fmt(at.pos)}" size="{self.site_size:g}"/>')
        robot.sensors += [f'<gyro name="{rn}_gyro" site="{self.site}"/>',
                          f'<accelerometer name="{rn}_acc" site="{self.site}"/>']

    def summary(self) -> str:
        return f"Imu: site {self.site}"


class ViewCamera(Component):
    kind = "view"

    def __init__(self, suffix: str, xyaxes: str, fovy: float | None = None, offset=(0, 0, 0),
                 mode: str | None = None) -> None:
        super().__init__()
        self.suffix, self.xyaxes, self.fovy, self.offset, self.mode = suffix, xyaxes, fovy, offset, mode

    def attach(self, robot, at) -> None:
        self.camera = f"{robot.name}_{self.suffix}"
        pos = np.asarray(at.pos, float) + np.asarray(self.offset, float)
        attrs = f' mode="{self.mode}"' if self.mode else ""
        fov = f' fovy="{self.fovy:g}"' if self.fovy else ""
        at.body.cameras.append(f'<camera name="{self.camera}"{attrs} pos="{fmt(pos)}" xyaxes="{self.xyaxes}"{fov}/>')

    def summary(self) -> str:
        return f"ViewCamera: {self.camera}"
