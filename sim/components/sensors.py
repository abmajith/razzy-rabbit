"""Sensors with a place on the vehicle, and view cameras.

Each sensor component takes a part from sim/parts/sensors.toml (datasheet noise, rates,
latency) plus how it is installed, and contributes one entry to the robot's `sensors` meta;
runtime/sensors.py builds the error models (sim/sensing/models.py) from those entries.

    Imu    gyro + accelerometer at a site (`<robot>_imu`); soft-mount frequency/damping; it
           also collects the rotor vibration sources of the robot (imbalance, blade ripple,
           arm bending mode) because that is what it measures
    Baro   barometric altitude at a site (`<robot>_baro`); static-port airspeed error
    Mag    compass; Earth field of the site + field from the battery current
    Gnss   receiver whose antenna sits at a mount (+ height); sky view by ray casting
    ViewCamera  a camera that is not a sensor part (downward view, chase view)
"""
from __future__ import annotations

import numpy as np

from .. import parts
from ..physics import propeller
from ..components.base import Component, fmt


def _site(at, name: str, size: float) -> str | object:
    """A site at a mount (at the body's centre of mass when the mount has no position)."""
    if at.pos is None:
        return lambda body: f'<site name="{name}" pos="{fmt(body.com())}" size="{size:g}"/>'
    if np.allclose(at.pos, 0):
        return f'<site name="{name}" size="{size:g}"/>'
    return f'<site name="{name}" pos="{fmt(at.pos)}" size="{size:g}"/>'


class Imu(Component):
    kind = "imu"

    def __init__(self, part: str | None = None, mount_hz: float = 1e4, mount_zeta: float = 0.15,
                 site_size: float = 0.008) -> None:
        """part: `imu.*` sensor part (None: only the site + ideal MuJoCo gyro/accelerometer).
        mount_hz / mount_zeta: soft-mount natural frequency and damping (1e4 = hard mounted)."""
        super().__init__()
        self.part_name = part
        self.spec = parts.get("imu", part) if part else None
        self.mount_hz, self.mount_zeta, self.site_size = mount_hz, mount_zeta, site_size

    def attach(self, robot, at) -> None:
        rn = robot.name
        self.site, self.gyro, self.acc = f"{rn}_imu", f"{rn}_gyro", f"{rn}_acc"
        at.body.sites.append(_site(at, self.site, self.site_size))
        robot.sensors += [f'<gyro name="{self.gyro}" site="{self.site}"/>',
                          f'<accelerometer name="{self.acc}" site="{self.site}"/>']

    def contribute(self, robot, meta: dict) -> None:
        if not self.spec:
            return
        entry = dict(self.spec, type="imu", name=self.name, part=self.part_name, site=self.site,
                     gyro=self.gyro, acc=self.acc, vibration=self.vibration(robot))
        meta["sensors"].append(entry)

    def vibration(self, robot) -> dict | None:
        """Rotor vibration sources as seen at the IMU (see sensing.models.RotorVibration)."""
        rotors = robot.info.get("rotors")
        if not rotors:
            return None
        mp = robot.mass_properties()
        com, r0 = mp["root_com"], rotors[0]
        D = propeller.diameter_m(r0.prop)
        rho = robot.environment["air_density"]
        return dict(rotor_pos=[list(r.hub - com) for r in rotors], spin=[r.spin for r in rotors],
                    mass=mp["root_mass"], inertia=mp["root_inertia"].tolist(), blades=r0.prop.get("blades", 2),
                    imbalance_gmm=r0.imbalance_gmm, thrust_ripple=r0.thrust_ripple,
                    arm_hz=robot.info.get("arm_mode_hz", 1e9), arm_zeta=robot.info.get("arm_zeta", 0.02),
                    mount_hz=self.mount_hz, mount_zeta=self.mount_zeta,
                    rotor_actuators=[r.actuator for r in rotors], kn=1.0 / (r0.prop["ct"] * rho * D ** 4))

    def summary(self) -> str:
        return f"Imu: {self.part_name or 'ideal'}, site {self.site}" + (
            f", soft mount {self.mount_hz:g} Hz" if self.spec and self.mount_hz < 1e4 else "")


class Baro(Component):
    kind = "baro"

    def __init__(self, part: str, static_cp: float = 0.3) -> None:
        super().__init__()
        self.part_name, self.spec, self.static_cp = part, parts.get("baro", part), static_cp

    def attach(self, robot, at) -> None:
        self.site = f"{robot.name}_baro"
        at.body.sites.append(_site(at, self.site, 0.006))

    def contribute(self, robot, meta: dict) -> None:
        meta["sensors"].append(dict(self.spec, type="baro", name=self.name, part=self.part_name,
                                    static_cp=self.static_cp, site=self.site))

    def summary(self) -> str:
        return f"Baro: {self.part_name}"


class Mag(Component):
    kind = "mag"

    def __init__(self, part: str, current_ut_per_a: float = 0.0) -> None:
        """current_ut_per_a: field at the compass per amp of battery current (wiring layout)."""
        super().__init__()
        self.part_name, self.spec, self.current = part, parts.get("mag", part), current_ut_per_a

    def attach(self, robot, at) -> None:
        self.body = at.body.name

    def contribute(self, robot, meta: dict) -> None:
        meta["sensors"].append(dict(self.spec, type="mag", name=self.name, part=self.part_name,
                                    current_ut_per_a=self.current,
                                    environment=dict(robot.environment.get("magnetic_field", {}))))

    def summary(self) -> str:
        return f"Mag: {self.part_name}"


class Gnss(Component):
    kind = "gnss"

    def __init__(self, part: str, antenna_height: float = 0.0) -> None:
        """antenna_height: antenna above the mount point [m]."""
        super().__init__()
        self.part_name, self.spec, self.height = part, parts.get("gnss", part), antenna_height

    def attach(self, robot, at) -> None:
        self.antenna = np.asarray(at.pos, float) + [0, 0, self.height]

    def contribute(self, robot, meta: dict) -> None:
        meta["sensors"].append(dict(self.spec, type="gnss", name=self.name, part=self.part_name,
                                    antenna=[float(x) for x in self.antenna]))

    def summary(self) -> str:
        return f"Gnss: {self.part_name}, antenna at {fmt(self.antenna)} m"


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
