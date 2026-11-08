"""Rotor: brushless motor + propeller (+ ESC), mounted at an arm tip.

Adds the motor (with its mount) and the prop as mass parts, a thrust site at the prop hub,
and a MuJoCo actuator whose command is thrust [N]:

    thrust follows the command through a first-order lag (motor time constant, from
    physics.dc_motor.speed_loop_time_constant in the sizing), up to `thrust_max`
    reaction torque about the hub = spin x yaw_coef x thrust, yaw_coef = cp D / (2 pi ct)

Its electrical load (power from thrust) and the battery-sag limit on thrust are applied at
run time by the generic power train (sim/runtime/power.py); its rotor drag by
sim/runtime/aero.py. Its imbalance and blade-pass ripple feed the IMU vibration model.
"""
from __future__ import annotations

import numpy as np

from .. import parts
from ..physics import propeller
from .base import Component, MassPart, box_inertia


class Rotor(Component):
    kind = "rotor"

    def __init__(self, motor: str, prop: str, spin: int, thrust_max: float, lag_s: float,
                 motor_mount_g: float = 0.0, hub_height: float = 0.03, esc_eff: float = 0.93,
                 imbalance_gmm: float = 0.5, thrust_ripple: float = 0.03, drag_coef: float = 0.012) -> None:
        """imbalance_gmm / thrust_ripple: vibration sources seen by the IMU (installation);
        drag_coef: rotor (blade-flapping) drag per newton of thrust per m/s of airspeed."""
        super().__init__()
        self.esc_eff, self.imbalance_gmm, self.thrust_ripple, self.drag_coef = esc_eff, imbalance_gmm, thrust_ripple, drag_coef
        self.motor_name, self.prop_name = motor, prop
        self.motor, self.prop = parts.get("motor", motor), parts.get("prop", prop)
        self.spin, self.thrust_max, self.lag_s = spin, thrust_max, lag_s
        self.mount_g, self.zp = motor_mount_g, hub_height

    def attach(self, robot, at) -> None:
        body, tip = at.body, at.pos
        i = robot.index("rotor")
        rn = robot.name
        D = propeller.diameter_m(self.prop)
        r_mot = self.motor["stator_mm"] / 1000 / 2 + 0.003
        m_mm = self.motor["mass_g"] / 1000 + self.mount_g / 1000           # motor + its mount
        body.mass_parts.append(MassPart(m_mm, tip + [0, 0, 0.012], box_inertia(m_mm, 2 * r_mot, 2 * r_mot, 0.02),
                                        f"motor {self.name}"))
        body.geoms.append(f'<geom type="cylinder" size="{r_mot:.4f} .011" pos="{tip[0]:.4f} {tip[1]:.4f} .012" '
                          'material="alu" contype="0" conaffinity="0"/>')
        m_prop = self.prop["mass_g"] / 1000
        zp = self.zp
        body.mass_parts.append(MassPart(m_prop, tip + [0, 0, zp], m_prop * D * D / 24 * np.diag([0.5, 0.5, 1.0]),
                                        f"prop {self.name}"))
        body.geoms.append(f'<geom name="{rn}_prop{i}" type="cylinder" size="{D/2:.4f} .003" '
                          f'pos="{tip[0]:.4f} {tip[1]:.4f} {zp}" material="prop" contype="0" conaffinity="0"/>')
        site = f"{rn}_rotor{i}"
        body.sites.append(f'<site name="{site}" pos="{tip[0]:.4f} {tip[1]:.4f} {zp}" size=".008"/>')
        self.k_yaw = propeller.yaw_moment_coef(D, self.prop["ct"], self.prop["cp"])
        tau = max(self.lag_s, 0.005)
        self.actuator = f"{rn}_m{i}"
        robot.actuators.append(
            f'<general name="{self.actuator}" site="{site}" gear="0 0 1 0 0 {self.spin * self.k_yaw:.5f}" '
            f'dyntype="filter" dynprm="{tau:.4f}" gaintype="fixed" gainprm="1" ctrlrange="0 {self.thrust_max:.3f}"/>')
        self.hub = np.array([tip[0], tip[1], zp])
        robot.info.setdefault("rotors", []).append(self)
        self.publish("hub", body, self.hub)

    def contribute(self, robot, meta: dict) -> None:
        loads = meta["power"]["loads"]
        group = next((ld for ld in loads if ld["type"] == "rotor" and ld["motor_name"] == self.motor_name
                      and ld["prop_name"] == self.prop_name), None)
        if group is None:                          # identical rotors share one load entry
            group = {"type": "rotor", "motor_name": self.motor_name, "prop_name": self.prop_name, "actuators": [],
                     "motor": self.motor, "prop": self.prop, "rho": robot.environment["air_density"],
                     "esc_eff": self.esc_eff, "t_max_ref": self.thrust_max}
            loads.append(group)
        group["actuators"].append(self.actuator)
        drag = meta.setdefault("rotor_drag", {"body": robot.root.name, "actuators": [], "rotor_drag": self.drag_coef})
        drag["actuators"].append(self.actuator)

    def summary(self) -> str:
        return (f"Rotor: {self.motor_name} + {self.prop_name}, spin {'+' if self.spin > 0 else '-'}, "
                f"max {self.thrust_max:.1f} N, lag {self.lag_s*1000:.0f} ms")
