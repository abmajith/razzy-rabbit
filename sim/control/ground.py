"""Ground-vehicle stages: Ackermann steering and wheel-speed control.

Ackermann (four steered corners, all wheels driven), from (speed v, curvature kappa):

    each steered wheel at (x_i, y_i):  delta_i = atan2(x_i kappa, 1 - y_i kappa)
    each wheel's speed:                omega_i = v |(1 - y_i kappa, x_i kappa)| / r

so every wheel rolls on a circle around the same centre (no scrubbing). Steering angles go
straight to the servos; the wheel-speed targets go to WheelSpeedPI.

WheelSpeedPI: duty u = omega_t / omega_0 (the DC motor's feed-forward: no-load speed per
unit duty) + kp e + integral(ki e), e = (omega_t - omega) / omega_0, integral clipped to
+-limit, u clipped to [-1, 1].
"""
from __future__ import annotations

import numpy as np

from .base import Controller


class Ackermann(Controller):
    name = "ackermann"
    defaults: dict = {}

    def bind(self, sim, robot) -> None:
        m = sim.m
        self.d = sim.d
        self.steer = np.array([m.actuator(f"{s}_p").id for s in robot["steer_xy"]])
        self.sxy = np.array(list(robot["steer_xy"].values()))
        self.wxy = np.array(robot["wheel_xy"])
        self.rw = robot["wheel_radius"]

    def update(self, t, dt, state, cmd):
        v, kappa = cmd
        self.d.ctrl[self.steer] = np.arctan2(self.sxy[:, 0] * kappa, 1 - self.sxy[:, 1] * kappa)
        return v * np.hypot(1 - self.wxy[:, 1] * kappa, self.wxy[:, 0] * kappa) / self.rw


class WheelSpeedPI(Controller):
    name = "wheel_speed_pi"
    defaults = {"kp": 1.5, "ki": 3.0, "limit": 0.6}

    def bind(self, sim, robot) -> None:
        m = sim.m
        self.d = sim.d
        self.drive = np.array([m.actuator(a).id for a in robot["drive"]])
        self.jdof = np.array([m.jnt_dofadr[m.joint(f"{w}_j").id] for w in robot["wheels"]])
        self.w0 = robot["omega0"]
        self.integ = np.zeros(len(self.drive))

    def update(self, t, dt, state, w_t):
        prm = self.params
        w = state.joint_velocity(self.jdof)
        e = (w_t - w) / self.w0
        self.integ = np.clip(self.integ + prm["ki"] * e * dt, -prm["limit"], prm["limit"])
        u = np.clip(w_t / self.w0 + prm["kp"] * e + self.integ, -1, 1)
        self.d.ctrl[self.drive] = u
        return u


class DriveCommand(Controller):
    """Top stage for a policy / joystick: set `.command = [speed, curvature]`
    (same as the C++ core's rover "setpoint" mode); curvature is clipped to the steering."""

    name = "drive_command"
    defaults = {"max_steer_deg": 38.0}

    def bind(self, sim, robot) -> None:
        sxy = np.array(list(robot["steer_xy"].values()))
        kmax = np.tan(np.radians(self.params["max_steer_deg"])) / np.max(np.abs(sxy[:, 0]))
        self.kmax = kmax / (1 + kmax * np.max(np.abs(sxy[:, 1])))
        self.command = np.zeros(2)

    def update(self, t, dt, state, x):
        return float(self.command[0]), float(np.clip(self.command[1], -self.kmax, self.kmax))
