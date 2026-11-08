"""Multirotor stages: geometric SE(3) control and the rotor mixer.

Se3Controller (Lee, Leok, McClamroch 2010), from a pose setpoint (p, v, yaw):

    a   = -kp e_p - kd (v - v_d) + g z       desired acceleration (e_p clipped to 3 m,
                                             tilt limited to 35 deg, a_z >= 2 m/s^2)
    F   = m a,  thrust T = F . body z        collective thrust along the current body axis
    R_d from b3 = F/|F| and the desired yaw  desired attitude
    e_R = vee(R_d^T R - R^T R_d) / 2         attitude error
    tau = J (-kR e_R - kW omega) + omega x J omega

m and J are the *nominal* (designed) values, read when the stage is bound: the controller
does not know the randomised vehicle. Output: wrench [T, tau_x, tau_y, tau_z].

Mixer: rotor thrusts from the wrench, f = A^-1 [T, tau] with A built from each rotor's
position relative to the centre of mass and its yaw torque per newton; clipped to [0, f_max]
where f_max is read live from the actuators (the power train lowers it as the battery sags).
"""
from __future__ import annotations

import numpy as np

from .base import Controller


class Se3Controller(Controller):
    name = "se3"
    defaults = {"kp": 5.0, "kd": 4.0, "kR": 140.0, "kW": 22.0, "max_tilt_deg": 35.0}

    def bind(self, sim, robot) -> None:
        m = sim.m
        bid = m.body(robot["body"]).id
        self.mass = float(m.body_subtreemass[bid])
        self.J = m.body_inertia[bid].copy()

    def update(self, t, dt, state, sp, a_ff=np.zeros(3)):
        prm = self.params
        R = state.R
        w = state.omega
        e_p = state.pos - sp["p"]
        n = np.linalg.norm(e_p)
        if n > 3.0:
            e_p *= 3.0 / n
        a = -prm["kp"] * e_p - prm["kd"] * (state.vel - sp["v"]) + a_ff + [0, 0, 9.81]
        a[2] = max(a[2], 2.0)
        hmax = np.tan(np.radians(prm["max_tilt_deg"])) * a[2]
        hn = np.linalg.norm(a[:2])
        if hn > hmax:
            a[:2] *= hmax / hn
        F = self.mass * a
        b3 = F / np.linalg.norm(F)
        yaw_d = sp["yaw"]
        b1c = np.array([np.cos(yaw_d), np.sin(yaw_d), 0.0])
        b2 = np.cross(b3, b1c)
        b2 /= np.linalg.norm(b2)
        Rd = np.column_stack([np.cross(b2, b3), b2, b3])
        E = 0.5 * (Rd.T @ R - R.T @ Rd)
        eR = np.array([E[2, 1], E[0, 2], E[1, 0]])
        tau = self.J * (-prm["kR"] * eR - prm["kW"] * w) + np.cross(w, self.J * w)
        T = F @ R[:, 2]
        return np.r_[T, tau]


class Mixer(Controller):
    name = "mixer"
    defaults: dict = {}

    def bind(self, sim, robot) -> None:
        m = sim.m
        self.m, self.d = m, sim.d
        bid = m.body(robot["body"]).id
        self.act = np.array([m.actuator(a).id for a in robot["actuators"]])
        A = np.zeros((4, 4))
        com = m.body_ipos[bid]                 # torques act about the centre of mass
        for i, aid in enumerate(self.act):
            x, y, _ = m.site_pos[m.actuator_trnid[aid, 0]] - com
            A[:, i] = [1.0, y, -x, m.actuator_gear[aid, 5]]
        self.Ainv = np.linalg.inv(A)

    def update(self, t, dt, state, wrench):
        fmax = self.m.actuator_ctrlrange[self.act, 1]   # live: battery sag lowers it
        f = np.clip(self.Ainv @ wrench, 0, fmax)
        self.d.ctrl[self.act] = f
        return f


class VelocityCommand(Controller):
    """Top stage for a policy / joystick: set `.command = [vx, vy, vz, yaw_rate]` (world frame);
    outputs a pose setpoint that holds the current position and tracks the velocity
    (same as the C++ core's "setpoint" mode)."""

    name = "velocity_command"
    defaults: dict = {}

    def bind(self, sim, robot) -> None:
        from .base import TruthState
        self.command = np.zeros(4)
        self.yaw = float(TruthState(sim, robot).yaw)

    def update(self, t, dt, state, x):
        c = self.command
        self.yaw += c[3] * dt
        return {"p": state.pos.copy(), "v": np.asarray(c[:3], float), "yaw": self.yaw}
