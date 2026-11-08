"""Aerodynamic effects of rotors that MuJoCo doesn't model.

Rotor (blade-flapping) drag: a spinning rotor moving through air tilts its thrust back and
produces a horizontal force opposing the airspeed in the rotor plane, roughly proportional
to total thrust:

    F = -k_drag * sum(T) * v_perp,   v_perp = airspeed minus its component along the body z axis

Body drag: quadratic drag of the whole vehicle in the wind field (physics.aero),
F = 1/2 rho CdA |v_rel| v_rel, assigned every step (rotor drag is added on top).
"""
from __future__ import annotations

import numpy as np


class BodyDrag:
    def __init__(self, sim, robot: dict) -> None:
        m = sim.m
        self.sim, self.CdA = sim, robot.get("drag_CdA", 0.04)
        self.bid = m.body(robot["body"]).id
        jnt = m.body_jntadr[self.bid]
        self.qa, self.va = m.jnt_qposadr[jnt], m.jnt_dofadr[jnt]

    def update(self, t: float, dt: float) -> None:
        sim = self.sim
        d = sim.d
        pos, vel = d.qpos[self.qa:self.qa + 3], d.qvel[self.va:self.va + 3]
        f = sim.wind.drag_force(pos, vel, self.CdA, t) if sim.wind is not None else 0.0
        d.xfrc_applied[self.bid, :3] = f


class RotorDrag:
    def __init__(self, sim, spec: dict) -> None:
        m = sim.m
        self.sim, self.k = sim, spec["rotor_drag"]
        self.bid = m.body(spec["body"]).id
        self.act = np.array([m.actuator(a).id for a in spec["actuators"]])

    def update(self, t: float, dt: float) -> None:
        sim = self.sim
        d, m = sim.d, sim.m
        T = d.actuator_force[self.act]
        R = d.xmat[self.bid].reshape(3, 3)
        vel = d.qvel[m.jnt_dofadr[m.body_jntadr[self.bid]]:][:3]
        if sim.wind is not None:
            vel = vel - sim.wind.velocity(d.xpos[self.bid], t)[0]
        z = R[:, 2]
        v_perp = vel - (vel @ z) * z
        d.xfrc_applied[self.bid, :3] += -self.k * float(np.sum(T)) * v_perp
