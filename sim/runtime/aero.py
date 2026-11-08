"""Aerodynamic effects of rotors that MuJoCo doesn't model.

Rotor (blade-flapping) drag: a spinning rotor moving through air tilts its thrust back and
produces a horizontal force opposing the airspeed in the rotor plane, roughly proportional
to total thrust:

    F = -k_drag * sum(T) * v_perp,   v_perp = airspeed minus its component along the body z axis

(Body drag in the wind is applied by the mission controller; see physics.aero.)
"""
from __future__ import annotations

import numpy as np


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
