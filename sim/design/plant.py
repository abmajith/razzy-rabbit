"""Runtime power-train models for designed vehicles ("plants").

They run after the controllers every physics step:
  * compute electrical power from what the actuators are actually doing,
  * integrate battery state of charge, update terminal voltage (OCV - I R),
  * feed voltage sag back into the actuators (less thrust / torque as the pack drains),
  * drone only: rotor drag (horizontal force proportional to thrust x relative airspeed).
"""
from __future__ import annotations

import numpy as np

from .common import Pack
from .drone import rotor_point


class DronePlant:
    def __init__(self, sim, robot: dict) -> None:
        p = robot["plant"]
        self.sim, self.p = sim, p
        m = sim.m
        self.bid = m.body(robot["body"]).id
        self.act = np.array([m.actuator(a).id for a in robot["actuators"]])
        self.pack = Pack(p["battery"])
        self.v_ref = self.pack.ocv(0.5)
        self.soc, self.v, self.i, self.power = 1.0, self.pack.ocv(1.0), 0.0, 0.0
        self.energy_wh = 0.0

    def update(self, t: float, dt: float) -> None:
        sim, p = self.sim, self.p
        d, m = sim.d, sim.m
        T = d.actuator_force[self.act]                      # actual (lagged) rotor thrust
        op = rotor_point(T, p["motor"], p["prop"], p["rho"])
        self.power = float(np.sum(op["Pe"])) / p["esc_eff"] + p["avionics_w"]
        self.v, self.i = self.pack.terminal(self.power, self.soc)
        self.soc = max(0.0, self.soc - self.i * dt / 3600.0 / self.pack.cap_ah)
        self.energy_wh += self.power * dt / 3600.0
        scale = (self.v / self.v_ref) ** 2 if self.soc > 0 else 0.0
        m.actuator_ctrlrange[self.act, 1] = p["t_max_ref"] * scale
        # rotor drag: opposes airspeed in the rotor plane, proportional to total thrust
        R = d.xmat[self.bid].reshape(3, 3)
        vel = d.qvel[m.jnt_dofadr[m.body_jntadr[self.bid]]:][:3]
        if sim.wind is not None:
            vel = vel - sim.wind.velocity(d.xpos[self.bid], t)[0]
        z = R[:, 2]
        v_perp = vel - (vel @ z) * z
        d.xfrc_applied[self.bid, :3] += -p["rotor_drag"] * float(np.sum(T)) * v_perp

    def status(self) -> str:
        return f"battery {self.soc*100:5.1f}%  {self.v:5.2f} V  {self.power:6.1f} W"


class RoverPlant:
    def __init__(self, sim, robot: dict) -> None:
        p = robot["plant"]
        self.sim, self.p = sim, p
        m = sim.m
        self.act = np.array([m.actuator(a).id for a in robot["drive"]])
        self.pack = Pack(p["battery"])
        self.v_nom = self.pack.v_nom
        mot = p["motor"]
        self.i0, self.i_s = mot["no_load_current"], mot["stall_current"] * p["v_ratio"]
        self.tau_s = p["tau_stall"]
        self.soc, self.v, self.i, self.power = 1.0, self.pack.ocv(1.0), 0.0, 0.0
        self.energy_wh = 0.0
        self.n_servo = len(robot["steer"])

    def update(self, t: float, dt: float) -> None:
        d, m, p = self.sim.d, self.sim.m, self.p
        tau = np.abs(d.actuator_force[self.act])
        u = np.abs(d.ctrl[self.act])
        i_mot = self.i0 + (self.i_s - self.i0) * np.minimum(tau / self.tau_s, 1.0)
        self.power = float(np.sum(self.v * u * i_mot)) + p["avionics_w"] + self.n_servo * p["servo_w"]
        self.v, self.i = self.pack.terminal(self.power, self.soc)
        self.soc = max(0.0, self.soc - self.i * dt / 3600.0 / self.pack.cap_ah)
        self.energy_wh += self.power * dt / 3600.0
        scale = self.v / self.v_nom if self.soc > 0 else 0.0
        m.actuator_gainprm[self.act, 0] = self.tau_s * scale        # stall torque follows voltage
        m.actuator_forcerange[self.act, 0] = -self.tau_s * scale
        m.actuator_forcerange[self.act, 1] = self.tau_s * scale

    def status(self) -> str:
        return f"battery {self.soc*100:5.1f}%  {self.v:5.2f} V  {self.power:6.1f} W"


def make_plant(sim, robot: dict):
    kind = robot["plant"]["kind"]
    return {"drone": DronePlant, "rover": RoverPlant}[kind](sim, robot)
