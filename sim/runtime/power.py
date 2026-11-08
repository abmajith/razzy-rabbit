"""Generic power train: one battery, every component's electrical load, voltage sag fed back.

Each component that draws power declares a *load* in the robot meta (`power.loads`):

    rotor       motor + prop + ESC: electrical power from the actual (lagged) rotor thrust,
                P = sum(V I) / eta_esc   (physics.propeller.rotor_point)
                feedback: max thrust ~ (V / V_ref)^2, V_ref = pack OCV at 50 % charge
    gearmotor   brushed DC drive: P = sum(V_pack u I(tau))   (physics.dc_motor)
                feedback: stall torque ~ V / V_nominal
    constant    boards, servos, sensors: a fixed draw, in named groups ("avionics", "servos")

Per physics step (after the controllers, before mj_step):

    P = dynamic loads + constant groups      ->  V, I = pack.terminal(P, soc)   (physics.battery)
    soc -= I dt / capacity                   ->  each load applies the new voltage to its actuators

A robot with rotors *and* wheels needs nothing special: both kinds of load are summed.
"""
from __future__ import annotations

import numpy as np

from ..physics.battery import Pack
from ..physics.dc_motor import gearmotor_current
from ..physics.propeller import rotor_point


class RotorLoad:
    def __init__(self, sim, spec: dict, pack: Pack) -> None:
        m = sim.m
        self.spec = spec
        self.act = np.array([m.actuator(a).id for a in spec["actuators"]])
        self.v_ref = pack.ocv(0.5)

    def power(self, d, v_bus: float) -> float:
        s = self.spec
        T = d.actuator_force[self.act]                      # actual (lagged) rotor thrust
        op = rotor_point(T, s["motor"], s["prop"], s["rho"])
        return float(np.sum(op["Pe"])) / s["esc_eff"]

    def apply_voltage(self, m, v: float, soc: float) -> None:
        scale = (v / self.v_ref) ** 2 if soc > 0 else 0.0
        m.actuator_ctrlrange[self.act, 1] = self.spec["t_max_ref"] * scale


class GearmotorLoad:
    def __init__(self, sim, spec: dict, pack: Pack) -> None:
        m = sim.m
        self.spec = spec
        self.act = np.array([m.actuator(a).id for a in spec["actuators"]])
        mot = spec["motor"]
        self.i0, self.i_s = mot["no_load_current"], mot["stall_current"] * spec["v_ratio"]
        self.tau_s = np.asarray(spec["tau_s"], float)        # per motor (differs when randomised)
        self.v_nom = pack.v_nom

    def power(self, d, v_bus: float) -> float:
        tau = np.abs(d.actuator_force[self.act])
        u = np.abs(d.ctrl[self.act])
        i_mot = gearmotor_current(tau, self.i0, self.i_s, self.tau_s, saturate=True)
        return float(np.sum(v_bus * u * i_mot))

    def apply_voltage(self, m, v: float, soc: float) -> None:
        scale = v / self.v_nom if soc > 0 else 0.0
        m.actuator_gainprm[self.act, 0] = self.tau_s * scale        # stall torque follows voltage
        m.actuator_forcerange[self.act, 0] = -self.tau_s * scale
        m.actuator_forcerange[self.act, 1] = self.tau_s * scale


LOADS = {"rotor": RotorLoad, "gearmotor": GearmotorLoad}


class PowerSystem:
    """Battery + loads; exposes soc, v, i (amps), power (W), energy_wh."""

    def __init__(self, sim, spec: dict) -> None:
        self.sim, self.spec = sim, spec
        self.pack = Pack(spec["battery"])
        self.dynamic = [LOADS[s["type"]](sim, s, self.pack) for s in spec["loads"] if s["type"] in LOADS]
        groups: dict[str, float] = {}
        for s in spec["loads"]:
            if s["type"] == "constant":
                groups[s["group"]] = groups.get(s["group"], 0) + s["watts"]
        self.constant = list(groups.values())           # summed per group, groups in order of first use
        self.constant_groups = groups
        self.soc, self.v, self.i, self.power = 1.0, self.pack.ocv(1.0), 0.0, 0.0
        self.energy_wh = 0.0

    def update(self, t: float, dt: float) -> None:
        d, m = self.sim.d, self.sim.m
        p = 0.0
        for k, load in enumerate(self.dynamic):
            p = load.power(d, self.v) if k == 0 else p + load.power(d, self.v)
        for w in self.constant:
            p += w
        self.power = p
        self.v, self.i = self.pack.terminal(self.power, self.soc)
        self.soc = max(0.0, self.soc - self.i * dt / 3600.0 / self.pack.cap_ah)
        self.energy_wh += self.power * dt / 3600.0
        for load in self.dynamic:
            load.apply_voltage(m, self.v, self.soc)

    def status(self) -> str:
        return f"battery {self.soc*100:5.1f}%  {self.v:5.2f} V  {self.power:6.1f} W"

    def budget(self) -> str:
        """Constant loads by group (W) and the battery, for summaries."""
        groups = ", ".join(f"{g} {w:g} W" for g, w in self.constant_groups.items())
        return f"{self.spec['battery'].get('part', 'battery')}: constant loads {groups}"
