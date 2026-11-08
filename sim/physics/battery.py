"""Battery packs.

A pack of `cells` in series behaves like an open-circuit voltage source with an internal
resistance:

    E(soc)  = cells x OCV_cell(soc)      OCV curve per chemistry (sim/parts/energy.toml)
    R       = cells x r_cell
    V       = E - I R                    terminal voltage under load

Delivering a power P: P = V I = E I - R I^2  ->  I = (E - sqrt(E^2 - 4 R P)) / (2 R).
If E^2 < 4 R P the pack cannot deliver P at all (it would collapse to V = E/2).
Charge used: d(soc)/dt = -I / capacity.
"""
from __future__ import annotations

import numpy as np

from .. import parts

CHEMISTRY: dict = parts.load()["chemistry"]   # {name: {nominal, cutoff, soc[], ocv[]}}


def ocv_cell(chem: str, soc):
    """Open-circuit voltage of one cell at state of charge soc (0..1)."""
    c = CHEMISTRY[chem]
    return np.interp(soc, c["soc"], c["ocv"])


class Pack:
    """Battery pack from a `battery.*` part: OCV(soc) x cells - I R."""

    def __init__(self, spec: dict) -> None:
        self.spec = spec
        self.chem = spec["chemistry"]
        self.cells = spec["cells"]
        self.cap_ah = spec["capacity_mah"] / 1000
        self.r = spec["r_cell"] * spec["cells"]
        self.i_max = spec["c_max"] * self.cap_ah
        self.mass = spec["mass_g"] / 1000
        c = CHEMISTRY[self.chem]
        self.v_nom = c["nominal"] * self.cells
        self.v_cut = c["cutoff"] * self.cells
        self.energy_wh = self.v_nom * self.cap_ah

    def ocv(self, soc) -> float:
        return float(ocv_cell(self.chem, soc)) * self.cells

    def terminal(self, power_w: float, soc: float) -> tuple[float, float]:
        """Terminal voltage and current while delivering `power_w` at state of charge `soc`."""
        e = self.ocv(soc)
        disc = e * e - 4 * self.r * power_w
        if disc <= 0:                       # pack cannot deliver this power at all
            return e / 2, e / (2 * self.r)
        i = (e - np.sqrt(disc)) / (2 * self.r)
        return e - i * self.r, i


def soc_at_cutoff(pack: Pack) -> float:
    """State of charge where OCV reaches the end-of-discharge level (cutoff + 0.1 V/cell)."""
    socs = np.linspace(0, 1, 201)
    v = np.array([pack.ocv(s) for s in socs])
    return float(socs[np.argmax(v >= pack.v_cut + 0.1 * pack.cells)])
