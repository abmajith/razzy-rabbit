"""Battery pack (and generic fixed loads).

Battery: a `battery.*` part placed on a mount (its box sits along the mount normal).
It publishes a `top` mount: things strapped on top of the pack (GNSS mast, companion
computer), slightly aft of the pack centre.

Payload: any fixed mass with an optional continuous electrical load and no geometry of its
own (electronics lumped inside a chassis, a sample payload...).
"""
from __future__ import annotations

import numpy as np

from .. import parts
from .base import Component, MassPart, box_inertia


class Battery(Component):
    kind = "battery"

    def __init__(self, part: str, visible: bool = True) -> None:
        super().__init__()
        self.part_name, self.spec, self.visible = part, parts.get("battery", part), visible

    def attach(self, robot, at) -> None:
        body = at.body
        bx, by, bz = (s / 1000 for s in self.spec["size_mm"])
        m = self.spec["mass_g"] / 1000
        pos = at.place(bz)
        body.mass_parts.append(MassPart(m, pos, box_inertia(m, bx, by, bz), f"battery {self.part_name}"))
        if self.visible:
            body.geoms.append(f'<geom type="box" size="{bx/2:.4f} {by/2:.4f} {bz/2:.4f}" pos="0 0 {pos[2]:.4f}" '
                              'material="accent" contype="0" conaffinity="0"/>')
        self.publish("top", body, np.asarray(at.pos, float) + [-0.02, 0, bz], (0, 0, 1))
        robot.info["battery"] = self.spec

    def contribute(self, robot, meta: dict) -> None:
        if meta["power"]["battery"] is not None:
            raise ValueError("one battery per robot (for now)")
        meta["power"]["battery"] = dict(self.spec, part=self.part_name)

    def summary(self) -> str:
        s = self.spec
        return f"Battery: {self.part_name} ({s['cells']}S {s['chemistry']}, {s['capacity_mah']:.0f} mAh, {s['mass_g']:.0f} g)"


class Payload(Component):
    kind = "payload"

    def __init__(self, mass_kg: float, power_w: float = 0.0, label: str = "payload") -> None:
        super().__init__()
        self.mass, self.power, self.label = mass_kg, power_w, label

    def attach(self, robot, at) -> None:
        at.body.mass_parts.append(MassPart(self.mass, np.asarray(at.pos, float), np.zeros((3, 3)), self.label))
        robot.info["avionics_w"] = robot.info.get("avionics_w", 0.0) + self.power

    def contribute(self, robot, meta: dict) -> None:
        if self.power:
            meta["power"]["loads"].append({"type": "constant", "group": "avionics", "label": self.label,
                                           "watts": self.power})

    def summary(self) -> str:
        return f"Payload: {self.label}, {self.mass:.2f} kg" + (f", {self.power:.0f} W" if self.power else "")
