"""What every component is made of: bodies, named mount points, mass parts.

A robot is a tree of rigid **bodies** (MuJoCo bodies). Components either create bodies
(structure: a frame, a chassis, a rocker-bogie, a steering knuckle, a wheel) or add things to
an existing body (devices: a rotor, a battery, a board, an IMU...). Every component that can
carry others publishes **named mount points**: a body, a position on it, and a direction in
which an attached part extends (a battery sits *up* on a deck, a camera sticks *forward*
out of the nose, a board hangs *down* under the plate).

    robot.add(Battery("lipo_6s_2200"), at=frame.mount("deck"))
    frame.mounts()          # -> names you can attach to

A body's mass properties come from one of three rules (`Body.inertial`):
    "parts"  explicit <inertial> summed from MassParts with the parallel-axis theorem
             (the quadrotor: every part's mass, position and own inertia)
    "box"    total of its MassParts, inertia of a uniform box of the body's size
             (the rover chassis: a lumped box)
    "geoms"  no <inertial>: MuJoCo infers it from geom masses (tubes, wheels)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------- helpers ---
def fmt(v) -> str:
    """Vector to MJCF text (4 significant digits, like the rest of the generator)."""
    return " ".join(f"{float(x):.4g}" for x in np.ravel(v))


def box_inertia(m: float, sx: float, sy: float, sz: float) -> np.ndarray:
    """Inertia of a uniform box about its centre."""
    return m / 12 * np.diag([sy * sy + sz * sz, sx * sx + sz * sz, sx * sx + sy * sy])


def rod_inertia(m: float, length: float, axis) -> np.ndarray:
    """Thin rod of given length along unit `axis`, about its centre."""
    u = np.asarray(axis, float)
    return m / 12 * length * length * (np.eye(3) - np.outer(u, u))


@dataclass
class MassPart:
    """A lump of mass on a body: mass [kg], position of its centre [m, body frame], and
    its inertia about its own centre [kg m^2]."""
    mass: float
    pos: np.ndarray
    inertia: np.ndarray
    label: str = ""


# -------------------------------------------------------------------------- bodies ---
@dataclass
class Body:
    name: str
    pos: np.ndarray
    parent: "Body | None" = None
    joint: str | None = None                 # MJCF joint element (or None: welded to parent)
    inertial: str = "geoms"                  # "parts" | "box" | "geoms"
    box_size: tuple = (0, 0, 0)              # for inertial == "box"
    mass_scale: float = 1.0                  # e.g. 1 + wiring fraction, fastener margin
    geoms: list = field(default_factory=list)       # MJCF strings, or callables(body) -> str
    sites: list = field(default_factory=list)
    cameras: list = field(default_factory=list)
    children: list = field(default_factory=list)
    mass_parts: list = field(default_factory=list)
    attrs: str = ""                          # extra body attributes, e.g. quat="..."
    pos_str: str = ""                        # exact MJCF text for pos (default: fmt(pos))
    order: int = 0                           # position among siblings in the MJCF tree (numbering only)

    # mass properties from MassParts (used by "parts"/"box" bodies and by summaries)
    def parts_mass(self) -> float:
        return sum(p.mass for p in self.mass_parts)

    def com(self) -> np.ndarray:
        if not self.mass_parts:
            return np.zeros(3)
        return sum(p.mass * p.pos for p in self.mass_parts) / sum(p.mass for p in self.mass_parts)

    def mass(self) -> float:
        return self.parts_mass() * self.mass_scale if self.inertial in ("parts", "box") else self.parts_mass()

    def inertia(self) -> np.ndarray:
        """Inertia about the body's centre of mass (parallel-axis sum, scaled)."""
        com = self.com()
        scale = self.mass_scale
        I = np.zeros((3, 3))
        for p in self.mass_parts:
            r = p.pos - com
            I += scale * (p.inertia + p.mass * (r @ r * np.eye(3) - np.outer(r, r)))
        return I

    def inertial_xml(self) -> str | None:
        if self.inertial == "parts":
            I = self.inertia()
            full = f"{I[0,0]:.6g} {I[1,1]:.6g} {I[2,2]:.6g} {I[0,1]:.6g} {I[0,2]:.6g} {I[1,2]:.6g}"
            return f'<inertial pos="{fmt(self.com())}" mass="{self.mass():.5f}" fullinertia="{full}"/>'
        if self.inertial == "box":
            lx, ly, lz = self.box_size
            mb = self.mass()
            Ib = mb / 12 * np.array([ly * ly + lz * lz, lx * lx + lz * lz, lx * lx + ly * ly])
            return f'<inertial pos="0 0 0" mass="{mb:.4f}" diaginertia="{fmt(Ib)}"/>'
        return None

    def xml(self, ind: str = "") -> list[str]:
        out = [f'{ind}<body name="{self.name}" pos="{self.pos_text()}"{self.attrs}>']
        inner = ind + "  "
        if self.joint:
            out.append(inner + self.joint)
        inert = self.inertial_xml()
        if inert:
            out.append(inner + inert)
        for item in self.geoms + self.sites + self.cameras:
            out.append(inner + (item(self) if callable(item) else item))
        for c in sorted(self.children, key=lambda c: c.order):     # stable: equal order keeps add order
            out += c.xml(inner)
        out.append(f"{ind}</body>")
        return out

    def pos_text(self) -> str:
        return self.pos_str or fmt(self.pos)

    def world_offset(self) -> np.ndarray:
        """Position of this body's origin in the root body frame (robot at rest, no rotations)."""
        p, b = np.zeros(3), self
        while b.parent is not None:
            p = p + b.pos
            b = b.parent
        return p


# -------------------------------------------------------------------------- mounts ---
@dataclass
class Mount:
    """A named place to attach a component: a body, a position on it (None = the body's
    centre of mass, resolved when the robot is written), and the direction an attached part
    extends in (+z sits on top, -z hangs below, +x sticks out forward)."""
    name: str
    body: Body
    pos: np.ndarray | None
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    owner: "Component | None" = None         # the component that published it
    pos_str: str = ""                        # exact MJCF text for a body created here

    def place(self, extent_along_normal: float) -> np.ndarray:
        """Centre of a part of the given size along the normal, touching the mount."""
        return np.asarray(self.pos, float) + self.normal * extent_along_normal / 2


# ---------------------------------------------------------------------- component ---
class Component:
    """Base class. Subclasses implement `attach(robot, at)`, which adds bodies, geoms,
    sites, actuators, sensors and MassParts to the robot, and may publish mounts."""

    kind = "component"

    def __init__(self) -> None:
        self.name: str | None = None
        self.robot = None
        self._mounts: dict[str, Mount] = {}

    def attach(self, robot, at: Mount | None) -> None:          # pragma: no cover - abstract
        raise NotImplementedError

    # mounts this component offers to others
    def publish(self, name: str, body: Body, pos, normal=(0, 0, 1), pos_str: str = "") -> Mount:
        m = Mount(name, body, None if pos is None else np.asarray(pos, float), np.asarray(normal, float),
                  owner=self, pos_str=pos_str)
        self._mounts[name] = m
        return m

    def contribute(self, robot, meta: dict) -> None:
        """Add what the runtime needs to the robot meta (an electrical load, a sensor
        entry...). Called once the robot is complete, so robot-wide facts (mass, centre of
        mass, other components) are available. Default: nothing."""

    def mount(self, name: str) -> Mount:
        if name not in self._mounts:
            raise KeyError(f"{type(self).__name__} '{self.name}' has no mount '{name}'; "
                           f"mounts: {list(self._mounts)}")
        return self._mounts[name]

    def mounts(self) -> list[str]:
        return list(self._mounts)

    def summary(self) -> str:
        return type(self).__name__
