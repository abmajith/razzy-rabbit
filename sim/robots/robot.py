"""Robot: assemble components into one MuJoCo model.

    robot = Robot("uav")
    frame = robot.add(QuadFrame(...))                          # creates the root body
    robot.add(Battery("lipo_6s_2200"), at=frame.mount("deck"))
    for arm in frame.arms:
        robot.add(Rotor(...), at=arm.tip)
    robot.summary()                                            # mass, CoM, parts list
    robot.write(builder, pos=[x, y, z])                        # body tree + actuators + sensors

The first component that creates a body without a parent becomes the root (it gets the
free joint). Components register what the runtime needs to know (actuator names, wheel
positions, rotor sites...) in `robot.info`; templates turn that into the scene meta.
"""
from __future__ import annotations

import math

import numpy as np

from ..components.base import Body, Component, Mount, fmt


class Robot:
    def __init__(self, name: str) -> None:
        self.name = name
        self.root: Body | None = None
        self.components: list[Component] = []
        self.actuators: list[str] = []
        self.sensors: list[str] = []
        self.equalities: list[str] = []
        self.contacts: list[str] = []
        self.materials: list[tuple] = []          # (name, rgba, kwargs) registered at write
        self.info: dict = {}                      # runtime facts collected from components
        self.environment: dict = {}               # site conditions components need (air density, magnetic field)
        self._counts: dict[str, int] = {}

    # ------------------------------------------------------------------ assembly ---
    def add(self, component: Component, at: Mount | None = None, name: str | None = None) -> Component:
        """Attach a component (at a named mount point of another component)."""
        if at is not None and not isinstance(at, Mount):
            raise TypeError(f"`at` must be a mount (e.g. frame.mount('deck')), got {at!r}")
        component.robot = self
        component.name = name or self.next_name(component.kind)
        component.attach(self, at)
        self.components.append(component)
        return component

    def next_name(self, kind: str) -> str:
        i = self._counts.get(kind, 0)
        self._counts[kind] = i + 1
        return f"{kind}{i}"

    def index(self, kind: str) -> int:
        """How many components of this kind were added before (0 for the first)."""
        return self._counts.get(kind, 1) - 1

    def new_body(self, name: str, parent: Body | None, pos, *, joint: str | None = None, inertial: str = "geoms",
                 pos_str: str = "", **kw) -> Body:
        b = Body(name=name, pos=np.asarray(pos, float), parent=parent, joint=joint, inertial=inertial,
                 pos_str=pos_str, **kw)
        if parent is None:
            if self.root is not None:
                raise ValueError(f"robot '{self.name}' already has a root body ({self.root.name})")
            self.root = b
        else:
            parent.children.append(b)
        return b

    def material(self, name: str, rgba, **kw) -> None:
        self.materials.append((name, rgba, kw))

    def bodies(self) -> list[Body]:
        out, stack = [], [self.root] if self.root else []
        while stack:
            b = stack.pop(0)
            out.append(b)
            stack = sorted(b.children, key=lambda c: c.order) + stack
        return out

    def runtime_meta(self) -> dict:
        """What the runtime needs, collected from the components (`contribute`):
        power  : {battery, loads: [rotor | gearmotor | constant ...]}   -> runtime/power.py
        sensors: [ {type, part, datasheet values, site/camera/antenna...} ] -> runtime/sensors.py
        compute, rotor_drag: when a Computer / Rotors are present."""
        meta: dict = {"power": {"battery": None, "loads": []}, "sensors": []}
        for c in self.components:
            c.contribute(self, meta)
        if meta["power"]["battery"] is None:
            del meta["power"]
        return meta

    # ---------------------------------------------------------------- inspection ---
    def mass_properties(self) -> dict:
        """Mass, CoM and inertia of the root body (the vehicle's main body), and total mass
        including child bodies that carry MassParts."""
        r = self.root
        return {"root_mass": r.mass(), "root_com": r.com(), "root_inertia": r.inertia(),
                "total_parts_mass": sum(b.mass() for b in self.bodies())}

    def cameras(self) -> list[str]:
        """Camera names, chase view first, then onboard cameras (what view/render use)."""
        names = []
        for c in self.components:
            for attr in ("camera", "rgb_camera", "depth_camera"):
                if hasattr(c, attr):
                    names.append(getattr(c, attr))
        return sorted(names, key=lambda n: (not n.endswith("_chase"), names.index(n)))

    def meta(self, **extra) -> dict:
        """Everything the runtime needs about this robot: runtime_meta() (power, sensors,
        compute, rotor drag) + body, mass, cameras, and the actuator lists the controller
        stages read (rotors: `actuators`; wheels: `drive`, `wheels`, ...). `extra` is merged in."""
        m = self.runtime_meta()
        m.update(body=self.root.name, mass=self.mass_properties()["root_mass"], cameras=self.cameras())
        rotors = self.info.get("rotors", [])
        if rotors:
            m.update(type="quadrotor", actuators=[r.actuator for r in rotors])
        wheels, steers = self.info.get("wheels", []), self.info.get("steers", [])
        if wheels:
            m.update(type="rocker_bogie" if steers else "wheeled", wheels=[w.body_name for w in wheels],
                     wheel_xy=[list(map(float, w.xy)) for w in wheels], wheel_radius=wheels[0].radius,
                     drive=[w.actuator for w in wheels], omega0=wheels[0].omega0, tau_stall=wheels[0].tau_s)
        if steers:
            m.update(steer=[s.actuator for s in steers], steer_xy={s.joint: list(map(float, s.xy)) for s in steers})
        m.update(extra)
        return m

    def compile(self):
        """Compile the robot alone (on an empty floor) into a MuJoCo model, e.g. to read
        exact totals: model.body_subtreemass[model.body(robot.name).id]."""
        import tempfile
        from pathlib import Path

        import mujoco

        from ..builder import Builder
        with tempfile.TemporaryDirectory() as tmp:
            b = Builder(f"robot_{self.name}", 0, out_root=Path(tmp))
            self.write(b, [0, 0, 1.0])
            return mujoco.MjModel.from_xml_path(str(b.write()))

    def summary(self) -> str:
        mp = self.mass_properties()
        m = self.compile()
        total = float(m.body_subtreemass[m.body(self.root.name).id])
        lines = [f"robot '{self.name}': {len(self.components)} components, {len(self.bodies())} bodies, "
                 f"{len(self.actuators)} actuators, total mass {total:.3f} kg",
                 f"  root body '{self.root.name}': {mp['root_mass']:.3f} kg, CoM {fmt(mp['root_com'])} m"]
        for c in self.components:
            lines.append(f"  - {c.name:12s} {c.summary()}")
        rt = self.runtime_meta()
        if "power" in rt:
            pw = rt["power"]
            loads = []
            for ld in pw["loads"]:
                if ld["type"] == "rotor":
                    loads.append(f"{len(ld['actuators'])} rotors (from thrust)")
                elif ld["type"] == "gearmotor":
                    loads.append(f"{len(ld['actuators'])} gearmotors (from torque)")
            groups: dict = {}
            for ld in pw["loads"]:
                if ld["type"] == "constant":
                    groups[ld["group"]] = groups.get(ld["group"], 0) + ld["watts"]
            loads += [f"{g} {w:g} W" for g, w in groups.items()]
            lines.append(f"  power: {pw['battery']['part']} -> " + ", ".join(loads))
        if rt["sensors"]:
            lines.append("  sensors: " + ", ".join(f"{e['type']} {e['part']}" for e in rt["sensors"]))
        return "\n".join(lines)

    # -------------------------------------------------------------------- output ---
    def write(self, b, pos, yaw: float | None = None) -> None:
        """Emit materials, the body tree, actuators, sensors, equalities and contact
        exclusions into a scene Builder."""
        from ..builder import robot_materials
        robot_materials(b)
        for name, rgba, kw in self.materials:
            b.material(name, rgba, **kw)
        root = self.root
        root.pos = np.asarray(pos, float)
        root.pos_str = fmt(pos)
        if yaw is not None:
            root.attrs = f' quat="{math.cos(yaw/2):.5f} 0 0 {math.sin(yaw/2):.5f}"'
        root.joint = f'<freejoint name="{self.name}_root"/>'
        b.world.append("\n    ".join(root.xml()))
        b.actuators += [a() if callable(a) else a for a in self.actuators]
        # ground truth of the main body (for controllers / rewards, not a sensor model)
        b.sensors += self.sensors + [f'<framepos name="{self.name}_pos" objtype="body" objname="{root.name}"/>',
                                     f'<framequat name="{self.name}_quat" objtype="body" objname="{root.name}"/>']
        b.equalities += self.equalities
        b.contacts += self.contacts
