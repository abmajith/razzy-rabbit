"""Load a generated scene and step it with its reference controller + wind."""
from __future__ import annotations

import json
import struct

import mujoco
import numpy as np

from .. import scenes as _scenes  # noqa: F401  (registers the scenes)
from ..builder import GENERATED
from ..control import CONTROLLERS
from ..physics.aero import WindField
from ..world.terrain import Terrain


def scene_dir(name: str, seed: int):
    return GENERATED / f"{name}-s{seed}"


def load_terrain(meta: dict, folder) -> Terrain | None:
    tm = meta.get("terrain")
    if not tm:
        return None
    raw = (folder / tm["file"]).read_bytes()
    nr, nc = struct.unpack("ii", raw[:8])
    z = np.frombuffer(raw[8:], np.float32).reshape(nr, nc).astype(float) * tm["elev"] + tm["zmin"]
    return Terrain(z, tm["half_x"], tm["half_y"])


class Sim:
    """A generated scene with its controller, plant, sensors and wind.

    variant=None runs the nominal vehicle. variant=k draws vehicle k from the mission's
    [uncertainty] ranges (deterministic per (seed, k)); reset(k) switches variants without
    reloading the scene.
    """

    def __init__(self, name: str, seed: int = 0, rebuild: bool = False, controller: bool = True,
                 sensors: bool = True, record: bool = False, variant: int | None = None) -> None:
        folder = scene_dir(name, seed)
        if rebuild or not (folder / "scene.xml").exists():
            from ..scenes import build
            print(f"[sim] generating {name} (seed {seed}) ...", flush=True)
            build(name, seed)
        self.folder, self.seed = folder, seed
        self.use_controller, self.use_sensors, self.record = controller, sensors, record
        self.m = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
        self.d = mujoco.MjData(self.m)
        self.meta0 = json.loads((folder / "meta.json").read_text())
        self.terrain = load_terrain(self.meta0, folder)
        from ..uncertainty import ModelSnapshot
        self._nominal = ModelSnapshot(self.m)
        self.reset(variant)

    def reset(self, variant: int | None = None) -> None:
        from .. import uncertainty as U
        m, d = self.m, self.d
        self._nominal.restore(m)
        mujoco.mj_setConst(m, d)
        mujoco.mj_resetData(m, d)
        self.variant, self.params, self.rejected = variant, {}, 0
        robot0 = self.meta0["robots"][0] if self.meta0.get("robots") else {}
        unc = self.meta0.get("uncertainty", {})
        if variant is not None:
            if not unc:
                raise SystemExit("this scene has no [uncertainty] spec (designed vehicles only)")
            rng = np.random.default_rng([self.seed, 0xD0A1, variant])
            self.params, self.rejected = U.sample(unc, robot0, rng)
        self.meta = U.apply_meta(self.meta0, self.params) if self.params else self.meta0
        self.wind = WindField(self.meta["wind"], self.terrain, self.seed) if "wind" in self.meta else None
        mujoco.mj_forward(m, d)
        # controllers are built against the NOMINAL model (what the designer knows) ...
        self.ctrls = []
        if self.use_controller and self.meta.get("controller") in CONTROLLERS:
            self.ctrls.append(CONTROLLERS[self.meta["controller"]](self, self.meta))
        # ... the physics gets the drawn vehicle
        if self.params:
            U.apply_model(m, d, self.robot, self.params)
            mujoco.mj_forward(m, d)
        # power train and rotor aerodynamics: generic, from what the components declared
        self.power = None
        if "power" in self.robot:
            from .power import PowerSystem
            self.power = PowerSystem(self, self.robot["power"])
            self.ctrls.append(self.power)
        if "rotor_drag" in self.robot:
            from .aero import RotorDrag
            self.ctrls.append(RotorDrag(self, self.robot["rotor_drag"]))
        self.sensors = None
        if self.use_sensors and self.robot.get("sensors"):    # sensor models, sampled after each step
            from .sensors import SensorSuite
            noise_seed = self.seed if variant is None else self.seed + 1_000_003 * (variant + 1)
            self.sensors = SensorSuite(self, self.robot, noise_seed, record=self.record)

    def param_vector(self) -> tuple[list[str], np.ndarray]:
        from ..uncertainty import flatten
        return flatten(self.params)

    @property
    def robot(self) -> dict:
        return self.meta["robots"][0]

    def step(self, n: int = 1) -> None:
        dt = self.m.opt.timestep
        for _ in range(n):
            t0 = self.d.time
            for c in self.ctrls:
                c.update(t0, dt)
            mujoco.mj_step(self.m, self.d)
            if self.sensors is not None:
                self.sensors.update(t0, dt)

    # ---- visual helpers ---------------------------------------------------------------
    def draw_wind(self, scn: mujoco.MjvScene, center, half: float = 12.0, n: int = 7, scale: float = 0.25) -> None:
        """Add a grid of wind arrows (coloured by speed) around `center` into a scene."""
        if self.wind is None:
            return
        g = np.linspace(-half, half, n)
        X, Y = np.meshgrid(g + center[0], g + center[1])
        P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, center[2])], 1)
        V = self.wind.velocity(P, self.d.time)
        vmax = max(self.wind.U * 1.4, 1.0)
        for p, v in zip(P, V):
            if scn.ngeom >= scn.maxgeom:
                break
            s = np.linalg.norm(v)
            if s < 1e-3:
                continue
            c = np.clip(s / vmax, 0, 1)
            geom = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                np.array([0.2 + 0.8 * c, 0.9 - 0.6 * c, 1.0 - 0.9 * c, 0.85], np.float32))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.06, p, p + v * scale)
            scn.ngeom += 1
