"""Load a generated scene and step it with its reference controller + wind."""
from __future__ import annotations

import json
import struct

import mujoco
import numpy as np

from . import scenes as _scenes  # noqa: F401  (registers sim/custom controllers)
from .builder import GENERATED
from .control import CONTROLLERS, TurbineSpin
from .terrain import Terrain
from .wind import WindField


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
    def __init__(self, name: str, seed: int = 0, rebuild: bool = False, controller: bool = True,
                 sensors: bool = True, record: bool = False) -> None:
        folder = scene_dir(name, seed)
        if rebuild or not (folder / "scene.xml").exists():
            from .scenes import build
            print(f"[sim] generating {name} (seed {seed}) ...", flush=True)
            build(name, seed)
        self.folder = folder
        self.m = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
        self.d = mujoco.MjData(self.m)
        self.meta = json.loads((folder / "meta.json").read_text())
        self.terrain = load_terrain(self.meta, folder)
        self.wind = WindField(self.meta["wind"], self.terrain, seed) if "wind" in self.meta else None
        mujoco.mj_forward(self.m, self.d)
        self.ctrls = []
        if controller and self.meta.get("controller") in CONTROLLERS:
            self.ctrls.append(CONTROLLERS[self.meta["controller"]](self, self.meta))
        if "rotor_speed" in self.meta:
            self.ctrls.append(TurbineSpin(self, self.meta))
        self.plant = None
        if "plant" in self.robot:          # designed vehicle: battery + motor model, runs last
            from .design.plant import make_plant
            self.plant = make_plant(self, self.robot)
            self.ctrls.append(self.plant)
        self.sensors = None
        if sensors and "sensors" in self.robot:   # grounded sensor models, sampled after each step
            from .sensing.suite import SensorSuite
            self.sensors = SensorSuite(self, self.robot, seed, record=record)

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
