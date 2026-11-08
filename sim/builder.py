"""Scene builder: collects MJCF fragments, merged vegetation meshes and metadata, then
writes a self-contained folder  sim/generated/<scene>-s<seed>/  that MuJoCo (Python or
C/C++) loads directly with  mj_loadXML("scene.xml").
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .mesh import MeshBatch

GENERATED = Path(__file__).resolve().parent / "generated"


def fmt(v) -> str:
    return " ".join(f"{float(x):.4g}" for x in np.ravel(v))


def lookat_xyaxes(offset, up=(0, 0, 1)) -> str:
    """xyaxes for a camera at `offset` (relative to target) that looks at the target."""
    z = np.asarray(offset, float)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return fmt(np.concatenate([x, y]))


@dataclass
class Lighting:
    """Randomised time-of-day / atmosphere (cheap domain randomisation)."""
    sun_dir: np.ndarray
    sun_rgb: np.ndarray
    sky_top: np.ndarray
    sky_horizon: np.ndarray
    ambient: float
    haze: float

    @staticmethod
    def sample(rng: np.random.Generator, overcast: float | None = None) -> "Lighting":
        elev = np.deg2rad(rng.uniform(30, 68))
        azim = rng.uniform(0, 2 * np.pi)
        sun_dir = -np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
        warm = 1.0 - (elev - np.deg2rad(30)) / np.deg2rad(38)   # 1 = low sun
        oc = rng.uniform(0, 0.6) if overcast is None else overcast
        sun_rgb = np.array([1.0, 0.93 - 0.12 * warm, 0.82 - 0.25 * warm]) * (0.95 - 0.45 * oc)
        sky_top = (1 - oc) * np.array([0.30, 0.52, 0.85]) + oc * np.array([0.62, 0.65, 0.70])
        sky_hor = (1 - oc) * np.array([0.80, 0.86, 0.93]) + oc * np.array([0.82, 0.83, 0.85])
        sky_hor = sky_hor + warm * np.array([0.10, 0.04, -0.06])
        return Lighting(sun_dir, sun_rgb, sky_top, np.clip(sky_hor, 0, 1),
                        0.28 + 0.25 * oc, 0.25 + 0.4 * oc)


@dataclass
class Builder:
    name: str
    seed: int
    out_root: Path = GENERATED
    timestep: float = 0.002
    assets: list[str] = field(default_factory=list)
    world: list[str] = field(default_factory=list)
    actuators: list[str] = field(default_factory=list)
    sensors: list[str] = field(default_factory=list)
    contacts: list[str] = field(default_factory=list)
    equalities: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.dir = self.out_root / f"{self.name}-s{self.seed}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._batches: dict[str, MeshBatch] = {}
        self._materials: set[str] = set()
        self.meta.update(scene=self.name, seed=self.seed)

    # ---- materials & meshes ---------------------------------------------------------
    def material(self, name: str, rgba, specular: float = 0.1, shininess: float = 0.1,
                 reflectance: float = 0.0, texture: str | None = None, texrepeat=None,
                 emission: float = 0.0) -> str:
        if name in self._materials:
            return name
        self._materials.add(name)
        extra = f' texture="{texture}"' if texture else ""
        if texrepeat is not None:
            extra += f' texrepeat="{fmt(texrepeat)}" texuniform="true"'
        self.assets.append(
            f'<material name="{name}" rgba="{fmt(rgba)}" specular="{specular}" shininess="{shininess}" '
            f'reflectance="{reflectance}" emission="{emission}"{extra}/>')
        return name

    def batch(self, material: str) -> MeshBatch:
        return self._batches.setdefault(material, MeshBatch())

    def _flush_batches(self) -> None:
        for mat, b in self._batches.items():
            if b.empty:
                continue
            fname = f"mesh_{mat}.obj"
            b.write_obj(self.dir / fname)
            self.assets.append(f'<mesh name="m_{mat}" file="{fname}" inertia="shell"/>')
            self.world.append(f'<geom name="veg_{mat}" type="mesh" mesh="m_{mat}" material="{mat}" '
                              f'contype="0" conaffinity="0" group="1"/>')
        self.meta["mesh_triangles"] = {m: b.ntri for m, b in self._batches.items()}

    # ---- environment ----------------------------------------------------------------
    def overview(self, pos, target, name: str = "overview") -> None:
        pos, target = np.asarray(pos, float), np.asarray(target, float)
        self.world.append(f'<camera name="{name}" pos="{fmt(pos)}" xyaxes="{lookat_xyaxes(pos - target)}"/>')

    def sky_and_sun(self, light: Lighting, shadow_extent: float = 60.0, castshadow: bool = True,
                    fill: float = 0.0) -> None:
        self.light = light
        self.assets.append(
            f'<texture name="sky" type="skybox" builtin="gradient" rgb1="{fmt(light.sky_top)}" '
            f'rgb2="{fmt(light.sky_horizon)}" width="512" height="3072"/>')
        d = light.sun_dir
        self.world.append(
            f'<light name="sun" directional="true" castshadow="{str(castshadow).lower()}" pos="{fmt(-d * 100)}" dir="{fmt(d)}" '
            f'diffuse="{fmt(light.sun_rgb)}" specular="0.2 0.2 0.2"/>')
        if fill > 0:  # diffuse sky fill (e.g. light scattered by greenhouse glass)
            for k, az in enumerate((0.6, 2.7, 4.8)):
                fd = -np.array([np.cos(az) * 0.5, np.sin(az) * 0.5, 0.85])
                self.world.append(f'<light name="fill{k}" directional="true" castshadow="false" dir="{fmt(fd)}" '
                                  f'diffuse="{fill:.2f} {fill:.2f} {fill * 0.97:.2f}" specular="0 0 0"/>')
        self._shadow_extent = shadow_extent
        self.meta["lighting"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                 for k, v in light.__dict__.items()}

    # ---- output ---------------------------------------------------------------------
    def write(self) -> Path:
        self._flush_batches()
        light = getattr(self, "light", None)
        amb = light.ambient if light else 0.3
        haze = light.haze if light else 0.3
        hor = fmt(light.sky_horizon) if light else "0.8 0.85 0.9"
        ext = getattr(self, "_shadow_extent", 30.0)
        xml = f"""<mujoco model="{self.name}">
  <!-- generated by sim/scenes.py  (scene={self.name}, seed={self.seed}) - do not edit by hand -->
  <compiler angle="degree" meshdir="." texturedir="." autolimits="true"/>
  <option timestep="{self.timestep}" integrator="implicitfast"/>
  <statistic extent="{ext}" center="0 0 2"/>
  <visual>
    <global offwidth="1920" offheight="1080" fovy="55" elevation="-20" azimuth="135"/>
    <quality shadowsize="8192" offsamples="8"/>
    <map znear="0.01" zfar="3000" haze="{haze:.3f}" shadowclip="1.0" shadowscale="0.6"/>
    <headlight ambient="{amb:.3f} {amb:.3f} {amb:.3f}" diffuse="0.25 0.25 0.25" specular="0 0 0"/>
    <rgba haze="{hor} 1"/>
  </visual>
  <asset>
    {chr(10).join('    ' + a for a in self.assets).strip()}
  </asset>
  <worldbody>
    {chr(10).join('    ' + w for w in self.world).strip()}
  </worldbody>
  <contact>
    {chr(10).join('    ' + c for c in self.contacts).strip()}
  </contact>
  <equality>
    {chr(10).join('    ' + e for e in self.equalities).strip()}
  </equality>
  <actuator>
    {chr(10).join('    ' + a for a in self.actuators).strip()}
  </actuator>
  <sensor>
    {chr(10).join('    ' + s for s in self.sensors).strip()}
  </sensor>
</mujoco>
"""
        path = self.dir / "scene.xml"
        path.write_text(xml)
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=1, default=_json_default))
        return path


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))
