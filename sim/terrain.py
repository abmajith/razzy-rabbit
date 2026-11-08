"""Procedural terrain: fractal noise heightfields + matching ground textures.

Everything is seeded so a (scene, seed) pair always produces byte-identical assets.
Heightfields are written in MuJoCo's raw binary format (int32 nrow, int32 ncol,
float32[nrow*ncol]) so the C/C++ side can load the very same files.

Conventions (verified against MuJoCo's mj_ray):
  * hfield data row 0 is at y = -half_y, column 0 at x = -half_x
  * a 2D texture is stretched over the whole hfield, image row 0 at y = +half_y
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def _upsample(grid: np.ndarray, n: int) -> np.ndarray:
    """Smooth (smoothstep-bilinear) upsampling of a (res+1)^2 lattice to n x n."""
    res = grid.shape[0] - 1
    t = np.linspace(0.0, res, n)
    i = np.minimum(t.astype(int), res - 1)
    f = t - i
    f = f * f * (3.0 - 2.0 * f)
    fy, fx = f[:, None], f[None, :]
    a = grid[np.ix_(i, i)]
    b = grid[np.ix_(i + 1, i)]
    c = grid[np.ix_(i, i + 1)]
    d = grid[np.ix_(i + 1, i + 1)]
    return a * (1 - fy) * (1 - fx) + b * fy * (1 - fx) + c * (1 - fy) * fx + d * fy * fx


def fbm(n: int, rng: np.random.Generator, octaves: int = 6, base: int = 3,
        persistence: float = 0.5) -> np.ndarray:
    """Fractal Brownian motion value noise in roughly [-1, 1], shape (n, n)."""
    out = np.zeros((n, n))
    amp, total = 1.0, 0.0
    for o in range(octaves):
        res = base * 2 ** o
        out += amp * _upsample(rng.standard_normal((res + 1, res + 1)), n)
        total += amp
        amp *= persistence
    return out / total


@dataclass
class Terrain:
    z: np.ndarray          # absolute heights in metres, shape (nrow, ncol); row = y, col = x
    half_x: float
    half_y: float
    base: float = 1.0      # thickness of the solid slab below the lowest point

    @property
    def zmin(self) -> float:
        return float(self.z.min())

    @property
    def elev(self) -> float:
        return max(float(self.z.max() - self.z.min()), 1e-3)

    def height(self, x, y):
        """Bilinear terrain height at world (x, y); the hfield geom sits at z = zmin."""
        nr, nc = self.z.shape
        cx = np.clip((np.asarray(x) + self.half_x) / (2 * self.half_x) * (nc - 1), 0, nc - 1.001)
        ry = np.clip((np.asarray(y) + self.half_y) / (2 * self.half_y) * (nr - 1), 0, nr - 1.001)
        c0, r0 = cx.astype(int), ry.astype(int)
        fx, fy = cx - c0, ry - r0
        z = self.z
        return (z[r0, c0] * (1 - fx) * (1 - fy) + z[r0, c0 + 1] * fx * (1 - fy)
                + z[r0 + 1, c0] * (1 - fx) * fy + z[r0 + 1, c0 + 1] * fx * fy)

    def grid_xy(self):
        nr, nc = self.z.shape
        xs = np.linspace(-self.half_x, self.half_x, nc)
        ys = np.linspace(-self.half_y, self.half_y, nr)
        return np.meshgrid(xs, ys)

    def write(self, out_dir: Path, name: str = "terrain") -> str:
        nr, nc = self.z.shape
        norm = ((self.z - self.zmin) / self.elev).astype(np.float32)
        fname = f"{name}.bin"
        (out_dir / fname).write_bytes(struct.pack("ii", nr, nc) + norm.tobytes())
        return fname

    def asset_xml(self, fname: str, name: str = "terrain") -> str:
        return (f'<hfield name="{name}" file="{fname}" '
                f'size="{self.half_x} {self.half_y} {self.elev:.4f} {self.base}"/>')

    def geom_xml(self, name: str = "terrain", material: str = "ground") -> str:
        return (f'<geom name="{name}" type="hfield" hfield="{name}" material="{material}" '
                f'pos="0 0 {self.zmin:.4f}" friction="1.0 0.02 0.001" group="0"/>')


def write_texture(out_dir: Path, fname: str, rgb_rows_y_up: np.ndarray) -> str:
    """Write a terrain texture given in hfield order (row 0 = -y); flips for MuJoCo."""
    img = np.clip(rgb_rows_y_up[::-1] * 255, 0, 255).astype(np.uint8)
    iio.imwrite(out_dir / fname, img)
    return fname


def colorize(t: Terrain, rng: np.random.Generator, n: int,
             palette: list[tuple[float, tuple[float, float, float]]],
             extra: np.ndarray | None = None, detail: float = 0.25) -> np.ndarray:
    """Map noise (+ optional per-pixel mask `extra` in [0,1]) through a colour ramp.

    palette: list of (threshold, rgb) sorted by threshold in [0, 1].
    Returns float image (n, n, 3) in hfield order.
    """
    v = 0.5 + 0.5 * fbm(n, rng, octaves=7, base=6, persistence=0.55)
    if extra is not None:
        v = np.clip(0.6 * v + 0.4 * extra, 0, 1)
    stops = np.array([p[0] for p in palette])
    cols = np.array([p[1] for p in palette])
    img = np.stack([np.interp(v, stops, cols[:, k]) for k in range(3)], axis=-1)
    # fine speckle for grain
    img *= (1.0 - detail * 0.5) + detail * rng.random((n, n, 1))
    return img


def resample(a: np.ndarray, n: int) -> np.ndarray:
    """Nearest-ish resample of a 2D field to (n, n) for texture masks."""
    r = np.linspace(0, a.shape[0] - 1, n).astype(int)
    c = np.linspace(0, a.shape[1] - 1, n).astype(int)
    return a[np.ix_(r, c)]
