"""Low-poly mesh batching.

Thousands of plants as individual MuJoCo geoms would blow past the renderer's geom
budget and slow every collision broad-phase. Instead, all purely visual vegetation of
one material is merged into a single triangle mesh (one OBJ, one geom, contype=0).
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _frame(axis: np.ndarray):
    """Orthonormal u, v with u x v = axis (right-handed)."""
    w = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0, 0]) if abs(w[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(helper, w)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    return u, v, w


class MeshBatch:
    def __init__(self) -> None:
        self._v: list[np.ndarray] = []
        self._f: list[np.ndarray] = []
        self.nv = 0

    @property
    def empty(self) -> bool:
        return self.nv == 0

    @property
    def ntri(self) -> int:
        return int(sum(len(f) for f in self._f))

    def add(self, V, F) -> None:
        V = np.asarray(V, dtype=float)
        F = np.asarray(F, dtype=np.int64)
        self._v.append(V)
        self._f.append(F + self.nv)
        self.nv += len(V)

    # ---- primitives (all CCW / outward facing) ------------------------------------
    def ellipsoid(self, c, r, R=None, seg: int = 10, rings: int = 6, jitter: float = 0.0,
                  rng: np.random.Generator | None = None) -> None:
        th = np.linspace(0, np.pi, rings + 1)[1:-1]
        ph = np.linspace(0, 2 * np.pi, seg, endpoint=False)
        ring = np.stack([np.outer(np.sin(th), np.cos(ph)), np.outer(np.sin(th), np.sin(ph)),
                         np.outer(np.cos(th), np.ones_like(ph))], axis=-1).reshape(-1, 3)
        unit = np.vstack([[0, 0, 1.0], ring, [0, 0, -1.0]])
        if jitter and rng is not None:  # lumpy foliage
            unit = unit * (1 + jitter * (rng.random((len(unit), 1)) - 0.5))
        V = unit * np.asarray(r, float)
        if R is not None:
            V = V @ np.asarray(R).T
        V = V + np.asarray(c, float)
        F = [[0, 1 + j, 1 + (j + 1) % seg] for j in range(seg)]
        for i in range(rings - 2):
            a, b = 1 + i * seg, 1 + (i + 1) * seg
            for j in range(seg):
                j2 = (j + 1) % seg
                F += [[a + j, b + j, b + j2], [a + j, b + j2, a + j2]]
        last, a = len(V) - 1, 1 + (rings - 2) * seg
        F += [[a + j, last, a + (j + 1) % seg] for j in range(seg)]
        self.add(V, F)

    def cone(self, base, radius: float, height: float, seg: int = 12, tilt=None) -> None:
        ph = np.linspace(0, 2 * np.pi, seg, endpoint=False)
        ring = np.stack([radius * np.cos(ph), radius * np.sin(ph), np.zeros(seg)], -1)
        V = np.vstack([ring, [0, 0, height], [0, 0, 0]])
        if tilt is not None:
            V = V @ np.asarray(tilt).T
        V = V + np.asarray(base, float)
        apex, ctr = seg, seg + 1
        F = [[j, (j + 1) % seg, apex] for j in range(seg)]
        F += [[ctr, (j + 1) % seg, j] for j in range(seg)]
        self.add(V, F)

    def cylinder(self, p0, p1, r0: float, r1: float | None = None, seg: int = 8) -> None:
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        r1 = r0 if r1 is None else r1
        u, v, _ = _frame(p1 - p0)
        ph = np.linspace(0, 2 * np.pi, seg, endpoint=False)
        dirs = np.outer(np.cos(ph), u) + np.outer(np.sin(ph), v)
        V = np.vstack([p0 + r0 * dirs, p1 + r1 * dirs])
        F = []
        for j in range(seg):
            j2 = (j + 1) % seg
            F += [[j, j2, seg + j2], [j, seg + j2, seg + j]]
        self.add(V, F)

    def box(self, c, half, R=None) -> None:
        hx, hy, hz = half
        V = np.array([[sx * hx, sy * hy, sz * hz] for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)])
        if R is not None:
            V = V @ np.asarray(R).T
        V = V + np.asarray(c, float)
        F = [[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4],
             [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]]
        self.add(V, F)

    def leaf(self, base, azimuth: float, elevation: float, length: float, width: float,
             droop: float = 0.4, n: int = 5) -> None:
        """Curved, tapered, double-sided leaf blade."""
        s = np.linspace(0, 1, n + 1)
        d = np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth),
                      np.sin(elevation)])
        side = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
        spine = np.asarray(base, float) + np.outer(s * length, d) - np.outer(droop * length * s ** 2, [0, 0, 1])
        w = width * np.sin(np.pi * np.clip(s, 0.02, 0.98)) ** 0.7
        w[0] *= 0.3
        L = spine - w[:, None] * side
        Rt = spine + w[:, None] * side
        V = np.vstack([L, Rt])
        F = []
        for k in range(n):
            F += [[k, k + 1, n + 1 + k + 1], [k, n + 1 + k + 1, n + 1 + k]]
        F = np.array(F)
        # back faces with duplicated vertices so smooth normals don't cancel
        self.add(V, F)
        self.add(V, F[:, ::-1])

    def blade(self, root, axis_dir, length: float, chord_root: float, chord_tip: float,
              thickness: float = 0.15, twist: float = 0.3, n: int = 10, m: int = 8) -> None:
        """Lofted airfoil-ish blade (elliptic section) along axis_dir; used for turbines."""
        u, v, w = _frame(np.asarray(axis_dir, float))
        rings = []
        for k in range(n + 1):
            s = k / n
            chord = chord_root + (chord_tip - chord_root) * s
            if s < 0.12:  # round root
                chord = chord_root * 0.55 + (chord - chord_root * 0.55) * (s / 0.12)
            a = twist * (1 - s)
            cu = np.cos(a) * u + np.sin(a) * v
            cv = -np.sin(a) * u + np.cos(a) * v
            ph = np.linspace(0, 2 * np.pi, m, endpoint=False)
            pts = (np.asarray(root, float) + w * length * s
                   + np.outer(0.5 * chord * np.cos(ph), cu)
                   + np.outer(0.5 * chord * thickness * np.sin(ph), cv))
            rings.append(pts)
        V = np.vstack(rings + [np.asarray(root, float) + w * length * 1.01])
        F = []
        for k in range(n):
            for j in range(m):
                j2 = (j + 1) % m
                a, b = k * m, (k + 1) * m
                F += [[a + j, a + j2, b + j2], [a + j, b + j2, b + j]]
        tip = len(V) - 1
        F += [[n * m + j, n * m + (j + 1) % m, tip] for j in range(m)]
        self.add(V, F)

    # ---- output ---------------------------------------------------------------------
    def write_obj(self, path: Path) -> None:
        V = np.vstack(self._v)
        F = np.vstack(self._f) + 1
        buf = io.StringIO()
        np.savetxt(buf, V, fmt="v %.4f %.4f %.4f")
        np.savetxt(buf, F, fmt="f %d %d %d")
        Path(path).write_text(buf.getvalue())
