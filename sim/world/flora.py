"""Procedural vegetation & props geometry (one tree, one rock...), written into merged mesh
batches. World features (world/features/) decide *where* these go.

Visual geometry goes into Builder mesh batches (no collision). Where a robot can hit
something (trunks, logs, rocks), a cheap primitive collision geom is added separately.
"""
from __future__ import annotations

import numpy as np

from ..builder import Builder, fmt
from ..mesh import rot_x, rot_y, rot_z


# name: (base rgb, jitter amplitude or None, material options). The order matters for the
# seeded jitter: PlantColours draws the jitter of these, in this order.
MATERIALS = {
    "bark":         ([0.30, 0.22, 0.15], 0.04, dict(specular=0.02)),
    "bark_birch":   ([0.82, 0.80, 0.74], None, dict(specular=0.05)),
    "leaf_a":       ([0.20, 0.42, 0.13], 0.06, dict(specular=0.15, shininess=0.2)),
    "leaf_b":       ([0.28, 0.50, 0.16], 0.06, dict(specular=0.15, shininess=0.2)),
    "leaf_c":       ([0.14, 0.30, 0.10], 0.06, dict(specular=0.1)),
    "conifer":      ([0.09, 0.24, 0.12], 0.03, dict(specular=0.05)),
    "conifer_b":    ([0.12, 0.30, 0.15], 0.03, dict(specular=0.05)),
    "lettuce":      ([0.42, 0.70, 0.22], 0.06, dict(specular=0.3, shininess=0.4)),
    "lettuce_dark": ([0.25, 0.52, 0.16], 0.06, dict(specular=0.3, shininess=0.4)),
    "weed":         ([0.45, 0.55, 0.12], 0.06, dict(specular=0.1)),
    "apple":        ([0.75, 0.10, 0.08], None, dict(specular=0.6, shininess=0.7)),
    "rock":         ([0.48, 0.46, 0.43], None, dict(specular=0.1)),
    "grass_tuft":   ([0.35, 0.50, 0.18], 0.06, dict(specular=0.05)),
    "dry_grass":    ([0.62, 0.58, 0.36], None, dict(specular=0.05)),
}
TREE_MATERIALS = ("bark", "bark_birch", "leaf_a", "leaf_b", "leaf_c", "conifer", "conifer_b")
GRASS_MATERIALS = ("grass_tuft", "dry_grass")


def plant_materials(b: Builder, rng: np.random.Generator | None = None, names=None) -> None:
    """Register vegetation materials (all, or only `names`). With an rng each colour is
    jittered (seeded variation, all materials, fixed order); without one the base colours are
    used. Materials already registered are kept."""
    for name, (rgb, jitter, opts) in MATERIALS.items():
        c = np.asarray(rgb, float)
        if rng is not None and jitter is not None:
            c = np.clip(c + rng.uniform(-jitter, jitter, 3) * np.array([0.5, 1, 0.5]), 0, 1)
        if names is None or name in names:
            b.material(name, [*c, 1], **opts)


# ---------------------------------------------------------------------------- trees ---
def broadleaf_tree(b: Builder, rng, x, y, z, height=6.0, crown=2.2, fruit=0.0,
                   collide=True, foliage=("leaf_a", "leaf_b", "leaf_c")) -> None:
    trunk_h = height * rng.uniform(0.35, 0.5)
    lean = rng.normal(0, 0.05, 2)
    top = np.array([x + lean[0] * trunk_h, y + lean[1] * trunk_h, z + trunk_h])
    tr = 0.08 + 0.03 * height / 6
    b.batch("bark").cylinder([x, y, z - 0.2], top, tr * 1.3, tr * 0.8, seg=8)
    # a few branches
    for _ in range(rng.integers(3, 6)):
        az, el = rng.uniform(0, 2 * np.pi), rng.uniform(0.5, 1.1)
        L = crown * rng.uniform(0.5, 0.9)
        p0 = top - [0, 0, rng.uniform(0, trunk_h * 0.3)]
        p1 = p0 + L * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        b.batch("bark").cylinder(p0, p1, tr * 0.5, tr * 0.2, seg=6)
    # crown: clustered lumpy ellipsoids
    cz = z + trunk_h + (height - trunk_h) * 0.45
    n = rng.integers(5, 9)
    for k in range(n):
        off = rng.normal(0, 1, 3) * np.array([crown * 0.45, crown * 0.45, (height - trunk_h) * 0.22])
        r = crown * rng.uniform(0.45, 0.7) * np.array([1, 1, rng.uniform(0.7, 1.0)])
        c = np.array([x, y, cz]) + off
        b.batch(foliage[k % len(foliage)]).ellipsoid(c, r, R=rot_z(rng.uniform(0, 6.3)),
                                                     seg=9, rings=6, jitter=0.35, rng=rng)
        if fruit > 0:
            m = rng.poisson(fruit)
            dirs = rng.normal(size=(m, 3))
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            dirs[:, 2] = np.abs(dirs[:, 2]) * -0.6 + dirs[:, 2] * 0.2
            for d in dirs:
                b.batch("apple").ellipsoid(c + d * r * 0.97, [0.045] * 3, seg=6, rings=4)
    if collide:
        b.world.append(f'<geom type="capsule" fromto="{x:.3f} {y:.3f} {z-.2:.3f} {fmt(top)}" size="{tr*1.2:.3f}" '
                       f'rgba="0 0 0 0" group="3"/>')
        b.world.append(f'<geom type="ellipsoid" pos="{x:.3f} {y:.3f} {cz:.3f}" size="{crown*0.9:.3f} {crown*0.9:.3f} '
                       f'{(height-trunk_h)*0.45:.3f}" rgba="0 0 0 0" group="3"/>')


def conifer(b: Builder, rng, x, y, z, height=12.0, collide=True) -> None:
    base_r = height * rng.uniform(0.18, 0.25)
    tr = 0.1 + 0.015 * height
    b.batch("bark").cylinder([x, y, z - 0.2], [x, y, z + height * 0.98], tr, tr * 0.25, seg=7)
    tiers = rng.integers(4, 7)
    start = height * rng.uniform(0.12, 0.25)
    mat = "conifer" if rng.random() < 0.6 else "conifer_b"
    for k in range(tiers):
        f = k / tiers
        zb = z + start + (height - start) * f * 0.92
        r = base_r * (1 - f * 0.85) * rng.uniform(0.9, 1.1)
        h = (height - start) / tiers * 2.2 * (1 - f * 0.3)
        tilt = rot_x(rng.normal(0, 0.04)) @ rot_y(rng.normal(0, 0.04))
        b.batch(mat).cone([x, y, zb], r, h, seg=11, tilt=tilt)
    if collide:
        b.world.append(f'<geom type="capsule" fromto="{x:.3f} {y:.3f} {z-.2:.3f} {x:.3f} {y:.3f} {z+height:.3f}" '
                       f'size="{tr*1.1:.3f}" rgba="0 0 0 0" group="3"/>')
        b.world.append(f'<geom type="capsule" fromto="{x:.3f} {y:.3f} {z+start+base_r*0.5:.3f} {x:.3f} {y:.3f} {z+height*0.75:.3f}" '
                       f'size="{base_r*0.55:.3f}" rgba="0 0 0 0" group="3"/>')


def birch(b: Builder, rng, x, y, z, height=9.0, collide=True) -> None:
    tr = 0.07 + 0.01 * height
    top = np.array([x, y, z + height * 0.9]) + np.r_[rng.normal(0, 0.15, 2), 0]
    b.batch("bark_birch").cylinder([x, y, z - 0.2], top, tr, tr * 0.4, seg=7)
    for k in range(rng.integers(4, 7)):
        c = top - [0, 0, height * rng.uniform(0.05, 0.4)] + np.r_[rng.normal(0, 0.6, 2), 0]
        b.batch("leaf_b" if k % 2 else "leaf_a").ellipsoid(
            c, [height * 0.12, height * 0.12, height * 0.2], seg=8, rings=6, jitter=0.4, rng=rng)
    if collide:
        b.world.append(f'<geom type="capsule" fromto="{x:.3f} {y:.3f} {z-.2:.3f} {fmt(top)}" size="{tr:.3f}" rgba="0 0 0 0" group="3"/>')


def rock(b: Builder, rng, x, y, z, size=0.5, collide=True) -> None:
    r = size * rng.uniform(0.6, 1.2, 3) * np.array([1, 1, 0.6])
    b.batch("rock").ellipsoid([x, y, z + r[2] * 0.3], r, R=rot_z(rng.uniform(0, 6.3)), seg=8, rings=5,
                              jitter=0.3, rng=rng)
    if collide:
        b.world.append(f'<geom type="ellipsoid" pos="{x:.3f} {y:.3f} {z + r[2]*0.3:.3f}" size="{fmt(r*0.9)}" '
                       f'rgba="0 0 0 0" group="3"/>')


def log(b: Builder, rng, x, y, z, length=5.0, collide=True) -> None:
    a = rng.uniform(0, np.pi)
    d = np.array([np.cos(a), np.sin(a), 0]) * length / 2
    r = rng.uniform(0.15, 0.3)
    p0, p1 = np.array([x, y, z + r * 0.7]) - d, np.array([x, y, z + r * 0.7]) + d
    b.batch("bark").cylinder(p0, p1, r, r * 0.8, seg=9)
    if collide:
        b.world.append(f'<geom type="capsule" fromto="{fmt(p0)} {fmt(p1)}" size="{r:.3f}" rgba="0 0 0 0" group="3"/>')


# ---------------------------------------------------------------------------- crops ---
def lettuce(b: Builder, rng, x, y, z, size=0.16) -> None:
    s = size * rng.uniform(0.75, 1.2)
    b.batch("lettuce_dark").ellipsoid([x, y, z + s * 0.25], [s, s, s * 0.55], seg=8, rings=5,
                                     jitter=0.3, rng=rng)
    n = rng.integers(6, 10)
    for k in range(n):
        az = 2 * np.pi * k / n + rng.normal(0, 0.2)
        b.batch("lettuce").leaf([x, y, z + s * 0.2], az, rng.uniform(0.5, 0.9), s * 1.3, s * 0.55,
                                droop=0.6, n=4)


def weed(b: Builder, rng, x, y, z) -> None:
    for k in range(rng.integers(3, 6)):
        b.batch("weed").leaf([x, y, z], rng.uniform(0, 6.3), rng.uniform(0.6, 1.3),
                             rng.uniform(0.08, 0.18), 0.025, droop=0.5, n=3)


def grass_tuft(b: Builder, rng, x, y, z, mat="grass_tuft", h=0.35) -> None:
    for k in range(rng.integers(5, 9)):
        b.batch(mat).leaf([x + rng.normal(0, 0.05), y + rng.normal(0, 0.05), z], rng.uniform(0, 6.3),
                          rng.uniform(1.1, 1.45), h * rng.uniform(0.6, 1.2), 0.012, droop=0.35, n=3)
