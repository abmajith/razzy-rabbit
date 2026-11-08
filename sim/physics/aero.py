"""Air drag and the analytic wind field  v(x, y, z, t)  built from superposed potential-flow elements.

    horizontal flow = log-law profile(AGL) * [ uniform stream
                                              + doublets (flow around tower/trunk cylinders)
                                              + divergence-free frozen turbulence (stream fn) ]
    vertical flow   = terrain-following component  u . grad(h)  decaying with height
    turbine wake    = Jensen top-hat velocity deficit downstream of a rotor

Everything is closed form and vectorised, so it is cheap enough to evaluate per physics
step for many bodies (and trivially portable to C / a GPU kernel later).
"""
from __future__ import annotations

import numpy as np

RHO_AIR = 1.225


def drag_force(CdA: float, v_rel, rho: float = RHO_AIR) -> np.ndarray:
    """Quadratic drag F = 1/2 rho CdA |v| v [N] for air velocity v_rel relative to the body."""
    v_rel = np.asarray(v_rel, float)
    return 0.5 * rho * CdA * np.linalg.norm(v_rel) * v_rel


class WindField:
    def __init__(self, cfg: dict, terrain=None, seed: int = 0, n_modes: int = 8) -> None:
        self.U = float(cfg.get("U", 0.0))
        self.heading = np.radians(cfg.get("heading_deg", 0.0))
        self.dir = np.array([np.cos(self.heading), np.sin(self.heading)])
        self.gust = float(cfg.get("gust", 0.0))
        self.z_ref = float(cfg.get("z_ref", 10.0))
        self.z0 = float(cfg.get("z0", 0.1))
        self.cyl = np.array(cfg.get("cylinders", []), float).reshape(-1, 4)  # x, y, R, top_z
        self.turbine = cfg.get("turbine")
        self.terrain = terrain
        rng = np.random.default_rng(seed + 7919)
        L = rng.uniform(12, 90, n_modes)                      # eddy wavelengths [m]
        ang = rng.uniform(0, 2 * np.pi, n_modes)
        self.k = (2 * np.pi / L)[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1)
        self.phase = rng.uniform(0, 2 * np.pi, n_modes)
        self.omega = rng.uniform(0.05, 0.4, n_modes)          # slow evolution on top of advection
        kn = np.linalg.norm(self.k, axis=1)
        self.amp = self.gust / kn / np.sqrt(n_modes / 2)      # stream-fn amplitude -> |v| ~ gust
        self.wphase = rng.uniform(0, 2 * np.pi, n_modes)

    def _ground(self, x, y):
        if self.terrain is None:
            return np.zeros_like(x)
        return self.terrain.height(x, y)

    def velocity(self, P, t: float) -> np.ndarray:
        P = np.atleast_2d(np.asarray(P, float))
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        h = np.maximum(z - self._ground(x, y), 0.05)
        prof = np.log((h + self.z0) / self.z0) / np.log((self.z_ref + self.z0) / self.z0)

        # complex potential: dW/dz = u - i v
        zc = x + 1j * y
        e = np.exp(-1j * self.heading)
        dW = np.full(len(P), self.U * e, dtype=complex)
        inside = np.zeros(len(P), bool)
        for cx, cy, R, top in self.cyl:
            rel = zc - (cx + 1j * cy)
            below = z < top
            r2 = np.abs(rel) ** 2
            dW = dW - np.where(below, self.U * np.conj(e) * R * R / np.where(r2 > 1e-9, rel, 1) ** 2, 0)
            inside |= below & (r2 < R * R)
        u, v = dW.real, -dW.imag

        # frozen turbulence: psi = sum a sin(k.(x - U t) + w t + phi);  u = dpsi/dy, v = -dpsi/dx
        xi = P[:, :2] - self.U * self.dir * t
        arg = xi @ self.k.T + self.omega * t + self.phase                # (N, modes)
        c = np.cos(arg) * self.amp
        u = u + c @ self.k[:, 1]
        v = v - c @ self.k[:, 0]
        w = 0.25 * (np.sin(arg + self.wphase) * self.amp) @ np.linalg.norm(self.k, axis=1) * 0.5

        u, v = u * prof, v * prof
        if self.terrain is not None:  # flow follows the terrain near the ground
            eps = 0.5
            gx = (self._ground(x + eps, y) - self._ground(x - eps, y)) / (2 * eps)
            gy = (self._ground(x, y + eps) - self._ground(x, y - eps)) / (2 * eps)
            w = w + (u * gx + v * gy) * np.exp(-h / 40.0)

        V = np.stack([u, v, w], 1)
        if self.turbine:
            hub = np.asarray(self.turbine["hub"], float)
            R, Ct, kw = self.turbine["R"], self.turbine["Ct"], self.turbine["k"]
            d3 = np.r_[self.dir, 0.0]
            rel = P - hub
            s = rel @ d3
            r = np.linalg.norm(rel - np.outer(s, d3), axis=1)
            Rw = R + kw * np.maximum(s, 0)
            deficit = (1 - np.sqrt(1 - Ct)) / (1 + kw * np.maximum(s, 0) / R) ** 2
            in_wake = (s > 0) & (r < Rw)
            V[in_wake, :2] *= (1 - deficit[in_wake])[:, None]
        V[inside] = 0.0
        return V

    def drag_force(self, pos, vel, CdA: float, t: float) -> np.ndarray:
        return drag_force(CdA, self.velocity(pos, t)[0] - np.asarray(vel))
