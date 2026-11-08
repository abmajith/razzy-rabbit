"""Path-following stages (the top of a chain): where should the vehicle go now?"""
from __future__ import annotations

import numpy as np

from .base import Controller


def _polyline(points):
    P = np.asarray(points, float)
    seg = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1) + 1e-9
    return P, np.r_[0, np.cumsum(seg)]


def _interp(P, s_cum, s):
    s = np.clip(s, 0, s_cum[-1])
    i = min(np.searchsorted(s_cum, s, side="right") - 1, len(P) - 2)
    f = (s - s_cum[i]) / (s_cum[i + 1] - s_cum[i])
    tan = P[i + 1] - P[i]
    return P[i] + f * tan, tan / (np.linalg.norm(tan[:2]) + 1e-9)


class FollowPath(Controller):
    """Multirotor: carrot-on-a-stick along a waypoint route, back and forth.

    - take-off: climb vertically to the first waypoint's height over `takeoff_s` seconds
    - then a carrot moves along the route at `cruise` m/s, slowing down when the vehicle
      lags behind (speed = cruise x clip(1.6 - lag/2, 0, 1)); at the ends it turns back
    - yaw follows the route direction at most `yaw_rate` rad/s
    - `agl=True`: waypoint heights are above the terrain under them

    Output: pose setpoint {"p": position, "v": velocity, "yaw": heading}.
    `err`: distance to the carrot (tracking error) after take-off.
    """

    name = "follow_path"
    defaults = {"waypoints": None, "agl": True, "cruise": 3.0, "takeoff_s": 2.5, "yaw_rate": 1.2}

    def __init__(self, waypoints=None, **params) -> None:
        super().__init__(waypoints=[list(map(float, w)) for w in waypoints] if waypoints is not None else None,
                         **params)

    def bind(self, sim, robot) -> None:
        from .base import TruthState
        p = self.params
        self.sim, self.agl, self.cruise = sim, p["agl"], p["cruise"]
        self.t_takeoff, self.yaw_rate = p["takeoff_s"], p["yaw_rate"]
        pos = TruthState(sim, robot).pos
        wps = np.array(p["waypoints"], float)
        p0 = pos.copy()
        first = wps[0].copy()
        if self.agl:
            p0[2] = 0.0
        lift = np.r_[p0[:2], first[2]]
        self.P, self.s = _polyline(np.vstack([lift, wps]))
        self.s_now, self.dirn = 0.0, 1
        self.yaw = float(np.arctan2(*(self.P[1, 1::-1] - self.P[0, 1::-1])))
        self.ground0 = pos[2]
        self.err = 0.0

    def target(self):
        p, tan = _interp(self.P, self.s, self.s_now)
        p = p.copy()
        if self.agl:
            p[2] += float(self.sim.terrain.height(p[0], p[1]))
        return p, tan

    def update(self, t, dt, state, x):
        p_d, tan = self.target()
        if t < self.t_takeoff:  # vertical take-off ramp
            p_d = state.pos.copy()
            p_d[2] = self.ground0 + (self.target()[0][2] - self.ground0) * (t / self.t_takeoff)
            v_d = np.zeros(3)
        else:
            lag = np.linalg.norm(state.pos - p_d)
            self.err = float(lag)
            speed = self.cruise * np.clip(1.6 - lag / 2.0, 0.0, 1.0)
            self.s_now += self.dirn * speed * dt
            if self.s_now >= self.s[-1] or self.s_now <= self.s[1]:
                self.dirn *= -1
                self.s_now = np.clip(self.s_now, self.s[1], self.s[-1])
            v_d = self.dirn * tan * speed
            if self.agl:
                v_d[2] = 0.0
            if np.linalg.norm(tan[:2]) > 1e-6:
                yaw_t = np.arctan2(self.dirn * tan[1], self.dirn * tan[0])
                dy = (yaw_t - self.yaw + np.pi) % (2 * np.pi) - np.pi
                self.yaw += np.clip(dy, -self.yaw_rate * dt, self.yaw_rate * dt)
        return {"p": p_d, "v": v_d, "yaw": self.yaw}


class PurePursuit(Controller):
    """Ground vehicle: pure pursuit around a closed waypoint loop.

    The loop is densified to `spacing` m; the nearest point is tracked within a window, and
    the target is the first point `lookahead` m ahead. Curvature  kappa = 2 sin(alpha) / Ld
    (alpha: bearing to the target relative to the heading), clipped to what the steering can
    do (`max_steer_deg` at the corner wheels). Speed drops to `slow_factor` x `speed` when
    the target is more than `slow_angle` rad off the heading.

    Output: (speed [m/s], curvature [1/m]). `err`: cross-track distance; `laps` completed.
    """

    name = "pure_pursuit"
    defaults = {"waypoints": None, "speed": 0.5, "lookahead": 1.2, "spacing": 0.2, "window": 60,
                "max_steer_deg": 38.0, "slow_angle": 0.6, "slow_factor": 0.4}

    def __init__(self, waypoints=None, **params) -> None:
        super().__init__(waypoints=[list(map(float, w)) for w in waypoints] if waypoints is not None else None,
                         **params)

    def bind(self, sim, robot) -> None:
        p = self.params
        wp = np.array(p["waypoints"], float)
        pts = []
        for a, b in zip(wp, np.roll(wp, -1, axis=0)):          # densify the closed loop
            n = max(2, int(np.linalg.norm(b - a) / p["spacing"]))
            pts += [a + (b - a) * k / n for k in range(n)]
        self.P = np.array(pts)
        bid = sim.m.body(robot["body"]).id
        self.idx = int(np.argmin(np.linalg.norm(self.P - sim.d.xpos[bid][:2], axis=1)))
        sxy = np.array(list(robot["steer_xy"].values()))
        kmax = np.tan(np.radians(p["max_steer_deg"])) / np.max(np.abs(sxy[:, 0]))
        self.kmax = kmax / (1 + kmax * np.max(np.abs(sxy[:, 1])))
        self.laps, self.err = 0, 0.0

    def update(self, t, dt, state, x):
        prm = self.params
        p = state.xy
        yaw = state.yaw
        n = len(self.P)
        win = (self.idx + np.arange(prm["window"])) % n
        dist = np.linalg.norm(self.P[win] - p, axis=1)
        k = int(np.argmin(dist))
        self.err = float(dist[k])                               # cross-track error
        if self.idx + k >= n:
            self.laps += 1
        self.idx = (self.idx + k) % n
        Ld, j = prm["lookahead"], self.idx
        for _ in range(n):
            if np.linalg.norm(self.P[j] - p) >= Ld:
                break
            j = (j + 1) % n
        tgt = self.P[j]
        alpha = (np.arctan2(tgt[1] - p[1], tgt[0] - p[0]) - yaw + np.pi) % (2 * np.pi) - np.pi
        kappa = float(np.clip(2 * np.sin(alpha) / Ld, -self.kmax, self.kmax))
        v = prm["speed"] * (1.0 if abs(alpha) < prm["slow_angle"] else prm["slow_factor"])
        return v, kappa
