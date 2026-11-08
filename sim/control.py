"""Reference controllers so every scene "does something" out of the box.

These are classical baselines (geometric SE(3) quadrotor control with a waypoint
mission, Ackermann pure pursuit for the rocker-bogie rover). They are the thing an RL policy has to
beat, and their outputs make handy demonstrations / residual-policy priors.
"""
from __future__ import annotations

import mujoco
import numpy as np


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


# ======================================================================================
class QuadController:
    """Geometric tracking controller on SE(3) (Lee, Leok, McClamroch 2010)."""

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, meta: dict) -> None:
        self.m, self.d = m, d
        self.bid = m.body(meta["body"]).id
        jnt = m.body_jntadr[self.bid]
        self.qa, self.va = m.jnt_qposadr[jnt], m.jnt_dofadr[jnt]
        self.act = np.array([m.actuator(a).id for a in meta["actuators"]])
        self.mass = float(m.body_subtreemass[self.bid])
        self.J = m.body_inertia[self.bid].copy()
        A = np.zeros((4, 4))
        com = m.body_ipos[self.bid]                 # torques act about the centre of mass
        for i, aid in enumerate(self.act):
            x, y, _ = m.site_pos[m.actuator_trnid[aid, 0]] - com
            A[:, i] = [1.0, y, -x, m.actuator_gear[aid, 5]]
        self.Ainv = np.linalg.inv(A)
        self.kp, self.kd, self.kR, self.kW = 5.0, 4.0, 140.0, 22.0

    @property
    def pos(self):
        return self.d.qpos[self.qa:self.qa + 3]

    @property
    def vel(self):
        return self.d.qvel[self.va:self.va + 3]

    def __call__(self, p_d, v_d, yaw_d, a_ff=np.zeros(3)) -> None:
        d = self.d
        R = d.xmat[self.bid].reshape(3, 3)
        w = d.qvel[self.va + 3:self.va + 6]  # body-frame angular velocity
        e_p = self.pos - p_d
        n = np.linalg.norm(e_p)
        if n > 3.0:
            e_p *= 3.0 / n
        a = -self.kp * e_p - self.kd * (self.vel - v_d) + a_ff + [0, 0, 9.81]
        a[2] = max(a[2], 2.0)
        hmax = np.tan(np.radians(35)) * a[2]
        hn = np.linalg.norm(a[:2])
        if hn > hmax:
            a[:2] *= hmax / hn
        F = self.mass * a
        b3 = F / np.linalg.norm(F)
        b1c = np.array([np.cos(yaw_d), np.sin(yaw_d), 0.0])
        b2 = np.cross(b3, b1c)
        b2 /= np.linalg.norm(b2)
        Rd = np.column_stack([np.cross(b2, b3), b2, b3])
        E = 0.5 * (Rd.T @ R - R.T @ Rd)
        eR = np.array([E[2, 1], E[0, 2], E[1, 0]])
        tau = self.J * (-self.kR * eR - self.kW * w) + np.cross(w, self.J * w)
        T = F @ R[:, 2]
        fmax = self.m.actuator_ctrlrange[self.act, 1]   # live: battery sag lowers it
        d.ctrl[self.act] = np.clip(self.Ainv @ np.r_[T, tau], 0, fmax)


class QuadMission:
    """Carrot-on-a-stick waypoint follower with take-off and ping-pong looping."""

    def __init__(self, sim, meta: dict) -> None:
        self.sim = sim
        rmeta = meta["robots"][0]
        self.ctl = QuadController(sim.m, sim.d, rmeta)
        self.cruise = meta.get("cruise", 3.0)
        self.agl = "waypoints_agl" in meta
        wps = np.array(meta["waypoints_agl"] if self.agl else meta["waypoints_abs"], float)
        p0 = self.ctl.pos.copy()
        first = wps[0].copy()
        if self.agl:
            p0[2] = 0.0
        else:
            first = first.copy()
        lift = np.r_[p0[:2], first[2]]
        self.P, self.s = _polyline(np.vstack([lift, wps]))
        self.s_now, self.dirn = 0.0, 1
        self.yaw = float(np.arctan2(*(self.P[1, 1::-1] - self.P[0, 1::-1])))
        self.t_takeoff = 2.5
        self.ground0 = self.ctl.pos[2]
        self.CdA = rmeta.get("drag_CdA", 0.04)
        self.err = 0.0          # distance to the carrot (tracking error), after take-off

    def target(self, t):
        p, tan = _interp(self.P, self.s, self.s_now)
        p = p.copy()
        if self.agl:
            p[2] += float(self.sim.terrain.height(p[0], p[1]))
        return p, tan

    def update(self, t: float, dt: float) -> None:
        p_d, tan = self.target(t)
        if t < self.t_takeoff:  # vertical take-off ramp
            p_d = self.ctl.pos.copy()
            p_d[2] = self.ground0 + (self.target(t)[0][2] - self.ground0) * (t / self.t_takeoff)
            v_d = np.zeros(3)
        else:
            lag = np.linalg.norm(self.ctl.pos - p_d)
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
                self.yaw += np.clip(dy, -1.2 * dt, 1.2 * dt)
        self.ctl(p_d, v_d, self.yaw)
        # aerodynamic drag in the wind field (assigned every step: the plant adds rotor drag on top)
        f = self.sim.wind.drag_force(self.ctl.pos, self.ctl.vel, self.CdA, t) if self.sim.wind is not None else 0.0
        self.sim.d.xfrc_applied[self.ctl.bid, :3] = f


class RoverAckermann:
    """Pure pursuit around a closed waypoint loop with 4-wheel Ackermann steering.

    Each steered wheel at (x_i, y_i) gets  delta_i = atan2(x_i k, 1 - y_i k)  and every wheel
    the speed  v * |(1 - y_i k, x_i k)| / r  for path curvature k. Wheel speeds are tracked
    with a PI loop around the DC-motor feed-forward  u = omega / omega_0.
    """

    def __init__(self, sim, meta: dict) -> None:
        self.sim = sim
        m = sim.m
        r = meta["robots"][0]
        self.bid = m.body(r["body"]).id
        self.drive = np.array([m.actuator(a).id for a in r["drive"]])
        self.jdof = np.array([m.jnt_dofadr[m.joint(f"{w}_j").id] for w in r["wheels"]])
        self.wxy = np.array(r["wheel_xy"])
        self.steer = np.array([m.actuator(f"{s}_p").id for s in r["steer_xy"]])
        self.sxy = np.array(list(r["steer_xy"].values()))
        self.rw, self.w0 = r["wheel_radius"], r["omega0"]
        self.v = meta.get("speed", 0.5)
        wp = np.array(meta["waypoints"], float)
        pts = []
        for a, b in zip(wp, np.roll(wp, -1, axis=0)):          # densify the closed loop
            n = max(2, int(np.linalg.norm(b - a) / 0.2))
            pts += [a + (b - a) * k / n for k in range(n)]
        self.P = np.array(pts)
        self.idx = int(np.argmin(np.linalg.norm(self.P - sim.d.xpos[self.bid][:2], axis=1)))
        self.integ = np.zeros(len(self.drive))
        kmax = np.tan(np.radians(38)) / np.max(np.abs(self.sxy[:, 0]))
        self.kmax = kmax / (1 + kmax * np.max(np.abs(self.sxy[:, 1])))
        self.laps = 0
        self.err = 0.0

    def update(self, t: float, dt: float) -> None:
        d = self.sim.d
        p = d.xpos[self.bid][:2]
        R = d.xmat[self.bid].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        n = len(self.P)
        win = (self.idx + np.arange(60)) % n
        dist = np.linalg.norm(self.P[win] - p, axis=1)
        k = int(np.argmin(dist))
        self.err = float(dist[k])                               # cross-track error
        if self.idx + k >= n:
            self.laps += 1
        self.idx = (self.idx + k) % n
        Ld, j = 1.2, self.idx
        for _ in range(n):
            if np.linalg.norm(self.P[j] - p) >= Ld:
                break
            j = (j + 1) % n
        tgt = self.P[j]
        alpha = (np.arctan2(tgt[1] - p[1], tgt[0] - p[0]) - yaw + np.pi) % (2 * np.pi) - np.pi
        kappa = float(np.clip(2 * np.sin(alpha) / Ld, -self.kmax, self.kmax))
        v = self.v * (1.0 if abs(alpha) < 0.6 else 0.4)
        d.ctrl[self.steer] = np.arctan2(self.sxy[:, 0] * kappa, 1 - self.sxy[:, 1] * kappa)
        w_t = v * np.hypot(1 - self.wxy[:, 1] * kappa, self.wxy[:, 0] * kappa) / self.rw
        w = d.qvel[self.jdof]
        e = (w_t - w) / self.w0
        self.integ = np.clip(self.integ + 3.0 * e * dt, -0.6, 0.6)
        d.ctrl[self.drive] = np.clip(w_t / self.w0 + 1.5 * e + self.integ, -1, 1)


CONTROLLERS = {"quad_waypoints": QuadMission, "rover_ackermann": RoverAckermann}
