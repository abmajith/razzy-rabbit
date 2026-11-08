"""Reference controllers so every scene "does something" out of the box.

These are classical baselines (geometric SE(3) quadrotor control, pure-pursuit skid
steer, damped-least-squares IK pick-and-place). They are the thing an RL policy has to
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
        for i, aid in enumerate(self.act):
            x, y, _ = m.site_pos[m.actuator_trnid[aid, 0]]
            A[:, i] = [1.0, y, -x, m.actuator_gear[aid, 5]]
        self.Ainv = np.linalg.inv(A)
        self.fmax = m.actuator_ctrlrange[self.act, 1]
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
        d.ctrl[self.act] = np.clip(self.Ainv @ np.r_[T, tau], 0, self.fmax)


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
        if self.sim.wind is not None:  # aerodynamic drag in the wind field
            f = self.sim.wind.drag_force(self.ctl.pos, self.ctl.vel, self.CdA, t)
            self.sim.d.xfrc_applied[self.ctl.bid, :3] = f


# ======================================================================================
class RoverSerpentine:
    """Pure-pursuit skid-steer along crop beds with headland U-turns."""

    def __init__(self, sim, meta: dict) -> None:
        self.sim = sim
        m = sim.m
        r = meta["robots"][0]
        self.bid = m.body(r["body"]).id
        self.left = np.array([m.actuator(a).id for a in r["actuators"] if a.endswith("l_v")])
        self.right = np.array([m.actuator(a).id for a in r["actuators"] if a.endswith("r_v")])
        self.W, self.rw = r["track"], r["wheel_radius"]
        self.v = meta.get("speed", 0.8)
        xa, xb = meta["x_turn"]
        pts = []
        lanes = meta["lanes_y"]
        x0 = float(sim.d.xpos[self.bid][0])
        for i, y in enumerate(lanes):
            xs = np.arange(xa, xb, 0.2) if i % 2 == 0 else np.arange(xb, xa, -0.2)
            if i == 0:
                xs = np.arange(x0, xb, 0.2)
            pts += [[x, y] for x in xs]
            if i + 1 < len(lanes):
                yc, rad = (y + lanes[i + 1]) / 2, abs(lanes[i + 1] - y) / 2
                xe = xb if i % 2 == 0 else xa
                sgn = 1 if i % 2 == 0 else -1
                for a in np.linspace(-np.pi / 2, np.pi / 2, 24):
                    pts.append([xe + sgn * rad * np.cos(a) * 1.0, yc + rad * np.sin(a) * np.sign(lanes[i + 1] - y)])
        self.P = np.array(pts)
        self.idx = 0

    def update(self, t: float, dt: float) -> None:
        d = self.sim.d
        p = d.xpos[self.bid][:2]
        R = d.xmat[self.bid].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        win = self.P[self.idx:self.idx + 40]
        self.idx += int(np.argmin(np.linalg.norm(win - p, axis=1)))
        if self.idx >= len(self.P) - 3:  # end of field: stop
            d.ctrl[self.left] = d.ctrl[self.right] = 0.0
            return
        Ld = 1.3
        j = self.idx
        while j < len(self.P) - 1 and np.linalg.norm(self.P[j] - p) < Ld:
            j += 1
        tgt = self.P[j]
        alpha = (np.arctan2(tgt[1] - p[1], tgt[0] - p[0]) - yaw + np.pi) % (2 * np.pi) - np.pi
        kappa = 2 * np.sin(alpha) / Ld
        v = self.v / (1 + 1.5 * abs(kappa)) if abs(alpha) < 1.2 else 0.0
        omega = v * kappa if abs(alpha) < 1.2 else 0.6 * np.sign(alpha)
        weff = 1.6 * self.W  # skid-steer slip factor
        d.ctrl[self.left] = (v - omega * weff / 2) / self.rw
        d.ctrl[self.right] = (v + omega * weff / 2) / self.rw


# ======================================================================================
class HarvestSequencer:
    """Rail travel -> IK pre-grasp -> approach -> grasp -> retract -> drop in crate."""

    RED = np.array([0.78, 0.07, 0.04, 1.0])
    HIDE = np.zeros(4)

    def __init__(self, sim, meta: dict) -> None:
        self.sim = sim
        m = sim.m
        r = meta["robots"][0]
        name = r["body"]
        self.name = name
        self.jn = r["joints"]                     # rail, lift_z, yaw, shoulder, elbow, pitch
        self.qadr = np.array([m.joint(j).qposadr[0] for j in self.jn])
        self.act = np.array([m.actuator(f"{j}_p").id for j in self.jn])
        self.fing = np.array([m.actuator(f"{name}_finger_{s}_p").id for s in "lr"])
        self.site = m.site(r["ee"]).id
        self.held = m.geom(f"{name}_held").id
        self.crate = [m.geom(f"{name}_crate{i}").id for i in range(r["n_basket"])]
        self.fruits = np.array(meta["fruits"])
        self.fgeom = [m.geom(g).id for g in meta["fruit_geoms"]]
        self.aisle = meta["aisle_y"]
        self.x0 = float(sim.d.xpos[m.body(name).id][0])
        self.lo = m.jnt_range[[m.joint(j).id for j in self.jn], 0]
        self.hi = m.jnt_range[[m.joint(j).id for j in self.jn], 1]
        self.scratch = mujoco.MjData(m)
        self.home = np.array([0.0, 0.5, 0.0, -1.2, 2.3, -1.1])
        self.q_cmd = self.home.copy()
        self.k, self.n_crate = 0, 0
        self._plan("travel")

    # ---- IK ---------------------------------------------------------------------------
    def ik(self, q0, target, approach, iters=80):
        m, sd = self.sim.m, self.scratch
        q = q0.copy()
        free = [1, 2, 3, 4, 5]
        for _ in range(iters):
            sd.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, sd)
            def err(qq):
                sd.qpos[self.qadr] = qq
                mujoco.mj_kinematics(m, sd)
                x_axis = sd.site_xmat[self.site].reshape(3, 3)[:, 0]
                return np.r_[sd.site_xpos[self.site] - target, 0.25 * np.cross(x_axis, approach)]
            e = err(q)
            if np.linalg.norm(e[:3]) < 2e-3:
                break
            Jn = np.zeros((6, len(free)))
            for c, i in enumerate(free):
                dq = q.copy()
                dq[i] += 1e-4
                Jn[:, c] = (err(dq) - e) / 1e-4
            step = Jn.T @ np.linalg.solve(Jn @ Jn.T + 1e-3 * np.eye(6), -e)
            q[free] = np.clip(q[free] + np.clip(step, -0.3, 0.3), self.lo[free], self.hi[free])
        return q

    # ---- state machine ----------------------------------------------------------------
    def _plan(self, state):
        self.state, self.t_state = state, None
        f = self.fruits[self.k]
        a = np.array([0.0, np.sign(f[1] - self.aisle), 0.0])
        q = self.q_cmd.copy()
        crate_world = None
        if state == "travel":
            q = self.home.copy()
            q[0] = f[0] - self.x0 - 0.12
            q[1] = np.clip(f[2] - 0.75, 0.05, 1.4)
            self.dur = max(1.5, abs(q[0] - self.q_cmd[0]) / 0.6)
        elif state == "pregrasp":
            q = self.ik(self.q_cmd, f - 0.13 * a, a); self.dur = 1.4
        elif state == "approach":
            q = self.ik(self.q_cmd, f - 0.005 * a, a); self.dur = 0.8
        elif state == "grasp":
            self.dur = 0.5
        elif state == "retract":
            q = self.ik(self.q_cmd, f - 0.18 * a + [0, 0, 0.06], a); self.dur = 0.8
        elif state == "place":
            crate_world = np.array([self.x0 + self.q_cmd[0] - 0.38, self.aisle, 0.65 + 0.0])
            q = self.ik(self.q_cmd, crate_world, np.array([-1.0, 0, 0])); self.dur = 1.6
        elif state == "release":
            self.dur = 0.4
        self.q_from, self.q_to = self.q_cmd.copy(), q

    NEXT = dict(travel="pregrasp", pregrasp="approach", approach="grasp", grasp="retract",
                retract="place", place="release", release="travel")

    def update(self, t: float, dt: float) -> None:
        m, d = self.sim.m, self.sim.d
        if self.t_state is None:
            self.t_state = t
        s = np.clip((t - self.t_state) / self.dur, 0, 1)
        sm = 0.5 - 0.5 * np.cos(np.pi * s)
        self.q_cmd = self.q_from + (self.q_to - self.q_from) * sm
        d.ctrl[self.act] = self.q_cmd
        opening = self.state in ("pregrasp", "approach", "release")
        d.ctrl[self.fing] = 0.035 if opening else 0.0
        if s >= 1.0 and (self.state != "travel" or abs(d.qpos[self.qadr[0]] - self.q_to[0]) < 0.02):
            if self.state == "grasp":
                m.geom_rgba[self.fgeom[self.k]] = self.HIDE
                m.geom_rgba[self.held] = self.RED
            if self.state == "release":
                m.geom_rgba[self.held] = self.HIDE
                m.geom_rgba[self.crate[self.n_crate % len(self.crate)]] = self.RED
                self.n_crate += 1
                self.k += 1
                if self.k >= len(self.fruits):  # field cleared: regrow and restart
                    self.k = 0
                    for g in self.fgeom:
                        m.geom_rgba[g] = self.RED
                    for g in self.crate:
                        m.geom_rgba[g] = self.HIDE
            self._plan(self.NEXT[self.state])


class TurbineSpin:
    def __init__(self, sim, meta):
        self.sim, self.speed = sim, meta.get("rotor_speed", 1.5)
        self.aid = sim.m.actuator("rotor_drive").id

    def update(self, t, dt):
        self.sim.d.ctrl[self.aid] = self.speed


CONTROLLERS = {"quad_waypoints": QuadMission, "rover_serpentine": RoverSerpentine,
               "harvest": HarvestSequencer}
