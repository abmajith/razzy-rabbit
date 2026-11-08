// Controller stages (port of sim/control/: path.py, multirotor.py, ground.py). Each stage
// reads the truth state from mjData and the previous stage's output from the Signal.
#include <algorithm>
#include <stdexcept>

#include "core.hpp"

namespace rz {

static bool invert4(const double A[16], double out[16]) {
  double M[4][8];
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 8; ++c) M[r][c] = c < 4 ? A[4 * r + c] : (c - 4 == r ? 1.0 : 0.0);
  for (int c = 0; c < 4; ++c) {
    int piv = c;
    for (int r = c + 1; r < 4; ++r)
      if (std::fabs(M[r][c]) > std::fabs(M[piv][c])) piv = r;
    if (std::fabs(M[piv][c]) < 1e-14) return false;
    if (piv != c)
      for (int k = 0; k < 8; ++k) std::swap(M[c][k], M[piv][k]);
    double s = 1.0 / M[c][c];
    for (int k = 0; k < 8; ++k) M[c][k] *= s;
    for (int r = 0; r < 4; ++r)
      if (r != c) {
        double f = M[r][c];
        for (int k = 0; k < 8; ++k) M[r][k] -= f * M[c][k];
      }
  }
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) out[4 * r + c] = M[r][c + 4];
  return true;
}

void BodyRef::init(const mjModel* m, int body) {
  bid = body;
  int jnt = m->body_jntadr[bid];
  qa = m->jnt_qposadr[jnt];
  va = m->jnt_dofadr[jnt];
}

double curvature_limit(const std::vector<std::array<double, 2>>& sxy, double max_steer_deg) {
  double sxmax = 0, symax = 0;
  for (const auto& q : sxy) { sxmax = std::max(sxmax, std::fabs(q[0])); symax = std::max(symax, std::fabs(q[1])); }
  double k0 = std::tan(max_steer_deg * DEG) / sxmax;
  return k0 / (1 + k0 * symax);
}

// ------------------------------------------------------------------- FollowPath ---
void FollowPath::init(const mjData* d, const std::vector<std::array<double, 3>>& wps) {
  const double* p = d->qpos + b.qa;
  std::array<double, 3> lift = {p[0], p[1], wps[0][2]};
  P.clear();
  P.push_back(lift);
  for (const auto& w : wps) P.push_back(w);
  s.assign(P.size(), 0.0);
  for (size_t i = 1; i < P.size(); ++i)
    s[i] = s[i - 1] + std::hypot(P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1]) + 1e-9;
  s_now = 0;
  dirn = 1;
  yaw = std::atan2(P[1][1] - P[0][1], P[1][0] - P[0][0]);
  ground0 = p[2];
  e = 0;
}

void FollowPath::target(double out[3], double tan[3]) const {
  double sq = clampd(s_now, 0.0, s.back());
  int i = int(std::upper_bound(s.begin(), s.end(), sq) - s.begin()) - 1;
  i = std::min(i, int(P.size()) - 2);
  double f = (sq - s[i]) / (s[i + 1] - s[i]);
  double t3[3] = {P[i + 1][0] - P[i][0], P[i + 1][1] - P[i][1], P[i + 1][2] - P[i][2]};
  double tn = std::hypot(t3[0], t3[1]) + 1e-9;
  for (int k = 0; k < 3; ++k) {
    out[k] = P[i][k] + f * t3[k];
    tan[k] = t3[k] / tn;
  }
  if (agl && terrain) out[2] += terrain->height(out[0], out[1]);
}

void FollowPath::update(StageCtx& c, Signal& sg, double t, double dt) {
  double p_d[3], tan[3], v_d[3] = {0, 0, 0};
  target(p_d, tan);
  const double* pos = c.d->qpos + b.qa;
  if (t < t_takeoff) {
    double z_goal = p_d[2];
    p_d[0] = pos[0];
    p_d[1] = pos[1];
    p_d[2] = ground0 + (z_goal - ground0) * (t / t_takeoff);
  } else {
    double lag = std::sqrt((pos[0] - p_d[0]) * (pos[0] - p_d[0]) + (pos[1] - p_d[1]) * (pos[1] - p_d[1]) +
                           (pos[2] - p_d[2]) * (pos[2] - p_d[2]));
    e = lag;
    double speed = cruise * clampd(1.6 - lag / 2.0, 0.0, 1.0);
    s_now += dirn * speed * dt;
    if (s_now >= s.back() || s_now <= s[1]) {
      dirn *= -1;
      s_now = clampd(s_now, s[1], s.back());
    }
    for (int k = 0; k < 3; ++k) v_d[k] = dirn * tan[k] * speed;
    if (agl) v_d[2] = 0.0;
    if (std::hypot(tan[0], tan[1]) > 1e-6) {
      double yaw_t = std::atan2(dirn * tan[1], dirn * tan[0]);
      double dy = floormod(yaw_t - yaw + PI, 2 * PI) - PI;
      yaw += clampd(dy, -yaw_rate * dt, yaw_rate * dt);
    }
  }
  for (int k = 0; k < 3; ++k) { sg.p[k] = p_d[k]; sg.v[k] = v_d[k]; }
  sg.yaw = yaw;
}

// -------------------------------------------------------------- VelocityCommand ---
void VelocityCommand::update(StageCtx& c, Signal& sg, double, double dt) {
  const std::vector<double>& cmd = *c.command;
  double vx = cmd.size() > 0 ? cmd[0] : 0, vy = cmd.size() > 1 ? cmd[1] : 0, vz = cmd.size() > 2 ? cmd[2] : 0;
  double wz = cmd.size() > 3 ? cmd[3] : 0;
  yaw += wz * dt;
  const double* pos = c.d->qpos + b.qa;
  for (int k = 0; k < 3; ++k) sg.p[k] = pos[k];
  sg.v[0] = vx; sg.v[1] = vy; sg.v[2] = vz;
  sg.yaw = yaw;
}

// ---------------------------------------------------------------- Se3Controller ---
void Se3Controller::update(StageCtx& c, Signal& sg, double, double) {
  const mjData* d = c.d;
  const double* R = d->xmat + 9 * b.bid;
  const double* w = d->qvel + b.va + 3;                 // body-frame angular velocity
  const double* p = d->qpos + b.qa;
  const double* v = d->qvel + b.va;
  double e_p[3] = {p[0] - sg.p[0], p[1] - sg.p[1], p[2] - sg.p[2]};
  double n = std::sqrt(e_p[0] * e_p[0] + e_p[1] * e_p[1] + e_p[2] * e_p[2]);
  if (n > 3.0)
    for (double& x : e_p) x *= 3.0 / n;
  double a[3];
  for (int k = 0; k < 3; ++k) a[k] = -kp * e_p[k] - kd * (v[k] - sg.v[k]);
  a[2] += 9.81;
  a[2] = std::max(a[2], 2.0);
  double hmax = std::tan(max_tilt_deg * DEG) * a[2];
  double hn = std::sqrt(a[0] * a[0] + a[1] * a[1]);
  if (hn > hmax) { a[0] *= hmax / hn; a[1] *= hmax / hn; }
  double F[3] = {mass * a[0], mass * a[1], mass * a[2]};
  double Fn = std::sqrt(F[0] * F[0] + F[1] * F[1] + F[2] * F[2]);
  double b3[3] = {F[0] / Fn, F[1] / Fn, F[2] / Fn};
  double b1c[3] = {std::cos(sg.yaw), std::sin(sg.yaw), 0.0};
  double b2[3] = {b3[1] * b1c[2] - b3[2] * b1c[1], b3[2] * b1c[0] - b3[0] * b1c[2], b3[0] * b1c[1] - b3[1] * b1c[0]};
  double b2n = std::sqrt(b2[0] * b2[0] + b2[1] * b2[1] + b2[2] * b2[2]);
  for (double& x : b2) x /= b2n;
  double b1[3] = {b2[1] * b3[2] - b2[2] * b3[1], b2[2] * b3[0] - b2[0] * b3[2], b2[0] * b3[1] - b2[1] * b3[0]};
  double Rd[9] = {b1[0], b2[0], b3[0], b1[1], b2[1], b3[1], b1[2], b2[2], b3[2]};   // columns b1 b2 b3
  double E[9];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      double s1 = 0, s2 = 0;
      for (int k = 0; k < 3; ++k) {
        s1 += Rd[3 * k + i] * R[3 * k + j];
        s2 += R[3 * k + i] * Rd[3 * k + j];
      }
      E[3 * i + j] = 0.5 * (s1 - s2);
    }
  double eR[3] = {E[7], E[2], E[3]};
  double Jw[3] = {J[0] * w[0], J[1] * w[1], J[2] * w[2]};
  double tau[3];
  for (int k = 0; k < 3; ++k) tau[k] = J[k] * (-kR * eR[k] - kW * w[k]);
  tau[0] += w[1] * Jw[2] - w[2] * Jw[1];
  tau[1] += w[2] * Jw[0] - w[0] * Jw[2];
  tau[2] += w[0] * Jw[1] - w[1] * Jw[0];
  sg.wrench[0] = F[0] * R[2] + F[1] * R[5] + F[2] * R[8];
  sg.wrench[1] = tau[0]; sg.wrench[2] = tau[1]; sg.wrench[3] = tau[2];
}

// ------------------------------------------------------------------------ Mixer ---
void Mixer::init(const mjModel* m, int body) {
  double A[16];
  const double* com = m->body_ipos + 3 * body;           // torques act about the centre of mass
  if (act.size() != 4) throw std::runtime_error("mixer: exactly 4 rotors supported");
  for (size_t i = 0; i < act.size(); ++i) {
    int aid = act[i];
    int site = m->actuator_trnid[2 * aid];
    double x = m->site_pos[3 * site] - com[0], y = m->site_pos[3 * site + 1] - com[1];
    A[0 * 4 + i] = 1.0;
    A[1 * 4 + i] = y;
    A[2 * 4 + i] = -x;
    A[3 * 4 + i] = m->actuator_gear[6 * aid + 5];
  }
  if (!invert4(A, Ainv)) throw std::runtime_error("quad mixer is singular");
}

void Mixer::update(StageCtx& c, Signal& sg, double, double) {
  for (size_t i = 0; i < act.size(); ++i) {
    double u = 0;
    for (int k = 0; k < 4; ++k) u += Ainv[4 * i + k] * sg.wrench[k];
    int aid = act[i];
    c.d->ctrl[aid] = clampd(u, 0.0, c.m->actuator_ctrlrange[2 * aid + 1]);   // live: battery sag lowers it
  }
}

// ------------------------------------------------------------------ PurePursuit ---
void PurePursuit::init(const mjData* d, const std::vector<std::array<double, 2>>& wp, double spacing) {
  P.clear();
  for (size_t i = 0; i < wp.size(); ++i) {
    const auto& a = wp[i];
    const auto& b = wp[(i + 1) % wp.size()];
    int n = std::max(2, int(std::hypot(b[0] - a[0], b[1] - a[1]) / spacing));
    for (int k = 0; k < n; ++k) P.push_back({a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n});
  }
  const double* p = d->xpos + 3 * bid;
  double best = 1e300;
  for (size_t i = 0; i < P.size(); ++i) {
    double dd = std::hypot(P[i][0] - p[0], P[i][1] - p[1]);
    if (dd < best) { best = dd; idx = int(i); }
  }
  nlaps = 0;
  e = 0;
}

void PurePursuit::update(StageCtx& c, Signal& sg, double, double) {
  const double* p = c.d->xpos + 3 * bid;
  const double* R = c.d->xmat + 9 * bid;
  double yaw = std::atan2(R[3], R[0]);
  int n = int(P.size());
  int kbest = 0;
  double dbest = 1e300;
  for (int k = 0; k < window; ++k) {
    const auto& q = P[(idx + k) % n];
    double dd = std::hypot(q[0] - p[0], q[1] - p[1]);
    if (dd < dbest) { dbest = dd; kbest = k; }
  }
  e = dbest;
  if (idx + kbest >= n) ++nlaps;
  idx = (idx + kbest) % n;
  const double Ld = lookahead;
  int j = idx;
  for (int it = 0; it < n; ++it) {
    if (std::hypot(P[j][0] - p[0], P[j][1] - p[1]) >= Ld) break;
    j = (j + 1) % n;
  }
  double alpha = floormod(std::atan2(P[j][1] - p[1], P[j][0] - p[0]) - yaw + PI, 2 * PI) - PI;
  sg.kappa = clampd(2 * std::sin(alpha) / Ld, -kmax, kmax);
  sg.speed = speed * (std::fabs(alpha) < slow_angle ? 1.0 : slow_factor);
}

void DriveCommand::update(StageCtx& c, Signal& sg, double, double) {
  const std::vector<double>& cmd = *c.command;
  sg.speed = cmd.size() > 0 ? cmd[0] : 0.0;
  sg.kappa = clampd(cmd.size() > 1 ? cmd[1] : 0.0, -kmax, kmax);
}

// -------------------------------------------------------------------- Ackermann ---
void Ackermann::update(StageCtx& c, Signal& sg, double, double) {
  const double kappa = sg.kappa;
  for (size_t i = 0; i < steer.size(); ++i)
    c.d->ctrl[steer[i]] = std::atan2(sxy[i][0] * kappa, 1 - sxy[i][1] * kappa);
  sg.wheel.resize(wxy.size());
  for (size_t i = 0; i < wxy.size(); ++i)
    sg.wheel[i] = sg.speed * std::hypot(1 - wxy[i][1] * kappa, wxy[i][0] * kappa) / rw;
}

void WheelSpeedPI::update(StageCtx& c, Signal& sg, double, double dt) {
  for (size_t i = 0; i < drive.size(); ++i) {
    double w_t = sg.wheel[i];
    double e = (w_t - c.d->qvel[jdof[i]]) / w0;
    integ[i] = clampd(integ[i] + ki * e * dt, -limit, limit);
    c.d->ctrl[drive[i]] = clampd(w_t / w0 + kp * e + integ[i], -1.0, 1.0);
  }
}

}  // namespace rz
