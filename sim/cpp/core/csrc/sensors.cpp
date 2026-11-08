// Sensor models + per-step sampling (port of sim/sensing/models.py and suite.py).
#include <algorithm>
#include <limits>

#include "core.hpp"

namespace rz {

void GaussMarkov::init(double sigma_, double tau_, int n, Rng& rng) {
  sigma = sigma_;
  tau = tau_;
  x.resize(n);
  for (double& v : x) v = sigma * rng.normal();
}

void GaussMarkov::step(double dt, Rng& rng) {
  double a = std::exp(-dt / tau), b = sigma * std::sqrt(1 - a * a);
  for (double& v : x) v = a * v + b * rng.normal();
}

template <int N>
void DelayLine<N>::drain(double t, std::vector<Item>* out) {
  std::vector<Item> ready;
  for (auto it = q.begin(); it != q.end();) {
    if (it->deliver <= t) { ready.push_back(*it); it = q.erase(it); }
    else ++it;
  }
  if (ready.empty()) return;
  std::sort(ready.begin(), ready.end(), [](const Item& a, const Item& b) { return a.capture < b.capture; });
  const Item& last = ready.back();
  if (!has_latest || last.capture >= latest_t) { has_latest = true; latest_t = last.capture; latest = last.v; }
  if (out) out->insert(out->end(), ready.begin(), ready.end());
}
template struct DelayLine<2>;
template struct DelayLine<4>;
template struct DelayLine<7>;
template struct DelayLine<12>;

static double transmissibility(double f, double fn, double z) {
  double r = f / fn, a = 2 * z * r;
  return std::sqrt((1 + a * a) / ((1 - r * r) * (1 - r * r) + a * a));
}
static double lowpass(double f, double fc) { double q = f / fc; return 1.0 / std::sqrt(1 + q * q * q * q); }

static void small_rotation(Rng& rng, double sig, double out[9]) {
  double a[3] = {sig * rng.normal(), sig * rng.normal(), sig * rng.normal()};
  double M[9] = {1, -a[2], a[1], a[2], 1, -a[0], -a[1], a[0], 1};
  for (int i = 0; i < 9; ++i) out[i] = M[i];
}

// ------------------------------------------------------------------------------ IMU ---
void ImuModel::init(const ImuSpec& s, Rng& rng) {
  rate = s.rate_hz;
  double sq = std::sqrt(rate);
  g_sig = s.gyro_noise_dps_rthz * DEG * sq;
  a_sig = s.accel_noise_ug_rthz * 1e-6 * G0 * sq;
  for (double& b : g_bias0) b = rng.normal() * s.gyro_bias_dps * DEG;
  for (double& b : a_bias0) b = rng.normal() * s.accel_bias_mg * 1e-3 * G0;
  g_walk.init(s.gyro_bias_instability_dph * DEG / 3600, s.bias_tau_s, 3, rng);
  a_walk.init(s.accel_bias_instability_ug * 1e-6 * G0, s.bias_tau_s, 3, rng);
  double se = s.scale_error_pct / 100, mis = s.misalign_deg * DEG;
  for (double* M : {Mg, Ma}) {
    double sc[3] = {1 + se * rng.normal(), 1 + se * rng.normal(), 1 + se * rng.normal()}, Rm[9];
    small_rotation(rng, mis, Rm);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) M[3 * i + j] = sc[i] * Rm[3 * i + j];
  }
  g_fs = s.gyro_range_dps * DEG;
  a_fs = s.accel_range_g * G0;
  g_lsb = 2 * g_fs / std::pow(2.0, s.bits);
  a_lsb = 2 * a_fs / std::pow(2.0, s.bits);
  saturated = 0;
}

void ImuModel::sample(const double gt[3], const double at[3], const double gv[3], const double av[3], double g[3],
                      double a[3], Rng& rng) {
  double dt = 1.0 / rate;
  g_walk.step(dt, rng);
  a_walk.step(dt, rng);
  double gi[3], ai[3];
  for (int k = 0; k < 3; ++k) { gi[k] = gt[k] + gv[k]; ai[k] = at[k] + av[k]; }
  bool sat = false;
  for (int i = 0; i < 3; ++i) {
    g[i] = Mg[3 * i] * gi[0] + Mg[3 * i + 1] * gi[1] + Mg[3 * i + 2] * gi[2] + g_bias0[i] + g_walk.x[i] +
           g_sig * rng.normal();
    a[i] = Ma[3 * i] * ai[0] + Ma[3 * i + 1] * ai[1] + Ma[3 * i + 2] * ai[2] + a_bias0[i] + a_walk.x[i] +
           a_sig * rng.normal();
    sat = sat || std::fabs(g[i]) > g_fs || std::fabs(a[i]) > a_fs;
  }
  if (sat) ++saturated;
  for (int i = 0; i < 3; ++i) {
    g[i] = std::nearbyint(clampd(g[i], -g_fs, g_fs) / g_lsb) * g_lsb;
    a[i] = std::nearbyint(clampd(a[i], -a_fs, a_fs) / a_lsb) * a_lsb;
  }
}

// -------------------------------------------------------------------- vibration -----
double RotorVibration::gain(double f) const {
  return transmissibility(f, arm_hz, arm_zeta) * transmissibility(f, mount_hz, mount_zeta) * lowpass(f, aaf_hz);
}

void RotorVibration::set_rotors(double t, const double* n_revs, const double* thrust) {
  size_t N = r.size();
  for (size_t i = 0; i < N; ++i) phase[i] += 2 * PI * n[i] * (t - t0);
  t0 = t;
  for (size_t i = 0; i < N; ++i) {
    n[i] = n_revs[i];
    T[i] = thrust[i];
    w[i] = 2 * PI * std::max(n[i], 1e-3);
    A[i] = U * w[i] * w[i] * gain(n[i]);
    Ar[i] = ripple * T[i] * gain(blades * n[i]);
  }
  latched = true;
}

void RotorVibration::at(double t, double gyro[3], double acc[3]) const {
  double F[3] = {0, 0, 0}, tau[3] = {0, 0, 0};
  for (size_t i = 0; i < r.size(); ++i) {
    double ph = phase[i] + w[i] * (t - t0);
    double c = std::cos(ph), s = std::sin(ph), pb = blades * ph + rip_off[i];
    double Fx = A[i] * c, Fy = A[i] * s * spin[i], Fz = Ar[i] * std::cos(pb);
    double Ix = A[i] / w[i] * s, Iy = -A[i] / w[i] * spin[i] * c, Iz = Ar[i] * std::sin(pb) / (blades * w[i]);
    F[0] += Fx; F[1] += Fy; F[2] += Fz;
    const auto& rr = r[i];
    tau[0] += rr[1] * Iz - rr[2] * Iy;
    tau[1] += rr[2] * Ix - rr[0] * Iz;
    tau[2] += rr[0] * Iy - rr[1] * Ix;
  }
  for (int k = 0; k < 3; ++k) acc[k] = F[k] / mass;
  for (int k = 0; k < 3; ++k) gyro[k] = Iinv[3 * k] * tau[0] + Iinv[3 * k + 1] * tau[1] + Iinv[3 * k + 2] * tau[2];
}

// ------------------------------------------------------------------- baro / mag -----
double BaroModel::sample(double h_true, double airspeed, Rng& rng) {
  drift.step(1.0 / rate, rng);
  double h = h_true + offset + drift.x[0] - cp * airspeed * airspeed / (2 * G0) + sig * rng.normal();
  return std::nearbyint(h / lsb) * lsb;
}

void MagModel::sample(const double R[9], double current, double out[3], Rng& rng) const {
  double b[3];
  for (int j = 0; j < 3; ++j) b[j] = R[j] * b_world[0] + R[3 + j] * b_world[1] + R[6 + j] * b_world[2];  // R^T b
  for (int i = 0; i < 3; ++i)
    out[i] = soft[3 * i] * b[0] + soft[3 * i + 1] * b[1] + soft[3 * i + 2] * b[2] + hard[i] + k_current[i] * current +
             sig * rng.normal();
}

// ----------------------------------------------------------------------------- GNSS ---
void dops(const std::vector<std::array<double, 3>>& u, double& hdop, double& vdop) {
  const double inf = std::numeric_limits<double>::infinity();
  hdop = vdop = inf;
  if (u.size() < 4) return;
  double N[16] = {0};
  for (const auto& s : u) {
    double h[4] = {-s[0], -s[1], -s[2], 1.0};
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) N[4 * i + j] += h[i] * h[j];
  }
  // Gauss-Jordan
  double M[4][8];
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 8; ++c) M[r][c] = c < 4 ? N[4 * r + c] : (c - 4 == r);
  for (int c = 0; c < 4; ++c) {
    int piv = c;
    for (int r = c + 1; r < 4; ++r)
      if (std::fabs(M[r][c]) > std::fabs(M[piv][c])) piv = r;
    if (std::fabs(M[piv][c]) < 1e-12) return;
    if (piv != c) for (int k = 0; k < 8; ++k) std::swap(M[c][k], M[piv][k]);
    double sc = 1.0 / M[c][c];
    for (int k = 0; k < 8; ++k) M[c][k] *= sc;
    for (int r = 0; r < 4; ++r)
      if (r != c) { double f = M[r][c]; for (int k = 0; k < 8; ++k) M[r][k] -= f * M[c][k]; }
  }
  hdop = std::sqrt(std::max(M[0][4] + M[1][5], 0.0));
  vdop = std::sqrt(std::max(M[2][6], 0.0));
}

void GnssModel::sats_at(double t, std::vector<std::array<double, 3>>& out) const {
  double a = az_rate * t, c = std::cos(a), s = std::sin(a);
  out.resize(sats.size());
  for (size_t i = 0; i < sats.size(); ++i)
    out[i] = {c * sats[i][0] - s * sats[i][1], s * sats[i][0] + c * sats[i][1], sats[i][2]};
}

RayScene::~RayScene() {
  if (d) mj_deleteData(d);
  if (m) mj_deleteModel(m);
}

double Sensors::sky_survival(const double p[3], const double u[3]) const {
  static const mjtByte groups[6] = {1, 1, 1, 0, 0, 0};  // visual geometry, not coarse colliders
  if (terrain && terrain->blocks(p, u)) return 0.0;
  double q[3] = {p[0], p[1], p[2]}, out = 1.0;
  int gid[1] = {-1};
  for (int layer = 0; layer < max_layers; ++layer) {
    double dist = mj_ray(rays->m, rays->d, q, u, groups, 1, bid, gid, nullptr);
    if (dist < 0) break;
    double op = rays->opacity[gid[0]];
    out *= 1.0 - (op < 0 ? foliage_loss : op);
    if (out < 0.02) return 0.0;
    for (int k = 0; k < 3; ++k) q[k] += u[k] * (dist + 0.05);
  }
  return out;
}

// --------------------------------------------------------------------- suite update ---
static void due(double& next, double t0, double t1, double rate, std::vector<double>& ts) {
  ts.clear();
  if (next < t0 - 1.0 / rate) next = t0;
  while (next < t1) { ts.push_back(next); next += 1.0 / rate; }
}

void Sensors::update(const mjModel* m, const mjData* d, const Wind* wind, double current, double t0, double dt) {
  const double t1 = t0 + dt;
  std::vector<double> ts;
  const double* gt = d->sensordata + gyro_adr;
  const double* at = d->sensordata + acc_adr;
  if (vib_on) {
    double nr[8], Tr[8];
    for (size_t i = 0; i < rotor_act.size(); ++i) {
      Tr[i] = std::max(d->actuator_force[rotor_act[i]], 0.0);
      nr[i] = std::sqrt(Tr[i] * kn);
    }
    vib.set_rotors(t0, nr, Tr);
  }
  if (imu_on) {
    due(next_imu, t0, t1, imu.rate, ts);
    for (double t : ts) {
      double gv[3] = {0, 0, 0}, av[3] = {0, 0, 0}, g[3], a[3];
      if (vib_on) vib.at(t, gv, av);
      imu.sample(gt, at, gv, av, g, a, r_imu);
      l_imu.push(t, {t, g[0], g[1], g[2], a[0], a[1], a[2]}, imu_spec.lat.sample(r_lat));
      if (record) log_imu.push_back({t, g[0], g[1], g[2], a[0], a[1], a[2], gt[0], gt[1], gt[2], at[0], at[1], at[2]});
    }
  }
  if (!(baro_on || mag_on || gnss_on)) return;
  const double* ps = d->site_xpos + 3 * site;
  const double* R = d->xmat + 9 * bid;
  double vl[3] = {0, 0, 0};
  if (dof >= 0) for (int k = 0; k < 3; ++k) vl[k] = d->qvel[dof + k];
  if (baro_on) {
    due(next_baro, t0, t1, baro.rate, ts);
    for (double t : ts) {
      double va[3] = {vl[0], vl[1], vl[2]};
      if (wind && wind->on) {
        double w[3];
        wind->velocity(ps, t, w);
        for (int k = 0; k < 3; ++k) va[k] -= w[k];
      }
      double h = baro.sample(ps[2], std::sqrt(va[0] * va[0] + va[1] * va[1] + va[2] * va[2]), r_baro);
      l_baro.push(t, {t, h}, baro.lat.sample(r_lat));
      if (record) log_baro.push_back({t, h, ps[2]});
    }
  }
  if (mag_on) {
    due(next_mag, t0, t1, mag.rate, ts);
    for (double t : ts) {
      double b[3];
      mag.sample(R, current, b, r_mag);
      l_mag.push(t, {t, b[0], b[1], b[2]}, mag.lat.sample(r_lat));
      if (record) {
        double bt[3];
        for (int j = 0; j < 3; ++j) bt[j] = R[j] * mag.b_world[0] + R[3 + j] * mag.b_world[1] + R[6 + j] * mag.b_world[2];
        log_mag.push_back({t, b[0], b[1], b[2], bt[0], bt[1], bt[2]});
      }
    }
  }
  if (gnss_on) {
    due(next_gnss, t0, t1, gnss.rate, ts);
    std::vector<std::array<double, 3>> u, vis;
    for (double t : ts) {
      double ant[3];
      for (int i = 0; i < 3; ++i)
        ant[i] = d->xpos[3 * bid + i] + R[3 * i] * antenna[0] + R[3 * i + 1] * antenna[1] + R[3 * i + 2] * antenna[2];
      gnss.sats_at(t, u);
      vis.clear();
      double surv_sum = 0;
      for (const auto& s : u) {
        double sv = rays ? sky_survival(ant, s.data()) : 1.0;
        surv_sum += sv;
        if (r_gnss.uniform() < sv) vis.push_back(s);
      }
      double blocked = 1.0 - surv_sum / double(u.size());
      double hd, vd;
      dops(vis, hd, vd);
      double gdt = 1.0 / gnss.rate;
      gnss.bias.step(gdt, r_gnss);
      gnss.mp.step(gdt, r_gnss);
      const double nan = std::numeric_limits<double>::quiet_NaN();
      std::array<double, 12> out = {t, 0.0, double(vis.size()), hd, blocked, nan, nan, nan, nan, nan, nan, 0};
      if (!(int(vis.size()) < gnss.min_sats || !std::isfinite(hd) || hd > 20)) {
        double k[3] = {hd / gnss.h_open, hd / gnss.h_open, vd / gnss.v_open};
        out[1] = 1.0;
        for (int i = 0; i < 3; ++i) {
          out[5 + i] = ant[i] + k[i] * (gnss.bias.x[i] + gnss.white * r_gnss.normal()) + blocked * gnss.mp.x[i];
          out[8 + i] = vl[i] + k[i] * gnss.vel_sig * r_gnss.normal();
        }
      }
      l_gnss.push(t, out, gnss.lat.sample(r_lat));
      if (record) log_gnss.push_back({t, out[1], out[2], hd, blocked, out[5], out[6], out[7], ant[0], ant[1], ant[2]});
    }
  }
}

}  // namespace rz
