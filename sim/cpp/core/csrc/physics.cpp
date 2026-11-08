// RNG, terrain lookup, analytic wind field (port of sim/wind.py) and power-train plants
// (port of sim/design/plant.py).
#include <algorithm>
#include <complex>

#include "core.hpp"

namespace rz {

// ------------------------------------------------------------------------------ RNG ---
static inline uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

Rng::Rng(uint64_t seed, uint64_t stream) {
  uint64_t k = splitmix64(splitmix64(seed) ^ (stream * 0xD1B54A32D192ED03ULL + 0x632BE59BD9B4E019ULL));
  key_[0] = uint32_t(k);
  key_[1] = uint32_t(k >> 32);
}

uint32_t Rng::next_u32() {
  if (pos_ == 4) {
    uint32_t c[4] = {uint32_t(ctr_), uint32_t(ctr_ >> 32), 0, 0}, k[2] = {key_[0], key_[1]};
    for (int r = 0; r < 10; ++r) {
      uint64_t p0 = uint64_t(0xD2511F53u) * c[0], p1 = uint64_t(0xCD9E8D57u) * c[2];
      uint32_t hi0 = uint32_t(p0 >> 32), lo0 = uint32_t(p0), hi1 = uint32_t(p1 >> 32), lo1 = uint32_t(p1);
      uint32_t n[4] = {hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0};
      c[0] = n[0]; c[1] = n[1]; c[2] = n[2]; c[3] = n[3];
      k[0] += 0x9E3779B9u;
      k[1] += 0xBB67AE85u;
    }
    for (int i = 0; i < 4; ++i) buf_[i] = c[i];
    ++ctr_;
    pos_ = 0;
  }
  return buf_[pos_++];
}

double Rng::uniform() {
  uint64_t a = next_u32() >> 5, b = next_u32() >> 6;    // 53 random bits
  return (double(a) * 67108864.0 + double(b)) * (1.0 / 9007199254740992.0);
}

double Rng::normal() {
  if (has_spare_) { has_spare_ = false; return spare_; }
  double u1 = uniform(), u2 = uniform();
  if (u1 < 1e-300) u1 = 1e-300;
  double r = std::sqrt(-2.0 * std::log(u1)), a = 2.0 * PI * u2;
  spare_ = r * std::sin(a);
  has_spare_ = true;
  return r * std::cos(a);
}

// -------------------------------------------------------------------------- terrain ---
double Terrain::height(double x, double y) const {
  double cx = clampd((x + hx) / (2 * hx) * (nc - 1), 0.0, nc - 1.001);
  double ry = clampd((y + hy) / (2 * hy) * (nr - 1), 0.0, nr - 1.001);
  int c0 = int(cx), r0 = int(ry);
  double fx = cx - c0, fy = ry - r0;
  const double* Z = z.data();
  return Z[r0 * nc + c0] * (1 - fx) * (1 - fy) + Z[r0 * nc + c0 + 1] * fx * (1 - fy) +
         Z[(r0 + 1) * nc + c0] * (1 - fx) * fy + Z[(r0 + 1) * nc + c0 + 1] * fx * fy;
}

bool Terrain::blocks(const double p[3], const double u[3]) const {
  const double step = std::min(2 * hx / (nc - 1), 2 * hy / (nr - 1));
  for (double s = step; s < 4 * (hx + hy); s += step) {
    double x = p[0] + s * u[0], y = p[1] + s * u[1], z = p[2] + s * u[2];
    if (z > zmax || std::fabs(x) > hx || std::fabs(y) > hy) return false;
    if (z < height(x, y)) return true;
  }
  return false;
}

// ----------------------------------------------------------------------------- wind ---
void Wind::velocity(const double p[3], double t, double out[3]) const {
  const double x = p[0], y = p[1], z = p[2];
  const double h = std::max(z - ground(x, y), 0.05);
  const double prof = std::log((h + z0) / z0) / std::log((z_ref + z0) / z0);
  using C = std::complex<double>;
  const C zc(x, y), e = std::exp(C(0, -heading));
  C dW = U * e;
  bool inside = false;
  for (const auto& c : cyl) {
    C rel = zc - C(c[0], c[1]);
    bool below = z < c[3];
    double r2 = std::norm(rel);
    if (below) dW -= U * std::conj(e) * c[2] * c[2] / ((r2 > 1e-9 ? rel : C(1, 0)) * (r2 > 1e-9 ? rel : C(1, 0)));
    inside = inside || (below && r2 < c[2] * c[2]);
  }
  double u = dW.real(), v = -dW.imag(), w = 0.0;
  const double xi0 = x - U * dirx * t, xi1 = y - U * diry * t;
  double ws = 0.0;
  for (size_t j = 0; j < kx.size(); ++j) {
    double arg = xi0 * kx[j] + xi1 * ky[j] + omega[j] * t + phase[j];
    double c = std::cos(arg) * amp[j];
    u += c * ky[j];
    v -= c * kx[j];
    ws += std::sin(arg + wphase[j]) * amp[j] * kn[j];
  }
  w = 0.25 * ws * 0.5;
  u *= prof;
  v *= prof;
  if (terrain && terrain->valid()) {
    const double eps = 0.5;
    double gx = (ground(x + eps, y) - ground(x - eps, y)) / (2 * eps);
    double gy = (ground(x, y + eps) - ground(x, y - eps)) / (2 * eps);
    w += (u * gx + v * gy) * std::exp(-h / 40.0);
  }
  if (turbine) {
    double rel[3] = {x - hub[0], y - hub[1], z - hub[2]};
    double s = rel[0] * dirx + rel[1] * diry;
    double q[3] = {rel[0] - s * dirx, rel[1] - s * diry, rel[2]};
    double r = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
    double sp = std::max(s, 0.0);
    double Rw = tR + tk * sp;
    double deficit = (1 - std::sqrt(1 - tCt)) / std::pow(1 + tk * sp / tR, 2);
    if (s > 0 && r < Rw) { u *= 1 - deficit; v *= 1 - deficit; }
  }
  if (inside) u = v = w = 0.0;
  out[0] = u; out[1] = v; out[2] = w;
}

void Wind::drag(const double p[3], const double vel[3], double CdA, double t, double f[3]) const {
  double vw[3];
  velocity(p, t, vw);
  double vr[3] = {vw[0] - vel[0], vw[1] - vel[1], vw[2] - vel[2]};
  double s = 0.5 * 1.225 * CdA * std::sqrt(vr[0] * vr[0] + vr[1] * vr[1] + vr[2] * vr[2]);
  for (int i = 0; i < 3; ++i) f[i] = s * vr[i];
}

// -------------------------------------------------------------------------- battery ---
double Pack::ocv(double soc) const {
  const auto& S = soc_tab;
  const auto& O = ocv_tab;
  double v;
  if (soc <= S.front()) v = O.front();
  else if (soc >= S.back()) v = O.back();
  else {
    size_t k = std::upper_bound(S.begin(), S.end(), soc) - S.begin();
    double f = (soc - S[k - 1]) / (S[k] - S[k - 1]);
    v = O[k - 1] + f * (O[k] - O[k - 1]);
  }
  return v * cells;
}

void Pack::terminal(double power, double soc, double& v_out, double& i_out) const {
  double e = ocv(soc);
  double disc = e * e - 4 * r * power;
  if (disc <= 0) { v_out = e / 2; i_out = e / (2 * r); return; }
  double i = (e - std::sqrt(disc)) / (2 * r);
  v_out = e - i * r;
  i_out = i;
}

void rotor_point(const RotorParams& p, double T, double& n, double& Pe) {
  T = std::max(T, 0.0);
  n = std::sqrt(T / (p.ct * p.rho * std::pow(p.D, 4)));
  double w = 2 * PI * n;
  double Pm = p.cp * p.rho * n * n * n * std::pow(p.D, 5);
  double Kt = 60.0 / (2 * PI * p.kv);
  double Q = w > 0 ? Pm / std::max(w, 1e-9) : 0.0;
  double I = Q / Kt + p.i0;
  double V = w * Kt + I * p.r_ohm;
  Pe = V * I;
}

// ------------------------------------------------------------------------- plants ---
void DronePlant::update(mjModel* m, mjData* d, const Wind* wind, double t, double dt) {
  double Tsum = 0, Pe_sum = 0;
  for (int a : act) {
    double T = d->actuator_force[a], n, Pe;
    rotor_point(rp, T, n, Pe);
    Pe_sum += Pe;
    Tsum += T;
  }
  power = Pe_sum / esc_eff + avionics_w;
  pack.terminal(power, soc, v, i);
  soc = std::max(0.0, soc - i * dt / 3600.0 / pack.cap_ah);
  energy_wh += power * dt / 3600.0;
  double scale = soc > 0 ? (v / v_ref) * (v / v_ref) : 0.0;
  for (int a : act) m->actuator_ctrlrange[2 * a + 1] = t_max_ref * scale;
  // rotor drag: opposes airspeed in the rotor plane, proportional to total thrust
  const double* R = d->xmat + 9 * bid;
  double vel[3] = {d->qvel[dof], d->qvel[dof + 1], d->qvel[dof + 2]};
  if (wind && wind->on) {
    double vw[3];
    wind->velocity(d->xpos + 3 * bid, t, vw);
    for (int k = 0; k < 3; ++k) vel[k] -= vw[k];
  }
  double z[3] = {R[2], R[5], R[8]};
  double vz = vel[0] * z[0] + vel[1] * z[1] + vel[2] * z[2];
  for (int k = 0; k < 3; ++k) d->xfrc_applied[6 * bid + k] += -rotor_drag * Tsum * (vel[k] - vz * z[k]);
}

void RoverPlant::update(mjModel* m, mjData* d, double, double dt) {
  double p = 0;
  for (size_t k = 0; k < act.size(); ++k) {
    int a = act[k];
    double tau = std::fabs(d->actuator_force[a]), u = std::fabs(d->ctrl[a]);
    double i_mot = i0 + (i_s - i0) * std::min(tau / tau_s[k], 1.0);
    p += v * u * i_mot;
  }
  power = p + avionics_w + n_servo * servo_w;
  pack.terminal(power, soc, v, i);
  soc = std::max(0.0, soc - i * dt / 3600.0 / pack.cap_ah);
  energy_wh += power * dt / 3600.0;
  double scale = soc > 0 ? v / pack.v_nom : 0.0;
  for (size_t k = 0; k < act.size(); ++k) {
    int a = act[k];
    m->actuator_gainprm[mjNGAIN * a] = tau_s[k] * scale;
    m->actuator_forcerange[2 * a] = -tau_s[k] * scale;
    m->actuator_forcerange[2 * a + 1] = tau_s[k] * scale;
  }
}

}  // namespace rz
