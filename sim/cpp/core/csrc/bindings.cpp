// Python bindings. Configuration arrives once per reset as a plain dict built by
// sim/core.py (names already resolved to ids); the hot loop never touches Python.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <limits>

#include "env.hpp"

namespace nb = nanobind;
using namespace rz;

namespace {

bool has(const nb::dict& d, const char* k) { return d.contains(k) && !d[k].is_none(); }
double num(const nb::dict& d, const char* k, double def) { return has(d, k) ? nb::cast<double>(d[k]) : def; }
double req(const nb::dict& d, const char* k) {
  if (!has(d, k)) throw std::runtime_error(std::string("razzy_core config: missing key '") + k + "'");
  return nb::cast<double>(d[k]);
}
std::vector<double> vec(const nb::dict& d, const char* k) {
  return has(d, k) ? nb::cast<std::vector<double>>(d[k]) : std::vector<double>{};
}
std::vector<int> ivec(const nb::dict& d, const char* k) {
  return has(d, k) ? nb::cast<std::vector<int>>(d[k]) : std::vector<int>{};
}
nb::dict sub(const nb::dict& d, const char* k) { return nb::cast<nb::dict>(d[k]); }
LatencySpec lat(const nb::dict& d) { return {num(d, "lat_lo", 0), num(d, "lat_hi", num(d, "lat_lo", 0))}; }

Pack parse_pack(const nb::dict& p) {
  Pack k;
  k.soc_tab = vec(p, "soc_tab");
  k.ocv_tab = vec(p, "ocv_tab");
  k.cells = int(req(p, "cells"));
  k.cap_ah = req(p, "cap_ah");
  k.r = req(p, "r");
  k.v_nom = req(p, "v_nom");
  return k;
}

void parse_sensors(Env& e, const nb::dict& c, uint64_t seed, const RayScene* rays, const Terrain* terrain) {
  auto s = std::make_unique<Sensors>();
  s->r_imu = Rng(seed, 1); s->r_baro = Rng(seed, 2); s->r_mag = Rng(seed, 3); s->r_gnss = Rng(seed, 4);
  s->r_depth = Rng(seed, 5); s->r_rgb = Rng(seed, 6); s->r_vib = Rng(seed, 8); s->r_lat = Rng(seed, 9);
  s->gyro_adr = int(req(c, "gyro_adr")); s->acc_adr = int(req(c, "acc_adr"));
  s->site = int(req(c, "site")); s->bid = int(req(c, "bid")); s->dof = int(num(c, "dof", -1));
  s->record = num(c, "record", 0) != 0;
  if (has(c, "imu")) {
    nb::dict q = sub(c, "imu");
    ImuSpec& p = s->imu_spec;
    p.rate_hz = req(q, "rate_hz"); p.gyro_noise_dps_rthz = req(q, "gyro_noise_dps_rthz");
    p.gyro_bias_dps = req(q, "gyro_bias_dps"); p.gyro_bias_instability_dph = req(q, "gyro_bias_instability_dph");
    p.accel_noise_ug_rthz = req(q, "accel_noise_ug_rthz"); p.accel_bias_mg = req(q, "accel_bias_mg");
    p.accel_bias_instability_ug = req(q, "accel_bias_instability_ug"); p.bias_tau_s = num(q, "bias_tau_s", 200);
    p.scale_error_pct = num(q, "scale_error_pct", 0); p.misalign_deg = num(q, "misalign_deg", 0);
    p.gyro_range_dps = req(q, "gyro_range_dps"); p.accel_range_g = req(q, "accel_range_g");
    p.bits = num(q, "bits", 16); p.aaf_hz = num(q, "aaf_hz", 1e9); p.lat = lat(q);
    s->imu.init(p, s->r_imu);
    s->imu_on = true;
  }
  if (has(c, "vib") && s->imu_on) {
    nb::dict q = sub(c, "vib");
    RotorVibration& v = s->vib;
    auto rp = vec(q, "rotor_pos");
    size_t n = rp.size() / 3;
    for (size_t i = 0; i < n; ++i) v.r.push_back({rp[3 * i], rp[3 * i + 1], rp[3 * i + 2]});
    v.spin = vec(q, "spin");
    v.mass = req(q, "mass");
    auto Iinv = vec(q, "inertia_inv");
    for (int k = 0; k < 9; ++k) v.Iinv[k] = Iinv[k];
    v.U = req(q, "imbalance_gmm") * 1e-6; v.ripple = req(q, "thrust_ripple"); v.blades = req(q, "blades");
    v.arm_hz = req(q, "arm_hz"); v.arm_zeta = num(q, "arm_zeta", 0.02);
    v.mount_hz = req(q, "mount_hz"); v.mount_zeta = req(q, "mount_zeta"); v.aaf_hz = s->imu_spec.aaf_hz;
    v.phase.resize(n); v.rip_off.resize(n);
    for (size_t i = 0; i < n; ++i) v.phase[i] = s->r_vib.uniform(0, 2 * PI);
    for (size_t i = 0; i < n; ++i) v.rip_off[i] = s->r_vib.uniform(0, 2 * PI);
    v.n.assign(n, 0); v.T.assign(n, 0); v.w.assign(n, 2 * PI * 1e-3); v.A.assign(n, 0); v.Ar.assign(n, 0);
    s->rotor_act = ivec(q, "rotor_act");
    s->kn = req(q, "kn");
    s->vib_on = true;
  }
  if (has(c, "baro")) {
    nb::dict q = sub(c, "baro");
    BaroModel& b = s->baro;
    double k = 1.0 / (1.2 * G0);
    b.rate = req(q, "rate_hz"); b.sig = req(q, "noise_pa") * k; b.lsb = num(q, "lsb_pa", 0.016) * k;
    b.drift.init(req(q, "drift_pa") * k, num(q, "drift_tau_s", 300), 1, s->r_baro);
    b.offset = s->r_baro.normal() * req(q, "offset_pa") * k;
    b.cp = num(q, "static_cp", 0.3); b.lat = lat(q);
    s->baro_on = true;
  }
  if (has(c, "mag")) {
    nb::dict q = sub(c, "mag");
    MagModel& g = s->mag;
    double B = req(q, "field_ut"), inc = req(q, "inclination_deg") * DEG, dec = req(q, "declination_deg") * DEG;
    g.b_world[0] = B * std::cos(inc) * std::sin(dec);
    g.b_world[1] = B * std::cos(inc) * std::cos(dec);
    g.b_world[2] = -B * std::sin(inc);
    g.rate = req(q, "rate_hz");
    for (double& h : g.hard) h = s->r_mag.normal() * req(q, "hard_iron_ut");
    for (int i = 0; i < 9; ++i) g.soft[i] = (i % 4 == 0 ? 1.0 : 0.0) + s->r_mag.normal() * req(q, "soft_iron_pct") / 100;
    double dv[3] = {s->r_mag.normal(), s->r_mag.normal(), s->r_mag.normal()};
    double dn = std::sqrt(dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]), kc = num(q, "current_ut_per_a", 0);
    for (int i = 0; i < 3; ++i) g.k_current[i] = kc * dv[i] / dn;
    g.sig = req(q, "noise_ut"); g.lat = lat(q);
    s->mag_on = true;
  }
  if (has(c, "gnss")) {
    nb::dict q = sub(c, "gnss");
    GnssModel& g = s->gnss;
    g.rate = req(q, "rate_hz");
    int n = int(req(q, "sats_open_sky"));
    double s0 = std::sin(num(q, "min_elevation_deg", 10) * DEG);
    for (int i = 0; i < n; ++i) {
      double az = s->r_gnss.uniform(0, 2 * PI), el = std::asin(s->r_gnss.uniform(s0, 1.0));
      g.sats.push_back({std::cos(el) * std::sin(az), std::cos(el) * std::cos(az), std::sin(el)});
    }
    dops(g.sats, g.h_open, g.v_open);
    double sig = req(q, "cep_m") / 1.1774;
    g.bias.init(sig * 0.9, num(q, "bias_tau_s", 60), 3, s->r_gnss);
    g.white = sig * 0.4;
    g.mp.init(req(q, "multipath_m"), num(q, "multipath_tau_s", 8), 3, s->r_gnss);
    g.vel_sig = req(q, "vel_acc_mps");
    g.min_sats = int(num(q, "min_sats", 5));
    g.az_rate = 2 * PI / (12 * 3600);
    g.lat = lat(q);
    s->foliage_loss = num(q, "foliage_loss", 0.5);
    auto ant = vec(c, "antenna");
    for (int k = 0; k < 3 && k < int(ant.size()); ++k) s->antenna[k] = ant[k];
    s->rays = rays;
    s->terrain = (rays && terrain && terrain->valid()) ? terrain : nullptr;
    s->gnss_on = true;
  }
  if (has(c, "depth")) {
    nb::dict q = sub(c, "depth");
    DepthSpec& p = s->depth;
    p.width = int(req(q, "width")); p.height = int(req(q, "height")); p.hfov = req(q, "hfov_deg");
    p.baseline = req(q, "baseline_m"); p.sub_sigma = req(q, "subpixel_sigma_px"); p.sub_bits = int(num(q, "subpixel_bits", 3));
    p.min_m = req(q, "min_m"); p.max_m = req(q, "max_m"); p.lr_check = num(q, "lr_check", 1) != 0;
    p.min_texture = num(q, "min_texture", 0); p.match_window = int(num(q, "match_window", 7));
    s->depth_on = true;
  }
  if (has(c, "rgb")) {
    nb::dict q = sub(c, "rgb");
    RgbSpec& p = s->rgb;
    p.hfov = req(q, "hfov_deg"); p.readout_ms = num(q, "readout_ms", 0); p.rolling = num(q, "rolling", 1) != 0;
    p.exp_lo = req(q, "exp_lo"); p.exp_hi = req(q, "exp_hi"); p.exp_ref = num(q, "exposure_ref_ms", 4);
    p.ae_target = num(q, "ae_target", 0.45); p.max_gain = num(q, "max_gain", 16); p.shot = num(q, "shot_noise_dn", 0.35);
    p.read = num(q, "read_noise_dn", 1.0); p.vignetting = num(q, "vignetting", 0);
    s->rgb_exposure = p.exp_hi / 1000 / 4;
    s->rgb_on = true;
  }
  e.sens = std::move(s);
}

void reset_env(Batch& B, int i, const nb::dict& c) {
  if (i < 0 || i >= int(B.envs.size())) throw std::out_of_range("env index");
  Env& e = *B.envs[i];
  e.chain.clear(); e.body_drag.reset(); e.rotor_drag.reset(); e.power.reset(); e.sens.reset();
  e.sig = Signal{};
  e.mode = int(num(c, "mode", CHAIN));
  e.body = int(num(c, "body", -1));
  uint64_t seed = uint64_t(nb::cast<int64_t>(c["seed"]));
  e.params = ModelParams{};
  if (has(c, "model")) {
    nb::dict p = sub(c, "model");
    ModelParams& mp = e.params;
    mp.mass_frac = num(p, "mass_frac", 1); mp.friction = num(p, "friction", 1);
    auto ifr = vec(p, "inertia_frac"), com = vec(p, "com_offset");
    for (int k = 0; k < 3; ++k) {
      if (ifr.size() == 3) mp.inertia_frac[k] = ifr[k];
      if (com.size() == 3) mp.com_offset[k] = com[k];
    }
    mp.thrust_coef = num(p, "thrust_coef", 1); mp.motor_lag = num(p, "motor_lag", 1); mp.yaw_coef = num(p, "yaw_coef", 1);
    mp.drive_torque = num(p, "drive_torque", 1); mp.steer_kp = num(p, "steer_kp", 1);
    mp.motor_gain = vec(p, "motor_gain"); mp.drive_gain = vec(p, "drive_gain");
    mp.thrust_act = ivec(p, "thrust_act"); mp.drive_act = ivec(p, "drive_act"); mp.steer_act = ivec(p, "steer_act");
  }
  e.apply_model(B.nominal);
  mj_resetData(e.m, e.d);
  mj_forward(e.m, e.d);
  // wind (physics) ------------------------------------------------------------------
  e.wind = Wind{};
  e.wind.terrain = &B.terrain;
  if (has(c, "wind")) {
    nb::dict w = sub(c, "wind");
    Wind& W = e.wind;
    W.on = true;
    W.U = req(w, "U"); W.heading = req(w, "heading"); W.dirx = std::cos(W.heading); W.diry = std::sin(W.heading);
    W.gust = num(w, "gust", 0); W.z_ref = num(w, "z_ref", 10); W.z0 = num(w, "z0", 0.1);
    auto cyl = vec(w, "cyl");
    for (size_t k = 0; k + 3 < cyl.size(); k += 4) W.cyl.push_back({cyl[k], cyl[k + 1], cyl[k + 2], cyl[k + 3]});
    W.kx = vec(w, "kx"); W.ky = vec(w, "ky"); W.phase = vec(w, "phase"); W.omega = vec(w, "omega");
    W.amp = vec(w, "amp"); W.wphase = vec(w, "wphase");
    W.kn.resize(W.kx.size());
    for (size_t k = 0; k < W.kx.size(); ++k) W.kn[k] = std::sqrt(W.kx[k] * W.kx[k] + W.ky[k] * W.ky[k]);
    if (has(w, "turbine_hub")) {
      auto hub = vec(w, "turbine_hub");
      W.turbine = true;
      for (int k = 0; k < 3; ++k) W.hub[k] = hub[k];
      W.tR = req(w, "turbine_R"); W.tCt = req(w, "turbine_Ct"); W.tk = req(w, "turbine_k");
    }
  }
  // controller chain: built against the NOMINAL model -------------------------------------
  if (has(c, "chain")) {
    for (nb::handle h : nb::cast<nb::list>(c["chain"])) {
      nb::dict q = nb::cast<nb::dict>(h);
      std::string type = nb::cast<std::string>(q["type"]);
      if (type == "follow_path") {
        auto st = std::make_unique<FollowPath>();
        st->b.init(B.nominal, int(req(q, "body")));
        st->terrain = B.terrain.valid() ? &B.terrain : nullptr;
        st->agl = num(q, "agl", 1) != 0;
        st->cruise = req(q, "cruise"); st->t_takeoff = req(q, "takeoff_s"); st->yaw_rate = req(q, "yaw_rate");
        auto w = vec(q, "waypoints");
        std::vector<std::array<double, 3>> wps;
        for (size_t k = 0; k + 2 < w.size(); k += 3) wps.push_back({w[k], w[k + 1], w[k + 2]});
        st->init(e.d, wps);
        e.chain.push_back(std::move(st));
      } else if (type == "velocity_command") {
        auto st = std::make_unique<VelocityCommand>();
        st->b.init(B.nominal, int(req(q, "body")));
        const double* R = e.d->xmat + 9 * st->b.bid;
        st->yaw = std::atan2(R[3], R[0]);
        e.chain.push_back(std::move(st));
      } else if (type == "se3") {
        auto st = std::make_unique<Se3Controller>();
        st->b.init(B.nominal, int(req(q, "body")));
        st->mass = B.nominal->body_subtreemass[st->b.bid];
        for (int k = 0; k < 3; ++k) st->J[k] = B.nominal->body_inertia[3 * st->b.bid + k];
        st->kp = req(q, "kp"); st->kd = req(q, "kd"); st->kR = req(q, "kR"); st->kW = req(q, "kW");
        st->max_tilt_deg = req(q, "max_tilt_deg");
        e.chain.push_back(std::move(st));
      } else if (type == "mixer") {
        auto st = std::make_unique<Mixer>();
        st->act = ivec(q, "act");
        st->init(B.nominal, int(req(q, "body")));
        e.chain.push_back(std::move(st));
      } else if (type == "pure_pursuit" || type == "drive_command") {
        auto sv = vec(q, "sxy");
        std::vector<std::array<double, 2>> sxy;
        for (size_t k = 0; k + 1 < sv.size(); k += 2) sxy.push_back({sv[k], sv[k + 1]});
        double kmax = curvature_limit(sxy, req(q, "max_steer_deg"));
        if (type == "drive_command") {
          auto st = std::make_unique<DriveCommand>();
          st->kmax = kmax;
          e.chain.push_back(std::move(st));
          continue;
        }
        auto st = std::make_unique<PurePursuit>();
        st->bid = int(req(q, "body"));
        st->kmax = kmax;
        st->speed = req(q, "speed"); st->lookahead = req(q, "lookahead"); st->window = int(req(q, "window"));
        st->slow_angle = req(q, "slow_angle"); st->slow_factor = req(q, "slow_factor");
        auto wp = vec(q, "waypoints");
        std::vector<std::array<double, 2>> wps;
        for (size_t k = 0; k + 1 < wp.size(); k += 2) wps.push_back({wp[k], wp[k + 1]});
        st->init(e.d, wps, req(q, "spacing"));
        e.chain.push_back(std::move(st));
      } else if (type == "ackermann") {
        auto st = std::make_unique<Ackermann>();
        st->steer = ivec(q, "steer");
        auto sv = vec(q, "sxy"), wv = vec(q, "wxy");
        for (size_t k = 0; k + 1 < sv.size(); k += 2) st->sxy.push_back({sv[k], sv[k + 1]});
        for (size_t k = 0; k + 1 < wv.size(); k += 2) st->wxy.push_back({wv[k], wv[k + 1]});
        st->rw = req(q, "rw");
        e.chain.push_back(std::move(st));
      } else if (type == "wheel_speed_pi") {
        auto st = std::make_unique<WheelSpeedPI>();
        st->drive = ivec(q, "drive"); st->jdof = ivec(q, "jdof");
        st->w0 = req(q, "w0"); st->kp = req(q, "kp"); st->ki = req(q, "ki"); st->limit = req(q, "limit");
        st->integ.assign(st->drive.size(), 0.0);
        e.chain.push_back(std::move(st));
      } else {
        throw std::runtime_error("razzy_core: unknown control stage '" + type + "'");
      }
    }
  }
  // aero effects + power train: get the drawn vehicle ------------------------------------
  if (has(c, "body_drag")) {
    nb::dict q = sub(c, "body_drag");
    e.body_drag = std::make_unique<BodyDrag>();
    e.body_drag->b.init(e.m, int(req(q, "body")));
    e.body_drag->CdA = req(q, "CdA");
  }
  if (has(c, "rotor_drag")) {
    nb::dict q = sub(c, "rotor_drag");
    e.rotor_drag = std::make_unique<RotorDrag>();
    e.rotor_drag->b.init(e.m, int(req(q, "body")));
    e.rotor_drag->act = ivec(q, "act");
    e.rotor_drag->k = req(q, "k");
  }
  if (has(c, "power")) {
    nb::dict q = sub(c, "power");
    auto& P = e.power = std::make_unique<PowerSystem>();
    P->pack = parse_pack(sub(q, "pack"));
    for (nb::handle h : nb::cast<nb::list>(q["loads"])) {
      nb::dict l = nb::cast<nb::dict>(h);
      Load ld;
      ld.act = ivec(l, "act");
      if (nb::cast<std::string>(l["type"]) == "rotor") {
        ld.kind = Load::ROTOR;
        ld.rp = {req(l, "kv"), req(l, "r_ohm"), req(l, "i0"), req(l, "D"), req(l, "ct"), req(l, "cp"), req(l, "rho")};
        ld.esc_eff = req(l, "esc_eff"); ld.t_max_ref = req(l, "t_max_ref");
        ld.v_ref = P->pack.ocv(0.5);
      } else {
        ld.kind = Load::GEARMOTOR;
        ld.tau_s = vec(l, "tau_s"); ld.i0 = req(l, "i0"); ld.i_s = req(l, "i_s");
        ld.v_nom = P->pack.v_nom;
      }
      P->loads.push_back(ld);
    }
    P->constant = vec(q, "constant");
    P->soc = 1; P->v = P->pack.ocv(1.0);
  }
  if (has(c, "sensors")) parse_sensors(e, sub(c, "sensors"), seed, B.rays.get(), &B.terrain);
  // actions --------------------------------------------------------------------------
  e.compute = ComputeLatency{};
  e.r_compute = Rng(seed, 7);
  if (has(c, "compute")) {
    nb::dict q = sub(c, "compute");
    e.compute.on = true;
    e.compute.rate = num(q, "rate_hz", 15);
    e.compute.lat = lat(q);
  }
  e.pending.clear();
  e.held_capture = -1e300;
  e.held.assign(e.mode == CTRL ? e.m->nu : 4, 0.0);
  e.imu_delivered.clear();
  B.write_outputs(i);
}

template <typename T>
nb::ndarray<nb::numpy, T> view(std::vector<T>& v, size_t rows, size_t cols, nb::handle owner) {
  size_t shape[2] = {rows, cols};
  return nb::ndarray<nb::numpy, T>(v.data(), cols == 0 ? 1 : 2, shape, owner);
}

template <size_t N>
nb::ndarray<nb::numpy, double> to_array(const std::vector<std::array<double, N>>& rows) {
  auto* buf = new double[std::max<size_t>(rows.size(), 1) * N];
  for (size_t i = 0; i < rows.size(); ++i) std::copy(rows[i].begin(), rows[i].end(), buf + i * N);
  nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
  size_t shape[2] = {rows.size(), N};
  return nb::ndarray<nb::numpy, double>(buf, 2, shape, owner);
}

Sensors& sensors_of(Batch& B, int i) {
  if (i < 0 || i >= int(B.envs.size()) || !B.envs[i]->sens) throw std::runtime_error("env has no sensors");
  return *B.envs[i]->sens;
}

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "razzy-rabbit C++ simulation core";
  m.def("mj_version", []() { return mj_version(); });

  nb::class_<Batch>(m, "Batch")
      .def(nb::init<const std::string&, int, int>(), nb::arg("physics_mjb"), nb::arg("n_envs"), nb::arg("threads"))
      .def_prop_ro("n_envs", [](Batch& b) { return int(b.envs.size()); })
      .def_prop_ro("threads", [](Batch& b) { return b.pool->size(); })
      .def_prop_ro("nq", [](Batch& b) { return b.nq; })
      .def_prop_ro("nv", [](Batch& b) { return b.nv; })
      .def_prop_ro("nu", [](Batch& b) { return b.nu; })
      .def_prop_ro("timestep", [](Batch& b) { return b.nominal->opt.timestep; })
      .def("set_terrain",
           [](Batch& b, nb::ndarray<double, nb::ndim<2>, nb::c_contig> z, double hx, double hy) {
             b.terrain.nr = int(z.shape(0)); b.terrain.nc = int(z.shape(1));
             b.terrain.z.assign(z.data(), z.data() + z.size());
             b.terrain.hx = hx; b.terrain.hy = hy;
             b.terrain.zmax = *std::max_element(b.terrain.z.begin(), b.terrain.z.end());
           })
      .def("load_rays",
           [](Batch& b, const std::string& xml, nb::ndarray<double, nb::ndim<1>, nb::c_contig> opacity) {
             char err[1000] = {0};
             auto rs = std::make_shared<RayScene>();
             rs->m = mj_loadXML(xml.c_str(), nullptr, err, sizeof(err));
             if (!rs->m) throw std::runtime_error(std::string("ray scene: ") + err);
             if (int(opacity.shape(0)) != rs->m->ngeom) throw std::runtime_error("opacity size != ngeom");
             // terrain is tested by a height-grid march (Terrain::blocks): far cheaper than
             // MuJoCo's hfield ray test, which dominated the GNSS cost
             if (b.terrain.valid())
               for (int g = 0; g < rs->m->ngeom; ++g)
                 if (rs->m->geom_type[g] == mjGEOM_HFIELD) rs->m->geom_group[g] = 5;
             rs->d = mj_makeData(rs->m);
             mj_forward(rs->m, rs->d);
             rs->opacity.assign(opacity.data(), opacity.data() + opacity.shape(0));
             b.rays = rs;
           })
      .def("reset", &reset_env, nb::arg("env"), nb::arg("config"))
      .def("step", [](Batch& b, int n) { nb::gil_scoped_release r; b.step(n); }, nb::arg("n") = 1,
           "advance every env by n physics steps (parallel, GIL released)")
      .def("act",
           [](Batch& b, nb::ndarray<double, nb::ndim<2>, nb::c_contig> a) {
             if (int(a.shape(0)) != int(b.envs.size())) throw std::runtime_error("act: need one row per env");
             for (size_t i = 0; i < a.shape(0); ++i) {
               std::vector<double> row(a.data() + i * a.shape(1), a.data() + (i + 1) * a.shape(1));
               b.envs[i]->submit(row);
             }
           },
           "submit actions (one row per env); delayed by the policy compute model if configured")
      // outputs: views into C++ buffers, refreshed by every step()
      .def_prop_ro("time", [](Batch& b) { return view(b.time, b.time.size(), 0, nb::find(&b)); }, nb::rv_policy::reference_internal)
      .def_prop_ro("qpos", [](Batch& b) { return view(b.qpos, b.envs.size(), b.nq, nb::find(&b)); }, nb::rv_policy::reference_internal)
      .def_prop_ro("qvel", [](Batch& b) { return view(b.qvel, b.envs.size(), b.nv, nb::find(&b)); }, nb::rv_policy::reference_internal)
      .def_prop_ro("ctrl", [](Batch& b) { return view(b.ctrl, b.envs.size(), b.nu, nb::find(&b)); }, nb::rv_policy::reference_internal)
      .def_prop_ro("err", [](Batch& b) { return view(b.err, b.err.size(), 0, nb::find(&b)); }, nb::rv_policy::reference_internal)
      .def_prop_ro("tilt_deg", [](Batch& b) { return view(b.tilt, b.tilt.size(), 0, nb::find(&b)); }, nb::rv_policy::reference_internal)
      .def_prop_ro("laps", [](Batch& b) { return view(b.laps, b.laps.size(), 0, nb::find(&b)); }, nb::rv_policy::reference_internal)
      .def_prop_ro("plant", [](Batch& b) { return view(b.plant, b.envs.size(), 5, nb::find(&b)); }, nb::rv_policy::reference_internal,
                   "soc, volts, amps, watts, Wh used")
      .def_prop_ro("imu", [](Batch& b) { return view(b.imu, b.envs.size(), 7, nb::find(&b)); }, nb::rv_policy::reference_internal,
                   "latest delivered: t_capture, gyro xyz, accel xyz")
      .def_prop_ro("baro", [](Batch& b) { return view(b.baro, b.envs.size(), 2, nb::find(&b)); }, nb::rv_policy::reference_internal)
      .def_prop_ro("mag", [](Batch& b) { return view(b.mag, b.envs.size(), 4, nb::find(&b)); }, nb::rv_policy::reference_internal)
      .def_prop_ro("gnss", [](Batch& b) { return view(b.gnss, b.envs.size(), 12, nb::find(&b)); }, nb::rv_policy::reference_internal,
                   "t_capture, fix, n_sats, hdop, sky_blocked, x y z, vx vy vz, 0")
      .def_prop_ro("vib", [](Batch& b) { return view(b.vib, b.vib.size(), 0, nb::find(&b)); }, nb::rv_policy::reference_internal)
      .def("imu_drain",
           [](Batch& b, int i) {
             auto& e = *b.envs.at(i);
             auto out = to_array(e.imu_delivered);
             e.imu_delivered.clear();
             return out;
           },
           "all IMU samples delivered since the last drain: (k, 7) t g a")
      .def("log",
           [](Batch& b, int i, const std::string& name) -> nb::object {
             Sensors& s = sensors_of(b, i);
             if (name == "imu") return nb::cast(to_array(s.log_imu));
             if (name == "baro") return nb::cast(to_array(s.log_baro));
             if (name == "mag") return nb::cast(to_array(s.log_mag));
             if (name == "gnss") return nb::cast(to_array(s.log_gnss));
             throw std::runtime_error("log: imu|baro|mag|gnss");
           })
      .def("sky_survival",
           [](Batch& b, int i, std::vector<double> p, nb::ndarray<double, nb::ndim<2>, nb::c_contig> dirs) {
             Sensors& s = sensors_of(b, i);
             std::vector<double> out(dirs.shape(0));
             for (size_t k = 0; k < dirs.shape(0); ++k) out[k] = s.sky_survival(p.data(), dirs.data() + 3 * k);
             return out;
           },
           "GNSS signal survival from point p towards each unit direction (testing/inspection)")
      .def("imu_saturated", [](Batch& b, int i) { return sensors_of(b, i).imu.saturated; })
      .def("rotor_speeds",
           [](Batch& b, int i) { return sensors_of(b, i).vib.n; })
      .def("depth_apply",
           [](Batch& b, int i, nb::ndarray<float, nb::ndim<2>, nb::c_contig> z, nb::object gray_obj) {
             Sensors& s = sensors_of(b, i);
             int h = int(z.shape(0)), w = int(z.shape(1));
             const float* gray = nullptr;
             nb::ndarray<float, nb::ndim<2>, nb::c_contig> g;
             if (!gray_obj.is_none()) { g = nb::cast<nb::ndarray<float, nb::ndim<2>, nb::c_contig>>(gray_obj); gray = g.data(); }
             auto* out = new float[h * w];
             depth_apply(s.depth, z.data(), gray, h, w, out, s.r_depth);
             nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<float*>(p); });
             size_t shape[2] = {size_t(h), size_t(w)};
             return nb::ndarray<nb::numpy, float>(out, 2, shape, owner);
           },
           nb::arg("env"), nb::arg("depth"), nb::arg("gray") = nb::none())
      .def("rgb_apply",
           [](Batch& b, int i, nb::ndarray<uint8_t, nb::ndim<3>, nb::c_contig> img, std::vector<double> omega) {
             Sensors& s = sensors_of(b, i);
             int h = int(img.shape(0)), w = int(img.shape(1));
             auto* out = new uint8_t[h * w * 3];
             rgb_apply(s.rgb, img.data(), h, w, omega.data(), out, s.rgb_exposure, s.rgb_gain, s.r_rgb);
             nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });
             size_t shape[3] = {size_t(h), size_t(w), 3};
             return nb::ndarray<nb::numpy, uint8_t>(out, 3, shape, owner);
           });
}
