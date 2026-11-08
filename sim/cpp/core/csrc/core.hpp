// razzy_core: batched MuJoCo environments with the designed vehicle's controller, power
// train ("plant"), sensors and latency, all in C++. Python only configures (at reset) and
// reads/writes flat arrays.
//
// Every model here is a port of the readable Python reference:
//   wind.cpp    <- sim/wind.py            control.cpp <- sim/control.py
//   plant.cpp   <- sim/design/plant.py    sensors.cpp <- sim/sensing/models.py + suite.py
//   camera.cpp  <- sim/sensing/models.py (StereoDepth, RgbCamera)
//   env.cpp     <- sim/runtime.py (step order) + sim/uncertainty.py (apply_model)
// Physics/controller/plant follow the Python operation order, so trajectories match the
// reference to rounding; sensor noise uses a counter-based Philox RNG (same statistics,
// different samples than numpy).
#pragma once

#include <mujoco/mujoco.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace rz {

constexpr double PI = 3.14159265358979323846;
constexpr double G0 = 9.80665;
constexpr double DEG = PI / 180.0;

inline double clampd(double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); }
inline double floormod(double a, double b) { return a - b * std::floor(a / b); }  // Python %

// ------------------------------------------------------------------------------ RNG ---
// Philox4x32-10 (Salmon et al. 2011): counter-based, so (seed, stream, n) -> same number
// on any thread/machine, and the same code ports to SYCL/Vulkan kernels.
class Rng {
 public:
  Rng() = default;
  Rng(uint64_t seed, uint64_t stream);
  uint32_t next_u32();
  double uniform();                       // [0, 1)
  double uniform(double lo, double hi) { return lo + (hi - lo) * uniform(); }
  double normal();                        // N(0, 1), Box-Muller
 private:
  uint32_t key_[2] = {0, 0};
  uint64_t ctr_ = 0;
  uint32_t buf_[4] = {0, 0, 0, 0};
  int pos_ = 4;
  bool has_spare_ = false;
  double spare_ = 0.0;
};

// -------------------------------------------------------------------------- terrain ---
struct Terrain {
  std::vector<double> z;  // row-major (nr, nc), row 0 at -y
  int nr = 0, nc = 0;
  double hx = 0, hy = 0, zmax = 0;
  bool valid() const { return nr > 1 && nc > 1; }
  double height(double x, double y) const;
  // does the ray p + s u (s > 0) pass below the terrain before leaving it?
  bool blocks(const double p[3], const double u[3]) const;
};

// ----------------------------------------------------------------------------- wind ---
struct Wind {
  bool on = false;
  double U = 0, heading = 0, dirx = 1, diry = 0, gust = 0, z_ref = 10, z0 = 0.1;
  std::vector<std::array<double, 4>> cyl;          // x, y, R, top_z
  bool turbine = false;
  double hub[3] = {0, 0, 0}, tR = 0, tCt = 0, tk = 0;
  std::vector<double> kx, ky, kn, phase, omega, amp, wphase;
  const Terrain* terrain = nullptr;
  void velocity(const double p[3], double t, double out[3]) const;
  void drag(const double p[3], const double v[3], double CdA, double t, double f[3]) const;
 private:
  double ground(double x, double y) const { return terrain && terrain->valid() ? terrain->height(x, y) : 0.0; }
};

// --------------------------------------------------------------------- controllers ---
struct QuadController {
  int bid = 0, qa = 0, va = 0;
  std::vector<int> act;
  double mass = 0, J[3] = {0, 0, 0}, Ainv[16];
  double kp = 5.0, kd = 4.0, kR = 140.0, kW = 22.0;
  void init(const mjModel* nominal, int body, const std::vector<int>& acts);
  // fmax: live per-actuator max thrust (from the env model, lowered by battery sag)
  void command(const mjModel* m, mjData* d, const double p_d[3], const double v_d[3], double yaw_d) const;
  const double* pos(const mjData* d) const { return d->qpos + qa; }
  const double* vel(const mjData* d) const { return d->qvel + va; }
};

struct QuadMission {
  QuadController ctl;
  std::vector<std::array<double, 3>> P;
  std::vector<double> s;
  double s_now = 0, yaw = 0, t_takeoff = 2.5, ground0 = 0, cruise = 3.0, err = 0;
  int dirn = 1;
  bool agl = true;
  const Terrain* terrain = nullptr;
  void init(const mjData* d, const std::vector<std::array<double, 3>>& wps, bool agl_, double cruise_);
  void target(double out_p[3], double out_tan[3]) const;
  void update(const mjModel* m, mjData* d, double t, double dt);
};

struct RoverAckermann {
  int bid = 0;
  std::vector<int> drive, jdof, steer;
  std::vector<std::array<double, 2>> wxy, sxy, P;
  double rw = 0.1, w0 = 1, v = 0.5, kmax = 1, err = 0;
  int idx = 0, laps = 0;
  std::vector<double> integ;
  void init(const mjData* d, const std::vector<std::array<double, 2>>& waypoints);
  void update(const mjModel* m, mjData* d, double t, double dt);
  void apply(mjData* d, double v_cmd, double kappa, double dt);   // shared low level
};

// --------------------------------------------------------------------------- plants ---
struct Pack {
  std::vector<double> soc_tab, ocv_tab;
  int cells = 1;
  double cap_ah = 1, r = 0.01, v_nom = 3.7;
  double ocv(double soc) const;
  void terminal(double power, double soc, double& v, double& i) const;
};

struct RotorParams { double kv, r_ohm, i0, D, ct, cp, rho; };
// operating point of one rotor at thrust T: rev/s, electrical power
void rotor_point(const RotorParams& p, double T, double& n, double& Pe);

struct DronePlant {
  int bid = 0, dof = 0;
  std::vector<int> act;
  Pack pack;
  RotorParams rp{};
  double esc_eff = 0.95, avionics_w = 0, t_max_ref = 1, rotor_drag = 0, v_ref = 1;
  double soc = 1, v = 0, i = 0, power = 0, energy_wh = 0;
  void update(mjModel* m, mjData* d, const Wind* wind, double t, double dt);
};

struct RoverPlant {
  std::vector<int> act;
  Pack pack;
  std::vector<double> tau_s;
  double i0 = 0, i_s = 1, avionics_w = 0, servo_w = 0;
  int n_servo = 0;
  double soc = 1, v = 0, i = 0, power = 0, energy_wh = 0;
  void update(mjModel* m, mjData* d, double t, double dt);
};

// -------------------------------------------------------------------------- sensors ---
struct GaussMarkov {
  double sigma = 0, tau = 1;
  std::vector<double> x;
  void init(double sigma_, double tau_, int n, Rng& rng);
  void step(double dt, Rng& rng);
};

struct LatencySpec {
  double lo = 0, hi = 0;  // seconds
  double sample(Rng& rng) const { return hi > lo ? rng.uniform(lo, hi) : lo; }
};

// measurement queue: visible `latency` after capture; jittered samples may overtake
template <int N>
struct DelayLine {
  struct Item { double deliver, capture; std::array<double, N> v; };
  std::deque<Item> q;
  bool has_latest = false;
  double latest_t = 0;
  std::array<double, N> latest{};
  void push(double tc, const std::array<double, N>& v, double lat) { q.push_back({tc + lat, tc, v}); }
  // moves delivered items (sorted by capture time) into out (if given); updates latest
  void drain(double t, std::vector<Item>* out);
};

struct ImuModel {
  double rate = 1000, g_sig = 0, a_sig = 0, g_fs = 0, a_fs = 0, g_lsb = 0, a_lsb = 0;
  double g_bias0[3] = {0, 0, 0}, a_bias0[3] = {0, 0, 0}, Mg[9], Ma[9];
  GaussMarkov g_walk, a_walk;
  long saturated = 0;
  void init(const struct ImuSpec& s, Rng& rng);
  void sample(const double gt[3], const double at[3], const double gv[3], const double av[3],
              double g[3], double a[3], Rng& rng);
};

struct ImuSpec {
  double rate_hz, gyro_noise_dps_rthz, gyro_bias_dps, gyro_bias_instability_dph, accel_noise_ug_rthz,
      accel_bias_mg, accel_bias_instability_ug, bias_tau_s, scale_error_pct, misalign_deg, gyro_range_dps,
      accel_range_g, bits, aaf_hz;
  LatencySpec lat;
};

struct RotorVibration {
  std::vector<std::array<double, 3>> r;
  std::vector<double> spin, phase, rip_off, n, T, w, A, Ar;
  double mass = 1, Iinv[9], U = 0, ripple = 0, blades = 2, arm_hz = 1e9, arm_zeta = 0.02, mount_hz = 1e9,
         mount_zeta = 0.15, aaf_hz = 1e9, t0 = 0;
  bool latched = false;
  double gain(double f) const;
  void set_rotors(double t, const double* n_revs, const double* thrust);
  void at(double t, double gyro[3], double acc[3]) const;
};

struct BaroModel {
  double rate = 50, sig = 0, lsb = 0, offset = 0, cp = 0.3;
  GaussMarkov drift;
  LatencySpec lat;
  double sample(double h_true, double airspeed, Rng& rng);
};

struct MagModel {
  double rate = 50, b_world[3], hard[3], soft[9], k_current[3], sig = 0;
  LatencySpec lat;
  void sample(const double R[9], double current, double out[3], Rng& rng) const;
};

struct GnssModel {
  double rate = 10, h_open = 1, v_open = 1, white = 0, vel_sig = 0, az_rate = 0;
  int min_sats = 5;
  std::vector<std::array<double, 3>> sats;
  GaussMarkov bias, mp;
  LatencySpec lat;
  void sats_at(double t, std::vector<std::array<double, 3>>& out) const;
};
void dops(const std::vector<std::array<double, 3>>& u, double& hdop, double& vdop);

struct ComputeLatency {
  bool on = false;
  double rate = 15;
  LatencySpec lat;
};

// static, shared data for GNSS sky-view ray casts (full visual model at rest)
struct RayScene {
  mjModel* m = nullptr;
  mjData* d = nullptr;
  std::vector<double> opacity;  // per geom of the full model: 1 solid, 0 see-through, <0 foliage
  ~RayScene();
};

struct DepthSpec {
  int width = 640, height = 480;
  double hfov = 73, baseline = 0.075, sub_sigma = 0.1, min_m = 0.35, max_m = 10, min_texture = 0;
  int sub_bits = 3, match_window = 7;
  bool lr_check = true;
};
struct RgbSpec {
  double hfov = 69, readout_ms = 28, exp_lo = 0.1, exp_hi = 33, exp_ref = 4, ae_target = 0.45, max_gain = 16,
         shot = 0.35, read = 1.2, vignetting = 0.3;
  bool rolling = true;
};
struct Sensors {
  bool imu_on = false, baro_on = false, mag_on = false, gnss_on = false, vib_on = false, record = false;
  int gyro_adr = 0, acc_adr = 0, site = 0, bid = 0, dof = -1;
  ImuModel imu; ImuSpec imu_spec{};
  RotorVibration vib; std::vector<int> rotor_act; double kn = 0;
  BaroModel baro; MagModel mag; GnssModel gnss;
  double antenna[3] = {0, 0, 0.08}, foliage_loss = 0.5;
  int max_layers = 4;
  Rng r_imu, r_baro, r_mag, r_gnss, r_vib, r_lat, r_depth, r_rgb, r_compute;
  double next_imu = 0, next_baro = 0, next_mag = 0, next_gnss = 0;
  DelayLine<7> l_imu;     // t g3 a3   (t duplicated for convenience)
  DelayLine<2> l_baro;    // t h
  DelayLine<4> l_mag;     // t b3
  DelayLine<12> l_gnss;   // t fix n hdop blocked x y z vx vy vz
  std::vector<std::array<double, 13>> log_imu;
  std::vector<std::array<double, 3>> log_baro;
  std::vector<std::array<double, 7>> log_mag;
  std::vector<std::array<double, 11>> log_gnss;
  const RayScene* rays = nullptr;
  const Terrain* terrain = nullptr;   // terrain occlusion by height-grid march (hfield hidden from mj_ray)
  DepthSpec depth; RgbSpec rgb;
  bool depth_on = false, rgb_on = false;
  double rgb_exposure = 0.008, rgb_gain = 1.0;
  // per-step work
  void update(const mjModel* m, const mjData* d, const Wind* wind, double current, double t0, double dt);
  double sky_survival(const double p[3], const double u[3]) const;
};

// -------------------------------------------------------------------------- cameras ---
void depth_apply(const DepthSpec& s, const float* z, const float* gray, int h, int w, float* out, Rng& rng);
void rgb_apply(const RgbSpec& s, const uint8_t* img, int h, int w, const double omega[3], uint8_t* out,
               double& exposure, double& gain, Rng& rng);

}  // namespace rz
