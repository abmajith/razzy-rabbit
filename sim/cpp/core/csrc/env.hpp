// One environment (own physics-model copy + data + controller/plant/sensors) and the
// batch that steps many of them on a thread pool.
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

#include "core.hpp"

namespace rz {

struct ModelParams {   // uncertainty draw applied to the env's model copy (sim/uncertainty.py)
  double mass_frac = 1, inertia_frac[3] = {1, 1, 1}, com_offset[3] = {0, 0, 0}, friction = 1;
  double thrust_coef = 1, motor_lag = 1, yaw_coef = 1, drive_torque = 1, steer_kp = 1;
  std::vector<double> motor_gain, drive_gain;
  std::vector<int> thrust_act, drive_act, steer_act;     // which actuators the factors apply to
};

enum Mode { CHAIN = 0, CTRL = 2 };   // CHAIN: the controller chain (incl. command stages); CTRL: raw actuators

struct Env {
  mjModel* m = nullptr;
  mjData* d = nullptr;
  int mode = CHAIN, body = -1;
  ModelParams params;
  std::vector<std::unique_ptr<Stage>> chain;
  Signal sig;
  Wind wind;
  std::unique_ptr<BodyDrag> body_drag;
  std::unique_ptr<RotorDrag> rotor_drag;
  std::unique_ptr<PowerSystem> power;
  std::unique_ptr<Sensors> sens;
  // actions (command stages / raw ctrl), optionally delayed by policy compute time
  ComputeLatency compute;
  Rng r_compute;
  struct Pending { double ready, capture; std::vector<double> a; };
  std::deque<Pending> pending;
  std::vector<double> held;
  double held_capture = -1e300;
  std::vector<std::array<double, 7>> imu_delivered;   // drained by Python (EKF-style consumers)
  ~Env();
  void apply_model(const mjModel* nominal);
  void step1();
  void submit(const std::vector<double>& a);
  void sync_outputs();
  double err() const;
  int laps() const;
};

class Pool {
 public:
  explicit Pool(int n);
  ~Pool();
  void run(int n_jobs, const std::function<void(int)>& fn);
  int size() const { return int(th_.size()) + 1; }
 private:
  void worker();
  std::vector<std::thread> th_;
  std::mutex mu_;
  std::condition_variable cv_, done_;
  const std::function<void(int)>* fn_ = nullptr;
  std::atomic<int> next_{0};
  int n_jobs_ = 0, busy_ = 0;
  uint64_t gen_ = 0;
  bool stop_ = false;
};

struct Batch {
  mjModel* nominal = nullptr;
  std::shared_ptr<RayScene> rays;
  Terrain terrain;
  std::vector<std::unique_ptr<Env>> envs;
  std::unique_ptr<Pool> pool;
  int nq = 0, nv = 0, nu = 0;
  // output buffers, refreshed after every step() (numpy views in Python)
  std::vector<double> time, qpos, qvel, ctrl, err, tilt, plant, imu, baro, mag, gnss, vib, laps;
  Batch(const std::string& phys_mjb, int n_envs, int threads);
  ~Batch();
  void step(int nsub);
  void write_outputs(int i);
};

}  // namespace rz
