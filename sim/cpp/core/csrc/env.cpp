// Per-step order mirrors the Python reference (sim/runtime/sim.py):
//   controller chain -> body drag -> power train -> rotor drag -> mj_step -> sensors (state at t0)
#include "env.hpp"

#include <limits>
#include <stdexcept>

namespace rz {

Env::~Env() {
  if (d) mj_deleteData(d);
  if (m) mj_deleteModel(m);
}

void Env::apply_model(const mjModel* nom) {
  mj_copyModel(m, nom);                       // start from the exact nominal model
  const ModelParams& p = params;
  if (body >= 0) {
    m->body_mass[body] *= p.mass_frac;
    for (int k = 0; k < 3; ++k) {
      m->body_inertia[3 * body + k] *= p.mass_frac * p.inertia_frac[k];
      m->body_ipos[3 * body + k] += p.com_offset[k];
    }
  }
  for (int g = 0; g < m->ngeom; ++g) m->geom_friction[3 * g] *= p.friction;
  for (size_t i = 0; i < p.thrust_act.size(); ++i) {
    int a = p.thrust_act[i];
    m->actuator_gainprm[mjNGAIN * a] *= p.thrust_coef * (p.motor_gain.empty() ? 1.0 : p.motor_gain[i]);
    m->actuator_dynprm[mjNDYN * a] *= p.motor_lag;
    m->actuator_gear[6 * a + 5] *= p.yaw_coef;
  }
  for (size_t i = 0; i < p.drive_act.size(); ++i) {
    int a = p.drive_act[i];
    double tq = p.drive_torque * (p.drive_gain.empty() ? 1.0 : p.drive_gain[i]);
    m->actuator_gainprm[mjNGAIN * a] *= tq;
    m->actuator_biasprm[mjNBIAS * a + 2] *= tq;
  }
  for (int a : p.steer_act) {
    m->actuator_gainprm[mjNGAIN * a] *= p.steer_kp;
    m->actuator_biasprm[mjNBIAS * a + 1] *= p.steer_kp;
  }
  mj_setConst(m, d);
}

void Env::submit(const std::vector<double>& a) {
  if (compute.on) pending.push_back({d->time + compute.lat.sample(r_compute), d->time, a});
  else { held = a; held_capture = d->time; }
}

void Env::step1() {
  const double t0 = d->time, dt = m->opt.timestep;
  if (!pending.empty()) {          // deliver actions whose compute finished
    // newest *observation* wins (an older action that arrives late never overrides it)
    for (auto it = pending.begin(); it != pending.end();) {
      if (it->ready <= t0) {
        if (it->capture >= held_capture) { held_capture = it->capture; held = it->a; }
        it = pending.erase(it);
      } else ++it;
    }
  }
  if (mode == CTRL) {
    if (held.size() == size_t(m->nu))
      for (int i = 0; i < m->nu; ++i) d->ctrl[i] = held[i];
  } else {
    StageCtx c{m, d, &held};
    for (auto& st : chain) st->update(c, sig, t0, dt);
  }
  if (body_drag) body_drag->update(d, wind, t0);
  if (power) power->update(m, d, dt);
  if (rotor_drag) rotor_drag->update(d, wind, t0);
  mj_step(m, d);
  if (sens) sens->update(m, d, &wind, power ? power->i : 0.0, t0, dt);
}

double Env::err() const {
  for (const auto& st : chain)
    if (st->has_err()) return st->err();
  return 0.0;
}

int Env::laps() const {
  for (const auto& st : chain)
    if (st->laps()) return st->laps();
  return 0;
}

void Env::sync_outputs() {
  if (!sens) return;
  double t = d->time;
  std::vector<DelayLine<7>::Item> got;
  sens->l_imu.drain(t, &got);
  for (const auto& it : got) imu_delivered.push_back(it.v);
  if (imu_delivered.size() > 20000) imu_delivered.erase(imu_delivered.begin(), imu_delivered.end() - 20000);
  sens->l_baro.drain(t, nullptr);
  sens->l_mag.drain(t, nullptr);
  sens->l_gnss.drain(t, nullptr);
}

// --------------------------------------------------------------------------- pool ---
Pool::Pool(int n) {
  for (int i = 0; i < n - 1; ++i) th_.emplace_back([this] { worker(); });
}

Pool::~Pool() {
  { std::lock_guard<std::mutex> lk(mu_); stop_ = true; ++gen_; }
  cv_.notify_all();
  for (auto& t : th_) t.join();
}

void Pool::worker() {
  uint64_t seen = 0;
  for (;;) {
    const std::function<void(int)>* fn;
    int n;
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [&] { return gen_ != seen; });
      seen = gen_;
      if (stop_) return;
      fn = fn_;
      n = n_jobs_;
      ++busy_;
    }
    for (int j; (j = next_.fetch_add(1)) < n;) (*fn)(j);
    { std::lock_guard<std::mutex> lk(mu_); if (--busy_ == 0) done_.notify_all(); }
  }
}

void Pool::run(int n_jobs, const std::function<void(int)>& fn) {
  if (th_.empty() || n_jobs == 1) { for (int j = 0; j < n_jobs; ++j) fn(j); return; }
  {
    std::lock_guard<std::mutex> lk(mu_);
    fn_ = &fn;
    n_jobs_ = n_jobs;
    next_ = 0;
    ++gen_;
  }
  cv_.notify_all();
  for (int j; (j = next_.fetch_add(1)) < n_jobs;) fn(j);
  std::unique_lock<std::mutex> lk(mu_);
  done_.wait(lk, [&] { return busy_ == 0 && next_ >= n_jobs; });
}

// -------------------------------------------------------------------------- batch ---
Batch::Batch(const std::string& path, int n, int threads) {
  nominal = mj_loadModel(path.c_str(), nullptr);
  if (!nominal) throw std::runtime_error("cannot load physics model " + path);
  nq = nominal->nq; nv = nominal->nv; nu = nominal->nu;
  for (int i = 0; i < n; ++i) {
    auto e = std::make_unique<Env>();
    e->m = mj_copyModel(nullptr, nominal);
    e->d = mj_makeData(e->m);
    mj_forward(e->m, e->d);
    envs.push_back(std::move(e));
  }
  pool = std::make_unique<Pool>(std::max(1, threads));
  const double nan = std::numeric_limits<double>::quiet_NaN();
  time.assign(n, 0); qpos.assign(n * nq, 0); qvel.assign(n * nv, 0); ctrl.assign(n * nu, 0);
  err.assign(n, 0); tilt.assign(n, 0); plant.assign(n * 5, nan); imu.assign(n * 7, nan); baro.assign(n * 2, nan);
  mag.assign(n * 4, nan); gnss.assign(n * 12, nan); vib.assign(n, 0); laps.assign(n, 0);
  for (int i = 0; i < n; ++i) write_outputs(i);
}

Batch::~Batch() {
  pool.reset();
  envs.clear();
  if (nominal) mj_deleteModel(nominal);
}

void Batch::step(int nsub) {
  pool->run(int(envs.size()), [&](int i) {
    Env& e = *envs[i];
    for (int k = 0; k < nsub; ++k) e.step1();
    e.sync_outputs();
    write_outputs(i);
  });
}

void Batch::write_outputs(int i) {
  Env& e = *envs[i];
  const mjData* d = e.d;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  time[i] = d->time;
  std::copy(d->qpos, d->qpos + nq, qpos.begin() + i * nq);
  std::copy(d->qvel, d->qvel + nv, qvel.begin() + i * nv);
  std::copy(d->ctrl, d->ctrl + nu, ctrl.begin() + i * nu);
  err[i] = e.err();
  laps[i] = e.laps();
  if (e.body >= 0) tilt[i] = std::acos(clampd(d->xmat[9 * e.body + 8], -1.0, 1.0)) / DEG;
  double* pl = &plant[5 * i];
  if (e.power) { pl[0] = e.power->soc; pl[1] = e.power->v; pl[2] = e.power->i; pl[3] = e.power->power; pl[4] = e.power->energy_wh; }
  if (e.sens) {
    const Sensors& s = *e.sens;
    auto put = [&](auto& line, double* dst, int n) {
      if (line.has_latest) std::copy(line.latest.begin(), line.latest.begin() + n, dst);
      else std::fill(dst, dst + n, nan);
    };
    put(s.l_imu, &imu[7 * i], 7);
    put(s.l_baro, &baro[2 * i], 2);
    put(s.l_mag, &mag[4 * i], 4);
    put(s.l_gnss, &gnss[12 * i], 12);
    if (s.vib_on && s.vib.latched) {
      double g[3], a[3];
      s.vib.at(d->time, g, a);
      vib[i] = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    }
  }
}

}  // namespace rz
