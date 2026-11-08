// Batched headless rollout benchmark: one shared mjModel, N independent mjData,
// a fixed pool of worker threads. This is the skeleton of a C++ RL environment
// server: replace `policy()` with your controller / network inference and read
// observations straight out of mjData (sensordata, qpos, qvel).
//
//   ./batch_rollout ../generated/orchard_uav-s0/scene.xml [n_envs] [threads] [steps]
#include <mujoco/mujoco.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

struct Rng {  // xorshift64*: tiny, per-env, reproducible
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 2685821657736338717ULL + 1) {}
  double uniform() {
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    return double((s * 2685821657736338717ULL) >> 11) * (1.0 / 9007199254740992.0);
  }
};

// Placeholder policy: mid-range control + exploration noise.
void policy(const mjModel* m, mjData* d, Rng& rng) {
  for (int i = 0; i < m->nu; ++i) {
    const mjtNum lo = m->actuator_ctrlrange[2 * i], hi = m->actuator_ctrlrange[2 * i + 1];
    const mjtNum mid = 0.5 * (lo + hi), span = 0.5 * (hi - lo);
    d->ctrl[i] = mid + 0.2 * span * (2.0 * rng.uniform() - 1.0);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s scene.xml [n_envs=64] [threads=hw] [steps=2000]\n", argv[0]);
    return 1;
  }
  const int n_envs = argc > 2 ? std::atoi(argv[2]) : 64;
  const int n_threads = argc > 3 ? std::atoi(argv[3]) : int(std::thread::hardware_concurrency());
  const int steps = argc > 4 ? std::atoi(argv[4]) : 2000;

  char err[1024] = {0};
  mjModel* m = mj_loadXML(argv[1], nullptr, err, sizeof(err));
  if (!m) { std::fprintf(stderr, "load error: %s\n", err); return 1; }
  std::printf("model: nq=%d nv=%d nu=%d ngeom=%d nbody=%d  dt=%.4f\n", m->nq, m->nv, m->nu, m->ngeom,
              m->nbody, m->opt.timestep);

  std::vector<mjData*> envs(n_envs);
  std::vector<Rng> rngs;
  for (int i = 0; i < n_envs; ++i) {
    envs[i] = mj_makeData(m);
    mj_forward(m, envs[i]);
    rngs.emplace_back(uint64_t(i) + 1);
  }

  std::atomic<int> next{0};
  auto worker = [&]() {
    for (int e; (e = next.fetch_add(1)) < n_envs;) {  // dynamic env assignment
      mjData* d = envs[e];
      for (int s = 0; s < steps; ++s) {
        policy(m, d, rngs[e]);
        mj_step(m, d);
      }
    }
  };

  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> pool;
  for (int t = 0; t < n_threads; ++t) pool.emplace_back(worker);
  for (auto& th : pool) th.join();
  const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  const double total = double(n_envs) * steps;
  std::printf("%d envs x %d steps on %d threads: %.2f s wall  ->  %.0f steps/s  (%.0fx real-time aggregate)\n",
              n_envs, steps, n_threads, wall, total / wall, total * m->opt.timestep / wall);

  for (auto* d : envs) mj_deleteData(d);
  mj_deleteModel(m);
  return 0;
}
