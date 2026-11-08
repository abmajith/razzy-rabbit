"""The C++ core (razzy_core) against the readable Python reference.

Physics, controllers and plants follow the reference's operation order, so trajectories,
battery state and power must agree to rounding. Sensor noise comes from a different RNG
(Philox), so sensors are checked statistically and against the datasheet."""
import math

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("razzy_core")

from sim.core import CoreBatch, CoreSim  # noqa: E402
from sim.design.common import components  # noqa: E402
from sim.runtime import Sim  # noqa: E402

DB = components()


@pytest.mark.parametrize("scene,variant,tol", [
    ("forest_grounded", None, 1e-9), ("forest_grounded", 3, 1e-9),
    ("mars_yard", None, 1e-6), ("mars_yard", 3, 1e-4)])
def test_trajectory_and_plant_match_python_reference(scene, variant, tol):
    py = Sim(scene, 0, sensors=False, variant=variant)
    cb = CoreBatch(scene, 1, 0, sensors=False)
    cb.reset([0], [variant])
    for _ in range(3):
        py.step(500)
        cb.step(500)
        assert abs(py.d.time - cb.time[0]) < 1e-12
        assert np.abs(py.d.qpos - cb.qpos[0]).max() < tol
        if py.plant is not None:
            assert cb.plant[0, 0] == pytest.approx(py.plant.soc, abs=1e-9)
            assert cb.plant[0, 3] == pytest.approx(py.plant.power, rel=1e-6)


def test_threads_do_not_change_results():
    out = []
    for threads in (1, 8):
        b = CoreBatch("forest_grounded", 8, 0, threads=threads, record=True)
        b.reset(variants=list(range(8)))
        b.step(800)
        out.append((b.qpos.copy(), b.imu.copy(), b.gnss.copy(), b.core.log(5, "imu")))
    for a, c in zip(out[0], out[1]):
        assert np.array_equal(a, c, equal_nan=True)


def test_gnss_sky_view_matches_python_ray_casts():
    py = Sim("forest_grounded", 0)
    cb = CoreBatch("forest_grounded", 1, 0)
    dirs = np.ascontiguousarray(py.sensors.gnss.sats_at(0.0))
    rng = np.random.default_rng(4)
    A, C = [], []
    for _ in range(25):
        x, y = rng.uniform(-60, 60, 2)
        p = np.array([x, y, py.terrain.height(x, y) + rng.uniform(1, 6)])
        A.append(py.sensors.sky_survival(p, dirs))
        C.append(cb.core.sky_survival(0, list(p), dirs))
    A, C = np.array(A), np.array(C)
    # vegetation: identical mj_ray casts. terrain: C++ marches the bilinear height grid instead
    # of MuJoCo's triangulated hfield test, so grazing rays over ridges may differ rarely
    assert np.mean(np.abs(A - C) > 1e-9) < 0.06
    assert abs(A.mean() - C.mean()) < 0.01


def test_sensor_statistics_in_flight():
    b = CoreBatch("forest_grounded", 1, 0, record=True)
    b.step(5000)                                                # 10 s
    imu = b.core.log(0, "imu")
    assert abs(len(imu) - 10_000) <= 2
    err = imu[2000:, 1:7] - imu[2000:, 7:13]
    spec = DB["imu"]["icm42688p"]
    floor_a = spec["accel_noise_ug_rthz"] * 1e-6 * 9.80665 * math.sqrt(1000)
    assert err[:, 5].std() == pytest.approx(floor_a, rel=0.25)  # z: noise only (ripple is isolated)
    assert 0.1 < err[:, 3:5].std() < 1.0                       # x/y: rotor imbalance vibration
    g = b.core.log(0, "gnss")
    assert g[:, 4].mean() > 0.6                                # forest canopy blocks most sky
    yard = CoreBatch("mars_yard", 1, 0, record=True)
    yard.step(5000)
    gy = yard.core.log(0, "gnss")
    assert gy[:, 1].all() and gy[:, 4].max() < 0.05            # open sky: always a fix
    eh = np.linalg.norm(gy[:, 5:7] - gy[:, 8:10], axis=1)
    assert np.median(eh) < 4 * DB["gnss"]["ublox_m10"]["cep_m"]
    t = yard.time[0]
    assert yard.gnss[0, 0] <= t - 0.06                         # latency respected


def test_setpoint_mode_and_compute_latency():
    b = CoreBatch("forest_grounded", 2, 0, mode="setpoint")
    b.step(1500)                                               # 3 s hovering on zero setpoints
    z0 = b.qpos[:, 2].copy()
    b.act(np.array([[0, 0, 1.0, 0], [0, 0, 0, 0]]))           # env 0 climbs at 1 m/s
    b.step(10)                                                 # 20 ms < 25 ms minimum inference time
    assert np.array_equal(b.qpos[0], b.qpos[1]) or np.allclose(b.ctrl[0], b.ctrl[1])   # not yet applied
    b.step(1000)
    climb = b.qpos[:, 2] - z0
    assert climb[0] > 1.0 and abs(climb[1]) < 0.3
    r = CoreBatch("mars_yard", 1, 0, mode="setpoint", sensors=False)
    x0 = r.qpos[0, :2].copy()
    r.act(np.array([[0.5, 0.0]]))
    r.step(2500)
    assert np.linalg.norm(r.qpos[0, :2] - x0) > 1.5


def test_camera_models_in_cpp():
    sim = CoreSim("forest_grounded", 0)
    su = sim.sensors
    spec = dict(sim.robot["sensors"]["depth_camera"])
    f = spec["width"] / 2 / math.tan(math.radians(spec["hfov_deg"]) / 2)
    for z in (2.0, 6.0):
        out = sim.batch.core.depth_apply(0, np.full((240, 320), z, np.float32), None)
        valid = out[out > 0]
        expect = z * z * spec["subpixel_sigma_px"] / (f * spec["baseline_m"])
        assert valid.size > 0.95 * out.size
        assert abs(valid.std() - expect) / expect < 0.3
    img = np.zeros((120, 160, 3), np.uint8)
    img[:, 80] = 255
    out = sim.batch.core.rgb_apply(0, img, [0.0, 2.0, 0.0])
    assert out.shape == img.shape and out.dtype == np.uint8
    assert su is not None and su.status()
