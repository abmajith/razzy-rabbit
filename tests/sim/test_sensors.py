"""Sensor models reproduce their datasheet statistics, and the scene-coupled parts (canopy
blocking GNSS, rotor vibration, latency) behave physically. Deterministic per seed."""
import math

import numpy as np
import pytest

pytest.importorskip("mujoco")

from sim.design.common import components  # noqa: E402
from sim.sensing import models as M  # noqa: E402
from sim.sensing.analysis import allan_deviation  # noqa: E402

DB = components()


def test_imu_noise_density_matches_datasheet_by_allan():
    spec = DB["imu"]["icm42688p"]
    imu = M.Imu(spec, np.random.default_rng(1))
    n = 60_000
    g = np.empty((n, 3))
    a = np.empty((n, 3))
    for i in range(n):
        g[i], a[i] = imu.sample(np.zeros(3), np.array([0, 0, 9.80665]))
    taus, ad_g = allan_deviation(g, spec["rate_hz"], taus=[1.0])
    _, ad_a = allan_deviation(a, spec["rate_hz"], taus=[1.0])
    nd_g = spec["gyro_noise_dps_rthz"] * math.pi / 180
    nd_a = spec["accel_noise_ug_rthz"] * 1e-6 * 9.80665
    # quantisation (16 bit at +-16 g = 4.8 mm/s^2 LSB) adds a little at 1 kHz
    assert np.allclose(ad_g[0], nd_g, rtol=0.25), (ad_g, nd_g)
    assert np.allclose(ad_a[0], nd_a, rtol=0.35), (ad_a, nd_a)


def test_delay_line_and_compute_latency():
    line = M.DelayLine()
    line.push(0.0, "a", 0.05)
    line.push(0.01, "b", 0.05)
    assert line.latest(0.049) is None
    assert line.latest(0.055)[1] == "a"
    assert line.latest(0.07)[1] == "b"
    c = M.ComputeLatency(dict(rate_hz=10, inference_ms=30), np.random.default_rng(0))
    c.submit(1.0, [1.0])
    assert c.current(1.02) is None
    assert c.current(1.031)[0] == 1.0


def test_stereo_depth_error_grows_with_range_squared():
    spec = dict(DB["depth_camera"]["oakd_lite_stereo"], lr_check=False)
    cam = M.StereoDepth(spec, np.random.default_rng(2))
    f, B = cam.f_native, cam.B
    for z in (2.0, 4.0, 8.0):
        out = cam.apply(np.full((240, 320), z))
        valid = out[out > 0]
        expect = z * z * spec["subpixel_sigma_px"] / (f * B)
        assert valid.size > 0.95 * out.size
        assert abs(valid.std() - expect) / expect < 0.3, (z, valid.std(), expect)
    assert (cam.apply(np.full((10, 10), spec["max_m"] + 1)) == 0).all()


def test_vibration_resonance_and_imbalance_force():
    assert M.transmissibility(100.0, 100.0, 0.02) == pytest.approx(25.0, rel=0.01)
    assert M.transmissibility(400.0, 100.0, 0.15) < 0.15          # isolator above its corner
    vib = dict(rotor_pos=[[0.1, 0, 0]], spin=[1], mass=1.0, inertia=np.eye(3) * 0.01, blades=2,
               imbalance_gmm=1.0, thrust_ripple=0.0, arm_hz=1e5, mount_hz=1e5, mount_zeta=0.1)
    v = M.RotorVibration(vib, aaf_hz=1e6, rng=np.random.default_rng(0))
    v.set_rotors(0.0, [200.0], [2.0])
    acc = np.array([v.at(t)[1] for t in np.linspace(0, 0.01, 400)])
    expect = 1e-6 * (2 * math.pi * 200) ** 2 / 1.0                # U w^2 / m
    assert np.abs(acc[:, :2]).max() == pytest.approx(expect, rel=0.02)


def test_gnss_open_sky_matches_cep_and_blocked_sky_loses_fix():
    spec = DB["gnss"]["ublox_m10"]
    g = M.Gnss(spec, np.random.default_rng(3))
    n = len(g.sats)
    err = []
    for k in range(60_000):                                      # 100 min at 10 Hz
        fx = g.sample(k / 10, np.zeros(3), np.zeros(3), np.ones(n))
        err.append(np.linalg.norm(fx["pos"][:2]))
    assert np.median(err) == pytest.approx(spec["cep_m"], rel=0.35)
    fx = g.sample(0.0, np.zeros(3), np.zeros(3), np.zeros(n))
    assert not fx["fix"]


def test_rolling_shutter_skews_rows_by_rotation():
    spec = dict(DB["rgb_camera"]["oakd_lite_rgb"], vignetting=0.0, read_noise_dn=0.0, shot_noise_dn=0.0,
                exposure_ms=[0.01, 0.01], exposure_ref_ms=0.01, ae_target=0.0)
    h, w = 120, 160
    img = np.zeros((h, w, 3), np.uint8)
    img[:, 80] = 255                                             # one vertical line
    cam = M.RgbCamera(spec, np.random.default_rng(0))
    yaw = 2.0                                                    # rad/s about camera y
    out = cam.apply(img, [0.0, yaw, 0.0])
    top, bot = out[0, :, 0].argmax(), out[-1, :, 0].argmax()
    f = M.focal_px(spec["hfov_deg"], w)
    expect = f * yaw * spec["readout_ms"] / 1000 * (h - 1) / h
    assert abs(abs(int(top) - int(bot)) - expect) <= 1.5, (top, bot, expect)


@pytest.fixture(scope="module")
def forest():
    from sim.runtime import Sim
    return Sim("forest_grounded", 0, record=True)


def test_canopy_blocks_gnss_in_forest_not_in_open(forest):
    from sim.runtime import Sim
    yard = Sim("mars_yard", 0)
    for s in (forest, yard):
        s.step(500)
    def survival(s):
        su = s.sensors
        p = s.d.xpos[su.bid] + np.array([0, 0, 0.1])
        return su.sky_survival(p, su.gnss.sats_at(0.0)).mean()
    assert survival(yard) > 0.95
    assert survival(forest) < 0.6


def test_flying_drone_sensors_run_at_spec_rates_and_are_deterministic(forest):
    from sim.runtime import Sim
    forest.step(1500)                                             # now 4 s in total, flying
    log = {k: np.asarray(v) for k, v in forest.sensors.log.items()}
    t = forest.d.time
    for k, rate in (("imu", 1000), ("baro", 50), ("gnss", 10)):
        assert abs(len(log[k]) - t * rate) <= 2, (k, len(log[k]), t * rate)
    imu = log["imu"][-1000:]
    vib = (imu[:, 4:6] - imu[:, 10:12]).std()
    assert 0.03 < vib < 3.0                                       # rotor vibration visible, sane
    assert forest.sensors.imu.saturated == 0
    again = Sim("forest_grounded", 0, record=True)
    again.step(2000)
    assert np.array_equal(np.asarray(again.sensors.log["imu"]), log["imu"])
    r = again.sensors.read(again.d.time)
    assert r["imu"][0] < again.d.time and r["gnss"][0] <= again.d.time - 0.06   # latency respected
