"""Uncertainty model: draws are reproducible, stay in range, are applied to the plant only
(controllers keep the nominal model), restore exactly, and infeasible draws are rejected."""
import numpy as np
import pytest

pytest.importorskip("mujoco")

from sim import uncertainty as U  # noqa: E402
from sim.runtime import Sim  # noqa: E402
from sim.runtime.sensors import by_type  # noqa: E402


@pytest.fixture(scope="module")
def drone():
    return Sim("forest_grounded", 0)


def _body(sim):
    return sim.m.body(sim.robot["body"]).id


def test_draws_are_deterministic_and_in_range(drone):
    unc = drone.meta0["uncertainty"]
    robot = drone.meta0["robots"][0]
    a, _ = U.sample(unc, robot, np.random.default_rng([0, 1]))
    b, _ = U.sample(unc, robot, np.random.default_rng([0, 1]))
    na, va = U.flatten(a)
    nb, vb = U.flatten(b)
    assert na == nb and np.array_equal(va, vb)
    for _ in range(50):
        p, _ = U.sample(unc, robot, np.random.default_rng(_))
        assert 1 - unc["mass_frac"] <= p["mass_frac"] <= 1 + unc["mass_frac"]
        assert unc["battery_capacity"][0] <= p["battery_capacity"] <= unc["battery_capacity"][1]
        assert len(p["motor_gain"]) == 4
        com = p["com_offset_mm"] * 1000
        assert np.hypot(com[0], com[1]) <= unc["com_offset_mm"][0] + 1e-9


def test_variant_applies_to_plant_not_controller_and_restores(drone):
    m, bid = drone.m, _body(drone)
    drone.reset(None)
    mass0, gain0 = float(m.body_mass[bid]), m.actuator_gainprm[:4, 0].copy()
    lat0 = by_type(drone.robot)["gnss"]["latency_ms"]
    drone.reset(3)
    p = drone.params
    assert m.body_mass[bid] == pytest.approx(mass0 * p["mass_frac"])
    assert np.allclose(m.actuator_gainprm[:4, 0], gain0 * p["thrust_coef"] * p["motor_gain"])
    assert drone.ctrls[0].ctl.mass == pytest.approx(mass0, rel=1e-6)   # controller: nominal
    lat = by_type(drone.robot)["gnss"]["latency_ms"]
    assert lat[0] == pytest.approx(lat0[0] + p["latency_ms_add"])
    assert by_type(drone.meta0["robots"][0])["gnss"]["latency_ms"] == lat0   # nominal untouched
    drone.reset(None)
    assert float(m.body_mass[bid]) == mass0 and np.array_equal(m.actuator_gainprm[:4, 0], gain0)


def test_same_variant_reproduces_trajectory(drone):
    out = []
    for _ in range(2):
        drone.reset(5)
        drone.step(1500)
        out.append(drone.d.qpos.copy())
    assert np.array_equal(out[0], out[1])
    drone.reset(6)
    drone.step(1500)
    assert not np.array_equal(out[0], drone.d.qpos)
    drone.reset(None)                                  # back to nominal == a freshly loaded scene
    drone.step(1500)
    fresh = Sim("forest_grounded", 0)
    fresh.step(1500)
    assert np.array_equal(fresh.d.qpos, drone.d.qpos)


def test_randomised_drones_still_fly(drone):
    for k in range(3):
        drone.reset(k)
        errs = []
        for i in range(4000):                                     # 8 s
            drone.step()
            if drone.d.time > 4 and i % 50 == 0:
                errs.append(drone.ctrls[0].err)
        assert max(errs) < 2.0, (k, max(errs), drone.params)


def test_infeasible_ranges_are_rejected_or_refused(drone):
    robot = drone.meta0["robots"][0]
    unc = {"mass_frac": [1.0, 1.8], "require_hover_throttle": 0.25, "max_attempts": 200}
    p, rejected = U.sample(unc, robot, np.random.default_rng(0))
    hover = robot["design"]["metrics"]["hover_throttle"] * p["mass_frac"]
    assert hover <= 0.25 and rejected > 0
    with pytest.raises(SystemExit):
        U.sample({"mass_frac": [3.0, 4.0], "require_twr_end": 1.8}, robot, np.random.default_rng(0))
    with pytest.raises(SystemExit):
        U.sample({"not_a_param": 0.1}, robot, np.random.default_rng(0))
    with pytest.raises(SystemExit):
        U.sample({"drive_torque": 0.1}, robot, np.random.default_rng(0))   # rover-only key


def test_rover_variants_drive():
    sim = Sim("mars_yard", 0, sensors=False)
    m = sim.m
    drive = [m.actuator(a).id for a in sim.robot["drive"]]
    sim.reset(2)
    tq = sim.params["drive_torque"] * sim.params["drive_gain"]
    assert np.allclose(sim.power.dynamic[0].tau_s, sim.meta0["robots"][0]["tau_stall"] * tq)
    sim.step(5000)                                                # 10 s
    assert sim.ctrls[0].err < 0.5
    assert np.all(np.abs(m.actuator_gainprm[drive, 0]) > 0)
