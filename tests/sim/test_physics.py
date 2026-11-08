"""sim/parts and sim/physics: the part database loads, and each equation gives the value
you'd get by hand (or the textbook limiting behaviour)."""
import math

import numpy as np
import pytest

from sim import parts
from sim.physics import aero, battery, beam, dc_motor, noise, propeller, vehicle, vibration


# ------------------------------------------------------------------------- parts ---
def test_parts_load_and_lookup():
    db = parts.load()
    for section in ("motor", "prop", "battery", "chemistry", "arm_tube", "frame", "wheel", "drive_motor",
                    "imu", "gnss", "compute"):
        assert section in db
    assert parts.get("motor", "m2004_1700kv")["kv"] == 1700
    assert "lipo_6s_2200" in parts.names("battery")
    with pytest.raises(KeyError, match="known"):
        parts.get("motor", "no_such_motor")
    p = parts.get("prop", "p5x3_bi")
    p["ct"] = -1                                   # copies: callers can't corrupt the database
    assert parts.get("prop", "p5x3_bi")["ct"] > 0


# --------------------------------------------------------------------- propeller ---
def test_propeller_thrust_power_roundtrip():
    prop, rho = parts.get("prop", "p5x3_bi"), 1.2
    D = propeller.diameter_m(prop)
    n = float(propeller.speed_for_thrust(2.0, D, prop["ct"], rho))
    assert prop["ct"] * rho * n ** 2 * D ** 4 == pytest.approx(2.0)          # T = ct rho n^2 D^4
    P = propeller.shaft_power(n, D, prop["cp"], rho)
    w = 2 * math.pi * n
    assert P / w == pytest.approx(propeller.drag_torque_coef(D, prop["cp"], rho) * w * w)   # Q = kq w^2


def test_rotor_point_power_balance():
    motor, prop = parts.get("motor", "m2004_1700kv"), parts.get("prop", "p5x3_bi")
    op = propeller.rotor_point(2.0, motor, prop, 1.2)
    assert float(op["Pe"]) > float(op["Pm"]) > 0                             # losses: I^2 R + I0
    Kt = dc_motor.torque_constant(motor["kv"])
    loss = float(op["I"]) ** 2 * motor["r_ohm"] + motor["i0"] * float(op["w"]) * Kt
    assert float(op["Pe"]) - float(op["Pm"]) == pytest.approx(loss, rel=1e-9)


# ---------------------------------------------------------------------- dc motor ---
def test_bldc_torque_constant_and_gearmotor_line():
    assert dc_motor.torque_constant(1000) == pytest.approx(60 / (2 * math.pi * 1000))
    gm = dc_motor.gearmotor_at_voltage(parts.get("drive_motor", "gm37_12v_50"), 1.0)
    assert dc_motor.gearmotor_speed(0.0, gm["tau_s"], gm["omega_0"]) == pytest.approx(gm["omega_0"])
    assert dc_motor.gearmotor_speed(gm["tau_s"], gm["tau_s"], gm["omega_0"]) == pytest.approx(0.0)
    assert dc_motor.gearmotor_current(gm["tau_s"], gm["i0"], gm["i_s"], gm["tau_s"]) == pytest.approx(gm["i_s"])
    assert dc_motor.gearmotor_current(9 * gm["tau_s"], gm["i0"], gm["i_s"], gm["tau_s"], saturate=True) == \
        pytest.approx(gm["i_s"])


# ----------------------------------------------------------------------- battery ---
def test_pack_terminal_voltage_power_balance():
    pack = battery.Pack(parts.get("battery", "lipo_6s_2200"))
    assert pack.ocv(1.0) == pytest.approx(6 * 4.20)
    v, i = pack.terminal(150.0, 0.5)
    assert v * i == pytest.approx(150.0)                                       # P = V I
    assert v == pytest.approx(pack.ocv(0.5) - i * pack.r)                       # V = E - I R
    v0, i0 = pack.terminal(0.0, 0.5)
    assert i0 == 0 and v0 == pytest.approx(pack.ocv(0.5))
    assert 0 < battery.soc_at_cutoff(pack) < 0.2


# -------------------------------------------------------------------------- beam ---
def test_beam_sections_and_cantilever():
    A, I = beam.round_tube(0.010, 0.0)                                         # solid rod
    assert A == pytest.approx(math.pi * 0.005 ** 2) and I == pytest.approx(math.pi * 0.005 ** 4 / 4)
    A, I, c = beam.square_tube(0.02, 0.01)                                     # wall = half side: solid
    assert A == pytest.approx(0.02 ** 2) and I == pytest.approx(0.02 ** 4 / 12) and c == 0.01
    E, I, L = 70e9, 1e-9, 0.2
    assert beam.cantilever_tip_deflection(10, L, E, I) == pytest.approx(10 * L ** 3 / (3 * E * I))
    k = 3 * E * I / L ** 3                                                     # massless beam: f = sqrt(k/m)/2pi
    assert beam.cantilever_first_mode_hz(E, I, L, 0.05, 0.0) == pytest.approx(math.sqrt(k / 0.05) / (2 * math.pi))


# ----------------------------------------------------------------------- vehicle ---
def test_vehicle_torques_and_tipping():
    m, g, r = 10.0, 9.81, 0.1
    assert vehicle.wheel_torque_slope(m, g, 0.0, 0.05, r, 4) == pytest.approx(vehicle.wheel_torque_rolling(m, g, 0.05, r, 4))
    assert vehicle.wheel_torque_slope(m, g, 90.0, 0.0, r, 1) == pytest.approx(m * g * r)
    assert vehicle.tip_over_angle_deg(0.5, 0.5) == pytest.approx(45.0)
    assert vehicle.tip_over_angle_deg(vehicle.track_for_tip_angle(0.3, 35) / 2, 0.3) == pytest.approx(35.0)


# ------------------------------------------------------------- aero / vibration ---
def test_drag_and_transmissibility():
    f = aero.drag_force(0.1, [10.0, 0, 0])
    assert f[0] == pytest.approx(0.5 * aero.RHO_AIR * 0.1 * 100)
    assert vibration.transmissibility(1e-6, 100, 0.1) == pytest.approx(1.0)
    assert vibration.transmissibility(100.0, 100, 0.05) == pytest.approx(math.sqrt(1 + 1 / 0.01), rel=1e-6)
    assert vibration.imbalance_force(1e-6, 1000.0) == pytest.approx(1.0)


def test_gauss_markov_stationary_sigma():
    gm = noise.GaussMarkov(2.0, 0.5, np.random.default_rng(0))
    xs = np.array([float(gm.step(0.05)) for _ in range(40000)])
    assert xs.std() == pytest.approx(2.0, rel=0.1)
