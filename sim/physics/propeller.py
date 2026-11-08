"""Propellers, and the operating point of a motor + propeller pair.

Static (hover) momentum coefficients, n in rev/s, D in m, rho in kg/m^3:

    T   = ct rho n^2 D^4               thrust
    P   = cp rho n^3 D^5               shaft power
    Q   = P / (2 pi n)                 shaft torque = kq omega^2, kq = cp rho D^5 / (8 pi^3)
    yaw reaction torque per thrust     Q / T = cp D / (2 pi ct)

Forward-flight effects (advance ratio) are not modelled; rotor drag covers them roughly.
"""
from __future__ import annotations

import math

import numpy as np

from . import dc_motor

INCH = 0.0254


def diameter_m(prop: dict) -> float:
    return prop["diameter_in"] * INCH


def speed_for_thrust(T, D: float, ct: float, rho: float):
    """Prop speed n [rev/s] producing thrust T [N]."""
    return np.sqrt(T / (ct * rho * D ** 4))


def shaft_power(n, D: float, cp: float, rho: float):
    """Shaft power [W] at prop speed n [rev/s]."""
    return cp * rho * n ** 3 * D ** 5


def drag_torque_coef(D: float, cp: float, rho: float) -> float:
    """kq [N m s^2]: shaft torque Q = kq omega^2."""
    return cp * rho * D ** 5 / (8 * math.pi ** 3)


def yaw_moment_coef(D: float, ct: float, cp: float) -> float:
    """Reaction torque per newton of thrust [m]."""
    return cp * D / (2 * math.pi * ct)


def inertia(m_prop: float, D: float) -> float:
    """Prop as a thin rod of length D spinning about its centre [kg m^2]."""
    return m_prop * D ** 2 / 24


def rotor_point(T, motor: dict, prop: dict, rho: float) -> dict:
    """Operating point of one motor + prop producing thrust T [N] (vectorised in T).
    Returns n [rev/s], w [rad/s], Q [N m], I [A], V [V], Pe (electrical) and Pm (shaft) [W]."""
    T = np.maximum(np.asarray(T, float), 0.0)
    D = diameter_m(prop)
    n = speed_for_thrust(T, D, prop["ct"], rho)
    w = 2 * np.pi * n
    Pm = shaft_power(n, D, prop["cp"], rho)
    Kt = dc_motor.torque_constant(motor["kv"])
    Q = np.where(w > 0, Pm / np.maximum(w, 1e-9), 0.0)
    I = dc_motor.bldc_current(Q, Kt, motor["i0"])
    V = dc_motor.bldc_voltage(w, I, Kt, motor["r_ohm"])
    return dict(n=n, w=w, Q=Q, I=I, V=V, Pe=V * I, Pm=Pm)
