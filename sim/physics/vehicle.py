"""Ground-vehicle mechanics: wheel torques, tip-over, steering clearance, ground pressure.

For a vehicle of mass m on n driven wheels of radius r (load shared equally):

    rolling      tau = m g crr r / n                           crr: rolling resistance coef.
    slope theta  tau = m g (sin theta + crr cos theta) r / n
    traction     tau = mu m g r / n                            mu: peak traction coefficient
    tip-over     angle = atan(half_base / h_cg)                half track (lateral) or wheelbase
    track for a tip angle   track = 2 h_cg tan(angle)
    corner wheel steered by delta swings inward by r sin(delta)
    contact pressure  p = wheel load / (width x contact length)
"""
from __future__ import annotations

import math


def wheel_torque_rolling(mass: float, g: float, crr: float, r: float, n_wheels: int) -> float:
    return mass * g * crr * r / n_wheels


def wheel_torque_slope(mass: float, g: float, slope_deg: float, crr: float, r: float, n_wheels: int) -> float:
    th = math.radians(slope_deg)
    return mass * g * (math.sin(th) + crr * math.cos(th)) * r / n_wheels


def wheel_torque_traction(mu: float, mass: float, g: float, r: float, n_wheels: int) -> float:
    return mu * mass * g * r / n_wheels


def tip_over_angle_deg(half_base: float, h_cg: float) -> float:
    return math.degrees(math.atan2(half_base, h_cg))


def track_for_tip_angle(h_cg: float, angle_deg: float) -> float:
    return 2 * h_cg * math.tan(math.radians(angle_deg))


def steer_swing(r: float, lock_deg: float) -> float:
    """How far a steered wheel's edge swings inward at full lock [m]."""
    return r * math.sin(math.radians(lock_deg))


def contact_pressure_kpa(wheel_load_n: float, width: float, contact_len: float) -> float:
    return wheel_load_n / (width * contact_len) / 1000
