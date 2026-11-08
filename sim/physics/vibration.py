"""Vibration transmission.

A mass on a spring-damper (natural frequency fn, damping ratio zeta) shaken at its base
with frequency f moves with amplitude ratio (transmissibility), r = f / fn:

    T(f) = sqrt( (1 + (2 zeta r)^2) / ((1 - r^2)^2 + (2 zeta r)^2) )

T ~ 1 well below fn, peaks at ~1/(2 zeta) at resonance, and falls off above sqrt(2) fn:
that is how a soft mount isolates an IMU, and why an arm whose bending mode sits near the
rotor speed shakes the whole frame.

Residual imbalance U [kg m] spinning at omega gives a rotating force F = U omega^2.
An n-th order low-pass filter (e.g. an IMU anti-alias filter) passes 1/sqrt(1 + (f/fc)^(2n)).
"""
from __future__ import annotations

import numpy as np


def transmissibility(f, fn: float, zeta: float):
    r = np.asarray(f, float) / fn
    return np.sqrt((1 + (2 * zeta * r) ** 2) / ((1 - r * r) ** 2 + (2 * zeta * r) ** 2))


def lowpass_gain(f, fc: float, order: int = 2):
    return 1.0 / np.sqrt(1 + (np.asarray(f, float) / fc) ** (2 * order))


def imbalance_force(U_kgm: float, omega):
    """Rotating force [N] of residual imbalance U [kg m] at omega [rad/s]."""
    return U_kgm * np.asarray(omega, float) ** 2
