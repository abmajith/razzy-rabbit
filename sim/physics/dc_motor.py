"""Electric motors.

Brushless DC motor (drone rotors), from its datasheet KV, winding resistance R and
no-load current I0:

    Kt = 60 / (2 pi KV)                torque constant [N m / A] (= back-EMF constant [V s/rad])
    I  = Q / Kt + I0                   current to deliver shaft torque Q
    V  = omega Kt + I R                terminal voltage: back-EMF + copper drop
    J domega/dt = Kt^2/R (V/Kt - omega) - kq omega^2      speed dynamics with a prop load;
    linearised around omega_h the time constant is
    tau = J / (2 kq omega_h + Kt^2 / R)

Brushed DC gearmotor (rover wheels), values at the output shaft:

    tau   = tau_s (u - omega / omega_0)       u in [-1, 1] = PWM duty
    I     = I0 + (I_s - I0) tau / tau_s
    stall torque tau_s, no-load speed omega_0 and stall current I_s scale with
    (supply voltage / rated voltage); the no-load current I0 does not.
"""
from __future__ import annotations

import math

import numpy as np


# ------------------------------------------------------------------ brushless DC ---
def torque_constant(kv: float) -> float:
    """Kt [N m/A] from KV [rpm/V]."""
    return 60.0 / (2 * math.pi * kv)


def bldc_current(Q, kt: float, i0: float):
    """Current [A] to deliver shaft torque Q [N m]."""
    return Q / kt + i0


def bldc_voltage(omega, I, kt: float, r_ohm: float):
    """Terminal voltage [V] at speed omega [rad/s] and current I [A]."""
    return omega * kt + I * r_ohm


def rotor_inertia(m_motor: float, r_rotor: float) -> float:
    """Outrunner bell approximated as a disc of the motor mass at radius r_rotor [kg m^2]."""
    return 0.5 * m_motor * r_rotor ** 2


def speed_loop_time_constant(J: float, kq: float, omega_h: float, kt: float, r_ohm: float) -> float:
    """First-order time constant [s] of motor + prop speed around hover speed omega_h."""
    return J / (2 * kq * omega_h + kt * kt / r_ohm) if omega_h > 0 else 0.05


# --------------------------------------------------------------------- gearmotor ---
def gearmotor_at_voltage(motor: dict, ratio: float) -> dict:
    """Torque-speed line of a gearmotor run at `ratio` x its rated voltage.
    Returns tau_s [N m], omega_0 [rad/s], i0, i_s [A]."""
    return {"tau_s": motor["stall_torque"] * ratio,
            "omega_0": motor["no_load_rpm"] * 2 * math.pi / 60 * ratio,
            "i0": motor["no_load_current"],
            "i_s": motor["stall_current"] * ratio}


def gearmotor_speed(tau, tau_s: float, omega_0: float):
    """Speed [rad/s] at full duty while delivering torque tau."""
    return omega_0 * (1 - tau / tau_s)


def gearmotor_duty(omega, tau, tau_s: float, omega_0: float):
    """Duty u needed to turn at omega while delivering torque tau."""
    return omega / omega_0 + tau / tau_s


def gearmotor_current(tau, i0: float, i_s, tau_s, saturate: bool = False):
    """Current [A] while delivering torque tau (saturate: clip at the stall current)."""
    if saturate:
        return i0 + (i_s - i0) * np.minimum(tau / tau_s, 1.0)
    return i0 + (i_s - i0) * tau / tau_s
