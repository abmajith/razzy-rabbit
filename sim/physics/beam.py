"""Beams: tube cross-sections and the cantilever formulas used for arms and rockers.

Sections (A area, I second moment of area, c distance from neutral axis to outer fibre):

    round tube  OD, ID :   A = pi/4 (OD^2 - ID^2)       I = pi/64 (OD^4 - ID^4)
    square tube a, wall t: A = a^2 - (a - 2t)^2         I = (a^4 - (a - 2t)^4) / 12,  c = a/2

Cantilever of length L, modulus E, tip load F (e.g. a motor at the end of an arm):

    tip deflection      delta = F L^3 / (3 E I)
    first bending mode  f = 1/(2 pi) sqrt(3 E I / (L^3 (m_tip + 0.236 m_beam)))   (Rayleigh)
    bending stress      sigma = M c / I,  M = F L at the root
"""
from __future__ import annotations

import math


def round_tube(od: float, id_: float) -> tuple[float, float]:
    """(A [m^2], I [m^4]) of a round tube, diameters in m."""
    I = math.pi / 64 * (od ** 4 - id_ ** 4)
    A = math.pi / 4 * (od ** 2 - id_ ** 2)
    return A, I


def square_tube(side: float, wall: float) -> tuple[float, float, float]:
    """(A [m^2], I [m^4], c [m]) of a square tube, dimensions in m."""
    A = side * side - (side - 2 * wall) ** 2
    I = (side ** 4 - (side - 2 * wall) ** 4) / 12
    return A, I, side / 2


def cantilever_tip_deflection(F: float, L: float, E: float, I: float) -> float:
    return (F * L ** 3) / (3 * E * I)


def cantilever_first_mode_hz(E: float, I: float, L: float, m_tip: float, m_beam: float) -> float:
    return 1 / (2 * math.pi) * math.sqrt(3 * E * I / (L ** 3 * (m_tip + 0.236 * m_beam)))


def bending_stress_mpa(M: float, c: float, I: float) -> float:
    return M * c / I / 1e6
