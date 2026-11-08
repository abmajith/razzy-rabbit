"""Noise and delay building blocks for sensor models.

White noise with density ND [unit/sqrt(Hz)] sampled at rate f has per-sample sigma
ND sqrt(f); its Allan deviation at tau = 1 s equals ND (how datasheets state it).

First-order Gauss-Markov process (slowly wandering biases, drift, multipath):

    x[k+1] = a x[k] + sigma sqrt(1 - a^2) w[k],   a = exp(-dt / tau),  w ~ N(0, 1)

stationary standard deviation sigma, correlation time tau.

Delay line: a measurement captured at t_c becomes visible at t_c + latency; with jittered
latency samples can overtake each other, and the newest *captured* one wins.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np


class GaussMarkov:
    """First-order Gauss-Markov process: stationary std `sigma`, correlation time `tau`."""

    def __init__(self, sigma, tau: float, rng: np.random.Generator, shape=()) -> None:
        self.sigma, self.tau, self.rng = np.asarray(sigma, float), float(tau), rng
        self.x = self.sigma * rng.standard_normal(shape)

    def step(self, dt: float) -> np.ndarray:
        a = math.exp(-dt / self.tau)
        self.x = a * self.x + self.sigma * math.sqrt(1 - a * a) * self.rng.standard_normal(self.x.shape)
        return self.x


class DelayLine:
    """Measurements become visible `latency` after capture (latency may jitter per sample).
    push(t_capture, value, latency); latest(t) -> (t_capture, value) of the newest delivered
    sample, or None. drain(t) -> all samples delivered since the previous drain."""

    def __init__(self, maxlen: int = 4096) -> None:
        self.q: deque = deque(maxlen=maxlen)
        self._latest = None

    def push(self, t_capture: float, value, latency: float) -> None:
        self.q.append((t_capture + latency, t_capture, value))

    def drain(self, t: float) -> list:
        out = []
        # samples can overtake each other when latency jitters: deliver every ready one
        ready = [s for s in self.q if s[0] <= t]
        if ready:
            self.q = deque((s for s in self.q if s[0] > t), maxlen=self.q.maxlen)
            ready.sort(key=lambda s: s[1])
            out = [(tc, v) for _, tc, v in ready]
            if self._latest is None or out[-1][0] >= self._latest[0]:
                self._latest = out[-1]
        return out

    def latest(self, t: float):
        self.drain(t)
        return self._latest


def latency_sampler(spec_ms, rng: np.random.Generator):
    """spec_ms: number or [min, max] in ms -> callable returning a latency in seconds."""
    if isinstance(spec_ms, (list, tuple)):
        lo, hi = float(spec_ms[0]) / 1000, float(spec_ms[1]) / 1000
        return lambda: float(rng.uniform(lo, hi))
    v = float(spec_ms) / 1000
    return lambda: v
