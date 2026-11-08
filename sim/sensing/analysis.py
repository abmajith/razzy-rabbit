"""Characterisation tools: run them on simulated logs AND on real logs, compare the numbers.

  allan_deviation  - IMU noise density / bias instability from a stationary log
  vibration_peaks  - dominant vibration lines (rotor 1x, blade pass, resonances)
  report           - one text summary of a recorded SensorSuite run
"""
from __future__ import annotations

import math

import numpy as np

DEG = math.pi / 180


def allan_deviation(x: np.ndarray, rate: float, taus=None) -> tuple[np.ndarray, np.ndarray]:
    """Overlapping Allan deviation of a rate signal x (N,) or (N, k) sampled at `rate` Hz.
    At tau = 1 s on the white-noise slope (-1/2) the value equals the noise density."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    n = len(x)
    if taus is None:
        taus = np.unique(np.logspace(0, math.log10(max(n // 9, 2)), 30).astype(int)) / rate
    c = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, 0)]) / rate     # integrated signal
    out = []
    for tau in taus:
        m = int(round(tau * rate))
        if m < 1 or 2 * m >= n:
            out.append(np.full(x.shape[1], np.nan))
            continue
        d = c[2 * m:] - 2 * c[m:-m] + c[:-2 * m]
        out.append(np.sqrt(0.5 * np.mean(d * d, 0)) / tau)
    return np.asarray(taus), np.asarray(out)


def vibration_peaks(x: np.ndarray, rate: float, k: int = 3) -> list[tuple[float, float]]:
    """Top-k spectral lines (frequency Hz, amplitude) of a signal, Hann-windowed."""
    x = np.asarray(x, float) - np.mean(x)
    w = np.hanning(len(x))
    X = np.abs(np.fft.rfft(x * w)) * 2 / w.sum()
    f = np.fft.rfftfreq(len(x), 1 / rate)
    X[f < 5] = 0
    peaks = []
    for _ in range(k):
        i = int(np.argmax(X))
        if X[i] <= 0:
            break
        peaks.append((float(f[i]), float(X[i])))
        lo, hi = np.searchsorted(f, [f[i] * 0.85, f[i] * 1.15])
        X[lo:hi] = 0
    return peaks


def report(suite) -> str:
    """Summary of a SensorSuite run with record=True: errors vs ground truth, with the
    datasheet values next to them so mismatches are obvious."""
    L = {k: np.asarray(v) for k, v in suite.log.items() if len(v)}
    sp = suite.spec
    lines = []
    if "imu" in L:
        imu, s, rate = L["imu"], sp["imu"], suite.imu.rate
        eg, ea = imu[:, 1:4] - imu[:, 7:10], imu[:, 4:7] - imu[:, 10:13]
        lines.append(f"IMU {s['part']} @ {rate:.0f} Hz, {len(imu)} samples, saturated {suite.imu.saturated}")
        nd_g = s["gyro_noise_dps_rthz"] * DEG * math.sqrt(rate)
        nd_a = s["accel_noise_ug_rthz"] * 1e-6 * 9.80665 * math.sqrt(rate)
        lines.append(f"  gyro  error std xyz [rad/s]  {_v(eg.std(0))}   white-noise floor {nd_g:.4f}")
        lines.append(f"  accel error std xyz [m/s^2]  {_v(ea.std(0))}   white-noise floor {nd_a:.4f}")
        if len(imu) > 4 * rate:
            for ax, name in ((0, "x"), (1, "y"), (2, "z")):
                pk = vibration_peaks(ea[:, ax], rate)
                lines.append(f"  accel {name} vibration lines: " + ", ".join(f"{f:.0f} Hz {a:.3f}" for f, a in pk))
        if suite.vib is not None:
            v = suite.vib
            lines.append(f"  rotors now {_v(v.n)} rev/s (blade pass x{v.blades}); arm mode {v.arm_hz:.0f} Hz, "
                         f"IMU mount {v.mount_hz:.0f} Hz")
    if "gnss" in L:
        g = L["gnss"]
        ok = g[:, 1] > 0
        lines.append(f"GNSS {sp['gnss']['part']}: fix {ok.mean()*100:.0f}% of epochs, sats mean {g[:, 2].mean():.1f}, "
                     f"sky blocked mean {g[:, 4].mean()*100:.0f}%")
        if ok.any():
            eh = np.linalg.norm(g[ok, 5:7] - g[ok, 8:10], axis=1)
            ev = np.abs(g[ok, 7] - g[ok, 10])
            lines.append(f"  horizontal error: median {np.median(eh):.2f} m, 95% {np.percentile(eh, 95):.2f} m "
                         f"(open-sky CEP50 spec {sp['gnss']['cep_m']} m);  vertical 95% {np.percentile(ev, 95):.2f} m")
    if "baro" in L:
        b = L["baro"]
        e = b[:, 1] - b[:, 2]
        lines.append(f"BARO {sp['baro']['part']}: error mean {e.mean():+.2f} m, std {e.std():.3f} m "
                     f"(noise spec {sp['baro']['noise_pa'] / (1.2 * 9.81):.3f} m)")
    if "mag" in L:
        mg = L["mag"]
        a, b = mg[:, 1:4], mg[:, 4:7]
        cosang = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        lines.append(f"MAG {sp['mag']['part']}: field direction error mean {ang.mean():.1f} deg, 95% "
                     f"{np.percentile(ang, 95):.1f} deg (hard/soft iron residual + motor current)")
    if "compute" in sp:
        c = sp["compute"]
        lines.append(f"COMPUTE {c['part']}: policy {c['rate_hz']} Hz, inference {c['inference_ms']} ms")
    return "\n".join(lines)


def _v(a) -> str:
    return "[" + " ".join(f"{x:.4f}" if abs(x) < 10 else f"{x:.0f}" for x in a) + "]"
