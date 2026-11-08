"""Shared helpers for vehicle sizing: config loading, battery chemistry, selection."""
from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np

CONFIGS = Path(__file__).resolve().parent / "configs"

# open-circuit voltage per cell vs state of charge, and nominal voltage
CHEMISTRY = {
    "lipo":    dict(nominal=3.7, cutoff=3.5,
                    soc=[0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                    ocv=[3.30, 3.55, 3.65, 3.74, 3.80, 3.88, 4.05, 4.20]),
    "liion":   dict(nominal=3.6, cutoff=3.0,
                    soc=[0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                    ocv=[3.00, 3.25, 3.40, 3.55, 3.65, 3.80, 4.00, 4.20]),
    "lifepo4": dict(nominal=3.2, cutoff=2.8,
                    soc=[0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                    ocv=[2.80, 3.00, 3.15, 3.25, 3.28, 3.30, 3.35, 3.60]),
}


def load_toml(path) -> dict:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = CONFIGS / p
    with open(p, "rb") as f:
        return tomllib.load(f)


def components() -> dict:
    return load_toml(CONFIGS / "components.toml")


def select(db: dict, section: str, choice) -> dict[str, dict]:
    """Pick entries of a database section: "all" or a list of names."""
    items = db[section]
    if choice == "all":
        return dict(items)
    missing = [c for c in choice if c not in items]
    if missing:
        raise SystemExit(f"unknown {section} in config: {missing}; known: {list(items)}")
    return {k: items[k] for k in choice}


def ocv_cell(chem: str, soc) -> np.ndarray:
    c = CHEMISTRY[chem]
    return np.interp(soc, c["soc"], c["ocv"])


class Pack:
    """Battery pack electrical model: OCV(soc) * cells - I * R."""

    def __init__(self, spec: dict) -> None:
        self.spec = spec
        self.chem = spec["chemistry"]
        self.cells = spec["cells"]
        self.cap_ah = spec["capacity_mah"] / 1000
        self.r = spec["r_cell"] * spec["cells"]
        self.i_max = spec["c_max"] * self.cap_ah
        self.mass = spec["mass_g"] / 1000
        c = CHEMISTRY[self.chem]
        self.v_nom = c["nominal"] * self.cells
        self.v_cut = c["cutoff"] * self.cells
        self.energy_wh = self.v_nom * self.cap_ah

    def ocv(self, soc) -> float:
        return float(ocv_cell(self.chem, soc)) * self.cells

    def terminal(self, power_w: float, soc: float) -> tuple[float, float]:
        """Terminal voltage and current delivering `power_w` at state of charge `soc`."""
        e = self.ocv(soc)
        disc = e * e - 4 * self.r * power_w
        if disc <= 0:                       # pack cannot deliver this power at all
            return e / 2, e / (2 * self.r)
        i = (e - np.sqrt(disc)) / (2 * self.r)
        return e - i * self.r, i


def fmt_table(rows: list[dict], cols: list[tuple[str, str, str]]) -> str:
    """cols: (key, header, format)."""
    head = "  ".join(f"{h:>{max(len(h), 8)}}" for _, h, _ in cols)
    lines = [head, "-" * len(head)]
    for r in rows:
        cells = []
        for k, h, f in cols:
            v = r.get(k, "")
            s = format(v, f) if isinstance(v, (int, float)) and f else str(v)
            cells.append(f"{s:>{max(len(h), 8)}}")
        lines.append("  ".join(cells))
    return "\n".join(lines)
