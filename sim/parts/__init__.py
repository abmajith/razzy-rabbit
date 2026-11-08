"""Part database: datasheet numbers only, one TOML file per domain.

    propulsion.toml   motor.*, prop.*
    energy.toml       battery.*, chemistry.* (open-circuit voltage curves)
    drive.toml        wheel.*, drive_motor.*, steer_servo.*
    structure.toml    arm_tube.*, rover_tube.*, frame
    electronics.toml  avionics.* (fixed payload boards)
    sensors.toml      imu.*, baro.*, mag.*, gnss.*, depth_camera.*, rgb_camera.*
    compute.toml      compute.*

No code anywhere names a specific part; mission files (sim/design/configs/) pick parts by
name and code reads their numbers from here. To add a part, add a `[section.name]` table to
the right file; to add a domain, add a new .toml file (it is picked up automatically).

    from sim import parts
    parts.get("motor", "m2004_1700kv")     # -> dict of datasheet values
    parts.names("battery")                 # -> ["lipo_3s_1000", ...]
    parts.load()                           # -> {section: {name: dict}} (all files merged)
"""
from __future__ import annotations

import copy
import tomllib
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _load() -> dict:
    db: dict = {}
    for f in sorted(HERE.glob("*.toml")):
        with open(f, "rb") as fh:
            for section, items in tomllib.load(fh).items():
                if section in db:
                    raise ValueError(f"section [{section}] defined twice (second time in {f.name})")
                db[section] = items
    return db


def load() -> dict:
    """All parts, merged: {section: {name: {key: value}}}. A copy: safe to modify."""
    return copy.deepcopy(_load())


def sections() -> list[str]:
    return sorted(_load())


def names(section: str) -> list[str]:
    if section not in _load():
        raise KeyError(f"no part section '{section}'; sections: {sections()}")
    return list(_load()[section])


def get(section: str, name: str) -> dict:
    """One part's datasheet values (a copy). Unknown names list the valid ones."""
    items = _load().get(section)
    if items is None:
        raise KeyError(f"no part section '{section}'; sections: {sections()}")
    if name not in items:
        raise KeyError(f"unknown {section} '{name}'; known: {list(items)}")
    return copy.deepcopy(items[name])
