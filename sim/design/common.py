"""Shared helpers for vehicle sizing: mission-file loading, part selection, tables.
(Battery chemistry and the pack model live in sim/physics/battery.py.)"""
from __future__ import annotations

import tomllib
from pathlib import Path

from .. import parts

CONFIGS = Path(__file__).resolve().parent / "configs"


def load_toml(path) -> dict:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = CONFIGS / p
    with open(p, "rb") as f:
        return tomllib.load(f)


def components() -> dict:
    """The part database (sim/parts/*.toml, merged)."""
    return parts.load()


def select(db: dict, section: str, choice) -> dict[str, dict]:
    """Pick entries of a database section: "all" or a list of names."""
    items = db[section]
    if choice == "all":
        return dict(items)
    missing = [c for c in choice if c not in items]
    if missing:
        raise SystemExit(f"unknown {section} in config: {missing}; known: {list(items)}")
    return {k: items[k] for k in choice}


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
