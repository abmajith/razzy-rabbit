"""Vehicle sizing: optimisers find feasible designs, constraints bite, and the generated
MuJoCo vehicles match the sizing model (mass, power) when simulated."""
import tomllib

import numpy as np
import pytest

pytest.importorskip("mujoco")

from sim.design import drone, rover  # noqa: E402
from sim.design.common import CONFIGS  # noqa: E402
from sim.runtime import Sim  # noqa: E402


def _write_variant(tmp_path, base: str, section: str, key: str, value) -> str:
    cfg = tomllib.loads((CONFIGS / base).read_text())
    cfg[section][key] = value
    lines = []
    for sec, kv in cfg.items():
        lines.append(f"[{sec}]")
        for k, v in kv.items():
            lines.append(f"{k} = {_toml(v)}")
    path = tmp_path / base
    path.write_text("\n".join(lines))
    return str(path)


def _toml(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, list):
        return "[" + ", ".join(_toml(x) for x in v) + "]"
    return repr(v)


@pytest.mark.parametrize("mod", [drone, rover])
def test_default_missions_have_feasible_designs(mod):
    cfg, designs = mod.optimize()
    feas = [d for d in designs if d.feasible]
    assert feas, mod.report(cfg, designs)
    assert all(not d.violations for d in feas)


def test_drone_constraints_respected():
    cfg, best = drone.best("drone_forest.toml")
    m, mis = best.m, cfg["mission"]
    assert m["span_m"] <= mis["max_span_m"]
    assert m["twr_end"] >= mis["min_twr"]
    assert m["endurance_min"] >= mis["min_endurance_min"]


def test_impossible_mission_reports_violations(tmp_path):
    path = _write_variant(tmp_path, "drone_forest.toml", "mission", "max_span_m", 0.12)
    cfg, designs = drone.optimize(path)
    assert not any(d.feasible for d in designs)
    assert "NO FEASIBLE DESIGN" in drone.report(cfg, designs)


def test_rover_steering_clearance():
    cfg, best = rover.best("rover_yard.toml")
    g = best.geo
    inner_edge = g["track"] / 2 - g["w"] / 2 - g["r"] * np.sin(np.radians(rover.STEER_LOCK_DEG))
    assert inner_edge > cfg["body"]["width_m"] / 2


@pytest.mark.parametrize("scene,tol", [("forest_grounded", 0.02), ("mars_yard", 0.03)])
def test_generated_vehicle_matches_sizing(scene, tol):
    sim = Sim(scene, 0)
    bid = sim.m.body(sim.robot["body"]).id
    designed = sim.robot["design"]["metrics"]["mass_kg"]
    assert abs(sim.m.body_subtreemass[bid] - designed) / designed < tol


def test_drone_battery_drains_and_power_matches_hover():
    sim = Sim("forest_grounded", 0)
    sim.step(int(8.0 / sim.m.opt.timestep))
    hover_w = sim.robot["design"]["metrics"]["hover_w"]
    assert 0.0 < sim.power.soc < 1.0
    assert sim.power.v < sim.power.pack.ocv(1.0)
    assert abs(sim.power.power - hover_w) / hover_w < 0.3     # cruising ~ hover power
