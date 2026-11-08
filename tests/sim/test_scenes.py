"""Smoke tests for sim/: every scene builds, compiles, and its controller does its job."""
import hashlib

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from sim.runtime import Sim, scene_dir  # noqa: E402
from sim.scenes import SCENES, build  # noqa: E402


@pytest.mark.parametrize("name", sorted(SCENES))
def test_scene_builds_and_controller_runs(name):
    sim = Sim(name, seed=0, rebuild=True)
    robot = sim.robot
    bid = sim.m.body(robot["body"]).id
    p0 = sim.d.xpos[bid].copy()

    sim.step(int(6.0 / sim.m.opt.timestep))          # 6 s of simulated time

    assert np.isfinite(sim.d.qpos).all() and np.isfinite(sim.d.qvel).all()
    p = sim.d.xpos[bid]
    if robot["type"] == "quadrotor":
        agl = p[2] - float(sim.terrain.height(p[0], p[1]))
        assert agl > 1.0, f"UAV did not take off (AGL {agl:.2f} m)"
        assert np.linalg.norm(p[:2] - p0[:2]) > 2.0, "UAV did not follow its mission"
    elif robot["type"] == "rocker_bogie":
        assert np.linalg.norm(p[:2] - p0[:2]) > 2.0, "rover did not drive its loop"

def test_same_seed_is_byte_identical():
    def digest():
        build("mars_yard", 0)
        folder = scene_dir("mars_yard", 0)
        h = hashlib.md5()
        for f in sorted(folder.glob("*")):
            if f.suffix in {".xml", ".obj", ".bin", ".png", ".json"}:
                h.update(f.read_bytes())
        return h.hexdigest()

    assert digest() == digest()
