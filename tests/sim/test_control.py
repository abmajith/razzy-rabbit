"""sim/control: stages chain and round-trip through the scene meta, the state source is
explicit, policy-style command stages drive both vehicles, and the C++ core only takes
chains it really implements."""
import copy

import numpy as np
import pytest

pytest.importorskip("mujoco")

from sim.control import (Ackermann, DriveCommand, FollowPath, Mixer, PurePursuit, Se3Controller,  # noqa: E402
                         VelocityCommand, WheelSpeedPI, from_spec)
from sim.runtime import Sim  # noqa: E402


def test_chain_spec_roundtrip_and_checks():
    chain = FollowPath([[0, 0, 2], [10, 0, 2]], cruise=2.0) >> Se3Controller(kp=6.0) >> Mixer()
    spec = chain.spec()
    assert [s["type"] for s in spec["stages"]] == ["follow_path", "se3", "mixer"]
    assert from_spec(spec).spec() == spec
    with pytest.raises(TypeError, match="unknown parameters"):
        Se3Controller(kpp=1.0)
    with pytest.raises(KeyError, match="unknown control stage"):
        from_spec({"stages": [{"type": "autopilot"}]})


def test_sensor_source_needs_an_estimator():
    sim = Sim("forest_grounded", 0, controller=False, sensors=False)
    chain = (VelocityCommand() >> Se3Controller() >> Mixer()).using("sensors")
    with pytest.raises(NotImplementedError, match="estimator"):
        chain.attach(sim)


def test_velocity_command_flies_the_quad():
    sim = Sim("forest_grounded", 0, controller=False, sensors=False)
    cmd = VelocityCommand()
    sim.ctrls.insert(0, (cmd >> Se3Controller() >> Mixer()).attach(sim))   # before drag / power train
    sim.step(1500)                                                          # 3 s: holds / lifts off
    z0 = float(sim.d.qpos[2])
    cmd.command = np.array([0.0, 0.0, 1.0, 0.0])                            # climb at 1 m/s
    sim.step(1000)
    assert float(sim.d.qpos[2]) - z0 > 1.2


def test_drive_command_drives_the_rover():
    sim = Sim("mars_yard", 0, controller=False, sensors=False)
    cmd = DriveCommand()
    sim.ctrls.insert(0, (cmd >> Ackermann() >> WheelSpeedPI()).attach(sim))
    bid = sim.m.body("rover").id
    x0 = sim.d.xpos[bid][:2].copy()
    cmd.command = np.array([0.5, 0.0])
    sim.step(2500)                                                          # 5 s
    assert np.linalg.norm(sim.d.xpos[bid][:2] - x0) > 1.5


def test_core_takes_only_chains_it_implements():
    pytest.importorskip("razzy_core")
    from sim.core import Unsupported, core_chain
    meta = Sim("mars_yard", 0, sensors=False).meta0
    assert core_chain(meta)[0] == "rover"
    changed = copy.deepcopy(meta)
    changed["control"]["stages"][2]["kp"] = 2.0                            # WheelSpeedPI gain
    with pytest.raises(Unsupported, match="non-default"):
        core_chain(changed)
    other = copy.deepcopy(meta)
    other["control"] = (PurePursuit([[0, 0], [5, 0], [5, 5]]) >> Ackermann()).spec()
    with pytest.raises(Unsupported, match="no C\\+\\+ version"):
        core_chain(other)
