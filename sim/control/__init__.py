"""Chainable controllers (see base.py).

    path.py        FollowPath (multirotor route), PurePursuit (ground loop)
    multirotor.py  Se3Controller, Mixer, VelocityCommand (policy / joystick input)
    ground.py      Ackermann, WheelSpeedPI, DriveCommand (policy / joystick input)

    chain = FollowPath(route, cruise=3.0) >> Se3Controller() >> Mixer()
    meta["control"] = chain.spec()          # stored with the scene
    chain = from_spec(meta["control"])      # rebuilt by the runtime
"""
from .base import Chain, Controller, TruthState
from .ground import Ackermann, DriveCommand, WheelSpeedPI
from .multirotor import Mixer, Se3Controller, VelocityCommand
from .path import FollowPath, PurePursuit

STAGES = {c.name: c for c in (FollowPath, PurePursuit, Se3Controller, Mixer, VelocityCommand, Ackermann,
                              WheelSpeedPI, DriveCommand)}


def from_spec(spec: dict) -> Chain:
    """Rebuild a chain from its stored spec ({"source", "stages": [{"type", params...}]})."""
    stages = []
    for st in spec["stages"]:
        st = dict(st)
        kind = st.pop("type")
        if kind not in STAGES:
            raise KeyError(f"unknown control stage '{kind}'; known: {sorted(STAGES)}")
        stages.append(STAGES[kind](**st))
    return Chain(stages, spec.get("source", "truth"))


__all__ = ["Chain", "Controller", "TruthState", "Ackermann", "DriveCommand", "WheelSpeedPI", "Mixer", "Se3Controller",
           "VelocityCommand", "FollowPath", "PurePursuit", "STAGES", "from_spec"]
