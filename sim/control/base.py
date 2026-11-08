"""Controllers as chainable stages.

A stage takes the output of the stage before it and produces its own:

    FollowPath(route) >> Se3Controller() >> Mixer()
       pose setpoint  ->   body wrench   ->  rotor thrusts (written to the actuators)

Every stage implements

    bind(sim, robot)              look up actuators / bodies, read the *nominal* model
                                  (what a real controller knows: the designed mass and inertia)
    update(t, dt, state, x) -> y  x: previous stage's output (None for the first stage)
    spec() -> dict                parameters, so a scene can store the chain in its meta

and stages chain with `>>`. A chain reads the vehicle state from one **source**:

    "truth"    exact MuJoCo state (what the reference controllers use today)
    "sensors"  estimated from the sensor models (needs a state estimator: not built yet)

That switch is the sim-to-real boundary made explicit: a controller that only works on
"truth" has not yet faced the real sensors. An RL policy replaces the first stage(s):
`policy >> Se3Controller() >> Mixer()` (see VelocityCommand / DriveCommand).
"""
from __future__ import annotations

import numpy as np


class TruthState:
    """Exact state of the robot's main body from MuJoCo (views, no copies)."""

    def __init__(self, sim, robot: dict) -> None:
        m = sim.m
        self.sim, self.m, self.d = sim, m, sim.d
        self.bid = m.body(robot["body"]).id
        jnt = m.body_jntadr[self.bid]
        self.qa, self.va = m.jnt_qposadr[jnt], m.jnt_dofadr[jnt]

    @property
    def pos(self):                      # world position of the free joint (body origin)
        return self.d.qpos[self.qa:self.qa + 3]

    @property
    def vel(self):                      # world linear velocity
        return self.d.qvel[self.va:self.va + 3]

    @property
    def omega(self):                    # body-frame angular velocity
        return self.d.qvel[self.va + 3:self.va + 6]

    @property
    def R(self):                        # body orientation (rotation matrix, body -> world)
        return self.d.xmat[self.bid].reshape(3, 3)

    @property
    def xy(self):                       # body origin in the horizontal plane
        return self.d.xpos[self.bid][:2]

    @property
    def yaw(self) -> float:
        R = self.R
        return np.arctan2(R[1, 0], R[0, 0])

    def joint_velocity(self, dofs):
        return self.d.qvel[dofs]


SOURCES = {"truth": TruthState}


class Controller:
    """Base class of a stage."""

    name = "stage"
    defaults: dict = {}

    def __init__(self, **params) -> None:
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise TypeError(f"{type(self).__name__}: unknown parameters {sorted(unknown)}; known: {sorted(self.defaults)}")
        self.params = {**self.defaults, **params}

    def bind(self, sim, robot: dict) -> None:
        pass

    def update(self, t: float, dt: float, state, x):          # pragma: no cover - abstract
        raise NotImplementedError

    def spec(self) -> dict:
        return {"type": self.name, **self.params}

    def __rshift__(self, other: "Controller") -> "Chain":
        return Chain([self, other])


class Chain(Controller):
    """Stages run in order every physics step; reads state from `source`."""

    name = "chain"

    def __init__(self, stages, source: str = "truth") -> None:
        self.stages = list(stages)
        self.source = source
        self.params = {}

    def __rshift__(self, other: Controller) -> "Chain":
        return Chain(self.stages + (other.stages if isinstance(other, Chain) else [other]), self.source)

    def using(self, source: str) -> "Chain":
        """Same stages, other state source ("truth" | "sensors")."""
        return Chain(self.stages, source)

    def spec(self) -> dict:
        return {"source": self.source, "stages": [s.spec() for s in self.stages]}

    def attach(self, sim) -> "Chain":
        """Bind every stage to a running Sim (call before randomising the model)."""
        if self.source not in SOURCES:
            raise NotImplementedError(f"state source '{self.source}' needs a state estimator, which is not built "
                                      f"yet; available: {list(SOURCES)}")
        robot = sim.robot
        self.state = SOURCES[self.source](sim, robot)
        for s in self.stages:
            s.bind(sim, robot)
        return self

    def update(self, t: float, dt: float, state=None, x=None):
        state = state or self.state
        for s in self.stages:
            x = s.update(t, dt, state, x)
        return x

    @property
    def err(self) -> float:
        """Tracking error of the first stage that reports one (path following)."""
        return next((s.err for s in self.stages if hasattr(s, "err")), 0.0)

    def stage(self, name: str) -> Controller:
        return next(s for s in self.stages if s.name == name)
