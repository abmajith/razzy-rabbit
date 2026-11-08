"""Tutorial example (docs/tutorial.md): a small quad assembled from parts, flying a trail
over a hilly meadow with scattered trees.

    uv run --extra sim python -m sim view meadow_scout
"""
from __future__ import annotations

import numpy as np

from sim.builder import lookat_xyaxes
from sim.components import Battery, Board, CameraModule, Gnss, Imu, Mag, QuadFrame, Rotor, ViewCamera
from sim.control import FollowPath, Mixer, Se3Controller
from sim.robots import Robot
from sim.scenes import scene
from sim.world import Clear, Uniform, World
from sim.world.features import DRY_SOIL, GRASS, Forest, Grass, Ground, Hills, PlantColours, Rocks, Sun, Trail, Wind


def scout() -> Robot:
    """A 5-inch quad: frame, battery, flight controller, depth camera, four rotors, sensors."""
    r = Robot("scout")
    r.environment = {"air_density": 1.2,
                     "magnetic_field": {"field_ut": 47.5, "inclination_deg": 62, "declination_deg": 2.5}}
    frame = r.add(QuadFrame("cf_10x8", arm_length=0.11, prop_d=0.127))
    battery = r.add(Battery("lipo_4s_1500"), at=frame.mount("deck"))
    r.add(Board("fc_stack_h7"), at=frame.mount("center"))
    r.add(CameraModule("oakd_lite", rgb="oakd_lite_rgb", depth="oakd_lite_stereo"), at=frame.mount("nose"))
    for arm in frame.arms:                     # thrust limit and motor lag computed from the parts
        r.add(Rotor("m2004_1700kv", "p5x3_bi", arm.spin, motor_mount_g=4), at=arm.tip)
    r.add(Imu("icm42688p", mount_hz=90), at=frame.mount("cg"))
    r.add(Mag("ist8310", current_ut_per_a=0.15), at=frame.mount("origin"))
    r.add(Gnss("ublox_m10", antenna_height=0.05), at=battery.mount("top"))
    chase = (-2.5, 0.0, 1.0)
    r.add(ViewCamera("chase", lookat_xyaxes(chase), offset=chase, mode="trackcom"), at=frame.mount("origin"))
    return r


@scene
def meadow_scout(seed: int = 0):
    """Hand-assembled 5-inch quad flying a trail over a hilly meadow (tutorial example)."""
    w = World("meadow_scout", seed, half_size=40.0, resolution=321)
    w.add(PlantColours())
    w.add(Sun())
    w.add(Hills(relief=3.0, base=3))
    trail = w.add(Trail(amplitude=(8, 3), scale=(12, 5)))
    w.add(Ground(GRASS, overlays={trail: DRY_SOIL}))
    w.add(Forest(spacing=7.0, clear=[Clear(trail, 4.0)], mix=(("broadleaf", 0.7), ("birch", 0.3))))
    w.add(Rocks(count=40, size=Uniform(0.2, 0.6), clear=[Clear(trail, 1.0)]))
    w.add(Grass(count=1500))
    w.add(Wind(speed=2.0, gust=0.5))

    route = trail.path(np.linspace(-34, 34, 50), height_agl=3.0)
    w.add_robot(scout(), *route[0][:2], clearance=0.1, drag_CdA=0.03,
                control=FollowPath(route, cruise=4.0) >> Se3Controller() >> Mixer(),
                uncertainty={"mass_frac": 0.05, "thrust_coef": 0.08, "motor_lag": 0.3,
                             "imu_noise": [0.9, 1.5], "latency_ms_add": [0, 15], "wind_speed": [0, 4]})
    w.overview([-50, -45, 30], [0, 0, 0])
    return w.builder
