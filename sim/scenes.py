"""The scene catalogue: the grounded scenes, whose vehicles are sized from real parts.
Each scene is a short composition of world features (sim/world) plus a sized vehicle
(sim/robots). It is deterministic in (seed) and writes
sim/generated/<scene>-s<seed>/{scene.xml, meta.json, terrain.bin, ground.png, mesh_*.obj}.

  forest_grounded  mixed forest on steep terrain, trail flown by the quad sized from
                   design/configs/drone_forest.toml
  mars_yard        rocky test yard driven by the rocker-bogie rover sized from rover_yard.toml

Scene files only *describe* a scene; building it takes seconds, and everything that runs
afterwards (viewer, render, training) is the C++ core loading the written files.
"""
from __future__ import annotations

import numpy as np

from .builder import Builder
from .world import Clear, PowerLaw, Uniform, World
from .world.features import (DRY_SOIL, FOREST_FLOOR, MARS_SOIL, Craters, DriveLoop, Forest, Grass, Ground, Hills,
                             Lander, Logs, PlantColours, Rocks, Sun, Trail, Wind)

SCENES: dict[str, callable] = {}


def scene(fn):
    SCENES[fn.__name__] = fn
    return fn


@scene
def forest_grounded(seed: int = 0) -> Builder:
    """Forest trail flown by the quad sized from sim/design/configs/drone_forest.toml."""
    from .design import drone as drone_design
    w = World("forest_grounded", seed, half_size=70.0, resolution=561)
    w.add(PlantColours())
    w.add(Sun(shadow_extent=40))
    w.add(Hills(relief=9.0, octaves=6, base=2))
    trail = w.add(Trail(amplitude=(16, 6), scale=(17, 6.5), cut_depth=0.25))
    w.add(Ground(FOREST_FLOOR, overlays={trail: DRY_SOIL}))
    # wide crowns keep the flight corridor open: conifers 5.5 m, broadleaf 4 m from the trail
    w.add(Forest(spacing=3.3, clear=[Clear(trail, 2.6)], narrow_near=(trail, {"conifer": 5.5, "broadleaf": 4.0})))
    w.add(Rocks(count=90, size=Uniform(0.3, 1.2), clear=[Clear(trail, 1.5)]))
    w.add(Logs(count=40, length=Uniform(3, 7), clear=[Clear(trail, 4.0)]))
    w.add(Grass(count=3500))
    w.add(Wind(speed=1.5, gust=0.5, z_ref=10, z0=1.0))

    route = trail.path(np.linspace(-62, 62, 90), height_agl=2.0)
    cfg, best = drone_design.best("drone_forest.toml")
    b = w.builder
    drone_design.to_mjcf(b, "uav", w.ground_point(*route[0][:2], clearance=0.1), best, cfg, chase_offset=(-2.5, 0.0, 1.0))
    w.overview([-82, -78, 50], [0, 0, 0])
    b.meta.update(controller="quad_waypoints", waypoints_agl=route, cruise=3.0)
    return b


@scene
def mars_yard(seed: int = 0) -> Builder:
    """Rocky test yard (JPL Mars Yard style) for the rocker-bogie rover sized from rover_yard.toml."""
    from .design import rover as rover_design
    cfg, best = rover_design.best("rover_yard.toml")
    w = World("mars_yard", seed, half_size=40.0, resolution=401)
    w.add(Sun(overcast=0.1, shadow_extent=30, sky_top=[0.55, 0.45, 0.40], sky_horizon=[0.85, 0.68, 0.52], haze=0.45))
    w.add(Hills(relief=1.2, octaves=6, base=2))
    w.add(Hills(relief=0.08, octaves=3, base=24))                     # small-scale roughness
    w.add(Craters(count=7, extent=30, radius=(2.5, 7), clear_radius=6))
    w.add(Ground(MARS_SOIL))
    loop = w.add(DriveLoop(corners=8, radius=(9, 14), start=(0.0, -3.0)))
    lander = Lander(pos=(0.0, 3.0))
    # power-law rock field; boulders the rover can't climb keep off the loop, all keep off the lander
    w.add(Rocks(count=900, size=PowerLaw(0.04, 0.09, a=2.0, cap=1.8), sink=0.3, colour=[0.42, 0.26, 0.18],
                clear=[Clear(loop, 2.2, larger_than=0.6 * best.m["climb_m"]), Clear(lander, 4.0)]))
    w.add(lander)

    b = w.builder
    start = w.ground_point(*loop.start, clearance=best.geo["H"] + 0.05)
    rover_design.to_mjcf(b, "rover", start, best, cfg, yaw=0.0)
    w.overview([-22, -26, 14], [0, 0, 0])
    b.meta.update(controller="rover_ackermann", waypoints=loop.path(), speed=cfg["mission"]["cruise_speed_mps"])
    return b


def build(name: str, seed: int = 0):
    if name not in SCENES:
        raise SystemExit(f"unknown scene {name!r}; choose from {', '.join(SCENES)}")
    b = SCENES[name](seed)
    return b.write()
