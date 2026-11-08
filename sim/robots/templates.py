"""Vehicle templates: build a common robot from its parts and its sizing result.

The sizing (sim/design/drone.py, rover.py: `evaluate`) decides the geometry from the parts
and the mission (arm length from the prop size, wheel spacing and track from the wheel size
and tip-over limits...). A template takes that result and assembles the robot from
components, so the simulated vehicle is exactly the one that was sized.

    from sim.design import drone
    cfg, design = drone.best("drone_forest.toml")            # or drone.evaluate(cfg, db, motor, prop, battery, tube)
    robot, meta = quadrotor("uav", cfg, design)
    print(robot.summary())

Each template returns (robot, meta): `meta` is what the runtime (plant, sensors, controller,
C++ core) reads from the scene's meta.json.
"""
from __future__ import annotations

from .. import parts
from ..builder import lookat_xyaxes
from ..components import (Baro, Battery, Board, CameraModule, Chassis, Computer, Gnss, Imu, Mag, Mast, Payload,
                          PropGuard, QuadFrame, RockerBogie, Rotor, SteeringServo, ViewCamera, WheelDrive)
from ..physics.battery import Pack
from .robot import Robot


def _site_environment(robot: Robot, cfg: dict) -> None:
    """Site conditions some components need (air density for rotors, magnetic field for the compass)."""
    robot.environment = {"air_density": cfg["mission"].get("air_density", 1.225),
                         "magnetic_field": dict(cfg.get("environment", {}))}


def _chosen(cfg: dict, kind: str) -> str | None:
    return cfg.get("sensors", {}).get(kind)


def quadrotor(name: str, cfg: dict, design, chase_offset=(-3.0, -1.8, 1.4)) -> tuple[Robot, dict]:
    """X quadrotor from a sizing result (design.drone.DroneDesign); sensors, their mounting
    and the policy computer come from the mission file's [sensors], [installation],
    [environment] and [compute] tables."""
    db = parts.load()
    g, m = design.geo, design.m
    inst = cfg.get("installation", {})
    robot = Robot(name)
    _site_environment(robot, cfg)
    frame = robot.add(QuadFrame(design.arm_tube, arm_length=g["L"], prop_d=g["D"],
                                guard_gap=g["cage_gap"] if g["cage"] else 0.0,
                                arm_mode_hz=m["arm_freq_hz"], arm_zeta=inst.get("arm_zeta", 0.02)))
    battery = robot.add(Battery(design.battery), at=frame.mount("deck"))
    # payload boards go where their part says ("mount" in sim/parts/electronics.toml)
    where = {"center": frame.mount("center"), "top": battery.mount("top"), "bottom": frame.mount("bottom"),
             "front": frame.mount("nose")}
    camera = None
    for item in g["payload"]:
        spot = db["avionics"][item]["mount"]
        if spot == "front":
            camera = robot.add(CameraModule(item, rgb=_chosen(cfg, "rgb_camera"), depth=_chosen(cfg, "depth_camera")),
                               at=where[spot])
        else:
            robot.add(Board(item), at=where[spot])
    for arm in frame.arms:
        robot.add(Rotor(design.motor, design.prop, arm.spin, thrust_max=m["t_max_n"], lag_s=m["tau_motor_s"],
                        motor_mount_g=db["frame"]["motor_mount_g"], esc_eff=db["frame"]["esc_efficiency"],
                        imbalance_gmm=inst.get("prop_imbalance_gmm", 0.5),
                        thrust_ripple=inst.get("thrust_ripple", 0.03)), at=arm.tip)
    if g["cage"]:
        robot.add(PropGuard(g["D"], g["cage_gap"], m["cage_kg"]), at=frame.mount("center"))
    # sensors (on the flight controller at the centre of mass, GNSS antenna on a short mast)
    robot.add(Imu(_chosen(cfg, "imu"), mount_hz=inst.get("imu_mount_hz", 1e4),
                  mount_zeta=inst.get("imu_mount_zeta", 0.15)), at=frame.mount("cg"))
    if _chosen(cfg, "baro"):
        robot.add(Baro(_chosen(cfg, "baro"), static_cp=inst.get("baro_static_cp", 0.3)), at=frame.mount("cg"))
    if _chosen(cfg, "mag"):
        robot.add(Mag(_chosen(cfg, "mag"), current_ut_per_a=inst.get("mag_current_ut_per_a", 0.0)),
                  at=frame.mount("origin"))
    if _chosen(cfg, "gnss"):
        robot.add(Gnss(_chosen(cfg, "gnss"), antenna_height=inst.get("gnss_antenna_height_m", 0.08)),
                  at=frame.mount("origin"))
    if cfg.get("compute", {}).get("part"):
        robot.add(Computer(cfg["compute"]["part"]))
    nadir = robot.add(ViewCamera("nadir", "0 -1 0 1 0 0", fovy=70), at=frame.mount("belly"))
    chase = robot.add(ViewCamera("chase", lookat_xyaxes(chase_offset), offset=chase_offset, mode="trackcom"),
                      at=frame.mount("origin"))

    # ---- runtime meta: power, sensors, compute and rotor drag come from the components ----
    rotors = robot.info["rotors"]
    meta = robot.runtime_meta()
    fpv = camera.rgb_camera if camera else nadir.camera
    meta.update(type="quadrotor", body=name, mass=robot.mass_properties()["root_mass"],
                actuators=[r.actuator for r in rotors], cameras=[chase.camera, fpv, nadir.camera],
                drag_CdA=(0.08 * m["span_m"] + 0.008) * (1.3 if g["cage"] else 1.0),
                design=dict(label=design.label, metrics=m))
    return robot, meta


def rocker_bogie(name: str, cfg: dict, design, steer_lock_deg: float | None = None) -> tuple[Robot, dict]:
    """Six-wheel rocker-bogie rover with four steered corners, from design.rover.RoverDesign."""
    from ..design.rover import STEER_LOCK_DEG
    lock = STEER_LOCK_DEG if steer_lock_deg is None else steer_lock_deg
    db = parts.load()
    g = design.geo
    bd = g["body"]
    inst = cfg.get("installation", {})
    pack = Pack(db["battery"][design.battery])
    ratio = pack.v_nom / db["drive_motor"][design.motor]["voltage"]
    r = g["r"]
    robot = Robot(name)
    _site_environment(robot, cfg)
    chassis = robot.add(Chassis(bd["length_m"], bd["width_m"], bd["height_m"], frame_tube=design.tube))
    center = chassis.mount("center")
    robot.add(Battery(design.battery, visible=False), at=center)          # inside the body box
    robot.add(Payload(bd["electronics_g"] / 1000, power_w=bd["avionics_w"], label="electronics"), at=center)
    robot.add(Payload(cfg["mission"]["payload_kg"], label="mission payload"), at=center)
    susp = robot.add(RockerBogie(r, g["w"], g["s"], g["H"], g["zb"], g["track"], bd["width_m"], bd["height_m"],
                                 design.tube), at=center)
    for side in "lr":
        for corner in (f"f{side}", f"m{side}", f"r{side}"):
            at = susp.mount(corner)
            if corner[0] in "fr":                          # front and rear corners steer
                servo = robot.add(SteeringServo(bd["steer_servo"], lock, wheel_r=r,
                                                fork_tube=susp.tube_geom if corner[0] == "f" else None), at=at)
                at = servo.mount(corner)
            robot.add(WheelDrive(design.wheel, design.motor, ratio), at=at)
    mast = robot.add(Mast(), at=chassis.mount("mast"))
    haz = robot.add(ViewCamera("hazcam", "0 -1 0 .707 0 .707", fovy=100), at=chassis.mount("nose_low"))
    robot.add(Imu(_chosen(cfg, "imu"), site_size=0.01), at=center)
    if _chosen(cfg, "mag"):
        robot.add(Mag(_chosen(cfg, "mag"), current_ut_per_a=inst.get("mag_current_ut_per_a", 0.0)), at=center)
    if _chosen(cfg, "gnss"):
        robot.add(Gnss(_chosen(cfg, "gnss"), antenna_height=inst.get("gnss_antenna_height_m", 0.08)), at=center)
    if cfg.get("compute", {}).get("part"):
        robot.add(Computer(cfg["compute"]["part"]))
    chase_offset = (-3.2, -2.2, 1.6)
    chase = robot.add(ViewCamera("chase", lookat_xyaxes(chase_offset), offset=chase_offset, mode="trackcom"),
                      at=chassis.mount("origin"))

    wheels, steers = robot.info["wheels"], robot.info["steers"]
    meta = robot.runtime_meta()
    meta.update(type="rocker_bogie", body=name, wheels=[w.body_name for w in wheels],
                wheel_xy=[list(map(float, w.xy)) for w in wheels], wheel_radius=r,
                drive=[w.actuator for w in wheels], steer=[s.actuator for s in steers],
                steer_xy={s.joint: list(map(float, s.xy)) for s in steers},
                omega0=wheels[0].omega0, tau_stall=wheels[0].tau_s,
                cameras=[chase.camera, mast.camera, haz.camera],
                design=dict(label=design.label, metrics=design.m))
    return robot, meta
