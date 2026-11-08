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
from ..components import (QUAD_X, Battery, Board, CameraModule, Chassis, Imu, Mast, Payload, PropGuard,  # noqa: F401
                          QuadFrame, RockerBogie, Rotor, SteeringServo, ViewCamera, WheelDrive)
from ..physics.battery import Pack
from .robot import Robot


def quadrotor(name: str, cfg: dict, design, chase_offset=(-3.0, -1.8, 1.4)) -> tuple[Robot, dict]:
    """X quadrotor from a sizing result (design.drone.DroneDesign)."""
    from ..sensing.suite import resolve
    db = parts.load()
    g, m = design.geo, design.m
    sensors = resolve(cfg, db)
    robot = Robot(name)
    frame = robot.add(QuadFrame(design.arm_tube, arm_length=g["L"], prop_d=g["D"],
                                guard_gap=g["cage_gap"] if g["cage"] else 0.0))
    battery = robot.add(Battery(design.battery), at=frame.mount("deck"))
    # payload boards go where their part says ("mount" in sim/parts/electronics.toml)
    where = {"center": frame.mount("center"), "top": battery.mount("top"), "bottom": frame.mount("bottom"),
             "front": frame.mount("nose")}
    camera = None
    for item in g["payload"]:
        spot = db["avionics"][item]["mount"]
        if spot == "front":
            camera = robot.add(CameraModule(item, rgb_vfov_deg=sensors.get("rgb_camera", {}).get("vfov_deg", 80),
                                            depth_vfov_deg=sensors.get("depth_camera", {}).get("vfov_deg", 58)),
                               at=where[spot])
        else:
            robot.add(Board(item), at=where[spot])
    for arm in frame.arms:
        robot.add(Rotor(design.motor, design.prop, arm.spin, thrust_max=m["t_max_n"], lag_s=m["tau_motor_s"],
                        motor_mount_g=db["frame"]["motor_mount_g"]), at=arm.tip)
    if g["cage"]:
        robot.add(PropGuard(g["D"], g["cage_gap"], m["cage_kg"]), at=frame.mount("center"))
    robot.add(Imu(), at=frame.mount("cg"))
    nadir = robot.add(ViewCamera("nadir", "0 -1 0 1 0 0", fovy=70), at=frame.mount("belly"))
    chase = robot.add(ViewCamera("chase", lookat_xyaxes(chase_offset), offset=chase_offset, mode="trackcom"),
                      at=frame.mount("origin"))

    # ---- runtime meta (read by plant, sensors, controller, C++ core) ----
    rotors = robot.info["rotors"]
    mp = robot.mass_properties()
    com, I = mp["root_com"], mp["root_inertia"]
    if camera is not None:
        if "rgb_camera" in sensors:
            sensors["rgb_camera"]["camera"] = camera.rgb_camera
        if "depth_camera" in sensors:
            sensors["depth_camera"]["camera"] = camera.depth_camera
    inst = sensors["installation"]
    mis = cfg["mission"]
    fpv = camera.rgb_camera if camera else nadir.camera
    meta = dict(type="quadrotor", body=name, mass=mp["root_mass"], actuators=[r.actuator for r in rotors],
                cameras=[chase.camera, fpv, nadir.camera],
                drag_CdA=(0.08 * m["span_m"] + 0.008) * (1.3 if g["cage"] else 1.0),
                design=dict(label=design.label, metrics=m),
                plant=dict(kind="drone", motor=db["motor"][design.motor], prop=db["prop"][design.prop],
                           battery=db["battery"][design.battery], rho=mis["air_density"],
                           esc_eff=db["frame"]["esc_efficiency"], esc_duty=db["frame"]["esc_max_duty"],
                           avionics_w=m["avionics_w"], t_max_ref=m["t_max_n"], rotor_drag=0.012),
                sensors=sensors,
                vibration=dict(rotor_pos=[list(r.hub - com) for r in rotors], spin=[r.spin for r in rotors],
                               mass=mp["root_mass"], inertia=I.tolist(),
                               blades=db["prop"][design.prop].get("blades", 2),
                               imbalance_gmm=inst.get("prop_imbalance_gmm", 0.5),
                               thrust_ripple=inst.get("thrust_ripple", 0.03),
                               arm_hz=m["arm_freq_hz"], arm_zeta=inst.get("arm_zeta", 0.02),
                               mount_hz=inst.get("imu_mount_hz", 1e4), mount_zeta=inst.get("imu_mount_zeta", 0.15)))
    return robot, meta


def rocker_bogie(name: str, cfg: dict, design, steer_lock_deg: float | None = None) -> tuple[Robot, dict]:
    """Six-wheel rocker-bogie rover with four steered corners, from design.rover.RoverDesign."""
    from ..design.rover import STEER_LOCK_DEG
    from ..sensing.suite import resolve
    lock = STEER_LOCK_DEG if steer_lock_deg is None else steer_lock_deg
    db = parts.load()
    g, m = design.geo, design.m
    bd = g["body"]
    pack = Pack(db["battery"][design.battery])
    ratio = pack.v_nom / db["drive_motor"][design.motor]["voltage"]
    r = g["r"]
    robot = Robot(name)
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
    robot.add(Imu(site_size=0.01), at=center)
    chase_offset = (-3.2, -2.2, 1.6)
    chase = robot.add(ViewCamera("chase", lookat_xyaxes(chase_offset), offset=chase_offset, mode="trackcom"),
                      at=chassis.mount("origin"))

    wheels, steers = robot.info["wheels"], robot.info["steers"]
    w0 = wheels[0]
    servo_spec = db["steer_servo"][bd["steer_servo"]]
    meta = dict(type="rocker_bogie", body=name, wheels=[w.body_name for w in wheels],
                wheel_xy=[list(map(float, w.xy)) for w in wheels], wheel_radius=r,
                drive=[w.actuator for w in wheels], steer=[s.actuator for s in steers],
                steer_xy={s.joint: list(map(float, s.xy)) for s in steers},
                omega0=w0.omega0, tau_stall=w0.tau_s,
                cameras=[chase.camera, mast.camera, haz.camera],
                design=dict(label=design.label, metrics=m),
                plant=dict(kind="rover", motor=db["drive_motor"][design.motor], battery=db["battery"][design.battery],
                           v_ratio=ratio, tau_stall=w0.tau_s, avionics_w=bd["avionics_w"], servo_w=servo_spec["power_w"]),
                sensors=resolve(cfg, db))
    return robot, meta
