"""sim/components + sim/robots: components attach at named mounts, masses add up, the
templates reproduce the sized vehicles, and a hand-assembled robot compiles."""
import mujoco
import pytest

from sim.builder import Builder
from sim.components import Battery, Board, Imu, QuadFrame, Rotor
from sim.design import drone, rover
from sim.robots import Robot, quadrotor, rocker_bogie


def _compile(robot: Robot, tmp_path) -> mujoco.MjModel:
    b = Builder("unit_robot", 0, out_root=tmp_path)
    b.world.append('<geom type="plane" size="5 5 .1"/>')
    robot.write(b, [0, 0, 0.3])
    return mujoco.MjModel.from_xml_path(str(b.write()))


def test_quad_template_matches_sizing():
    cfg, d = drone.best("drone_forest.toml")
    robot, meta = quadrotor("uav", cfg, d)
    assert robot.mass_properties()["root_mass"] == pytest.approx(d.m["mass_kg"], rel=1e-9)
    assert len(meta["actuators"]) == 4 and len(robot.info["rotors"]) == 4
    assert "deck" in robot.components[0].mounts() and "tip_FR" in robot.components[0].mounts()
    assert "uav" in robot.summary() and "Rotor" in robot.summary()


def test_rover_template_wheels_and_mass():
    cfg, d = rover.best("rover_yard.toml")
    robot, meta = rocker_bogie("rover", cfg, d)
    assert [w[:2] for w in meta["wheels"]] == ["ro"] * 6
    xs = sorted(round(x, 6) for x, _ in meta["wheel_xy"])
    assert xs[0] == pytest.approx(-d.geo["s"]) and xs[-1] == pytest.approx(d.geo["s"])
    assert all(abs(abs(y) - d.geo["track"] / 2) < 1e-9 for _, y in meta["wheel_xy"])
    assert len(meta["steer"]) == 4


def test_mounts_are_named_and_checked():
    robot = Robot("t")
    frame = robot.add(QuadFrame("cf_8x6", arm_length=0.1, prop_d=0.127))
    with pytest.raises(KeyError, match="mounts:"):
        frame.mount("roof")
    with pytest.raises(TypeError, match="mount"):
        robot.add(Battery("lipo_4s_1500"), at=(0, 0, 0.1))


def test_hand_built_quad_compiles_and_masses_add_up(tmp_path):
    robot = Robot("scout")
    frame = robot.add(QuadFrame("cf_8x6", arm_length=0.11, prop_d=0.127))
    m0 = frame.body.parts_mass()
    bat = robot.add(Battery("lipo_4s_1500"), at=frame.mount("deck"))
    robot.add(Board("fc_stack_h7"), at=frame.mount("center"))
    robot.add(Board("pi_zero2w"), at=bat.mount("top"))
    for arm in frame.arms:
        robot.add(Rotor("m2004_1700kv", "p5x3_bi", arm.spin, thrust_max=7.0, lag_s=0.02, motor_mount_g=4), at=arm.tip)
    robot.add(Imu(), at=frame.mount("cg"))
    parts_added = (bat.spec["mass_g"] + 25 + 11 + 4 * (20 + 4 + 3.5)) / 1000   # battery, FC, Pi, 4x motor+mount+prop
    assert frame.body.parts_mass() == pytest.approx(m0 + parts_added)
    m = _compile(robot, tmp_path)
    bid = m.body("scout").id
    assert m.body_mass[bid] == pytest.approx(frame.body.mass(), rel=1e-4)       # incl. wiring margin
    assert m.nu == 4 and m.sensor("scout_gyro").id >= 0
    # battery sits on the deck, its top mount is above it
    assert bat.mount("top").pos[2] > frame.mount("deck").pos[2]


def test_components_declare_power_loads_and_sensors():
    cfg, d = drone.best("drone_forest.toml")
    _, meta = quadrotor("uav", cfg, d)
    loads = meta["power"]["loads"]
    rot = [ld for ld in loads if ld["type"] == "rotor"]
    assert len(rot) == 1 and len(rot[0]["actuators"]) == 4            # identical rotors share one load
    avionics = sum(ld["watts"] for ld in loads if ld["type"] == "constant" and ld["group"] == "avionics")
    assert avionics == pytest.approx(d.m["avionics_w"])                 # the sizing's avionics budget
    types = [e["type"] for e in meta["sensors"]]
    assert sorted(types) == sorted(["imu", "baro", "mag", "gnss", "rgb_camera", "depth_camera"])
    imu = next(e for e in meta["sensors"] if e["type"] == "imu")
    assert imu["vibration"]["arm_hz"] == pytest.approx(d.m["arm_freq_hz"]) and len(imu["vibration"]["spin"]) == 4
    assert meta["compute"]["part"] == cfg["compute"]["part"]
    cfg, d = rover.best("rover_yard.toml")
    _, meta = rocker_bogie("rover", cfg, d)
    loads = meta["power"]["loads"]
    gear = [ld for ld in loads if ld["type"] == "gearmotor"]
    assert len(gear) == 1 and len(gear[0]["tau_s"]) == 6
    servos = [ld for ld in loads if ld["type"] == "constant" and ld["group"] == "servos"]
    assert len(servos) == 4
