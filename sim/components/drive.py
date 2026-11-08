"""Wheel drive (gearmotor + wheel) and corner steering servo.

WheelDrive: a wheel body spinning on a hinge, driven by a brushed DC gearmotor modelled as a
MuJoCo actuator with the motor's torque-speed line (physics.dc_motor):

    tau = tau_s u - (tau_s / omega_0) omega       u in [-1, 1] = duty
    (affine actuator: gain tau_s, bias -tau_s/omega_0 x joint velocity)

tau_s and omega_0 are the gearmotor's values at the actual pack voltage. The wheel + motor
mass (times the fastener margin) sits in the tyre geom. Names: `<robot>_w_<corner>` body,
`..._j` joint, `..._drive` actuator, `..._qd` wheel-speed sensor.

SteeringServo: a steering knuckle body turning on a vertical hinge, positioned by a servo
(MuJoCo position actuator with kp = stall torque / 0.25 rad). It publishes a `wheel` mount
where the WheelDrive goes. Names: `<robot>_steer_<corner>` body/joint, `..._p` actuator.
"""
from __future__ import annotations

from .. import parts
from ..physics import dc_motor
from .base import Component


class SteeringServo(Component):
    kind = "steering"

    def __init__(self, servo: str, lock_deg: float, wheel_r: float, fork_tube=None) -> None:
        """fork_tube: optional function (p0, p1) -> tube geom, adding the fork leg from the
        servo down to the wheel axle (today's rover has it on the front corners only)."""
        super().__init__()
        self.servo_name, self.spec, self.lock = servo, parts.get("steer_servo", servo), lock_deg
        self.wheel_r, self.fork_tube = wheel_r, fork_tube

    def attach(self, robot, at) -> None:
        corner = at.name
        self.joint = f"{robot.name}_steer_{corner}"
        lock = self.lock
        body = robot.new_body(self.joint, at.body, at.pos, pos_str=at.pos_str,
                              joint=f'<joint name="{self.joint}" type="hinge" axis="0 0 1" range="-{lock:.0f} {lock:.0f}" '
                                    'damping="0.5" armature="0.005"/>')
        r_wheel = self.wheel_r
        m = self.spec["mass_g"] / 1000
        body.geoms.append(f'<geom type="box" size=".022 .016 .02" pos="0 0 {0.6 * r_wheel + 0.02:.4f}" material="carbon" '
                          f'mass="{m:.4f}" contype="0" conaffinity="0"/>')
        if self.fork_tube:
            body.geoms.append(self.fork_tube([0, 0, 0.6 * r_wheel], [0, 0, 0.02]))
        tq = self.spec["stall_torque"]
        self.actuator = f"{self.joint}_p"
        robot.actuators.append(f'<position name="{self.actuator}" joint="{self.joint}" kp="{tq / 0.25:.2f}" '
                               f'kv="0.3" ctrlrange="-0.7 0.7" forcerange="{-tq} {tq}"/>')
        self.xy = body.world_offset()[:2]
        robot.info.setdefault("steers", []).append(self)
        self.publish(corner, body, [0, 0, 0])          # the wheel mount keeps the corner name

    def contribute(self, robot, meta: dict) -> None:
        meta["power"]["loads"].append({"type": "constant", "group": "servos", "label": self.servo_name,
                                       "watts": self.spec["power_w"]})

    def summary(self) -> str:
        return f"SteeringServo: {self.servo_name}, +-{self.lock:.0f} deg"


class WheelDrive(Component):
    kind = "wheel"

    def __init__(self, wheel: str, gearmotor: str, voltage_ratio: float, margin: float = 1.08) -> None:
        super().__init__()
        self.wheel_name, self.motor_name = wheel, gearmotor
        self.wheel, self.motor = parts.get("wheel", wheel), parts.get("drive_motor", gearmotor)
        self.ratio, self.margin = voltage_ratio, margin
        gm = dc_motor.gearmotor_at_voltage(self.motor, voltage_ratio)
        self.tau_s, self.omega0 = gm["tau_s"], gm["omega_0"]

    def attach(self, robot, at) -> None:
        corner = at.name
        wn = f"{robot.name}_w_{corner}"
        self.body_name, self.joint = wn, f"{wn}_j"
        r, w = self.wheel["diameter_mm"] / 2000, self.wheel["width_mm"] / 1000
        m = (self.wheel["mass_g"] / 1000 + self.motor["mass_g"] / 1000) * self.margin
        body = robot.new_body(wn, at.body, at.pos, pos_str=at.pos_str or "0 0 0",
                              joint=f'<joint name="{self.joint}" type="hinge" axis="0 1 0" damping="0.05" armature="0.01"/>')
        body.geoms += [f'<geom name="{wn}_tire" type="cylinder" size="{r:.4f} {w/2:.4f}" euler="90 0 0" material="rubber" '
                       f'mass="{m:.4f}" friction="1.0 0.02 0.002" condim="3"/>',
                       f'<geom type="cylinder" size="{r*0.6:.4f} {w/2+.002:.4f}" euler="90 0 0" material="alu" mass="0" '
                       'contype="0" conaffinity="0"/>',
                       *[f'<geom type="box" size=".006 {w/2+.003:.4f} {r*0.95:.4f}" euler="0 {k*30} 0" material="rubber" '
                         'mass="0" contype="0" conaffinity="0"/>' for k in range(6)]]          # grousers (visual)
        tau_s, w0 = self.tau_s, self.omega0
        self.actuator = f"{wn}_drive"
        robot.actuators.append(
            f'<general name="{self.actuator}" joint="{self.joint}" gaintype="fixed" gainprm="{tau_s:.4f}" '
            f'biastype="affine" biasprm="0 0 {-tau_s / w0:.5f}" ctrlrange="-1 1" forcerange="{-tau_s:.4f} {tau_s:.4f}"/>')
        robot.sensors.append(f'<jointvel name="{wn}_qd" joint="{self.joint}"/>')
        self.xy = body.world_offset()[:2]
        self.radius = r
        robot.info.setdefault("wheels", []).append(self)

    def contribute(self, robot, meta: dict) -> None:
        loads = meta["power"]["loads"]
        group = next((ld for ld in loads if ld["type"] == "gearmotor" and ld["motor_name"] == self.motor_name), None)
        if group is None:
            group = {"type": "gearmotor", "motor_name": self.motor_name, "actuators": [], "motor": self.motor,
                     "v_ratio": self.ratio, "tau_s": []}
            loads.append(group)
        group["actuators"].append(self.actuator)
        group["tau_s"].append(self.tau_s)

    def summary(self) -> str:
        return (f"WheelDrive: {self.wheel_name} + {self.motor_name} "
                f"(stall {self.tau_s:.1f} N m, no-load {self.omega0:.1f} rad/s at {self.ratio:.2f} x rated V)")
