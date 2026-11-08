"""Robot building blocks. Each component is a physical thing you could buy or build; it
adds bodies, geometry, mass, actuators and sensors to a Robot and publishes named mount
points for other components (see base.py).

Structure (create bodies):   QuadFrame, Chassis, RockerBogie, SteeringServo, WheelDrive
Devices (on a body):         Rotor, Battery, Payload, Board, CameraModule, Computer, Mast,
                             PropGuard, ViewCamera
Sensors:                     Imu, Baro, Mag, Gnss (cameras: CameraModule)
"""
from .base import Body, Component, MassPart, Mount, box_inertia, rod_inertia
from .chassis import Chassis, Mast
from .drive import SteeringServo, WheelDrive
from .electronics import Board, CameraModule, Computer
from .energy import Battery, Payload
from .frame import QUAD_X, PropGuard, QuadFrame
from .rocker_bogie import RockerBogie
from .rotor import Rotor
from .sensors import Baro, Gnss, Imu, Mag, ViewCamera

__all__ = ["Body", "Component", "MassPart", "Mount", "box_inertia", "rod_inertia", "Chassis", "Mast",
           "SteeringServo", "WheelDrive", "Board", "CameraModule", "Battery", "Payload", "QUAD_X", "PropGuard",
           "QuadFrame", "RockerBogie", "Rotor", "Imu", "ViewCamera", "Baro", "Gnss", "Mag", "Computer"]
