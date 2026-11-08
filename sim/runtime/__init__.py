"""Python reference engine: load a generated scene and step it.

    sim.py      Sim: controller -> power train -> physics step -> sensors, per step
    power.py    PowerSystem: battery + every component's electrical load, voltage sag fed back
    aero.py     RotorDrag: blade-flapping drag of rotors in relative wind
    sensors.py  SensorSuite: the sensor models of the robot's sensor components, with latency
"""
from .sim import Sim, load_terrain, scene_dir

__all__ = ["Sim", "load_terrain", "scene_dir"]
