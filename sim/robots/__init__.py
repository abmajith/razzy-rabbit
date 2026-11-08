"""Assemble components into robots (robot.py) and build common vehicles from their sizing
result (templates.py: quadrotor, rocker_bogie)."""
from .robot import Robot
from .templates import quadrotor, rocker_bogie

__all__ = ["Robot", "quadrotor", "rocker_bogie"]
