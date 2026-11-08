"""World features, applied in the order they are added to a World.

terrain shapes   Hills, Craters                       (edit the height grid)
paths            Trail, DriveLoop                     (regions to keep clear of + mission routes)
ground           Ground (+ palettes)                  (freezes the terrain, paints it)
vegetation       PlantColours, Forest, Grass
objects          Rocks, Logs, Lander
sky              Sun, Wind
"""
from .ground import DRY_SOIL, FOREST_FLOOR, GRASS, MARS_SOIL, Ground
from .objects import Lander, Logs, Rocks
from .paths import DriveLoop, Trail
from .sky import Sun, Wind
from .terrain_shapes import Craters, Hills
from .vegetation import Forest, Grass, PlantColours

__all__ = ["DRY_SOIL", "FOREST_FLOOR", "GRASS", "MARS_SOIL", "Ground", "Lander", "Logs", "Rocks", "DriveLoop",
           "Trail", "Sun", "Wind", "Craters", "Hills", "Forest", "Grass", "PlantColours"]
