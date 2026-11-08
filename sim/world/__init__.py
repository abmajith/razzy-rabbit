"""Composable environments: a World plus features applied in order (see world.py).

    from sim.world import World, Clear, Uniform, PowerLaw
    from sim.world.features import Sun, Hills, Trail, Ground, Forest, Rocks, ...

terrain.py (heightfields, textures), flora.py (tree / rock / grass geometry) and
sampling.py (placement, size distributions) are the building blocks the features use.
"""
from .sampling import Clear, PowerLaw, Uniform
from .world import Feature, World

__all__ = ["World", "Feature", "Clear", "PowerLaw", "Uniform"]
