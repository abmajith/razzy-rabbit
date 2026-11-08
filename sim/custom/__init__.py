"""Developer scenes & controllers.

Every module in this package is imported automatically (see the end of sim/scenes.py),
so a new file here that uses @scene / CONTROLLERS[...] shows up in `python -m sim list`
without touching the core modules. See docs/custom_scenes.md.
"""
import importlib
import pkgutil


def load_all() -> None:
    for mod in pkgutil.iter_modules(__path__):
        if not mod.name.startswith("_"):
            importlib.import_module(f"{__name__}.{mod.name}")
