"""Your own scenes: every module here (except names starting with "_") is imported when
the scene catalogue loads, so a `@scene` function in it appears in `python -m sim list`.
Start from meadow_scout.py and docs/tutorial.md."""
import importlib
import pkgutil


def load_all() -> None:
    for mod in pkgutil.iter_modules(__path__):
        if not mod.name.startswith("_"):
            importlib.import_module(f"{__name__}.{mod.name}")
