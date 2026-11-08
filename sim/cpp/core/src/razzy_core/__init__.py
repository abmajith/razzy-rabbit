"""C++ simulation core. Import mujoco first so its libmujoco is the one we bind to."""
import mujoco as _mujoco  # noqa: F401

from ._core import *  # noqa: F401,F403
from ._core import mj_version as _mj_version

if _mj_version() != _mujoco.mj_version():
    raise ImportError(f"razzy_core was built against MuJoCo {_mj_version()}, but {_mujoco.mj_version()} is installed;"
                      " rebuild with: uv sync --extra sim --reinstall-package razzy-core")
