"""Pick the rendering GPU before any OpenGL context exists.

On hybrid machines (e.g. AMD iGPU driving the display + Intel Arc / NVIDIA card) Mesa
defaults to the display GPU for both GLFW (window) and EGL (headless). We prefer a
discrete card: NVIDIA > Intel > AMD among cards that are not the boot VGA device.

Override with  RAZZY_GPU=igpu  (use the display GPU) or  RAZZY_GPU=<egl device id>.
Must be called before mujoco creates a GL context (i.e. before Renderer / viewer).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

VENDORS = {"0x10de": "NVIDIA", "0x8086": "Intel", "0x1002": "AMD"}
PREF = ["0x10de", "0x8086", "0x1002"]
CACHE = Path(__file__).resolve().parent / "generated" / ".gpu.json"

_PROBE = """
from mujoco.gl_context import GLContext
c = GLContext(16, 16); c.make_current()
from OpenGL import GL
print(GL.glGetString(GL.GL_RENDERER).decode())
"""


def discrete_card() -> dict | None:
    cards = []
    for c in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        dev = c / "device"
        try:
            vendor = (dev / "vendor").read_text().strip()
            boot = (dev / "boot_vga").read_text().strip() if (dev / "boot_vga").exists() else "0"
        except OSError:
            continue
        if boot == "1" or vendor not in VENDORS:
            continue
        pci = dev.resolve().name                               # 0000:03:00.0
        cards.append(dict(vendor=vendor, name=VENDORS[vendor],
                          prime="pci-" + pci.replace(":", "_").replace(".", "_")))
    cards.sort(key=lambda c: PREF.index(c["vendor"]))
    return cards[0] if cards else None


def _egl_renderer(i: int) -> str:
    env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID=str(i))
    try:
        r = subprocess.run([sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, timeout=20)
        return r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else ""
    except (subprocess.TimeoutExpired, IndexError):
        return ""


def _egl_device_for(name: str) -> int | None:
    try:
        cache = json.loads(CACHE.read_text())
        if cache.get("name") == name:
            return cache["egl_device"]
    except (OSError, ValueError, KeyError):
        pass
    for i in range(6):
        rend = _egl_renderer(i)
        if not rend:
            break
        if name.lower() in rend.lower() and "llvmpipe" not in rend:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_text(json.dumps({"name": name, "egl_device": i, "renderer": rend}))
            return i
    return None


def select(backend: str) -> str:
    """Configure env vars for `backend` ('glfw' or 'egl'). Returns a description."""
    choice = os.environ.get("RAZZY_GPU", "auto")
    if choice == "igpu":
        return "display GPU (RAZZY_GPU=igpu)"
    if backend == "egl":
        if choice.isdigit():
            os.environ["MUJOCO_EGL_DEVICE_ID"] = choice
            return f"EGL device {choice} (RAZZY_GPU)"
        if "MUJOCO_EGL_DEVICE_ID" in os.environ:
            return f"EGL device {os.environ['MUJOCO_EGL_DEVICE_ID']} (MUJOCO_EGL_DEVICE_ID)"
        card = discrete_card()
        if card:
            i = _egl_device_for(card["name"])
            if i is not None:
                os.environ["MUJOCO_EGL_DEVICE_ID"] = str(i)
                return f"{card['name']} discrete GPU (EGL device {i})"
        return "default EGL device"
    # glfw / windowed: Mesa PRIME offload by PCI tag
    if "DRI_PRIME" in os.environ:
        return f"DRI_PRIME={os.environ['DRI_PRIME']}"
    card = discrete_card()
    if card:
        os.environ["DRI_PRIME"] = card["prime"]
        if card["vendor"] == "0x10de":   # NVIDIA proprietary driver uses its own offload vars
            os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
            os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
        return f"{card['name']} discrete GPU ({card['prime']})"
    return "default GPU"
