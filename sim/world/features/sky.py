"""Sky, sun and wind."""
from __future__ import annotations

import numpy as np

from ...builder import Lighting
from ..world import Feature


class Sun(Feature):
    """Random time of day: sun elevation 30-68 deg, random azimuth, overcast fraction
    (random unless given). Optional sky colours / haze override the sampled ones (e.g. a
    dusty Mars sky). `shadow_extent`: area [m] the sun's shadow map covers."""

    def __init__(self, overcast: float | None = None, shadow_extent: float = 60.0, sky_top=None, sky_horizon=None,
                 haze: float | None = None) -> None:
        self.overcast, self.shadow_extent = overcast, shadow_extent
        self.sky_top, self.sky_horizon, self.haze = sky_top, sky_horizon, haze

    def apply(self, w) -> None:
        light = Lighting.sample(w.rng, overcast=self.overcast)
        if self.sky_top is not None:
            light.sky_top = np.array(self.sky_top)
        if self.sky_horizon is not None:
            light.sky_horizon = np.array(self.sky_horizon)
        if self.haze is not None:
            light.haze = self.haze
        w.builder.sky_and_sun(light, shadow_extent=self.shadow_extent)
        self.light = light


class Wind(Feature):
    """Analytic wind field for the scene (physics/aero.py): mean speed [m/s] at height z_ref,
    log-law profile with roughness z0, turbulence ~ gust [m/s]; heading random unless given."""

    def __init__(self, speed: float, gust: float = 0.5, z_ref: float = 10.0, z0: float = 1.0,
                 heading_deg: float | None = None) -> None:
        self.speed, self.gust, self.z_ref, self.z0, self.heading = speed, gust, z_ref, z0, heading_deg

    def apply(self, w) -> None:
        heading = float(w.rng.uniform(0, 360)) if self.heading is None else float(self.heading)
        w.builder.meta["wind"] = dict(U=self.speed, heading_deg=heading, gust=self.gust, z_ref=self.z_ref, z0=self.z0)
