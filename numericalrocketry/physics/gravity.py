"""WGS84 gravity model.

The reference simulation this project targets defaults to (and this
project's own reference flight implicitly uses, since its saved simulation
conditions have no gravity override) the WGS84 model rather than a constant
9.80665 m/s^2 -- gravity varies about +/-0.5% with latitude and falls off
slightly with altitude, which is a real, if small, contributor to any
apogee/timing gap against the reference flight.
"""

import math

# Mean earth radius (m), used for the altitude correction. This assumes a
# spherical earth, a small additional error but consistent with the model
# being reproduced.
_REARTH_M = 6371000.0


def wgs84_gravity(altitude_msl_m: float, latitude_deg: float) -> float:
    """Latitude- and altitude-corrected gravitational acceleration (m/s^2).

    `altitude_msl_m` is height above sea level (not launch-site AGL).
    """
    latitude_rad = math.radians(latitude_deg)
    sin2lat = math.sin(latitude_rad) ** 2
    g0 = 9.7803267714 * (1.0 + 0.00193185138639 * sin2lat) / math.sqrt(1.0 - 0.00669437999013 * sin2lat)
    return g0 * (_REARTH_M / (_REARTH_M + altitude_msl_m)) ** 2
