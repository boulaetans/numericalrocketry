"""Barrowman-style drag decomposition.

Skin friction, nose/base pressure, wave, and per-fin drag components, combined
into a single reference-area-normalized Cd by total_drag_coefficient().
"""

import math

from .atmosphere import air_viscosity, mach_number, reynolds_number, air_density
from ..rocket.geometry import RocketGeometry, fin_mac_properties, fin_planform_area_m2


# Surface-finish roughness heights, in meters.
ROUGHNESS_HEIGHT_M = {
    "rough": 500e-6,
    "roughunfinished": 250e-6,
    "unfinished": 150e-6,
    "normal": 60e-6,
    "smooth": 20e-6,
    "polished": 2e-6,
    "finishpolished": 0.5e-6,
}


def _roughness_correction(mach: float) -> float:
    """Compressibility correction for the roughness-limited Cf, ported from
    BarrowmanDragCalculator.calculateRoughnessCorrection()."""
    if mach < 0.9:
        return 1.0 - 0.1 * mach ** 2
    if mach > 1.1:
        return 1.0 / (1.0 + 0.18 * mach ** 2)
    c1 = 1.0 - 0.1 * 0.9 ** 2
    c2 = 1.0 / (1.0 + 0.18 * 1.1 ** 2)
    return c2 * (mach - 0.9) / 0.2 + c1 * (1.1 - mach) / 0.2


def skin_friction_coefficient(
    reynolds: float,
    mach: float,
    surface_finish: str = "normal",
    reference_length_m: float = 1.0,
) -> float:
    reynolds = max(reynolds, 1.0)
    mach = max(0.0, mach)
    perfect_finish = surface_finish in ("polished", "finishpolished")

    if perfect_finish:
        if reynolds < 1.0e4:
            cf = 1.33e-2
        elif reynolds < 5.39e5:
            cf = 1.328 / math.sqrt(reynolds)
        else:
            cf = 1.0 / (1.50 * math.log(reynolds) - 5.6) ** 2 - 1700.0 / reynolds

        c1 = 1.0
        c2 = 1.0
        if mach < 1.1 and reynolds > 1.0e6:
            if reynolds < 3.0e6:
                c1 = 1.0 - 0.1 * mach ** 2 * (reynolds - 1.0e6) / 2.0e6
            else:
                c1 = 1.0 - 0.1 * mach ** 2
        if mach > 0.9 and reynolds > 1.0e6:
            if reynolds < 3.0e6:
                c2 = 1.0 + ((1.0 / (1.0 + 0.045 * mach ** 2) ** 0.25) - 1.0) * (reynolds - 1.0e6) / 2.0e6
            else:
                c2 = 1.0 / (1.0 + 0.045 * mach ** 2) ** 0.25
    else:
        if reynolds < 1.0e4:
            cf = 1.48e-2
        else:
            cf = 1.0 / (1.50 * math.log(reynolds) - 5.6) ** 2

        c1 = 1.0 - 0.1 * mach ** 2 if mach < 1.1 else 1.0
        c2 = 1.0 / (1.0 + 0.15 * mach ** 2) ** 0.58 if mach > 0.9 else 1.0

    if mach < 0.9:
        cf *= c1
    elif mach < 1.1:
        cf *= c2 * (mach - 0.9) / 0.2 + c1 * (1.1 - mach) / 0.2
    else:
        cf *= c2

    roughness_height_m = ROUGHNESS_HEIGHT_M.get(surface_finish, ROUGHNESS_HEIGHT_M["normal"])
    cf_roughness = 0.032 * (roughness_height_m / max(reference_length_m, 1e-9)) ** 0.2 * _roughness_correction(mach)

    if perfect_finish:
        if reynolds > 1.0e6 and cf_roughness > cf:
            return cf_roughness
        return cf
    return max(cf, cf_roughness)


def wave_drag_component(geometry: RocketGeometry, mach: float) -> float:
    if mach <= 1.0:
        return 0.0
    projected_area = geometry.base_area_m2
    slenderness = max(geometry.fineness_ratio, 3.0)
    if mach < 1.5:
        k_wave = 0.18 * (1.0 + 0.5 * (mach - 1.0))
    else:
        k_wave = 0.18 * (1.25 + 0.1 / (mach - 1.0))
    slender_factor = 1.0 / slenderness
    if geometry.fineness_ratio > 12.0:
        slender_factor *= 0.8
    return k_wave * (projected_area / geometry.reference_area_m2) * slender_factor * math.sqrt(max(0.0, mach ** 2 - 1.0))


def nose_pressure_drag_coefficient(geometry: RocketGeometry, mach: float) -> float:
    # sinphi is the LOCAL slope discontinuity at the nose/body-tube junction
    # (sampled at 99% of nose length), not an overall half-angle.
    # Smooth-blending shapes (ellipse, tangent ogive, von Karman) have sinphi
    # near zero even though their overall shape isn't.
    sinphi = geometry.nose_sinphi()
    if geometry.nose_type in ("von_karman", "ogive"):
        base = 0.06
    elif geometry.nose_type == "conical":
        base = 0.10
    elif geometry.nose_type == "elliptical":
        base = 0.08
    elif geometry.nose_type == "hemisphere":
        base = 0.25
    else:
        base = 0.12

    c_pres_m0 = 0.8 * sinphi ** 2
    if mach < 0.8:
        return c_pres_m0
    if mach < 1.0:
        return 0.5 * base * mach ** 4 + c_pres_m0
    if geometry.nose_type == "conical" and geometry.nose_length_m > 0.0:
        # Supersonic conical/ogive formula (M >= 1.32 branch):
        #   Cd = 2.1*sinphi^2 + 0.5*sinphi/sqrt(M^2-1)
        # which DIVIDES by sqrt(M^2-1), correctly plateauing toward 2.1*sinphi^2
        # as M grows (an earlier version of this project multiplied instead,
        # the wrong physical trend).
        beta = math.sqrt(max(mach ** 2 - 1.0, 1e-9))
        return 2.1 * sinphi ** 2 + 0.5 * sinphi / beta
    return base + 0.3 * math.sqrt(max(0.0, mach ** 2 - 1.0))


def _base_pressure_coefficient(mach: float) -> float:
    """Base-drag Cd(M): purely Mach-dependent, no powered/unpowered branching."""
    if mach <= 1.0:
        return 0.12 + 0.13 * mach ** 2
    return 0.25 / mach


def _stagnation_cd(mach: float) -> float:
    """Stagnation-pressure Cd(M) for a blunt/square fin leading edge."""
    if mach <= 1.0:
        pressure = 1.0 + mach ** 2 / 4.0 + mach ** 4 / 40.0
    else:
        pressure = 1.84 - 0.76 / mach ** 2 + 0.166 / mach ** 4 + 0.035 / mach ** 6
    return 0.85 * pressure


def base_drag_component(geometry: RocketGeometry, powered: bool, mach: float) -> float:
    # `powered` kept for call-site compatibility; the base-drag formula has
    # no engine-state dependence, so it is unused here.
    coefficient = _base_pressure_coefficient(mach)
    return coefficient * (geometry.base_area_m2 / geometry.reference_area_m2)


def fin_friction_drag_coefficient(geometry: RocketGeometry, cf: float) -> float:
    """Fin friction drag: cf * (1 + 2*thickness/MAC) * 2*finArea/refArea, summed over N fins."""
    if geometry.fin_count <= 0:
        return 0.0
    area_per_fin = fin_planform_area_m2(geometry)
    mac_length, _mac_lead, _cos_gamma_lead, _cos_gamma_mid = fin_mac_properties(geometry)
    if area_per_fin <= 0.0 or mac_length <= 0.0:
        return 0.0
    per_fin = cf * (1.0 + 2.0 * geometry.fin_thickness_m / mac_length) * 2.0 * area_per_fin / geometry.reference_area_m2
    return geometry.fin_count * per_fin


def fin_pressure_drag_coefficient(geometry: RocketGeometry, mach: float) -> float:
    """Fin leading-edge pressure drag."""
    if geometry.fin_count <= 0 or geometry.fin_span_m <= 0.0 or geometry.fin_thickness_m <= 0.0:
        return 0.0

    cross_section = geometry.fin_cross_section
    if cross_section in ("airfoil", "rounded"):
        if mach < 0.9:
            cd = (1.0 - mach ** 2) ** -0.417 - 1.0
        elif mach < 1.0:
            cd = 1.0 - 1.785 * (mach - 0.9)
        else:
            cd = 1.214 - 0.502 / mach ** 2 + 0.1095 / mach ** 4
    else:
        # Square/blunt leading edge: real stagnation-pressure coefficient.
        cd = _stagnation_cd(mach)

    _mac_length, _mac_lead, cos_gamma_lead, _cos_gamma_mid = fin_mac_properties(geometry)
    cd *= cos_gamma_lead ** 2

    per_fin = cd * geometry.fin_span_m * geometry.fin_thickness_m / geometry.reference_area_m2
    return geometry.fin_count * per_fin


def fin_base_drag_coefficient(geometry: RocketGeometry, powered: bool, mach: float) -> float:
    """Fin trailing-edge base drag."""
    if geometry.fin_count <= 0 or geometry.fin_span_m <= 0.0 or geometry.fin_thickness_m <= 0.0:
        return 0.0

    base_cd = _base_pressure_coefficient(mach)
    cross_section = geometry.fin_cross_section
    if cross_section == "square":
        cd = base_cd
    elif cross_section == "rounded":
        cd = base_cd / 2.0
    else:
        cd = 0.0  # airfoil: assumed zero base drag

    per_fin = cd * geometry.fin_span_m * geometry.fin_thickness_m / geometry.reference_area_m2
    return geometry.fin_count * per_fin


def total_drag_coefficient(
    speed_m_s: float,
    altitude_m: float,
    geometry: RocketGeometry,
    surface_finish: str,
    powered: bool,
) -> float:
    if abs(speed_m_s) < 1e-8:
        return 0.0

    density = air_density(altitude_m)
    viscosity = air_viscosity(altitude_m)
    mach = mach_number(speed_m_s, altitude_m)
    reynolds = reynolds_number(density, speed_m_s, geometry.length_m, viscosity)

    cf = skin_friction_coefficient(reynolds, mach, surface_finish, reference_length_m=geometry.length_m)

    body_drag = cf * geometry.body_form_factor * (geometry.wetted_body_area_m2 / geometry.reference_area_m2)
    fin_friction_drag = fin_friction_drag_coefficient(geometry, cf)
    fin_pressure_drag = fin_pressure_drag_coefficient(geometry, mach)
    fin_base_drag = fin_base_drag_coefficient(geometry, powered, mach)
    nose_drag = nose_pressure_drag_coefficient(geometry, mach)
    base_drag = base_drag_component(geometry, powered, mach)
    wave_drag = wave_drag_component(geometry, mach)
    return body_drag + fin_friction_drag + fin_pressure_drag + fin_base_drag + nose_drag + base_drag + wave_drag
