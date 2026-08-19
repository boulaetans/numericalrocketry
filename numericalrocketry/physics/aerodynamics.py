"""Aerodynamic force and stability calculations.

The functions in this module convert state + geometry into aerodynamic loads,
including Barrowman CP/CN terms used by the 6DOF integrator.
"""

import math
from dataclasses import dataclass

import numpy as np

from .atmosphere import air_density, mach_number
from .drag import total_drag_coefficient
from ..rocket.geometry import (
    RocketGeometry,
    body_planform_geometry,
    fin_aspect_ratio,
    fin_geometry_stations,
    fin_mac_properties,
    fin_planform_area_m2,
    nose_profile_integration,
)
from ..simulation.state import FlightConditions, SimulationState


@dataclass(frozen=True)
class AerodynamicLoads:
    force_body_n: np.ndarray
    moment_body_n_m: np.ndarray
    cp_location_body_m: np.ndarray
    cn_alpha_per_rad: float
    drag_coefficient: float


@dataclass(frozen=True)
class AeroBreakdown:
    """Per-component aerodynamic breakdown returned by barrowman_aero_components().

    All positions are axial distances from the nose tip (positive aft).
    CN_alpha values are per radian in the small-angle linearised sense.
    static_margin_cal is in calibers (body diameters); positive means the CP
    is aft of the CG, i.e. the rocket is statically stable.
    """
    cn_alpha_nose: float
    cp_x_nose_m: float
    cn_alpha_fins: float
    cp_x_fins_m: float
    cn_alpha_total: float
    cp_x_total_m: float
    static_margin_cal: float


def _inverse_rotate(orientation_body_to_world, vector_world: np.ndarray) -> np.ndarray:
    return orientation_body_to_world.conjugate().rotate_vector(vector_world)


def compute_flight_conditions(
    state: SimulationState, wind_world_m_s: np.ndarray, site_altitude_m: float = 0.0
) -> FlightConditions:
    relative_air_velocity_world = state.velocity_world_m_s - wind_world_m_s
    relative_air_velocity_body = _inverse_rotate(state.orientation_body_to_world, relative_air_velocity_world)
    speed = float(np.linalg.norm(relative_air_velocity_body))
    # The atmosphere is always queried at altitude-above-sea-level (rocket
    # position, which is AGL from the launch pad, plus the launch site's own
    # MSL elevation), not at the raw AGL position.
    altitude_msl_m = state.position_world_m[2] + site_altitude_m
    dynamic_pressure = 0.5 * air_density(altitude_msl_m) * speed ** 2

    axial_speed = float(relative_air_velocity_body[0])
    lateral_speed = float(np.linalg.norm(relative_air_velocity_body[1:]))
    if speed <= 1e-9:
        angle_of_attack = 0.0
        sideslip = 0.0
    else:
        # AOA = acos(true signed axial component / speed), not abs(axial).
        # Ranges the full [0, 180 deg]: a rocket flying axial-speed-backward
        # (e.g. deep in a tumble) has AOA near 180 deg, not near 0 deg.
        angle_of_attack = math.acos(max(-1.0, min(1.0, axial_speed / speed)))
        sideslip = math.atan2(float(relative_air_velocity_body[1]), axial_speed)

    return FlightConditions(
        airspeed_body_m_s=relative_air_velocity_body,
        wind_world_m_s=wind_world_m_s.copy(),
        dynamic_pressure_pa=dynamic_pressure,
        mach=mach_number(speed, altitude_msl_m),
        angle_of_attack_rad=angle_of_attack,
        sideslip_rad=sideslip,
    )


def _nose_cp_x_from_tip(geometry: RocketGeometry) -> float:
    """Return the nose-cone center of pressure measured from the nose tip.

    Uses Barrowman (1967) slender-body theory.  The exact result depends on
    how quickly the cross-sectional area grows along the nose:

      Conical       r(x) ~ x           area grows quadratically from tip
                    -> loading concentrated near base -> CP at 2/3 L

      Ellipsoidal   r(x) ~ sqrt(x)     area grows steeply from the tip
      (quarter-     -> loading concentrated near tip  -> CP at 1/3 L
       ellipse)

      Ogive         similar to conical but blunter -> CP ~ 0.466 L (Barrowman
                    TRP Table IV value for tangent ogive at typical fineness)

      Parabolic     intermediate -> CP ~ 0.5 L

      Von Kármán /  empirical ~0.437 L (Niskanen 2009 Table 3.1)
      Haack
    """
    L = geometry.nose_length_m
    nt = geometry.nose_type
    if nt == "conical":
        return (2.0 / 3.0) * L
    if nt == "elliptical":
        # Quarter-ellipse: r = R*sqrt(2x/L - x²/L²).  Barrowman integral gives
        # x_CP = (2/S_ref) * ∫ x dA/dx dx / CN_alpha = L/3.
        return L / 3.0
    if nt == "parabolic":
        return 0.5 * L
    if nt in ("ogive", "von_karman"):
        # Exact volumetric CP:
        # x_cp = (length*A1 - fullVolume) / (A1 - A0), with A0=0 for a nose,
        # so x_cp = L - fullVolume/A1. fullVolume comes from numerically
        # integrating the real profile, so this is fineness-ratio-dependent
        # rather than a fixed fraction of L.
        a1 = geometry.reference_area_m2
        if a1 <= 0.0:
            return 0.466 * L
        full_volume, _wetted_area, _plan_area, _plan_center = nose_profile_integration(geometry)
        return L - full_volume / a1
    # Default: use the Barrowman TRP Table IV tangent-ogive value for unknown shapes.
    return 0.466 * L


_CNA_SUBSONIC_MACH = 0.9
_CNA_SUPERSONIC_MACH = 1.5
_MIN_BETA = 0.25
# NOTE: a commonly-cited 17.5 deg "stall angle" constant is NOT a force
# clamp in the reference this project targets -- it's used only by
# tumble-detection event logic, unrelated to force calculation. The actual
# CN clamp used during force calculation is 20 deg and applies ONLY to the
# fin contribution (the nose/body term has no clamp at all -- see
# _galejs_body_lift_components below for why it doesn't need one).
_FIN_STALL_ANGLE_RAD = 20.0 * math.pi / 180.0
_BODY_LIFT_K = 1.1


def _beta(mach: float) -> float:
    """Prandtl-Glauert compressibility factor."""
    if mach < 1.0:
        return max(_MIN_BETA, math.sqrt(max(1.0 - mach * mach, 0.0)))
    return max(_MIN_BETA, math.sqrt(max(mach * mach - 1.0, 0.0)))


def _body_fin_interference_factor(tau: float, mach: float) -> float:
    """Combined fin-in-body/body-in-fin normal-force multiplier K_fB.

    Squared (1+tau) below Mach 0.9 (NACA Report 1307 eq 14/21), blending
    linearly down to the unsquared fin-in-body-only term (1+tau) by Mach 1.5
    since the supersonic body contribution isn't modeled here, then constant
    above that.
    """
    fin_in_body = 1.0 + tau
    if mach <= _CNA_SUBSONIC_MACH:
        return fin_in_body ** 2
    if mach >= _CNA_SUPERSONIC_MACH:
        return fin_in_body
    body_in_fin = tau * fin_in_body
    weight = (_CNA_SUPERSONIC_MACH - mach) / (_CNA_SUPERSONIC_MACH - _CNA_SUBSONIC_MACH)
    return fin_in_body + weight * body_in_fin


def _fin_cp_poly_coefficients(aspect_ratio: float) -> tuple[float, float, float, float, float, float]:
    """5th-order CP-position polynomial coefficients.

    Fitted (offline) to match quarter-chord CP at Mach 0.5 and the supersonic
    AR-beta formula at Mach 2, with matching first derivative at both ends
    and zero 2nd/3rd derivative at Mach 2. Coefficients are used as
    poly[0] + poly[1]*M + poly[2]*M^2 + ...
    """
    ar = aspect_ratio
    denom = (1.0 - 3.4641 * ar) ** 2
    if denom <= 1e-12:
        return 0.25, 0.0, 0.0, 0.0, 0.0, 0.0
    p0 = (9.16049 * (-0.588838 + ar) * (-0.20624 + ar)) / denom
    p1 = (-31.6049 * (-0.705375 + ar) * (-0.198476 + ar)) / denom
    p2 = (55.3086 * (-0.711482 + ar) * (-0.196772 + ar)) / denom
    p3 = (-39.5062 * (-0.72074 + ar) * (-0.194245 + ar)) / denom
    p4 = (12.8395 * (-0.725688 + ar) * (-0.19292 + ar)) / denom
    p5 = (-1.58025 * (-0.728769 + ar) * (-0.192105 + ar)) / denom
    return p0, p1, p2, p3, p4, p5


def _fin_cp_fraction(mach: float, aspect_ratio: float) -> float:
    """Fraction of the MAC (from its leading edge) where the fin CP sits.

    Quarter-chord below Mach 0.5, an AR/beta-dependent empirical formula
    above Mach 2, and a 5th-order interpolation polynomial matching both in
    between.
    """
    if mach <= 0.5:
        return 0.25
    if mach >= 2.0:
        beta = _beta(mach)
        denom = 2.0 * aspect_ratio * beta - 1.0
        if abs(denom) < 1e-9:
            return 0.25
        return (aspect_ratio * beta - 0.67) / denom
    value = 0.0
    x = 1.0
    for coefficient in _fin_cp_poly_coefficients(aspect_ratio):
        value += coefficient * x
        x *= mach
    return value


def _fin_cn_alpha(geometry: RocketGeometry, mach: float = 0.0) -> float:
    """Normal-force slope for the complete fin set (all N fins combined).

    Subsonic branch (M<=0.9):

        CNa1 = 2*pi*s² / Aref / (1 + sqrt(1 + (1-M²)*(s²/(S*cosGamma))²))

    summed over N evenly-spaced fins in the total-angle-of-attack plane, which
    reduces to N/2 * CNa1 (the classical Barrowman 1967 total-CNa result),
    then scaled by the Mach-blended body-fin interference factor K_fB.

    This project targets subsonic flight, so the Mach number used inside the
    formula is clamped at the subsonic/transonic boundary (0.9) rather than
    also implementing transonic-interpolated and supersonic branches; K_fB
    itself is still blended across its full range.
    """
    s = geometry.fin_span_m
    N = geometry.fin_count
    d = geometry.diameter_m
    r_body = d / 2.0

    if s <= 0.0 or N <= 0 or d <= 0.0:
        return 0.0

    area_per_fin = fin_planform_area_m2(geometry)
    if area_per_fin <= 0.0:
        return 0.0

    _mac_length, _mac_lead, _cos_gamma_lead, cos_gamma_mid = fin_mac_properties(geometry)
    if cos_gamma_mid <= 1e-9:
        return 0.0

    ref_area = geometry.reference_area_m2
    mach_clamped = min(max(mach, 0.0), _CNA_SUBSONIC_MACH)
    lambda_term = (s ** 2 / (area_per_fin * cos_gamma_mid)) ** 2
    sq = math.sqrt(1.0 + (1.0 - mach_clamped ** 2) * lambda_term)
    cna1_per_fin = 2.0 * math.pi * s ** 2 / ref_area / (1.0 + sq)

    cn_alpha_base = 0.5 * N * cna1_per_fin

    tau = r_body / max(s + r_body, 1e-9)
    k_fb = _body_fin_interference_factor(tau, mach)
    return cn_alpha_base * k_fb


def _fin_cp_x_from_nose_tip(geometry: RocketGeometry, mach: float = 0.0) -> float:
    """Return the fin set CP measured from the nose tip.

    The fin panel aerodynamic center sits at the quarter chord of the MAC
    only below Mach 0.5; above that it shifts aft with Mach number, using the
    same MAC/leading-edge computation (exact for freeform, closed-form for
    trapezoidal) as the drag model in drag.py, so both stay consistent with
    the same geometry.
    """
    x_le = geometry.fin_leading_edge_x_m   # root LE position from nose tip
    mac_length, mac_lead, _cos_gamma_lead, _cos_gamma_mid = fin_mac_properties(geometry)
    aspect_ratio = fin_aspect_ratio(geometry)
    cp_fraction = _fin_cp_fraction(mach, aspect_ratio)
    return x_le + mac_lead + cp_fraction * mac_length


def _galejs_body_lift_components(
    geometry: RocketGeometry, aoa_rad: float, mach: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Body-lift CN contribution of the nose and body tube (the Galejs
    extension to Barrowman theory).

    Returns ((cn_nose_lift, cp_nose_lift_x), (cn_tube_lift, cp_tube_lift_x)).
    Unlike the fin/nose Barrowman terms, this is deliberately NOT wrapped in
    barrowman_aero_components()'s AOA-independent per-radian API: it is
    fundamentally nonlinear in AOA (proportional to sin(AOA)*sinc(AOA), not
    AOA itself) and needs the instantaneous AOA to evaluate, so it's computed
    directly where AOA is available (compute_aerodynamic_loads).

    Also unlike the linear Barrowman terms, no stall clamp applies here (raw
    AOA is used, un-min()'d) -- it doesn't need one, since sin(AOA)*sinc(AOA)
    = sin(AOA)^2/AOA is itself bounded and turns over well before AOA
    approaches 180 deg, unlike an unclamped linear CN_alpha*AOA term.
    """
    sin_aoa = math.sin(aoa_rad)
    sinc_aoa = 1.0 if aoa_rad < 0.001 else sin_aoa / aoa_rad
    # Suppresses the term near-motionless (Mach < 0.05) at high AOA (> 45
    # deg), to avoid instability turning around at apogee.
    mul = 1.0
    if mach < 0.05 and aoa_rad > math.pi / 4.0:
        mul = (mach / 0.05) ** 2
    sin_sinc = sin_aoa * sinc_aoa

    ref_area = geometry.reference_area_m2
    if ref_area <= 0.0:
        return (0.0, 0.0), (0.0, 0.0)

    _full_volume, _wetted_area, nose_planform_area, nose_planform_center = nose_profile_integration(geometry)
    cn_nose_lift = mul * _BODY_LIFT_K * nose_planform_area / ref_area * sin_sinc

    tube_length = max(0.0, geometry.length_m - geometry.nose_length_m)
    tube_planform_area = geometry.diameter_m * tube_length  # integral(2*r(x))dx for constant r
    tube_planform_center = geometry.nose_length_m + 0.5 * tube_length
    cn_tube_lift = mul * _BODY_LIFT_K * tube_planform_area / ref_area * sin_sinc

    return (cn_nose_lift, nose_planform_center), (cn_tube_lift, tube_planform_center)


def _roll_damping_coefficient(geometry: RocketGeometry, roll_rate_rad_s: float, velocity_m_s: float, mach: float) -> float:
    """Roll damping moment coefficient.

    Subsonic branch only (Mach clamped at 0.9), consistent with _fin_cn_alpha.
    Near apogee, low airspeed combined with a relatively large roll rate can
    push the fin tips well past stall, so that regime sums the (stall-capped)
    per-station chords separately instead.
    """
    if abs(roll_rate_rad_s) < 0.1 or velocity_m_s <= 1e-9:
        return 0.0

    stations = fin_geometry_stations(geometry)
    chord_lengths = stations.chord_length_stations_m
    n_stations = len(chord_lengths)
    if n_stations == 0:
        return 0.0

    ref_area = geometry.reference_area_m2
    ref_length = geometry.diameter_m
    if ref_area <= 0.0 or ref_length <= 0.0:
        return 0.0

    abs_rate = abs(roll_rate_rad_s)
    body_radius = stations.body_radius_m
    span = geometry.fin_span_m
    stall_angle = 15.0 * math.pi / 180.0

    if abs_rate * (body_radius + span) / velocity_m_s > stall_angle:
        total = 0.0
        for i in range(n_stations):
            dist = body_radius + span * i / n_stations
            aoa = min(abs_rate * dist / velocity_m_s, stall_angle)
            total += chord_lengths[i] * dist * aoa
        total *= (span / n_stations) * 2.0 * math.pi / _beta(mach) / (ref_area * ref_length)
        return math.copysign(total, roll_rate_rad_s)

    mach_clamped = min(mach, _CNA_SUBSONIC_MACH)
    beta = _beta(mach_clamped)
    return 2.0 * math.pi * roll_rate_rad_s * stations.roll_sum_m4 / (ref_area * ref_length * velocity_m_s * beta)


def _pitch_yaw_damping_moments_nm(
    geometry: RocketGeometry,
    pitch_rate_rad_s: float,
    yaw_rate_rad_s: float,
    velocity_m_s: float,
    dynamic_pressure_pa: float,
    nose_tip_moment_y_nm: float,
    nose_tip_moment_z_nm: float,
) -> tuple[float, float]:
    """Pitch/yaw damping moments (N*m) to subtract from the body y/z moments.

    Two deliberate deviations from a fully literal damping-multiplier model:

    1. The reference model's damping multiplier is nominally computed about
       a configurable "pitch center," but that field is never actually set
       anywhere in the reference implementation -- it stays at the nose tip
       for the whole simulation. So the multiplier (and the clamp bound,
       taken before the later CG shift) is always about the NOSE TIP, never
       the true CG -- reproduced literally here (no `cg_x_m` param; caller
       passes the nose-tip-referenced moment).
    2. The reference model clamps with a signed `min(computed, Cm)` then
       `sign(rate)`. In this project's simplified single-total-AOA-plane
       force model that lets the clamped "magnitude" go negative when the
       nose-tip moment is negative, so `sign(rate)*negative` can amplify the
       restoring moment instead of damping it -- verified to cause runaway
       divergence under wind. Clamps against `abs(nose-tip moment)` instead,
       preserving the clamp's real purpose (never let damping overshoot past
       canceling the current restoring moment) while always producing
       genuine damping.
    """
    if velocity_m_s <= 1e-9:
        return 0.0, 0.0

    ref_area = geometry.reference_area_m2
    ref_length = geometry.diameter_m
    if ref_area <= 0.0 or ref_length <= 0.0:
        return 0.0, 0.0

    cache_diameter, cache_length = body_planform_geometry(geometry)
    mul = 0.275 * cache_diameter / (ref_area * ref_length) * (cache_length ** 4)

    fin_area = fin_planform_area_m2(geometry)
    if fin_area > 0.0 and geometry.fin_count > 0:
        mac_length, mac_lead, _cos_gamma_lead, _cos_gamma_mid = fin_mac_properties(geometry)
        fin_midchord_x = geometry.fin_leading_edge_x_m + mac_lead + 0.5 * mac_length
        mul += (
            0.6 * min(geometry.fin_count, 4) * fin_area * fin_midchord_x ** 3
            / (ref_area * ref_length)
        )

    mul *= 3.0  # Higher damping yields much more realistic apogee turn behavior.

    q_aref_lref = dynamic_pressure_pa * ref_area * ref_length

    pitch_component_nm = mul * (pitch_rate_rad_s / velocity_m_s) ** 2 * q_aref_lref
    pitch_component_nm = min(pitch_component_nm, abs(nose_tip_moment_y_nm))
    pitch_damping_nm = math.copysign(pitch_component_nm, pitch_rate_rad_s)

    yaw_component_nm = mul * (yaw_rate_rad_s / velocity_m_s) ** 2 * q_aref_lref
    yaw_component_nm = min(yaw_component_nm, abs(nose_tip_moment_z_nm))
    yaw_damping_nm = math.copysign(yaw_component_nm, yaw_rate_rad_s)

    return pitch_damping_nm, yaw_damping_nm


def barrowman_aero_components(geometry: RocketGeometry, cg_x_m: float | None = None, mach: float = 0.0) -> AeroBreakdown:
    """Compute the full component-wise Barrowman aerodynamic breakdown.

    Returns per-component CN_alpha and CP position together with the
    moment-weighted total CP and (optionally) the static stability margin.

    Parameters
    ----------
    geometry : RocketGeometry
        Complete rocket geometry (nose, body, fins).
    cg_x_m : float, optional
        CG axial position from nose tip (m).  Used to compute static_margin_cal.
        Pass None to skip the margin computation (returns NaN).
    mach : float, optional
        Flight Mach number, used for the fin CN_alpha compressibility term,
        the Mach-blended body-fin interference factor, and the Mach-dependent
        fin CP shift. Defaults to 0.0 (incompressible).

    Returns
    -------
    AeroBreakdown
        Per-component and total CN_alpha / CP_x (from nose tip) and static margin.
    """
    cn_nose = 2.0                          # Barrowman slender-body theory: exact for all axisymmetric shapes
    cp_nose = _nose_cp_x_from_tip(geometry)

    cn_fins = _fin_cn_alpha(geometry, mach)
    cp_fins = _fin_cp_x_from_nose_tip(geometry, mach)

    cn_total = cn_nose + cn_fins

    # Weighted CP: CN-moment-balance gives total CP (Barrowman 1967, Eq. 12).
    if cn_total > 0.0:
        cp_total = (cn_nose * cp_nose + cn_fins * cp_fins) / cn_total
    else:
        cp_total = cp_nose

    # Static margin in calibers: positive = CP aft of CG = stable.
    if cg_x_m is not None and geometry.diameter_m > 0.0:
        margin = (cp_total - cg_x_m) / geometry.diameter_m
    else:
        margin = float("nan")

    return AeroBreakdown(
        cn_alpha_nose=cn_nose,
        cp_x_nose_m=cp_nose,
        cn_alpha_fins=cn_fins,
        cp_x_fins_m=cp_fins,
        cn_alpha_total=cn_total,
        cp_x_total_m=cp_total,
        static_margin_cal=margin,
    )


def barrowman_normal_force_slope(geometry: RocketGeometry) -> float:
    """Total CN_alpha for the rocket (nose + fins combined).

    Delegates to the component-wise breakdown so both functions stay in sync.
    """
    bd = barrowman_aero_components(geometry)
    return bd.cn_alpha_total


def barrowman_center_of_pressure_x(geometry: RocketGeometry) -> float:
    """Total CP axial position from the nose tip (m).

    Delegates to the component-wise breakdown so both functions stay in sync.
    """
    bd = barrowman_aero_components(geometry)
    return bd.cp_x_total_m


def compute_static_margin(cp_x_m: float, cg_x_m: float, diameter_m: float) -> float:
    """Return the static stability margin in calibers.

    Positive means CP is aft of CG (stable).  One caliber = one body diameter.
    """
    if diameter_m <= 0.0:
        return float("nan")
    return (cp_x_m - cg_x_m) / diameter_m


def compute_aerodynamic_loads(
    state: SimulationState,
    geometry: RocketGeometry,
    wind_world_m_s: np.ndarray,
    drag_coefficient: float | None = None,
    site_altitude_m: float = 0.0,
    surface_finish: str = "normal",
) -> tuple[FlightConditions, AerodynamicLoads]:
    flight_conditions = compute_flight_conditions(state, wind_world_m_s, site_altitude_m=site_altitude_m)
    speed = float(np.linalg.norm(flight_conditions.airspeed_body_m_s))
    if speed <= 1e-9:
        zero = np.zeros(3, dtype=float)
        return flight_conditions, AerodynamicLoads(zero, zero, zero, 0.0, 0.0)

    q = flight_conditions.dynamic_pressure_pa
    reference_area = geometry.reference_area_m2
    body_velocity = flight_conditions.airspeed_body_m_s
    velocity_hat = body_velocity / speed

    if state.recovery_deployed:
        # The post-deployment landing phase does more than discard body
        # drag: it also never generates a normal force or moment and never
        # updates rotational state at all (rotation velocity/orientation are
        # simply frozen for the rest of the descent -- see
        # integrator._state_increment, which zeroes the rotational
        # derivatives whenever recovery is deployed). Total force here is
        # therefore pure axial drag opposing the relative airspeed.
        if drag_coefficient is None:
            drag_coefficient = 0.0
        drag_force = -q * reference_area * drag_coefficient * velocity_hat
        zero = np.zeros(3, dtype=float)
        return flight_conditions, AerodynamicLoads(
            force_body_n=drag_force,
            moment_body_n_m=zero,
            cp_location_body_m=zero,
            cn_alpha_per_rad=0.0,
            drag_coefficient=drag_coefficient,
        )

    if drag_coefficient is None:
        # Use the rocket's actual configured surface finish, not a hardcoded
        # placeholder -- roughness height materially affects skin-friction Cf.
        drag_coefficient = total_drag_coefficient(
            speed_m_s=speed,
            altitude_m=state.position_world_m[2] + site_altitude_m,
            geometry=geometry,
            surface_finish=surface_finish,
            powered=False,
        )

    drag_force = -q * reference_area * drag_coefficient * velocity_hat

    # Component-wise Barrowman breakdown; pass CG x and Mach for static-margin
    # logging and the compressibility/interference/CP-shift terms.
    cg_x_m = float(state.mass_properties.center_of_gravity_m[0])
    mach = flight_conditions.mach
    aero_bd = barrowman_aero_components(geometry, cg_x_m=cg_x_m, mach=mach)
    lateral_velocity = body_velocity[1:]
    lateral_speed = np.linalg.norm(lateral_velocity)
    normal_force = np.zeros(3, dtype=float)
    cn_total = 0.0
    cp_total = aero_bd.cp_x_nose_m
    if lateral_speed > 1e-9:
        # Normal force acts in the SAME direction as the local crossflow
        # (unlike drag, which opposes the axial flow): a positive-margin
        # rocket's CP-aft-of-CG fins are pushed by the crossflow like a
        # weathervane's tail, swinging the tail with the flow and the nose
        # into it, which is what produces a restoring (weathercocking) moment.
        lateral_direction = lateral_velocity / lateral_speed
        aoa = flight_conditions.angle_of_attack_rad

        # Each component's own (CN, CP) is computed separately and summed as
        # independent moment contributions -- mathematically identical to a
        # CP-weighted combination (CN_total*CP_total = sum(CN_i*CP_i) by
        # construction of a weight-preserving average), but avoids threading
        # AOA through barrowman_aero_components()'s AOA-independent
        # per-radian API.
        #
        # Nose: raw, unclamped AOA -- no stall clamp applies to the nose/body
        # Barrowman term.
        cn_nose = aero_bd.cn_alpha_nose * aoa
        components = [(cn_nose, aero_bd.cp_x_nose_m)]

        # Galejs body-lift terms (nose + body tube), naturally AOA-bounded.
        nose_lift, tube_lift = _galejs_body_lift_components(geometry, aoa, mach)
        components.append(nose_lift)
        components.append(tube_lift)

        # Fins: CN = cna * min(AOA, 20 deg) -- the one real stall clamp
        # applied during force calculation, and it only applies to the fin
        # contribution.
        cn_fins = aero_bd.cn_alpha_fins * min(aoa, _FIN_STALL_ANGLE_RAD)
        components.append((cn_fins, aero_bd.cp_x_fins_m))

        cn_total = sum(cn for cn, _ in components)
        if cn_total > 0.0:
            cp_total = sum(cn * cp for cn, cp in components) / cn_total
        normal_force_magnitude = q * reference_area * cn_total
        normal_force[1:] = normal_force_magnitude * lateral_direction

    total_force_body = drag_force + normal_force
    cp_location = np.array([cp_total, 0.0, 0.0], dtype=float)
    cg_location = state.mass_properties.center_of_gravity_m
    arm = cp_location - cg_location
    total_moment_body = np.cross(arm, normal_force)

    # Angular-rate-dependent aerodynamic damping (pitch, yaw, roll). The
    # static CP/CN model above only captures the restoring moment from the
    # CP-CG offset; without this, angle-of-attack oscillations that should
    # decay (especially in the low-dynamic-pressure region near apogee)
    # instead persist or grow. Pitch/yaw/roll damping is a body-frame
    # ("rocket coordinates") quantity -- the world-frame rotation velocity
    # needs to be inverse-rotated into body frame before deriving
    # pitch/yaw/roll rate -- state stores angular velocity in world frame
    # (see state.py), so convert here.
    omega = _inverse_rotate(state.orientation_body_to_world, state.angular_velocity_world_rad_s)
    # Damping is computed on the pitch/yaw moment BEFORE the CG shift applied
    # later -- i.e. against the moment about the NOSE TIP (x=0), not the CG.
    # Since that CG shift is linear and independent of the damping term,
    # subtracting damping (clamped against the nose-tip value) from the
    # CG-referenced moment below gives the same final CG-referenced result;
    # see _pitch_yaw_damping_moments_nm's docstring.
    nose_tip_moment_body = np.cross(cp_location, normal_force)
    pitch_damping_nm, yaw_damping_nm = _pitch_yaw_damping_moments_nm(
        geometry,
        float(omega[1]),
        float(omega[2]),
        speed,
        q,
        float(nose_tip_moment_body[1]),
        float(nose_tip_moment_body[2]),
    )
    total_moment_body[1] -= pitch_damping_nm
    total_moment_body[2] -= yaw_damping_nm

    roll_damping_coefficient = _roll_damping_coefficient(geometry, float(omega[0]), speed, mach)
    if roll_damping_coefficient != 0.0:
        total_moment_body[0] -= roll_damping_coefficient * q * reference_area * geometry.diameter_m

    # Diagnostic only (unused elsewhere): instantaneous effective CN slope,
    # not a true constant now that the Galejs term is AOA-nonlinear.
    aoa_for_slope = flight_conditions.angle_of_attack_rad
    cn_alpha_effective = cn_total / aoa_for_slope if aoa_for_slope > 1e-9 else aero_bd.cn_alpha_total

    return flight_conditions, AerodynamicLoads(
        force_body_n=total_force_body,
        moment_body_n_m=total_moment_body,
        cp_location_body_m=cp_location,
        cn_alpha_per_rad=cn_alpha_effective,
        drag_coefficient=drag_coefficient,
    )