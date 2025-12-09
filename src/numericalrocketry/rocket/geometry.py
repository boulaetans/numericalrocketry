"""Rocket geometry: RocketGeometry data model, nose-profile integration, and
fin planform/mass-CG sampling."""

import math
from dataclasses import dataclass

from ..constants import PI


@dataclass(frozen=True)
class RocketGeometry:
    length_m: float
    diameter_m: float
    nose_length_m: float
    fin_count: int
    fin_thickness_m: float
    fin_area_m2: float
    fin_span_m: float
    fin_root_chord_m: float = 0.0
    fin_tip_chord_m: float = 0.0
    fin_sweep_length_m: float = 0.0
    fin_leading_edge_x_m: float = 0.0
    nose_type: str = "ogive"
    fin_cross_section: str = "rounded"
    fin_points_m: tuple[tuple[float, float], ...] = ()

    @property
    def reference_area_m2(self) -> float:
        return PI * (self.diameter_m / 2.0) ** 2

    @property
    def base_area_m2(self) -> float:
        return self.reference_area_m2

    @property
    def fineness_ratio(self) -> float:
        if self.diameter_m <= 0.0:
            return 1.0
        return max(1.0, self.length_m / self.diameter_m)

    @property
    def mean_fin_chord_m(self) -> float:
        if self.fin_root_chord_m > 0.0 and self.fin_tip_chord_m > 0.0:
            return 0.5 * (self.fin_root_chord_m + self.fin_tip_chord_m)
        if self.fin_span_m <= 0.0:
            return self.fin_area_m2
        return self.fin_area_m2 / self.fin_span_m

    @property
    def wetted_body_area_m2(self) -> float:
        cylinder_area = PI * self.diameter_m * max(0.0, self.length_m - self.nose_length_m)
        _full_volume, nose_wetted_area, _planform_area, _planform_center = nose_profile_integration(self)
        return cylinder_area + nose_wetted_area

    @property
    def body_form_factor(self) -> float:
        # 1 + 1/(2*fineness_ratio).
        fr = self.fineness_ratio
        return 1.0 + 1.0 / (2.0 * fr)

    def nose_joint_angle_rad(self) -> float:
        diameter = self.diameter_m
        length = self.nose_length_m
        if length <= 0.0:
            return 0.0
        if self.nose_type == "conical":
            return math.atan((diameter / 2.0) / length)
        if self.nose_type in ("ogive", "von_karman"):
            return 0.0
        if self.nose_type == "elliptical":
            return math.atan((diameter / 2.0) / math.sqrt(max(length ** 2 - (diameter / 2.0) ** 2, 1e-12)))
        if self.nose_type == "parabolic":
            return math.atan(diameter / (2.0 * length))
        if self.nose_type == "hemisphere":
            return 0.5 * math.pi
        return math.atan((diameter / 2.0) / length)

    def _nose_radius_at(self, x_m: float) -> float:
        """Nose profile radius at axial distance x from the tip (m).

        Covers conical, tangent-ogive, elliptical, full-parabola, and
        von Karman / LD-Haack (Haack series with param=0) shapes.
        """
        length = self.nose_length_m
        r_aft = self.diameter_m / 2.0
        if length <= 0.0:
            return r_aft
        x = min(max(x_m, 0.0), length)
        if self.nose_type == "conical":
            return r_aft * x / length
        if self.nose_type == "parabolic":
            # Full parabola (param=1): r = R*(2x/L - (x/L)^2)/(2-1).
            return r_aft * (2.0 * x / length - (x / length) ** 2)
        if self.nose_type == "hemisphere":
            frac = x / length
            return r_aft * math.sqrt(max(1.0 - (1.0 - frac) ** 2, 0.0))
        if self.nose_type == "ogive":
            # Tangent ogive: circular-arc profile. The general formula
            # R = sqrt((L^2+r^2)*(L^2+r^2)/(4r^2)) reduces, for a true
            # tangent ogive, to the classical circle radius rho = (L^2+r^2)/(2r).
            rho = (length ** 2 + r_aft ** 2) / (2.0 * r_aft) if r_aft > 0.0 else 0.0
            y0 = math.sqrt(max(rho ** 2 - length ** 2, 0.0))
            return math.sqrt(max(rho ** 2 - (length - x) ** 2, 0.0)) - y0
        if self.nose_type == "von_karman":
            # Haack series with param=0 (LD-Haack / von Karman).
            theta = math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * x / length)))
            return r_aft * math.sqrt(max((theta - math.sin(2.0 * theta) / 2.0) / math.pi, 0.0))
        # Elliptical and default: r = R*sqrt(2x/L - (x/L)^2).
        frac = x / length
        return r_aft * math.sqrt(max(2.0 * frac - frac ** 2, 0.0))

    def nose_sinphi(self) -> float:
        """Sine of the local nose/body-tube slope discontinuity at the base.

        Samples the profile radius at 99% of the nose length and compares it
        to the aft (body) radius over the last 1% of the length, rather than
        using any single closed-form half-angle. Only the tangent ogive is
        special-cased to exactly 0; every other shape, including von Karman,
        is computed from the sampled profile even though it also blends
        smoothly, since its curvature at 99% length is not exactly tangent
        the way a true tangent ogive's is.
        """
        length = self.nose_length_m
        if length <= 0.0:
            return 0.0
        if self.nose_type == "ogive":
            return 0.0
        r_aft = self.diameter_m / 2.0
        r_near = self._nose_radius_at(0.99 * length)
        dr = r_aft - r_near
        dx = 0.01 * length
        return dr / math.hypot(dr, dx)


_NOSE_DIVISIONS = 128  # Number of frustum slices for the nose-profile integration


def nose_profile_integration(geometry: RocketGeometry, divisions: int = _NOSE_DIVISIONS) -> tuple[float, float, float, float]:
    """Numerically integrate the nose's outer profile of revolution.

    Treats the nose as a solid (non-hollow) profile starting from radius 0 at
    the tip -- the outer-profile volume/area, not the material-only
    hollow-shell volume, which is all the aerodynamic CP/wetted-area/planform
    calculations actually need.

    Returns (full_volume_m3, wetted_area_m2, planform_area_m2, planform_center_m),
    where planform_center_m is measured from the nose tip.
    """
    length = geometry.nose_length_m
    if length <= 0.0 or divisions <= 0:
        return 0.0, 0.0, 0.0, 0.0

    volume_sum = 0.0
    wet_area = 0.0
    plan_area = 0.0
    plan_moment = 0.0

    dx = length / divisions
    r_prev = geometry._nose_radius_at(0.0)
    for n in range(divisions):
        x1 = n * dx
        x2 = (n + 1) * dx
        r1 = r_prev
        r2 = geometry._nose_radius_at(x2)
        r_prev = r2

        # Frustum volume (deferred *PI/3 factor).
        volume_sum += dx * (r1 * r1 + r1 * r2 + r2 * r2)

        # Wetted (lateral) area of the frustum slice (deferred *PI factor).
        wet_area += (r1 + r2) * math.hypot(r1 - r2, dx)

        # Planform (side-view) area and its first moment (deferred *1 factor;
        # no extra constant multiplier needed here).
        d_area = dx * (r1 + r2)
        plan_area += d_area
        plan_moment += d_area * x1 + 2.0 * dx * dx * (r1 / 6.0 + r2 / 3.0)

    full_volume = volume_sum * math.pi / 3.0
    wetted_area = wet_area * math.pi
    planform_center = plan_moment / plan_area if plan_area > 0.0 else 0.0
    return full_volume, wetted_area, plan_area, planform_center


def nose_shell_mass_cg_m(
    geometry: RocketGeometry,
    wall_thickness_m: float,
    shoulder_length_m: float = 0.0,
    shoulder_radius_m: float = 0.0,
    shoulder_thickness_m: float = 0.0,
    shoulder_capped: bool = False,
    divisions: int = _NOSE_DIVISIONS,
) -> float:
    """Real mass-CG of the nose cone's hollow shell (+ rear shoulder), from the nose tip.

    `nose_profile_integration()` above deliberately treats the nose as a
    solid outer-profile revolution -- correct for the aerodynamic wetted
    area/CP/planform terms, which use the *external* profile only. Mass CG is
    a different quantity: the nose cone is a hollow shell of the given wall
    `thickness`, and its real centroid is NOT at 0.5*length for any tapered
    (non-cylindrical) profile -- more shell material sits toward the wider
    (aft) end. Integrates the hollow-frustum shell directly (inner radius =
    outer radius minus the wall thickness projected along the local slope,
    `height = thickness*hyp/l`).

    If a rear shoulder is present (it extends past the nose's own length,
    into the body tube), also adds its ring + end-cap mass, volume-weighted
    against the base shell (same material => volume-weighted == mass-weighted).
    A mass override on the component (as Green Eggs' nose cone has) replaces
    only the total *weight*, never this shape-integrated *position*.
    """
    length = geometry.nose_length_m
    if length <= 0.0 or divisions <= 0:
        return 0.5 * length

    volume_sum = 0.0  # deferred *pi/3 factor, cancels in the shell_cg ratio below
    cgx_sum = 0.0
    dx = length / divisions
    r_prev = geometry._nose_radius_at(0.0)
    for n in range(divisions):
        x1 = n * dx
        x2 = (n + 1) * dx
        l = x2 - x1
        r1o = r_prev
        r2o = geometry._nose_radius_at(x2)
        r_prev = r2o

        hyp = math.hypot(r2o - r1o, l)
        height = wall_thickness_m * hyp / l if l > 0.0 else 0.0
        r1i = max(r1o - height, 0.0)
        r2i = max(r2o - height, 0.0)

        vol_full = l * (r1o * r1o + r1o * r2o + r2o * r2o)
        vol_inner = l * (r1i * r1i + r1i * r2i + r2i * r2i)
        cg_full = (
            l * (r1o**2 + 2.0 * r1o * r2o + 3.0 * r2o**2) / (4.0 * (r1o**2 + r1o * r2o + r2o**2))
            if vol_full > 1e-15 else l / 2.0
        )
        cg_inner = (
            l * (r1i**2 + 2.0 * r1i * r2i + 3.0 * r2i**2) / (4.0 * (r1i**2 + r1i * r2i + r2i**2))
            if vol_inner > 1e-15 else l / 2.0
        )

        d_v = vol_full - vol_inner
        d_cg = (cg_full * vol_full - cg_inner * vol_inner) / d_v if d_v > 0.0 else l / 2.0
        volume_sum += d_v
        cgx_sum += d_v * (x1 + d_cg)

    if volume_sum <= 0.0:
        return 0.5 * length
    shell_cg = cgx_sum / volume_sum

    if shoulder_length_m <= 0.0:
        return shell_cg

    shoulder_inner_r = max(shoulder_radius_m - shoulder_thickness_m, 0.0)
    ring_volume = math.pi * max(shoulder_radius_m**2 - shoulder_inner_r**2, 0.0) * shoulder_length_m
    ring_cg = length + 0.5 * shoulder_length_m

    total_volume = volume_sum * math.pi / 3.0 + ring_volume
    total_moment = shell_cg * (volume_sum * math.pi / 3.0) + ring_cg * ring_volume

    if shoulder_capped:
        cap_volume = math.pi * shoulder_inner_r**2 * shoulder_thickness_m
        cap_cg = length + shoulder_length_m - 0.5 * shoulder_thickness_m
        total_volume += cap_volume
        total_moment += cap_cg * cap_volume

    return total_moment / total_volume if total_volume > 0.0 else shell_cg


def polygon_area_m2(points: tuple[tuple[float, float], ...]) -> float:
    """Shoelace-formula planform area of a closed fin polygon."""
    if len(points) < 3:
        return 0.0
    area2 = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + (points[0],)):
        area2 += x1 * y2 - x2 * y1
    return abs(area2) * 0.5


def polygon_centroid_m2(points: tuple[tuple[float, float], ...]) -> tuple[float, float, float]:
    """Shoelace-formula (x_centroid, y_centroid, area) of a closed polygon."""
    if len(points) < 3:
        return 0.0, 0.0, 0.0
    area2 = 0.0
    cx = 0.0
    cy = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + (points[0],)):
        cross = x1 * y2 - x2 * y1
        area2 += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if area2 == 0.0:
        return 0.0, 0.0, 0.0
    area = area2 * 0.5
    return cx / (3.0 * area2), cy / (3.0 * area2), abs(area)


def fin_planform_area_m2(geometry: RocketGeometry) -> float:
    """Single-fin planform area, preferring the exact freeform polygon if present."""
    if geometry.fin_points_m:
        return polygon_area_m2(geometry.fin_points_m)
    if geometry.fin_area_m2 > 0.0:
        return geometry.fin_area_m2
    return 0.5 * (geometry.fin_root_chord_m + geometry.fin_tip_chord_m) * geometry.fin_span_m


def fin_set_mass_cg_m(
    geometry: RocketGeometry,
    thickness_m: float,
    cross_section_relative_volume: float,
    bulk_density_kg_m3: float,
    tab_length_m: float,
    tab_height_m: float,
    tab_front_offset_m: float,
    fillet_radius_m: float,
    fillet_density_kg_m3: float,
    body_radius_m: float,
) -> float:
    """Real per-fin mass-CG, from the fin's own root-chord leading edge.

    The true centroid is a mass-weighted blend of three physically distinct
    pieces, none of which sit at the fin's geometric midpoint for a
    non-rectangular (freeform/swept) planform:

    * bulk planform (the visible fin polygon, `fin_points_m`) -- true area
      centroid via the shoelace formula (which reduces to a plain polygon
      centroid here since Green Eggs' fin sits on a constant-radius body
      tube), extruded by `thickness_m` and the cross-section's relative
      volume factor (e.g. 0.99 for a rounded cross section).
    * through-the-wall tab: a rectangle of `tab_length_m` x `tab_height_m`
      starting at `tab_front_offset_m` from the fin's leading edge,
      thickness_m deep, but with NO cross-section relative-volume factor
      applied (the tab is inside the airframe, not an exposed aerodynamic
      surface).
    * root fillet: a constant cross-section (since body radius is constant
      along Green Eggs' fin root), in its own (usually different) fillet
      material, spanning the fin's *full* axial extent -- i.e. the fillet is
      treated as running the whole swept fin footprint, not just the
      physical root-chord contact line, even where a swept-back tip
      wouldn't actually touch the body. Reproduced literally here even
      though it's a small (~0.01 g) contributor either way.

    A mass override on the fin set, as Green Eggs has (documented in the
    .ork file as "paint and glue fillet allowance"), replaces only the
    reported total weight -- the position is still this literal per-piece
    mass-weighted average of the un-overridden material masses.
    """
    bulk_x, _bulk_y, bulk_area = polygon_centroid_m2(geometry.fin_points_m)
    bulk_volume = bulk_area * thickness_m * cross_section_relative_volume
    bulk_mass = bulk_volume * bulk_density_kg_m3

    tab_trail_offset_m = tab_front_offset_m + tab_length_m
    tab_x = 0.5 * (tab_front_offset_m + tab_trail_offset_m)
    tab_volume = tab_length_m * tab_height_m * thickness_m
    tab_mass = tab_volume * bulk_density_kg_m3

    fin_xs = [x for x, _y in geometry.fin_points_m]
    fillet_length_m = (max(fin_xs) - min(fin_xs)) if fin_xs else 0.0

    hyp = fillet_radius_m + body_radius_m
    inner_arc = math.asin(fillet_radius_m / hyp) if hyp > 0.0 else 0.0
    outer_arc = math.acos(fillet_radius_m / hyp) if hyp > 0.0 else 0.0
    triangle_area = math.tan(outer_arc) * fillet_radius_m**2 / 2.0
    # Each fin has a fillet on both sides (factor of 2).
    fillet_cross_area = 2.0 * (
        triangle_area - outer_arc * fillet_radius_m**2 / 2.0 - inner_arc * body_radius_m**2 / 2.0
    )
    fillet_volume = max(fillet_cross_area, 0.0) * fillet_length_m
    fillet_x = 0.5 * fillet_length_m
    fillet_mass = fillet_volume * fillet_density_kg_m3

    total_mass = bulk_mass + tab_mass + fillet_mass
    if total_mass <= 0.0:
        return 0.5 * geometry.fin_root_chord_m
    return (bulk_mass * bulk_x + tab_mass * tab_x + fillet_mass * fillet_x) / total_mass


def _horizontal_span_intersections(points: tuple[tuple[float, float], ...], y: float) -> tuple[float, float] | None:
    """Return (x_min, x_max) where the polygon boundary crosses span position y."""
    xs: list[float] = []
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        if y1 == y2:
            if y1 == y:
                xs.extend([x1, x2])
            continue
        lo, hi = (y1, y2) if y1 < y2 else (y2, y1)
        if lo <= y <= hi:
            t = (y - y1) / (y2 - y1)
            xs.append(x1 + t * (x2 - x1))
    if not xs:
        return None
    return min(xs), max(xs)


def fin_points_with_root(geometry: RocketGeometry) -> tuple[tuple[float, float], ...]:
    """Return the fin's boundary polygon, synthesizing one for trapezoidal fins.

    There's no separate closed-form path needed for trapezoidal fins: a
    trapezoid's realized point polygon (root leading edge, tip leading edge,
    tip trailing edge, root trailing edge) can always be sampled the same way
    a freeform polygon is. Synthesizing that polygon here lets freeform and
    trapezoidal fins share one numerical-integration code path.
    """
    if geometry.fin_points_m:
        return geometry.fin_points_m
    c_r = geometry.fin_root_chord_m
    c_t = geometry.fin_tip_chord_m
    span = geometry.fin_span_m
    sweep = geometry.fin_sweep_length_m
    if (c_r + c_t) <= 0.0 or span <= 0.0:
        return ()
    return (
        (0.0, 0.0),
        (sweep, span),
        (sweep + c_t, span),
        (c_r, 0.0),
    )


_FIN_DIVISIONS = 48  # Number of span-wise sample stations per fin


@dataclass(frozen=True)
class FinStationGeometry:
    """Discretized fin-span sampling."""
    mac_length_m: float
    mac_lead_x_m: float
    mac_span_m: float
    cos_gamma_lead: float
    cos_gamma_mid: float
    roll_sum_m4: float
    span_stations_m: tuple[float, ...]
    chord_length_stations_m: tuple[float, ...]
    body_radius_m: float


def fin_geometry_stations(geometry: RocketGeometry, divisions: int = _FIN_DIVISIONS) -> FinStationGeometry:
    """Sample one fin's chord across the span.

    Produces the MAC length/leading-edge position/spanwise position, the
    cosine of both the leading-edge and mid-chord sweep angles, and the roll
    damping "rollSum" integral, all from per-station chord data (root radius
    offset by the body radius, per-station chord length, etc.).
    """
    points = fin_points_with_root(geometry)
    body_radius = geometry.diameter_m / 2.0
    if not points or divisions < 2:
        return FinStationGeometry(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, (), (), body_radius)

    ys = [p[1] for p in points]
    span = max(ys) - min(ys)
    if span <= 0.0:
        return FinStationGeometry(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, (), (), body_radius)

    y0 = min(ys)
    dy = span / (divisions - 1)

    chord_lead: list[float] = []
    chord_trail: list[float] = []
    for i in range(divisions):
        y = min(y0 + i * dy, y0 + span)
        bounds = _horizontal_span_intersections(points, y)
        if bounds is None:
            x_le = chord_lead[-1] if chord_lead else 0.0
            x_te = chord_trail[-1] if chord_trail else 0.0
        else:
            x_le, x_te = bounds
        chord_lead.append(x_le)
        chord_trail.append(x_te)

    mac_length = 0.0
    mac_lead = 0.0
    mac_span = 0.0
    cos_gamma_sum = 0.0
    cos_gamma_lead_sum = 0.0
    roll_sum = 0.0
    area = 0.0
    chord_length_stations: list[float] = []
    span_stations: list[float] = []

    for i in range(divisions):
        length = max(0.0, chord_trail[i] - chord_lead[i])
        y = i * dy
        span_stations.append(y)
        chord_length_stations.append(length)

        mac_length += length * length
        mac_span += y * length
        mac_lead += chord_lead[i] * length
        area += length
        roll_sum += length * (body_radius + y) ** 2

        if i > 0:
            dx_mid = (chord_trail[i] + chord_lead[i]) / 2.0 - (chord_trail[i - 1] + chord_lead[i - 1]) / 2.0
            hyp_mid = math.hypot(dx_mid, dy)
            if hyp_mid > 0.0:
                cos_gamma_sum += dy / hyp_mid

            dx_lead = chord_lead[i] - chord_lead[i - 1]
            hyp_lead = math.hypot(dx_lead, dy)
            if hyp_lead > 0.0:
                cos_gamma_lead_sum += dy / hyp_lead

    mac_length *= dy
    mac_span *= dy
    mac_lead *= dy
    area *= dy
    roll_sum *= dy

    if area > 1e-12:
        mac_length /= area
        mac_span /= area
        mac_lead /= area
    else:
        mac_length = 0.0
        mac_span = 0.0
        mac_lead = 0.0

    cos_gamma_mid = cos_gamma_sum / (divisions - 1) if divisions > 1 else 1.0
    cos_gamma_lead = cos_gamma_lead_sum / (divisions - 1) if divisions > 1 else 1.0

    return FinStationGeometry(
        mac_length_m=mac_length,
        mac_lead_x_m=mac_lead,
        mac_span_m=mac_span,
        cos_gamma_lead=cos_gamma_lead,
        cos_gamma_mid=cos_gamma_mid,
        roll_sum_m4=roll_sum,
        span_stations_m=tuple(span_stations),
        chord_length_stations_m=tuple(chord_length_stations),
        body_radius_m=body_radius,
    )


def fin_mac_properties(geometry: RocketGeometry) -> tuple[float, float, float, float]:
    """Return (mac_length_m, mac_lead_x_m, cos_gamma_lead, cos_gamma_mid) for one fin."""
    stations = fin_geometry_stations(geometry)
    return stations.mac_length_m, stations.mac_lead_x_m, stations.cos_gamma_lead, stations.cos_gamma_mid


def fin_aspect_ratio(geometry: RocketGeometry) -> float:
    """Fin aspect ratio AR = 2*span^2/area."""
    area = fin_planform_area_m2(geometry)
    if area <= 0.0:
        return 0.0
    return 2.0 * geometry.fin_span_m ** 2 / area


def body_planform_geometry(geometry: RocketGeometry) -> tuple[float, float]:
    """Return (cache_diameter_m, cache_length_m) for the pitch/yaw damping multiplier.

    The area-averaged "diameter" of the whole axisymmetric body (nose + body
    tube), computed as total planform area / total length.
    """
    cache_length = geometry.length_m
    if cache_length <= 0.0:
        return 0.0, 0.0
    _full_volume, _wetted_area, nose_planform_area, _planform_center = nose_profile_integration(geometry)
    tube_length = max(0.0, geometry.length_m - geometry.nose_length_m)
    tube_planform_area = geometry.diameter_m * tube_length  # l*(r+r) = l*diameter
    total_area = nose_planform_area + tube_planform_area
    cache_diameter = total_area / cache_length
    return cache_diameter, cache_length