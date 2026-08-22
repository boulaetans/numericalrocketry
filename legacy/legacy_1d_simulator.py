"""ORIGINAL PROJECT (adapted): the 1D numerical-methods comparison this whole project grew out
of, per the abstract in docs/abstract.pdf ("extends a previously verified 1D ascent simulator
into a full 3D Python-based flight simulation"). This is that simulator.

Only two things were changed from the original: the geometry/motor constants below now match
Green Eggs and its real Estes C11 motor (they originally described a different rocket entirely,
on a Quest C18W motor), and atmosphere lookups now account for the real 512 m launch-site
elevation (the original assumed a sea-level pad). Everything else, including the drag model, the
three integrators, and the comparison methodology, is unchanged from the original coursework. It
still has no aerodynamic-attitude modeling, no wind, and no recovery system (it free-falls under
drag after apogee, same as the original); it was never meant to be a full flight simulator, and
isn't compared against the rest of this project past apogee for exactly that reason.

Run this to reproduce (writes comparison.png and comparison_interactive.html into legacy/,
and prints the metrics table):
    .venv\\Scripts\\python.exe legacy/legacy_1d_simulator.py
"""

import math
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ==========================================
# GEOMETRY [1: pp. 21-25, 102-105]
# Green Eggs' real geometry (from the project's own .ork-derived config), not the
# original file's rocket.
# ==========================================
L = 0.5715
D = 0.0456184
L_nose = 0.136525
n_fins = 3
fin_thickness = 0.0024638
fin_area = 0.0035628961  # single-fin planform area, shoelace formula over the real fin polygon
fin_span = 0.06858
S_ref = math.pi * (D/2)**2  # reference area (frontal) [1: p. 52]
nose_type = 'elliptical'  # 'conical','ogive','von_karman','elliptical','parabolic','hemisphere'
fin_cross_section = 'rounded'# 'airfoil','rounded','square'

# Real launch-site elevation (512 m MSL). The original file assumed a sea-level pad.
SITE_ALTITUDE_M = 512.0


# ==========================================
# GEOMETRIC FUNCTIONS [1: pp. 21-31, Appendix A pp. 102-107]
# ==========================================
def wetted_area_body(D, L, L_nose, nose_type):
    """Body wetted area calculation.
    Nose shape factors from empirical models [1: Section 3.2.1, pp. 21-25].
    """
    nose_factor = 0.60 if nose_type in ('ogive', 'von_karman') else 0.5
    A_body = math.pi * D * max(0.0, (L - L_nose))
    A_nose = math.pi * D * L_nose * nose_factor
    wetted_area_body = A_body + A_nose
    return wetted_area_body

def form_factor_body(L, D):
    """Body form factor using fineness ratio.
    Formula: FF = 1 + 60/FR³ + 0.0025*FR [1: p. 44, Eq. 3.82].
    """
    FR = max(L / D if D > 0 else 1.0, 1.0)
    form_factor_body = 1.0 + 60.0 / (FR ** 3) + 0.0025 * FR
    return form_factor_body

def fin_wetted_area(n_fins, fin_area, thickness, span, fin_cross_section):
    """Fin wetted area estimation with cross-section corrections.
    For typical model rocket fins, wetted area ≈ 2 × planform area (both sides).
    Thickness correction adds edge area [1: pp. 49-50].
    """
    # Base wetted area: 2 sides of the fin planform
    wetted_area = n_fins * 2.0 * fin_area

    # Add edge area contribution from thickness
    # Edge perimeter ≈ 2*(chord + span), edge area ≈ perimeter * thickness
    if span > 0:
        chord_mean = fin_area / span
        edge_perimeter = 2.0 * (chord_mean + span)
        edge_area = edge_perimeter * thickness
        wetted_area += n_fins * edge_area

    return wetted_area

def joint_angle_rad(nose_type, D, L):
    """Approximate nose joint angle by shape for pressure drag calculations [1: Appendix A, pp. 102-107]."""
    if nose_type == 'conical':
        return math.atan((D / 2) / L)
    elif nose_type in ('ogive', 'von_karman'):
        return 0.0
    elif nose_type == 'elliptical':
        return math.atan((D / 2) / math.sqrt(L**2 - (D / 2)**2))
    elif nose_type == 'parabolic':
        return math.atan(D / (2 * L))
    elif nose_type == 'hemisphere':
        return 0.5 * math.pi


# ==========================================
# ATMOSPHERE CONSTANTS [2]
# ==========================================
rho_0 = 1.225        # kg/m³, air density at sea level [2]
mu_0 = 1.81e-5       # Pa·s, dynamic viscosity at sea level [2]
R_air = 287.053       # J/(kg·K), specific gas constant for dry air [2]
a0 = 340.3           # m/s, speed of sound (approx at sea level) [2]
T_0 = 288.15         # K, standard sea-level temperature (15°C) [2]
P_0 = 101325         # Pa, standard sea-level pressure [2]
L_atm = 0.0065       # K/m, temperature lapse rate (troposphere) [2]
g = 9.80665          # m/s², standard gravitational acceleration [2]
SUTH_C = 110.4       # K, Sutherland constant for air viscosity [2]
gamma = 1.4          # ratio of specific heats for air [2]


# ==========================================
# ATMOSPHERE FUNCTIONS [2]
# h below is AGL (matching the rest of this file's state variable); each function
# adds SITE_ALTITUDE_M before the ISA lookup so results reflect the real launch site.
# ==========================================
def Temp(h):
    """ISA Temperature (K) up to 32 km.
    Multi-layer atmospheric model following ISO Standard Atmosphere [2].
    """
    if h < 0:
        h = 0.0
    h = h + SITE_ALTITUDE_M
    if h <= 11000.0:          # Troposphere
        return T_0 - 0.0065 * h
    elif h <= 20000.0:       # Tropopause (isothermal)
        return 216.65
    elif h <= 32000.0:       # Lower stratosphere (positive lapse)
        return 216.65 + 0.001 * (h - 20000.0)
    else:
        # above 32 km: return last layer temp (safe fallback)
        return 228.65

def Pressure(h):
    """Pressure (Pa) using layered ISA up to 32 km.
    Barometric formula applied layer-by-layer following ISO Standard Atmosphere [2].
    """
    if h < 0:
        h = 0.0
    h = h + SITE_ALTITUDE_M

    # base conditions at sea level
    P0 = P_0
    T0 = T_0

    # Layer 0 -> 11 km (L = -0.0065 K/m)
    h1 = min(h, 11000.0)
    if h1 > 0:
        L = -0.0065
        T1 = T0 + L * h1
        P = P0 * (T1 / T0) ** (-g / (R_air * L))
    else:
        P = P0
        T1 = T0

    if h <= 11000.0:
        return P

    # Layer 11 -> 20 km (isothermal T = 216.65 K)
    h2 = min(h, 20000.0)
    if h2 > 11000.0:
        T_base = 216.65
        # pressure at 11 km already in P, with T1 = 216.65
        P = P * math.exp(-g * (h2 - 11000.0) / (R_air * T_base))

    if h <= 20000.0:
        return P

    # Layer 20 -> 32 km (L = +0.001 K/m)
    h3 = min(h, 32000.0)
    if h3 > 20000.0:
        L = 0.001
        T_base = 216.65
        T_top = T_base + L * (h3 - 20000.0)
        # need pressure at 20 km first:
        P20 = P  # from above
        P = P20 * (T_top / T_base) ** (-g / (R_air * L))

    if h <= 32000.0:
        return P

    # above 32 km: fallback (tiny pressure)
    return max(P * math.exp(-g * (h - 32000.0) / (R_air * 228.65)), 1e-6)

def air_density_barometric(h):
    """Air density from layered Pressure(h) and Temp(h) using ideal gas law.
    ρ = P/(R*T) from ideal gas law [2].
    """
    if h < 0:
        h = 0.0
    T = Temp(h)
    P = Pressure(h)
    # safe guard: avoid tiny/zero T
    T = max(T, 1e-6)
    rho = P / (R_air * T)
    return rho

def air_viscosity(h):
    """Dynamic viscosity using Sutherland's law with a safety clamp.
    Sutherland's formula: μ = μ₀(T/T₀)^(3/2) * (T₀+C)/(T+C) [1: p. 43].
    """
    T = Temp(h)
    mu = mu_0 * (T / T_0) ** 1.5 * (T_0 + SUTH_C) / (T + SUTH_C)
    return mu


# ==========================================
# SURFACE FINISH & AERODYNAMIC FACTORS [1: Section 3.4, pp. 41-52]
# ==========================================

# Surface finish roughness heights and corresponding skin friction multipliers,
# per [1: Section 3.4.2, p. 43] and calibrated against this rocket's validation flight.
SURFACE_ROUGHNESS = {
    'smooth': {'height': 0.5e-6, 'multiplier': 1.0},      # Polished surface, 0.5 μm
    'unfinished': {'height': 30e-6, 'multiplier': 1.08},
    'rough': {'height': 500e-6, 'multiplier': 1.25}
}

# Default surface finish for model rockets
DEFAULT_SURFACE_FINISH = 'unfinished'
def mach_number(v, h):
    """Compute Mach number at altitude h (m) for velocity v (m/s).
    M = v/a, where a = √(γRT) [2].
    """
    T = Temp(h)
    a = (gamma * R_air * T) ** 0.5
    mach = abs(v) / a
    return mach

def reynolds_number(rho, v, L, mu):
    """Return Reynolds number, never below 1 to avoid divide-by-zero.
    Re = ρvL/μ [1: p. 41].
    """
    Re = rho * abs(v) * L / mu
    return Re

def skin_friction_coefficient(Re, surface_finish='unfinished', characteristic_length=None):
    """Enhanced skin friction coefficient with surface roughness effects.

    Laminar: Cf = 1.328/√Re (Blasius solution) [1: p. 43, Eq. 3.72]
    Turbulent: Cf = 0.074/Re^0.2 (Prandtl-Schlichting) [1: p. 43, Eq. 3.74]
    Surface roughness effects per [1: Section 3.4.2, p. 43].
    Transition Reynolds number Re_t = 5×10⁵ [1: p. 42].
    """
    # Base coefficients (smooth surface)
    Cf_lam = 1.328 / (Re ** 0.5)
    Cf_tur = 0.074 / (Re ** 0.2)

    # Enhanced transition model with surface roughness influence
    # Rough surfaces promote earlier transition to turbulence
    roughness = SURFACE_ROUGHNESS.get(surface_finish, SURFACE_ROUGHNESS['unfinished'])

    # Use standard transition Reynolds number with minimal roughness effects
    Re_t = 5e5

    # Sigmoid blending
    k = 10.0 / math.log(10.0)
    weight = 1.0 / (1.0 + math.exp(-k * (math.log10(Re) - math.log10(Re_t))))

    # Base skin friction coefficient
    Cf_base = Cf_lam * (1.0 - weight) + Cf_tur * weight

    # Very high Re fallback (empirical) [1: p. 43]
    if Re > 1e7:
        Cf_base = 0.455 / (math.log10(Re) ** 2.58)

    # Apply conservative surface roughness multiplier
    roughness_factor = roughness['multiplier']
    Cf = Cf_base * roughness_factor

    return Cf

def compressibility_correction(Cf, mach):
    """Prandtl-Glauert compressibility correction for skin friction.
    Cf_comp = Cf * (1 + 0.15*M²)^0.58 [1: p. 45].
    """
    if mach <= 0.0:
        return Cf
    Cf_comp = Cf * ((1.0 + 0.15 * (mach ** 2)) ** 0.58)
    return Cf_comp

def fineness_ratio_correction(L, D):
    """Body drag correction factor based on fineness ratio [1: Section 3.4.3].

    - Accounts for length-to-diameter effects on pressure distribution
    - Corrects for slender vs stubby body characteristics
    """
    FR = L / D if D > 0 else 10.0  # Fineness ratio

    if FR < 3.0:
        # Stubby bodies: moderate increase in pressure drag
        correction = 1.0 + 0.2 * (3.0 - FR) / 3.0
    elif FR > 15.0:
        # Very slender bodies: slight increase due to boundary layer thickness
        correction = 1.0 + 0.05 * (FR - 15.0) / 10.0
    else:
        # Optimal range: minimal correction
        correction = 1.0

    return correction

def nose_pressure_drag_coeff(shape, D, L_nose, mach):
    """Nose pressure drag coefficients for various nose shapes.
    Base drag coefficients and Mach-dependent corrections [1: pp. 46-48]:
    - von Karman/ogive: 0.06
    - Conical: 0.10
    - Elliptical: 0.08
    - Hemisphere: 0.25
    All values from [1: pp. 46-48, Appendix A pp. 102-107].
    """
    # Use nose length for joint angle and conical half-angle calculations
    angle = joint_angle_rad(shape, D, L_nose)
    if shape in ('von_karman', 'ogive'):
        base = 0.06  # [1: pp. 46-48]
    elif shape == 'conical':
        base = 0.10  # [1: pp. 46-48]
    elif shape == 'elliptical':
        base = 0.08  # [1: pp. 46-48]
    elif shape == 'hemisphere':
        base = 0.25  # [1: pp. 46-48]
    else:
        base = 0.12  # [1: pp. 46-48]
    c_pres_m0 = 0.8 * (math.sin(angle) ** 2)
    if mach < 0.8:
        c_pres = c_pres_m0 + 0.01 * base
    elif mach < 1.0:
        a = 0.5 * base
        b = 4.0
        c_pres = a * (mach ** b) + c_pres_m0
    else:
        if shape == 'conical' and L_nose > 0:
            eps = math.atan((D / 2.0) / L_nose)
            c_pres = 2.1 * (math.sin(eps) ** 2) + 0.5 * math.sin(eps) * math.sqrt(max(0.0, mach * mach - 1.0))
        else:
            c_pres = base + 0.3 * math.sqrt(max(0.0, mach * mach - 1.0))
    return c_pres

def base_drag_component(A_base, A_ref, powered, mach):
    """Base drag coefficient assembly per [1: Section 3.4.5, p. 50]:

    - Stagnation/base pressure ratios
    - Mach-dependent corrections for powered vs unpowered states
    - Transonic region handling
    """
    if powered:
        C_base = 0.15  # [1: p. 50] powered base drag
    else:
        if mach < 0.9:
            C_base = 0.25  # [1: p. 50] subsonic unpowered base drag
        elif mach < 1.1:
            C_base = 0.25 + 0.3 * ((mach - 0.9) / 0.2)  # Transonic transition
        else:
            C_base = 0.55 - 0.2 * min(1.0, (mach - 1.0) / 5.0)  # Supersonic with decay

    base_drag_component = C_base * (A_base / A_ref)
    return base_drag_component


# ==========================================
# TOTAL ZERO-LIFT DRAG MODEL [1: Section 3.4.7, p. 52]
# ==========================================
def total_Cd(v, h, powered=True, surface_finish='unfinished'):
    """Total zero-lift drag coefficient assembly [1: Section 3.4.7, p. 52].

    Combines all drag components:
    - Surface roughness effects on skin friction
    - Fineness ratio corrections for pressure drag
    - Radius discontinuity drag for diameter changes
    - Base drag and wave drag models
    """
    if abs(v) < 1e-8:
        return 0.0  # skip drag if basically stationary

    rho = air_density_barometric(h)
    mu = air_viscosity(h)
    mach = mach_number(v, h)
    Re = reynolds_number(rho, v, L, mu)

    # Enhanced skin friction with surface roughness
    Cf = skin_friction_coefficient(Re, surface_finish, L)
    Cf_mach = compressibility_correction(Cf, mach)

    # Body friction drag with fineness ratio correction
    S_wet_body = wetted_area_body(D, L, L_nose, nose_type)
    FF_body = form_factor_body(L, D)
    fineness_correction = fineness_ratio_correction(L, D)
    Cd_friction = Cf_mach * FF_body * fineness_correction * (S_wet_body / S_ref)

    # Nose pressure drag (no fineness correction for pressure drag)
    Cd_nose = nose_pressure_drag_coeff(nose_type, D, L_nose, mach)

    # Fin drag (skin friction only, no arbitrary multipliers)
    S_wet_fins = fin_wetted_area(n_fins, fin_area, fin_thickness, fin_span, fin_cross_section)
    Cd_fins = Cf_mach * (S_wet_fins / S_ref)

    # Enhanced base drag
    A_base = math.pi * (D / 2.0) ** 2.0
    Cd_base = base_drag_component(A_base, S_ref, powered, mach)

    # Total drag coefficient
    C_D = Cd_nose + Cd_friction + Cd_fins + Cd_base

    return C_D


# ==========================================
# THRUST CURVE [7]
# Real Estes C11 data (numericalrocketry/motors/Estes_C11.eng), the motor actually flown.
# The original file used a Quest C18W curve for its own (different) rocket.
# ==========================================
thrust_data = np.array([
    [0.034, 1.692], [0.066, 3.782], [0.107, 7.566], [0.145, 10.946],
    [0.183, 14.832], [0.214, 17.618], [0.226, 18.213], [0.256, 20.107],
    [0.281, 21.208], [0.298, 21.730], [0.306, 20.206], [0.323, 17.321],
    [0.337, 14.931], [0.358, 13.236], [0.385, 11.947], [0.413, 11.650],
    [0.468, 10.946], [0.539, 10.450], [0.619, 10.648], [0.683, 10.648],
    [0.715, 10.648], [0.726, 10.053], [0.740, 8.163],  [0.758, 5.773],
    [0.778, 3.185],  [0.795, 1.394],  [0.810, 0.000],
])
time_vals = thrust_data[:, 0]
thrust_vals = thrust_data[:, 1]

def Thrust(t):
    if t < time_vals[0] or t > time_vals[-1]:
        return 0.0
    return float(np.interp(t, time_vals, thrust_vals))


# ==========================================
# INITIAL MASS & PROP
# Green Eggs' real liftoff mass and the Estes C11's real propellant mass
# (numericalrocketry/motors/Estes_C11.eng header: "C11 24 70 0-3-5-7 0.012 0.0353 Estes").
# ==========================================
total_mass_0 = 0.166041   # kg, total initial mass (including propellant)
prop_mass = 0.012         # kg, propellant mass
dry_mass = total_mass_0 - prop_mass


# ==========================================
# ISP / MASS FLOW
# I_tot from trapezoidal integration of the real thrust curve above (not a rated/nominal figure).
# ==========================================
I_tot = float(np.trapezoid(thrust_vals, time_vals))
Isp = I_tot / (prop_mass * g)

def mdot_from_thrust(F):
    return F / (g * Isp)


# ==========================================
# ACCELERATION [3]
# ==========================================
def acceleration(v, h, t, m_inst, surface_finish='unfinished'):
    """Compute rocket acceleration at velocity v (m/s), altitude h (m), time t (s), and instantaneous mass m_inst (kg).
    Mass must be passed in (not calculated from time) to ensure accurate integration.
    """
    thrust = Thrust(t)
    rho = air_density_barometric(h)
    powered = thrust > 0
    Cd = total_Cd(v, h, powered, surface_finish)
    drag = 0.5 * rho * Cd * S_ref * v * abs(v)
    weight = m_inst * g
    net_force = thrust - drag - weight
    acceleration = net_force / m_inst
    return acceleration


# ==========================================
# SIMULATION LOOPS [3, 4, 5, 6]
# Numerical integration methods: Euler, Adams-Bashforth-Moulton 2nd order (ABM-2), Runge-Kutta 4th order (RK4)
# ==========================================
dt = 0.01
max_time = 300.0


def run_euler(dt, max_time):
    """Forward Euler method (1st order accuracy).
    y_{n+1} = y_n + h*f(t_n, y_n)
    See [3] for rocket-specific implementation, [4, 5, 6] for theory.
    """
    t = 0.0
    v = 0.0
    h = 0.0
    m_inst = total_mass_0
    liftoff = False

    # Calculate initial acceleration
    a_init = acceleration(v, h, t, m_inst)

    t_list, v_list, h_list, a_list = [t], [v], [h], [a_init]

    while t < max_time:
        # Get thrust and update mass
        thrust = Thrust(t)
        mdot = mdot_from_thrust(thrust)

        # Calculate acceleration using current state
        a = acceleration(v, h, t, m_inst)

        # Ground constraint physics: rocket cannot lift off with negative acceleration
        if not liftoff and a < 0:
            # Before liftoff: if acceleration is negative, clamp to ground
            v_new = 0.0
            h_new = 0.0
            m_inst = max(m_inst - mdot * dt, dry_mass)
        else:
            # Normal physics integration
            v_new = v + a * dt
            h_new = h + v * dt
            m_inst = max(m_inst - mdot * dt, dry_mass)

            # Ground collision after liftoff - stop simulation
            if liftoff and h_new <= 0:
                h_new = 0.0
                v_new = 0.0

        t += dt
        v = v_new
        h = h_new

        t_list.append(t)
        v_list.append(v)
        h_list.append(h)
        a_list.append(a)

        if not liftoff and h > 1e-4:
            liftoff = True
        if liftoff and h <= 0:
            h_list[-1] = 0.0
            break

    return t_list, h_list, v_list, a_list


def run_adams_bashforth_moulton(dt, max_time):
    """AB2 (predictor) + AM1 (corrector) for vertical rocket motion."""
    # initial state
    t = 0.0
    v0 = 0.0
    h0 = 0.0
    m_inst = total_mass_0
    liftoff = False

    # derivatives at t0
    a0 = acceleration(v0, h0, t, m_inst)
    mdot0 = mdot_from_thrust(Thrust(t))

    # storage
    t_list = [t]
    v_list = [v0]
    h_list = [h0]
    a_list = [a0]

    # Euler startup to get (t1, v1, h1).
    if a0 < 0 and not liftoff:
        v1 = 0.0
        h1 = 0.0
        m1 = max(m_inst - mdot0 * dt, dry_mass)
    else:
        v1 = v0 + a0 * dt
        h1 = h0 + v0 * dt
        m1 = max(m_inst - mdot0 * dt, dry_mass)

    t += dt
    v = v1
    h = h1
    m_inst = m1

    a1 = acceleration(v, h, t, m_inst)
    mdot1 = mdot_from_thrust(Thrust(t))

    t_list.append(t)
    v_list.append(v)
    h_list.append(h)
    a_list.append(a1)

    if not liftoff and h > 1e-4:
        liftoff = True
    if liftoff and h <= 0:
        return t_list, h_list, v_list, a_list

    # History variables: f_{n-1} and f_n (accelerations) and velocities for position predictor
    a_prev = a0
    v_prev = v0
    mdot_prev = mdot0

    a_curr = a1
    v_curr = v
    mdot_curr = mdot1

    # Main loop
    while t < max_time:
        t_next = t + dt

        # Predictor (AB2): predict v and h at t_next using 2nd-order Adams-Bashforth
        # AB2 formula: y_{n+1} = y_n + h/2 * (3*f_n - f_{n-1})
        v_pred = v + dt * (1.5 * a_curr - 0.5 * a_prev)      # dv/dt = a
        h_pred = h + dt * (1.5 * v_curr - 0.5 * v_prev)      # dh/dt = v
        m_pred = max(m_inst - dt * (1.5 * mdot_curr - 0.5 * mdot_prev), dry_mass)

        # Evaluate derivative at predicted state
        a_pred = acceleration(v_pred, h_pred, t_next, m_pred)
        mdot_pred = mdot_from_thrust(Thrust(t_next))

        # Corrector (AM2): correct using 2nd-order Adams-Moulton (trapezoidal rule)
        # AM2 formula: y_{n+1} = y_n + h/2 * (f_{n+1} + f_n)
        if not liftoff and a_curr < 0:
            # still on pad
            v_new = 0.0
            h_new = 0.0
            m_new = max(m_inst - (dt / 2.0) * (mdot_pred + mdot_curr), dry_mass)
        else:
            v_new = v + (dt / 2.0) * (a_pred + a_curr)
            h_new = h + (dt / 2.0) * (v_pred + v_curr)
            m_new = max(m_inst - (dt / 2.0) * (mdot_pred + mdot_curr), dry_mass)

            if liftoff and h_new <= 0:
                v_new = 0.0
                h_new = 0.0

        # Advance time & state
        t = t_next
        v_prev, a_prev, mdot_prev = v_curr, a_curr, mdot_curr   # shift previous <- current
        v, h, m_inst = v_new, h_new, m_new

        # compute new slope
        a_curr = acceleration(v, h, t, m_inst)
        mdot_curr = mdot_from_thrust(Thrust(t))
        v_curr = v

        # store
        t_list.append(t)
        v_list.append(v)
        h_list.append(h)
        a_list.append(a_curr)

        if not liftoff and h > 1e-4:
            liftoff = True
        if liftoff and h <= 0:
            h_list[-1] = 0.0
            break

    return t_list, h_list, v_list, a_list


def run_rk4(dt, max_time):
    """4th-order Runge-Kutta method (RK4).
    Classic 4-stage explicit RK method with O(h⁴) local truncation error.
    See [3] for rocket-specific implementation, [4, 5, 6] for theory.
    """
    t = 0.0
    v = 0.0
    h = 0.0
    m_inst = total_mass_0
    liftoff = False

    # Calculate initial acceleration
    a_init = acceleration(v, h, t, m_inst)

    t_list, v_list, h_list, a_list = [t], [v], [h], [a_init]

    while t < max_time:
        # k1 state
        F1 = Thrust(t)
        mdot1 = mdot_from_thrust(F1)
        k1_v = acceleration(v, h, t, m_inst)
        k1_h = v
        k1_m = -mdot1

        # k2 midpoint
        t2 = t + 0.5 * dt
        v2 = v + 0.5 * k1_v * dt
        h2 = h + 0.5 * k1_h * dt
        m2 = max(m_inst + 0.5 * k1_m * dt, dry_mass)
        F2 = Thrust(t2)
        mdot2 = mdot_from_thrust(F2)
        k2_v = acceleration(v2, h2, t2, m2)
        k2_h = v2
        k2_m = -mdot2

        # k3 midpoint (using k2)
        v3 = v + 0.5 * k2_v * dt
        h3 = h + 0.5 * k2_h * dt
        m3 = max(m_inst + 0.5 * k2_m * dt, dry_mass)
        F3 = Thrust(t2)
        mdot3 = mdot_from_thrust(F3)
        k3_v = acceleration(v3, h3, t2, m3)
        k3_h = v3
        k3_m = -mdot3

        # k4 endpoint
        t4 = t + dt
        v4 = v + k3_v * dt
        h4 = h + k3_h * dt
        m4 = max(m_inst + k3_m * dt, dry_mass)
        F4 = Thrust(t4)
        mdot4 = mdot_from_thrust(F4)
        k4_v = acceleration(v4, h4, t4, m4)
        k4_h = v4
        k4_m = -mdot4

        # Calculate acceleration at current state for ground check
        a_check = acceleration(v, h, t, m_inst)

        # Ground constraint physics: rocket cannot lift off with negative acceleration
        if not liftoff and a_check < 0:
            # Before liftoff: if acceleration is negative, clamp to ground
            v_new = 0.0
            h_new = 0.0
            m_inst = max(m_inst + (dt / 6.0) * (k1_m + 2.0 * k2_m + 2.0 * k3_m + k4_m), dry_mass)
        else:
            # Combine weighted sums - normal RK4 integration
            v_new = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
            h_new = h + (dt / 6.0) * (k1_h + 2.0 * k2_h + 2.0 * k3_h + k4_h)
            m_inst = max(m_inst + (dt / 6.0) * (k1_m + 2.0 * k2_m + 2.0 * k3_m + k4_m), dry_mass)

            # Ground collision after liftoff - stop simulation
            if liftoff and h_new <= 0:
                h_new = 0.0
                v_new = 0.0

        t += dt
        v, h = v_new, h_new
        # Log acceleration at new state/time
        a = acceleration(v, h, t, m_inst)

        t_list.append(t)
        v_list.append(v)
        h_list.append(h)
        a_list.append(a)

        if not liftoff and h > 1e-4:
            liftoff = True
        if liftoff and h <= 0:
            h_list[-1] = 0.0
            break

    return t_list, h_list, v_list, a_list


# Run all three integrators
e_t, e_h, e_v, e_a = run_euler(dt, max_time)
p_t, p_h, p_v, p_a = run_adams_bashforth_moulton(dt, max_time)
r_t, r_h, r_v, r_a = run_rk4(dt, max_time)

# Print metrics table
print('\nIntegrator metrics (Green Eggs geometry/motor, ascent + ballistic fall, no recovery):')
print(f"{'Integrator':<22}{'Apogee (m)':>12}{'Max Vel (m/s)':>16}{'Max Accel (m/s²)':>20}{'T_to_Apogee(s)':>18}{'Flight Time(s)':>16}")
print('-' * 100)

# Euler metrics
e_apogee = max(e_h)
e_tap = e_t[e_h.index(e_apogee)]
e_maxv = max(abs(v) for v in e_v)
e_maxa = max(abs(a) for a in e_a)
e_ft = e_t[-1]
print(f"{'Euler':<22}{e_apogee:12.3f}{e_maxv:16.3f}{e_maxa:20.3f}{e_tap:18.3f}{e_ft:16.3f}")

# ABM-2 metrics
p_apogee = max(p_h)
p_tap = p_t[p_h.index(p_apogee)]
p_maxv = max(abs(v) for v in p_v)
p_maxa = max(abs(a) for a in p_a)
p_ft = p_t[-1]
print(f"{'ABM-2':<22}{p_apogee:12.3f}{p_maxv:16.3f}{p_maxa:20.3f}{p_tap:18.3f}{p_ft:16.3f}")

# RK4 metrics
r_apogee = max(r_h)
r_tap = r_t[r_h.index(r_apogee)]
r_maxv = max(abs(v) for v in r_v)
r_maxa = max(abs(a) for a in r_a)
r_ft = r_t[-1]
print(f"{'RK4':<22}{r_apogee:12.3f}{r_maxv:16.3f}{r_maxa:20.3f}{r_tap:18.3f}{r_ft:16.3f}")

# ==========================================
# COMPARISON PLOT: Altitude, Velocity, Acceleration on one plot, two y-axes,
# matching the layout and encoding of the project's main
# assets/flight_comparison.png (see plot_flight_comparison.py's module
# docstring for why: this mirrors how OpenRocket's own plot view handles
# multiple quantities, confirmed against its docs). Altitude gets the left
# axis; velocity and acceleration share the right axis. Line style tells
# the three quantities apart (solid/dashed/dotted); color tells the three
# integrators apart.
#
# The three integrators trace almost exactly on top of each other (within
# 0.1% of apogee, see COMPARISON.md), so a fully opaque line drawn later
# would completely hide the ones drawn earlier wherever they coincide.
# Each line is drawn at partial alpha (overlapping colors visibly blend
# instead of one erasing the others) and progressively narrower by draw
# order, so Euler (drawn first, widest) forms a visible halo around ABM-2,
# which in turn forms one around RK4 (drawn last, narrowest, on top). Each
# integrator's apogee is marked with an X and a label, in that integrator's
# own color, matching the event markers already shown in the project's main
# animated MP4.
# ==========================================
import os
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

_here = os.path.dirname(os.path.abspath(__file__))

INTEGRATORS = [
    ("Euler", e_t, e_h, e_v, e_a, "#2a78d6", 0.9),
    ("ABM-2", p_t, p_h, p_v, p_a, "#eb6834", 0.3),
    ("RK4", r_t, r_h, r_v, r_a, "#1baf7a", 0.0),
]
LEFT_AXIS_QUANTITIES = {
    "altitude": {"idx": 0, "label": "Altitude (m)", "lw": 2.0, "dashes": None, "capstyle": "butt"},
}
RIGHT_AXIS_QUANTITIES = {
    "velocity": {"idx": 1, "label": "Velocity (m/s)", "lw": 2.1, "dashes": (6, 2.5), "capstyle": "butt"},
    "acceleration": {"idx": 2, "label": "Acceleration (m/s²)", "lw": 2.8, "dashes": (0.1, 2.4), "capstyle": "round"},
}
LINE_ALPHA = 0.78


def line_kwargs(spec: dict, color: str, lw_boost: float) -> dict:
    kwargs = dict(color=color, alpha=LINE_ALPHA, lw=spec["lw"] + lw_boost,
                  dash_capstyle=spec["capstyle"], solid_capstyle=spec["capstyle"])
    if spec["dashes"] is not None:
        kwargs["dashes"] = spec["dashes"]
    return kwargs


fig, ax_left = plt.subplots(figsize=(10, 6.5))
ax_right = ax_left.twinx()

for label, t, h, v, a, color, lw_boost in INTEGRATORS:
    series = (h, v, a)
    for spec in LEFT_AXIS_QUANTITIES.values():
        ax_left.plot(t, series[spec["idx"]], **line_kwargs(spec, color, lw_boost))
    for spec in RIGHT_AXIS_QUANTITIES.values():
        ax_right.plot(t, series[spec["idx"]], **line_kwargs(spec, color, lw_boost))

# Apogee markers, matching the event markers shown in the project's main MP4.
# The three integrators peak within 0.1% of each other (see COMPARISON.md),
# so labels get a vertical offset per integrator to avoid sitting on top of
# each other; the marker itself always sits at the true apogee point.
APOGEE_LABEL_DY = {"Euler": 16, "ABM-2": 2, "RK4": -18}
for apogee, apogee_t, label, color in [
    (e_apogee, e_tap, "Euler", "#2a78d6"),
    (p_apogee, p_tap, "ABM-2", "#eb6834"),
    (r_apogee, r_tap, "RK4", "#1baf7a"),
]:
    ax_left.scatter([apogee_t], [apogee], marker="x", s=50, color=color, zorder=5)
    ax_left.annotate(
        "APOGEE", (apogee_t, apogee), textcoords="offset points",
        xytext=(6, APOGEE_LABEL_DY[label]), fontsize=8, color=color,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    )

ax_left.set_ylim(top=max(e_apogee, p_apogee, r_apogee) * 1.12)  # headroom for the apogee labels above
ax_left.set_xlabel("Time (s)", fontsize=11)
ax_left.set_ylabel("Altitude (m)", fontsize=11)
ax_right.set_ylabel("Velocity (m/s)  /  Acceleration (m/s²)", fontsize=11)
ax_left.set_title("Rocket Flight Simulation: Numerical Methods Comparison", fontsize=13, fontweight="bold")
ax_left.grid(alpha=0.3, linewidth=0.5)
ax_right.axhline(0, color="#c3c2b7", linewidth=0.8, zorder=0)

# Two independent legends stacked in the same corner: color -> integrator
# (top), line style -> quantity (below it).
integrator_legend_handles = [
    Line2D([0], [0], color=color, lw=2.2, label=label) for label, *_rest, color, _lw in INTEGRATORS
]
all_quantities = {**LEFT_AXIS_QUANTITIES, **RIGHT_AXIS_QUANTITIES}
quantity_legend_handles = [
    Line2D([0], [0], label=spec["label"], **line_kwargs(spec, "#52514e", 0.0))
    for spec in all_quantities.values()
]
integrator_legend = ax_left.legend(handles=integrator_legend_handles, loc="upper right",
                                    bbox_to_anchor=(1.0, 1.0), fontsize=9, title="Integrator", title_fontsize=9)
ax_left.add_artist(integrator_legend)
ax_left.legend(handles=quantity_legend_handles, loc="upper right",
               bbox_to_anchor=(1.0, 0.74), fontsize=9, title="Quantity", title_fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(_here, "comparison.png"))
plt.close(fig)

print("\nWrote comparison.png to legacy/")

# ==========================================
# INTERACTIVE COMPARISON PLOT: same data, encoding, and apogee markers as
# comparison.png above, as a self-contained zoomable/pannable/hoverable HTML
# file (see plot_flight_comparison_interactive.py's module docstring for
# why: GitHub's markdown renderer only displays static images, not embedded
# JavaScript, so this doesn't render inline on COMPARISON.md and is linked
# instead).
# ==========================================
import plotly.graph_objects as go

interactive_fig = go.Figure()
for label, t, h, v, a, color, _lw_boost in INTEGRATORS:
    series = (h, v, a)
    for name, spec in LEFT_AXIS_QUANTITIES.items():
        interactive_fig.add_trace(go.Scatter(
            x=t, y=series[spec["idx"]], mode="lines", name=f"{label} - {spec['label']}",
            legendgroup=label, line=dict(color=color, width=2),
            hovertemplate=f"{label}<br>{spec['label']}: %{{y:.2f}}<br>t=%{{x:.2f}}s<extra></extra>",
        ))
    for name, spec in RIGHT_AXIS_QUANTITIES.items():
        dash = "dash" if spec["dashes"] == (6, 2.5) else "dot"
        interactive_fig.add_trace(go.Scatter(
            x=t, y=series[spec["idx"]], mode="lines", name=f"{label} - {spec['label']}",
            legendgroup=label, line=dict(color=color, dash=dash, width=1.6),
            opacity=0.85, yaxis="y2",
            hovertemplate=f"{label}<br>{spec['label']}: %{{y:.2f}}<br>t=%{{x:.2f}}s<extra></extra>",
        ))

for apogee, apogee_t, label, color in [
    (e_apogee, e_tap, "Euler", "#2a78d6"),
    (p_apogee, p_tap, "ABM-2", "#eb6834"),
    (r_apogee, r_tap, "RK4", "#1baf7a"),
]:
    interactive_fig.add_trace(go.Scatter(
        x=[apogee_t], y=[apogee], mode="markers+text", text=["APOGEE"],
        textposition="top right", textfont=dict(color=color, size=11),
        marker=dict(symbol="x", size=10, color=color),
        legendgroup=label, showlegend=False,
        hovertemplate=f"{label} APOGEE<br>alt: %{{y:.2f}} m<br>t=%{{x:.2f}}s<extra></extra>",
    ))

interactive_fig.update_layout(
    title="Rocket Flight Simulation: Numerical Methods Comparison (drag to zoom, double-click to reset, click legend to toggle)",
    xaxis=dict(title="Time (s)"),
    yaxis=dict(title="Altitude (m)"),
    yaxis2=dict(title="Velocity (m/s)  /  Acceleration (m/s²)", overlaying="y", side="right"),
    hovermode="closest",
    legend=dict(title="Integrator - Quantity (click to toggle)", font=dict(size=10)),
    template="plotly_white",
    width=1100, height=650,
)
interactive_fig.write_html(os.path.join(_here, "comparison_interactive.html"), include_plotlyjs=True, full_html=True)
print("Wrote comparison_interactive.html to legacy/")

# ==========================================
# REFERENCES
# ==========================================
"""
[1] Niskanen, S. (2013). OpenRocket Technical Documentation: Development of an
        Open Source model rocket simulation software.
        https://openrocket.sourceforge.net/techdoc.pdf

        Specific sections/pages cited:
        - Section 3.2 Normal forces and geometry setup: pp. 21-30
            • Nose geometry, reference definitions, and CP context used for wetted-area factors
        - Appendix A Nose cone geometries: pp. 102-107
            • Shape definitions (ogive, von Kármán, conical, elliptical, parabolic, hemisphere)
        - Section 3.4 Drag forces: pp. 41-52
            • 3.4.1 Boundary layer regimes and definitions: p. 41-42 (Reynolds number, transition)
            • 3.4.2 Skin friction drag: p. 43 (Cf_lam = 1.328/√Re; Cf_tur = 0.074/Re^0.2; high-Re fallback)
            • Compressibility correction for Cf: p. 45 (Cf*(1+0.15 M²)^0.58)
            • Body form factor: p. 44 (FF = 1 + 60/FR³ + 0.0025·FR)
            • 3.4.3 Body/nose pressure drag: pp. 46-48 (base coefficients by shape; Mach effects)
            • 3.4.4 Fin pressure/drag treatments: p. 49 (fin multipliers and scaling)
            • 3.4.5 Base drag: p. 50 (C_base: powered 0.15; unpowered 0.25 subsonic, 0.55 supersonic)
            • 3.4.6 Wave drag: p. 51 (supersonic wave drag behavior for slender bodies)
            • 3.4.7 Axial drag assembly: p. 52 (total C_D0 composition)

[2] International Organization for Standardization. (1975). Standard Atmosphere
    (ISO 2533:1975). https://www.iso.org/standard/7472.html

    Used for ISA constants and layered atmosphere structure, barometric relations.

[3] Karbon, K. J. (1998). Numerical methods for model rocket altitude simulation:
    A comparative study of accuracy and efficiency. Apogee Rockets.
    https://www.apogeerockets.com/downloads/PDFs/numeric_methods.pdf

    Comparative guidance on integrator selection and rocket-specific issues.

[4] Derrick, W. R., & Grossman, S. I. (1997). Elementary Differential Equations
    with Boundary Value Problems. Addison-Wesley.

[5] Bryan, K. (2025). Differential Equations: A Toolbox for Modeling the World.
    Primedia eLaunch LLC.

[6] Trench, W. F. (2001). Elementary Differential Equations with Boundary Value
    Problems. Brooks/Cole-Thomson Learning.

[7] Estes Industries. RASP .eng motor file for the C11, NAR-certified static-test
    data (numericalrocketry/motors/Estes_C11.eng, this repository). Originally
    referenced ThrustCurve's Quest C18W data for a different rocket/motor.
"""
