"""Simple atmosphere utilities used by drag and Mach/Reynolds calculations."""

import math

from ..constants import (
    G0,
    GAMMA_AIR,
    LAPSE_STRATOSPHERE,
    LAPSE_TROPOSPHERE,
    LOWER_STRATOSPHERE_TOP,
    MU0,
    P0,
    R_AIR,
    SUTHERLAND_C,
    T0,
    TROPOPAUSE_TOP,
    TROPOSPHERE_TOP,
)


def temperature_isa(altitude_m: float) -> float:
    altitude_m = max(0.0, altitude_m)
    if altitude_m <= TROPOSPHERE_TOP:
        return T0 + LAPSE_TROPOSPHERE * altitude_m
    if altitude_m <= TROPOPAUSE_TOP:
        return 216.65
    if altitude_m <= LOWER_STRATOSPHERE_TOP:
        return 216.65 + LAPSE_STRATOSPHERE * (altitude_m - TROPOPAUSE_TOP)
    return 228.65


def pressure_isa(altitude_m: float) -> float:
    altitude_m = max(0.0, altitude_m)

    temperature = T0
    pressure = P0

    if altitude_m <= TROPOSPHERE_TOP:
        temperature_top = temperature_isa(altitude_m)
        return pressure * (temperature_top / temperature) ** (-G0 / (R_AIR * LAPSE_TROPOSPHERE))

    temperature_top = temperature_isa(TROPOSPHERE_TOP)
    pressure = pressure * (temperature_top / temperature) ** (-G0 / (R_AIR * LAPSE_TROPOSPHERE))

    if altitude_m <= TROPOPAUSE_TOP:
        return pressure * math.exp(-G0 * (altitude_m - TROPOSPHERE_TOP) / (R_AIR * 216.65))

    pressure *= math.exp(-G0 * (TROPOPAUSE_TOP - TROPOSPHERE_TOP) / (R_AIR * 216.65))

    if altitude_m <= LOWER_STRATOSPHERE_TOP:
        temperature_base = 216.65
        temperature_top = temperature_isa(altitude_m)
        return pressure * (temperature_top / temperature_base) ** (-G0 / (R_AIR * LAPSE_STRATOSPHERE))

    pressure *= (temperature_isa(LOWER_STRATOSPHERE_TOP) / 216.65) ** (-G0 / (R_AIR * LAPSE_STRATOSPHERE))
    return max(pressure * math.exp(-G0 * (altitude_m - LOWER_STRATOSPHERE_TOP) / (R_AIR * 228.65)), 1e-6)


def air_density(altitude_m: float) -> float:
    temperature = max(temperature_isa(altitude_m), 1e-6)
    pressure = pressure_isa(altitude_m)
    return pressure / (R_AIR * temperature)


def air_viscosity(altitude_m: float) -> float:
    temperature = temperature_isa(altitude_m)
    return MU0 * (temperature / T0) ** 1.5 * (T0 + SUTHERLAND_C) / (temperature + SUTHERLAND_C)


def speed_of_sound(altitude_m: float) -> float:
    return math.sqrt(GAMMA_AIR * R_AIR * temperature_isa(altitude_m))


def mach_number(speed_m_s: float, altitude_m: float) -> float:
    return abs(speed_m_s) / speed_of_sound(altitude_m)


def reynolds_number(density_kg_m3: float, speed_m_s: float, length_m: float, viscosity_pa_s: float) -> float:
    if viscosity_pa_s <= 0.0:
        return 1.0
    return max(1.0, density_kg_m3 * abs(speed_m_s) * length_m / viscosity_pa_s)