"""Motor thrust curve loading utilities."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..constants import G0


def _parse_eng_header(line: str) -> tuple[float, float]:
    """Return (propellant_mass_kg, total_mass_kg) from a RASP .eng header line.

    RASP header format: name diameter length delays prop_mass total_mass manufacturer.
    The header line must contain exactly 7 whitespace-separated fields, and
    propellant mass may not exceed total mass; both are treated as hard
    errors rather than silently substituting a placeholder value.
    """
    parts = line.split()
    if len(parts) != 7:
        raise ValueError(
            "Illegal motor file format. Motor header line must contain 7 fields: "
            "designation diameter length delays propellantWeight totalWeight manufacturer"
        )

    propellant_mass_kg = float(parts[4])
    total_mass_kg = float(parts[5])

    if propellant_mass_kg > total_mass_kg:
        raise ValueError("Propellant weight exceeds total weight in motor file")

    return propellant_mass_kg, total_mass_kg


@dataclass(frozen=True)
class ThrustCurve:
    time_s: np.ndarray
    thrust_n: np.ndarray

    def thrust(self, time_s: float) -> float:
        if time_s < float(self.time_s[0]) or time_s > float(self.time_s[-1]):
            return 0.0
        return float(np.interp(time_s, self.time_s, self.thrust_n))

    @property
    def burn_time_s(self) -> float:
        return float(self.time_s[-1])


@dataclass(frozen=True)
class MotorModel:
    thrust_curve: ThrustCurve
    total_impulse_ns: float
    propellant_mass_kg: float
    total_mass_kg: float | None = None

    @property
    def specific_impulse_s(self) -> float:
        return self.total_impulse_ns / (self.propellant_mass_kg * G0)

    def thrust(self, time_s: float) -> float:
        return self.thrust_curve.thrust(time_s)

    def mass_flow_rate(self, thrust_n: float) -> float:
        if self.propellant_mass_kg <= 0.0 or self.total_impulse_ns <= 0.0:
            return 0.0
        return thrust_n / (G0 * self.specific_impulse_s)


def load_motor_from_eng_file(eng_path: str | Path) -> MotorModel | None:
    """Load a simple RASP-style .eng file into a MotorModel."""
    path = Path(eng_path)
    if not path.exists():
        return None

    time_s: list[float] = []
    thrust_n: list[float] = []
    header_propellant_mass_kg: float | None = None
    header_total_mass_kg: float | None = None

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue

        parts = line.split()
        if header_propellant_mass_kg is None:
            # The first non-comment, non-blank line is unconditionally the
            # header (not content-sniffed); a malformed header raises rather
            # than falling back to a guess.
            header_propellant_mass_kg, header_total_mass_kg = _parse_eng_header(line)
            continue

        if len(parts) < 2:
            continue
        try:
            t_val = float(parts[0])
            f_val = float(parts[1])
        except ValueError:
            continue
        time_s.append(t_val)
        thrust_n.append(f_val)

    if not time_s:
        return None

    time_arr = np.asarray(time_s, dtype=float)
    thrust_arr = np.asarray(thrust_n, dtype=float)
    if time_arr.size != thrust_arr.size:
        return None

    if time_arr.size > 1 and time_arr[0] > time_arr[-1]:
        time_arr = time_arr[::-1]
        thrust_arr = thrust_arr[::-1]

    curve = ThrustCurve(time_s=time_arr, thrust_n=thrust_arr)
    total_impulse_ns = float(np.trapezoid(thrust_arr, time_arr))

    if header_propellant_mass_kg is None or header_total_mass_kg is None:
        raise ValueError(f"Motor file {path} has no header line")

    return MotorModel(
        thrust_curve=curve,
        total_impulse_ns=total_impulse_ns,
        propellant_mass_kg=header_propellant_mass_kg,
        total_mass_kg=header_total_mass_kg,
    )