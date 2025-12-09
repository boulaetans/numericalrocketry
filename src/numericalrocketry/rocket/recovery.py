"""Recovery-device (parachute) deployment configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RecoveryModel:
    deploy_event: str
    deploy_delay_s: float
    deploy_altitude_m: float
    chute_cd: float
    chute_area_m2: float
    motor_ejection_delay_s: float
