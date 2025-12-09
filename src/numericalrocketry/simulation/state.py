"""Simulation state containers shared between dynamics and integrators."""

from dataclasses import dataclass, field

import numpy as np

from ..physics.quaternion import Quaternion


@dataclass(frozen=True)
class MassProperties:
    mass_kg: float
    center_of_gravity_m: np.ndarray
    inertia_body_kg_m2: np.ndarray


@dataclass(frozen=True)
class FlightConditions:
    airspeed_body_m_s: np.ndarray
    wind_world_m_s: np.ndarray
    dynamic_pressure_pa: float
    mach: float
    angle_of_attack_rad: float
    sideslip_rad: float


@dataclass
class SimulationState:
    time_s: float
    position_world_m: np.ndarray
    velocity_world_m_s: np.ndarray
    orientation_body_to_world: Quaternion = field(default_factory=Quaternion.identity)
    # Rotation velocity is tracked in WORLD frame (integrated directly in
    # world coordinates, not body coordinates) -- see integrator.py for
    # the full RK4 orientation/angular-velocity update.
    angular_velocity_world_rad_s: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    mass_properties: MassProperties = field(
        default_factory=lambda: MassProperties(
            mass_kg=0.0,
            center_of_gravity_m=np.zeros(3, dtype=float),
            inertia_body_kg_m2=np.eye(3, dtype=float),
        )
    )
    propellant_mass_kg: float = 0.0
    on_rail: bool = True
    has_lifted_off: bool = False
    recovery_deployed: bool = False

    def copy(self) -> "SimulationState":
        return SimulationState(
            time_s=self.time_s,
            position_world_m=self.position_world_m.copy(),
            velocity_world_m_s=self.velocity_world_m_s.copy(),
            orientation_body_to_world=self.orientation_body_to_world,
            angular_velocity_world_rad_s=self.angular_velocity_world_rad_s.copy(),
            mass_properties=MassProperties(
                mass_kg=self.mass_properties.mass_kg,
                center_of_gravity_m=self.mass_properties.center_of_gravity_m.copy(),
                inertia_body_kg_m2=self.mass_properties.inertia_body_kg_m2.copy(),
            ),
            propellant_mass_kg=self.propellant_mass_kg,
            on_rail=self.on_rail,
            has_lifted_off=self.has_lifted_off,
            recovery_deployed=self.recovery_deployed,
        )