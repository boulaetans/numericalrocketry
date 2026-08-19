"""Rigid-body dynamics equations for translational and rotational motion."""

from dataclasses import dataclass

import numpy as np

from .gravity import wgs84_gravity
from ..simulation.state import SimulationState


@dataclass(frozen=True)
class ForceMomentResult:
    force_body_n: np.ndarray
    moment_body_n_m: np.ndarray


@dataclass(frozen=True)
class StateDerivative:
    position_dot_world: np.ndarray
    velocity_dot_world: np.ndarray
    angular_acceleration_body_rad_s2: np.ndarray
    angular_acceleration_world_rad_s2: np.ndarray


def rigid_body_state_derivative(
    state: SimulationState,
    force_moment: ForceMomentResult,
    site_altitude_m: float = 0.0,
    launch_latitude_deg: float = 0.0,
) -> StateDerivative:
    orientation = state.orientation_body_to_world.normalized()
    force_world = orientation.rotate_vector(force_moment.force_body_n)
    mass = max(state.mass_properties.mass_kg, 1e-9)
    # Gravity is queried from the WGS84 model at the rocket's current world
    # position (launch-site latitude, MSL altitude = AGL position + launch
    # site elevation) every step, not a constant 9.80665 -- see gravity.py.
    altitude_msl_m = float(state.position_world_m[2]) + site_altitude_m
    gravity_world = np.array([0.0, 0.0, -wgs84_gravity(altitude_msl_m, launch_latitude_deg)], dtype=float)
    velocity_dot_world = force_world / mass + gravity_world

    # Angular acceleration is moment / inertia component-wise in body
    # ("rocket") coordinates, with NO omega x I*omega gyroscopic coupling
    # term -- the reference this project targets omits it entirely, so this
    # deliberately does too rather than adding "more complete" dynamics.
    # The result is then rotated into world coordinates, since angular
    # velocity is integrated in WORLD frame (see integrator.py's
    # orientation/angular-velocity integration).
    inertia = state.mass_properties.inertia_body_kg_m2
    angular_acceleration_body = np.linalg.solve(inertia, force_moment.moment_body_n_m)
    angular_acceleration_world = orientation.rotate_vector(angular_acceleration_body)

    return StateDerivative(
        position_dot_world=state.velocity_world_m_s.copy(),
        velocity_dot_world=velocity_dot_world,
        angular_acceleration_body_rad_s2=angular_acceleration_body,
        angular_acceleration_world_rad_s2=angular_acceleration_world,
    )