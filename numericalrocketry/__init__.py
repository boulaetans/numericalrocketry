"""Public package exports for the numericalrocketry simulation toolkit."""

from .physics.aerodynamics import AeroBreakdown, AerodynamicLoads, barrowman_aero_components, compute_aerodynamic_loads, compute_flight_conditions, compute_static_margin
from .constants import G0
from .physics.drag import total_drag_coefficient
from .rocket.geometry import RocketGeometry
from .simulation.integrator import run_6dof_rk4
from .rocket.mass_model import ComponentMass, ComponentMassModel
from .rocket.propulsion import MotorModel, ThrustCurve, load_motor_from_eng_file
from .physics.quaternion import Quaternion
from .rocket.recovery import RecoveryModel
from .rocket.rocket_config import DEFAULT_GREEN_EGG_CONFIG, RocketConfig, green_egg_config
from .simulation.state import FlightConditions, MassProperties, SimulationState

__all__ = [
    "AerodynamicLoads",
    "AeroBreakdown",
    "barrowman_aero_components",
    "compute_static_margin",
    "FlightConditions",
    "G0",
    "MassProperties",
    "MotorModel",
    "RecoveryModel",
    "Quaternion",
    "RocketConfig",
    "RocketGeometry",
    "DEFAULT_GREEN_EGG_CONFIG",
    "ComponentMass",
    "ComponentMassModel",
    "green_egg_config",
    "load_motor_from_eng_file",
    "SimulationState",
    "ThrustCurve",
    "total_drag_coefficient",
    "compute_aerodynamic_loads",
    "compute_flight_conditions",
    "run_6dof_rk4",
]