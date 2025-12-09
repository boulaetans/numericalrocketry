"""6DOF flight integration loop with event handling and recovery deployment.

Advances rigid-body state using RK4 with an adaptive time-step algorithm
(angle-step / roll-rate / pitch-rate constraints) and records key mission
events (ignition, burnout, rail clear, apogee, recovery deployment,
touchdown).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np

from ..physics.aerodynamics import barrowman_aero_components, compute_aerodynamic_loads
from ..physics.dynamics import ForceMomentResult, rigid_body_state_derivative
from ..rocket.geometry import RocketGeometry
from ..rocket.mass_model import ComponentMassModel
from ..rocket.propulsion import MotorModel
from ..physics.quaternion import Quaternion
from ..rocket.recovery import RecoveryModel
from .state import SimulationState

# Adaptive time-step limits, reproduced unchanged from the reference stepper.
MIN_TIME_STEP = 0.001
MAX_ROLL_STEP_ANGLE = 2 * 28.32 * math.pi / 180
MAX_ROLL_RATE_CHANGE = 2 * math.pi / 180
MAX_PITCH_YAW_CHANGE = 4 * math.pi / 180

# 2 cm relative displacement before LIFTOFF is flagged, and 1 cm of descent
# below the running-max altitude before APOGEE is flagged (backdated to the
# previous step's time).
LIFTOFF_THRESHOLD_M = 0.02
APOGEE_HYSTERESIS_M = 0.01


@dataclass(frozen=True)
class _StateIncrement:
    """One RK4 stage's sampled derivatives (and, for orientation, the state's
    own angular velocity -- see _integrate_step_rk4's docstring)."""
    position_world_m: np.ndarray                    # = velocity_world_m_s (position's derivative)
    velocity_world_m_s: np.ndarray                   # linear acceleration (velocity's derivative)
    angular_velocity_world_rad_s: np.ndarray         # state's angular velocity AT this stage, WORLD frame (orientation's rotation-vector generator)
    angular_acceleration_world_rad_s2: np.ndarray    # angular velocity's derivative, WORLD frame (integrated directly, like linear velocity)
    angular_acceleration_body_rad_s2: np.ndarray     # same instant, BODY ("rocket") frame -- only used for two of the adaptive time-step limits (roll-rate-change, pitch/yaw-rate-change)
    propellant_mass_kg: float


@dataclass
class SimulationEvents:
    ignition_time_s: float | None = None
    burnout_time_s: float | None = None
    rail_clear_time_s: float | None = None
    apogee_time_s: float | None = None
    recovery_deploy_time_s: float | None = None
    touchdown_time_s: float | None = None


@dataclass
class SimulationConfig:
    launch_rail_length_m: float = 1.0
    launch_direction_world: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float))
    align_body_x_to_launch_direction: bool = True
    # Recommended max body rotation allowed per adaptive time step.
    max_angle_step_rad: float = 3 * math.pi / 180
    # Launch site elevation above sea level (m). The atmosphere is queried
    # at rocket-AGL-position + this value, not at raw AGL.
    site_altitude_m: float = 0.0
    # Launch site latitude (deg), used by the WGS84 gravity model (gravity.py).
    launch_latitude_deg: float = 0.0
    # Rocket's actual configured surface finish (see aerodynamics.compute_aerodynamic_loads).
    surface_finish: str = "normal"


def _quaternion_from_vector_alignment(source: np.ndarray, target: np.ndarray) -> Quaternion:
    source_norm = np.linalg.norm(source)
    target_norm = np.linalg.norm(target)
    if source_norm <= 0.0 or target_norm <= 0.0:
        return Quaternion.identity()

    source_unit = source / source_norm
    target_unit = target / target_norm
    dot = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))

    if dot > 1.0 - 1e-10:
        return Quaternion.identity()

    if dot < -1.0 + 1e-10:
        # 180 degree rotation: pick any axis orthogonal to source.
        candidate = np.array([0.0, 1.0, 0.0], dtype=float)
        if abs(float(np.dot(source_unit, candidate))) > 0.9:
            candidate = np.array([0.0, 0.0, 1.0], dtype=float)
        axis = np.cross(source_unit, candidate)
        axis_norm = np.linalg.norm(axis)
        if axis_norm <= 0.0:
            return Quaternion.identity()
        axis = axis / axis_norm
        return Quaternion(0.0, float(axis[0]), float(axis[1]), float(axis[2])).normalized()

    axis = np.cross(source_unit, target_unit)
    w = 1.0 + dot
    return Quaternion(float(w), float(axis[0]), float(axis[1]), float(axis[2])).normalized()


def _add_scaled_state(state: SimulationState, increment: _StateIncrement, scale: float, mass_model: ComponentMassModel) -> SimulationState:
    """Build an RK4 sub-stage evaluation state, offset from `state` by `scale`
    (a partial time step, e.g. dt/2) times the given derivative sample.

    Orientation is updated via the exponential map, LEFT-multiplied (new
    rotation composed before the existing orientation) since this state's
    angular velocity is WORLD-frame.
    """
    rotation_vector = increment.angular_velocity_world_rad_s * scale
    orientation = Quaternion.from_rotation_vector(rotation_vector).multiply(state.orientation_body_to_world).normalized()

    propellant_mass = max(0.0, state.propellant_mass_kg + scale * increment.propellant_mass_kg)
    mass_properties = mass_model.from_propellant_mass(propellant_mass)

    next_state = state.copy()
    next_state.position_world_m = state.position_world_m + scale * increment.position_world_m
    next_state.velocity_world_m_s = state.velocity_world_m_s + scale * increment.velocity_world_m_s
    next_state.orientation_body_to_world = orientation
    next_state.angular_velocity_world_rad_s = state.angular_velocity_world_rad_s + scale * increment.angular_acceleration_world_rad_s2
    next_state.propellant_mass_kg = propellant_mass
    next_state.mass_properties = mass_properties
    return next_state


def _state_increment(
    state: SimulationState,
    geometry: RocketGeometry,
    motor: MotorModel,
    mass_model: ComponentMassModel,
    wind_world_m_s: np.ndarray,
    recovery_model: RecoveryModel | None,
    site_altitude_m: float = 0.0,
    on_rail: bool = False,
    launch_direction_world: np.ndarray | None = None,
    surface_finish: str = "normal",
    launch_latitude_deg: float = 0.0,
) -> _StateIncrement:
    """Sample one RK4 stage's state derivatives: aero + thrust + recovery-drag
    forces, rigid-body acceleration, and propellant burn rate."""
    flight_conditions, aero_loads = compute_aerodynamic_loads(
        state, geometry, wind_world_m_s, site_altitude_m=site_altitude_m, surface_finish=surface_finish
    )

    thrust = motor.thrust(state.time_s)
    thrust_force_body = np.array([thrust, 0.0, 0.0], dtype=float)

    recovery_force_body = np.zeros(3, dtype=float)
    if state.recovery_deployed and recovery_model is not None and recovery_model.chute_area_m2 > 0.0:
        # Post-deployment landing phase: add deployed-device drag as
        # q * Cd * A opposite the relative wind.
        airspeed_body = flight_conditions.airspeed_body_m_s
        speed = float(np.linalg.norm(airspeed_body))
        if speed > 1e-9:
            velocity_hat = airspeed_body / speed
            q = flight_conditions.dynamic_pressure_pa
            recovery_force_body = -q * recovery_model.chute_cd * recovery_model.chute_area_m2 * velocity_hat

    total_force_body = aero_loads.force_body_n + thrust_force_body + recovery_force_body
    total_moment_body = aero_loads.moment_body_n_m

    derivative = rigid_body_state_derivative(
        state,
        ForceMomentResult(force_body_n=total_force_body, moment_body_n_m=total_moment_body),
        site_altitude_m=site_altitude_m,
        launch_latitude_deg=launch_latitude_deg,
    )

    angular_velocity_world = state.angular_velocity_world_rad_s.copy()
    angular_acceleration_world = derivative.angular_acceleration_world_rad_s2
    angular_acceleration_body = derivative.angular_acceleration_body_rad_s2
    velocity_dot_world = derivative.velocity_dot_world

    if state.recovery_deployed:
        # The post-deployment landing phase never generates a moment, never
        # updates the rotation velocity (angular acceleration is always the
        # zero vector), and never touches the orientation quaternion at all
        # -- rotational state is simply frozen at whatever it was when
        # recovery deployed, for the rest of the descent. Zeroing the
        # acceleration freezes the persisted angular-velocity state at its
        # last pre-deployment value; zeroing the orientation-update
        # generator (independently of that persisted value) freezes the
        # orientation itself.
        angular_acceleration_body = np.zeros(3, dtype=float)
        angular_acceleration_world = np.zeros(3, dtype=float)
        angular_velocity_world = np.zeros(3, dtype=float)
    elif on_rail and launch_direction_world is not None:
        # While still on the launch rod (after liftoff, before rail
        # clearance), project linear acceleration onto the rod direction and
        # zero angular acceleration, inside every RK4 substage -- not just
        # clamped after the full step completes.
        projection = float(np.dot(velocity_dot_world, launch_direction_world))
        velocity_dot_world = projection * launch_direction_world
        angular_acceleration_body = np.zeros(3, dtype=float)
        angular_acceleration_world = np.zeros(3, dtype=float)

    if state.propellant_mass_kg <= 0.0 or thrust <= 0.0:
        propellant_dot = 0.0
    else:
        propellant_dot = -motor.mass_flow_rate(thrust)

    return _StateIncrement(
        position_world_m=derivative.position_dot_world,
        velocity_world_m_s=velocity_dot_world,
        angular_velocity_world_rad_s=angular_velocity_world,
        angular_acceleration_world_rad_s2=angular_acceleration_world,
        angular_acceleration_body_rad_s2=angular_acceleration_body,
        propellant_mass_kg=propellant_dot,
    )


def _integrate_step_rk4(
    state: SimulationState,
    geometry: RocketGeometry,
    motor: MotorModel,
    mass_model: ComponentMassModel,
    dt_s: float,
    wind_model: Callable[[float, float], np.ndarray],
    recovery_model: RecoveryModel | None,
    site_altitude_m: float = 0.0,
    on_rail: bool = False,
    launch_direction_world: np.ndarray | None = None,
    surface_finish: str = "normal",
    launch_latitude_deg: float = 0.0,
) -> SimulationState:
    """Advance `state` by one full RK4 step (4 stages, k1..k4), including the
    orientation update (exponential map of the dt-weighted average angular
    velocity, applied as a single left-multiply -- see the comment below)."""
    wind_1 = wind_model(state.time_s, state.position_world_m[2])
    k1 = _state_increment(state, geometry, motor, mass_model, wind_1, recovery_model, site_altitude_m, on_rail, launch_direction_world, surface_finish, launch_latitude_deg)

    state_k2 = _add_scaled_state(state, k1, 0.5 * dt_s, mass_model)
    state_k2.time_s = state.time_s + 0.5 * dt_s
    wind_2 = wind_model(state_k2.time_s, state_k2.position_world_m[2])
    k2 = _state_increment(state_k2, geometry, motor, mass_model, wind_2, recovery_model, site_altitude_m, on_rail, launch_direction_world, surface_finish, launch_latitude_deg)

    state_k3 = _add_scaled_state(state, k2, 0.5 * dt_s, mass_model)
    state_k3.time_s = state.time_s + 0.5 * dt_s
    wind_3 = wind_model(state_k3.time_s, state_k3.position_world_m[2])
    k3 = _state_increment(state_k3, geometry, motor, mass_model, wind_3, recovery_model, site_altitude_m, on_rail, launch_direction_world, surface_finish, launch_latitude_deg)

    state_k4 = _add_scaled_state(state, k3, dt_s, mass_model)
    state_k4.time_s = state.time_s + dt_s
    wind_4 = wind_model(state_k4.time_s, state_k4.position_world_m[2])
    k4 = _state_increment(state_k4, geometry, motor, mass_model, wind_4, recovery_model, site_altitude_m, on_rail, launch_direction_world, surface_finish, launch_latitude_deg)

    position_increment = (dt_s / 6.0) * (k1.position_world_m + 2.0 * k2.position_world_m + 2.0 * k3.position_world_m + k4.position_world_m)
    velocity_increment = (dt_s / 6.0) * (k1.velocity_world_m_s + 2.0 * k2.velocity_world_m_s + 2.0 * k3.velocity_world_m_s + k4.velocity_world_m_s)
    angular_velocity_increment = (dt_s / 6.0) * (
        k1.angular_acceleration_world_rad_s2
        + 2.0 * k2.angular_acceleration_world_rad_s2
        + 2.0 * k3.angular_acceleration_world_rad_s2
        + k4.angular_acceleration_world_rad_s2
    )
    propellant_increment = (dt_s / 6.0) * (
        k1.propellant_mass_kg
        + 2.0 * k2.propellant_mass_kg
        + 2.0 * k3.propellant_mass_kg
        + k4.propellant_mass_kg
    )

    # Orientation: exponential map of the dt-weighted RK4 average of the
    # stage angular velocities (WORLD frame), applied once via a single
    # LEFT-multiply (new rotation composed before the existing orientation)
    # onto the ORIGINAL base orientation.
    rotation_vector_sum = (dt_s / 6.0) * (
        k1.angular_velocity_world_rad_s
        + 2.0 * k2.angular_velocity_world_rad_s
        + 2.0 * k3.angular_velocity_world_rad_s
        + k4.angular_velocity_world_rad_s
    )

    next_state = state.copy()
    next_state.time_s = state.time_s + dt_s
    next_state.position_world_m = state.position_world_m + position_increment
    next_state.velocity_world_m_s = state.velocity_world_m_s + velocity_increment
    next_state.orientation_body_to_world = Quaternion.from_rotation_vector(rotation_vector_sum).multiply(
        state.orientation_body_to_world
    ).normalized()
    next_state.angular_velocity_world_rad_s = state.angular_velocity_world_rad_s + angular_velocity_increment
    next_state.propellant_mass_kg = max(0.0, state.propellant_mass_kg + propellant_increment)
    next_state.mass_properties = mass_model.from_propellant_mass(next_state.propellant_mass_kg)
    return next_state


def _select_adaptive_time_step(
    state: SimulationState,
    k1: _StateIncrement,
    user_time_step_s: float,
    config: SimulationConfig,
    max_time_step_to_next_event: float,
    on_rail: bool,
    previous_time_step: float,
) -> float:
    """Adaptive RK4 step-size selection.

    Picks the minimum of: the user time step (divided by 5 on the rail), the
    distance to the next scheduled event, the max angle-step limit, the max
    roll-step limit, the max roll-rate-change limit, the max pitch/yaw-change
    limit, 1/10th of the rail length (while on the rail), and 1.5x the
    previous step. Then snaps to the event boundary if very close to it, and
    enforces MIN_TIME_STEP as a floor.
    """
    candidates = [max(user_time_step_s / 5.0 if on_rail else user_time_step_s, MIN_TIME_STEP)]
    candidates.append(max_time_step_to_next_event)

    # These dt limits use body-frame ("rocket coordinates") rates -- inverse-
    # rotate the world-frame rotation velocity before deriving pitch/yaw/roll
    # rate.
    angular_velocity = state.orientation_body_to_world.conjugate().rotate_vector(state.angular_velocity_world_rad_s)
    lateral_pitch_rate = math.hypot(float(angular_velocity[1]), float(angular_velocity[2]))
    if lateral_pitch_rate > 0.0:
        candidates.append(config.max_angle_step_rad / lateral_pitch_rate)

    roll_rate = abs(float(angular_velocity[0]))
    if roll_rate > 0.0:
        candidates.append(MAX_ROLL_STEP_ANGLE / roll_rate)

    roll_accel = abs(float(k1.angular_acceleration_body_rad_s2[0]))
    if roll_accel > 0.0:
        candidates.append(MAX_ROLL_RATE_CHANGE / roll_accel)

    pitch_yaw_accel = max(abs(float(k1.angular_acceleration_body_rad_s2[1])), abs(float(k1.angular_acceleration_body_rad_s2[2])))
    if pitch_yaw_accel > 0.0:
        candidates.append(MAX_PITCH_YAW_CHANGE / pitch_yaw_accel)

    if on_rail:
        speed = float(np.linalg.norm(state.velocity_world_m_s))
        if speed > 0.0:
            candidates.append(config.launch_rail_length_m / speed / 10.0)

    if previous_time_step > 0.0:
        candidates.append(1.5 * previous_time_step)

    time_step = min(candidates)

    min_time_step = user_time_step_s / 20.0
    if abs(max_time_step_to_next_event - time_step) < min_time_step:
        time_step = max_time_step_to_next_event
    if time_step < min_time_step:
        time_step = min_time_step

    return time_step


def _apply_launch_rail_constraint(
    state: SimulationState,
    launch_origin_world_m: np.ndarray,
    launch_direction_world: np.ndarray,
    config: SimulationConfig,
) -> tuple[SimulationState, bool]:
    """Clamp state to the launch rod's axis and report whether it has cleared
    the rail (axial travel >= rail length)."""
    arm = state.position_world_m - launch_origin_world_m
    axial_distance = float(np.dot(arm, launch_direction_world))
    axial_distance = max(0.0, axial_distance)
    constrained_position = launch_origin_world_m + axial_distance * launch_direction_world

    velocity_axial = float(np.dot(state.velocity_world_m_s, launch_direction_world))
    constrained_velocity = max(0.0, velocity_axial) * launch_direction_world

    constrained_state = state.copy()
    constrained_state.position_world_m = constrained_position
    constrained_state.velocity_world_m_s = constrained_velocity
    constrained_state.angular_velocity_world_rad_s = np.zeros(3, dtype=float)

    cleared = axial_distance >= config.launch_rail_length_m
    return constrained_state, cleared


def run_6dof_rk4(
    initial_state: SimulationState,
    geometry: RocketGeometry,
    motor: MotorModel,
    mass_model: ComponentMassModel,
    dt_s: float,
    max_time_s: float,
    recovery_model: RecoveryModel | None = None,
    wind_model: Callable[[float, float], np.ndarray] | None = None,
    config: SimulationConfig | None = None,
) -> Dict[str, List[np.ndarray | float]]:
    """Run the full flight (pad through touchdown) and return a time-history
    dict of per-step state, phase, and mission events. The main entry point
    for this module."""
    if wind_model is None:
        wind_model = lambda _time_s, _altitude_m: np.zeros(3, dtype=float)
    if config is None:
        config = SimulationConfig()

    state = initial_state.copy()
    state.mass_properties = mass_model.from_propellant_mass(state.propellant_mass_kg)
    lifted_off = state.has_lifted_off
    rail_cleared = False
    launch_direction_world = config.launch_direction_world.astype(float)
    launch_norm = np.linalg.norm(launch_direction_world)
    if launch_norm <= 0.0:
        launch_direction_world = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        launch_direction_world = launch_direction_world / launch_norm

    if config.align_body_x_to_launch_direction:
        state.orientation_body_to_world = _quaternion_from_vector_alignment(
            np.array([1.0, 0.0, 0.0], dtype=float), launch_direction_world
        )

    launch_origin_world = state.position_world_m.copy()
    event_log = SimulationEvents()
    ignition_guess = float(motor.thrust_curve.time_s[0])
    burnout_guess = float(motor.thrust_curve.time_s[-1])
    # An RK4 step-boundary event is queued at every sample point of the
    # motor's thrust curve on ignition, in addition to burnout.
    thrust_curve_breakpoints = [float(t) for t in motor.thrust_curve.time_s]
    previous_thrust = motor.thrust(state.time_s)
    previous_time_step = 0.0
    max_altitude_so_far = float(state.position_world_m[2])
    previous_time_s = state.time_s

    history: Dict[str, List[np.ndarray | float]] = {
        "time_s": [state.time_s],
        "position_world_m": [state.position_world_m.copy()],
        "velocity_world_m_s": [state.velocity_world_m_s.copy()],
        "orientation_wxyz": [state.orientation_body_to_world.as_array().copy()],
        "angular_velocity_world_rad_s": [state.angular_velocity_world_rad_s.copy()],
        "mass_kg": [state.mass_properties.mass_kg],
        "propellant_mass_kg": [state.propellant_mass_kg],
        "phase": ["pad"],
        "recovery_deployed": [state.recovery_deployed],
        "cp_x_m": [float("nan")],
        "static_margin_cal": [float("nan")],
    }

    def should_deploy_recovery() -> bool:
        if recovery_model is None or state.recovery_deployed:
            return False

        event_name = recovery_model.deploy_event
        now = state.time_s

        if event_name == "never":
            return False

        if event_name == "launch":
            return now >= recovery_model.deploy_delay_s

        if event_name == "apogee":
            if event_log.apogee_time_s is None:
                return False
            return now >= event_log.apogee_time_s + recovery_model.deploy_delay_s

        if event_name == "altitude":
            descending = float(state.velocity_world_m_s[2]) < 0.0
            return descending and state.position_world_m[2] <= recovery_model.deploy_altitude_m

        # Default trigger for parachutes: the motor's ejection charge.
        if event_log.burnout_time_s is None:
            return False
        trigger = event_log.burnout_time_s + recovery_model.motor_ejection_delay_s + recovery_model.deploy_delay_s
        return now >= trigger

    while state.time_s < max_time_s:
        wind_now = wind_model(state.time_s, state.position_world_m[2])
        on_rail = lifted_off and not rail_cleared
        k1 = _state_increment(
            state, geometry, motor, mass_model, wind_now, recovery_model,
            config.site_altitude_m, on_rail, launch_direction_world, config.surface_finish,
            config.launch_latitude_deg,
        )

        # Distance to the next known event boundary (ignition/thrust-curve
        # samples/burnout/end of sim), used as the step size's upper bound.
        upcoming = [
            t for t in (ignition_guess, burnout_guess, max_time_s, *thrust_curve_breakpoints)
            if t > state.time_s
        ]
        max_time_step_to_next_event = min(upcoming) - state.time_s if upcoming else max_time_s - state.time_s

        dt = _select_adaptive_time_step(
            state, k1, dt_s, config, max_time_step_to_next_event, on_rail=on_rail, previous_time_step=previous_time_step
        )
        state = _integrate_step_rk4(
            state, geometry, motor, mass_model, dt, wind_model, recovery_model,
            config.site_altitude_m, on_rail, launch_direction_world, config.surface_finish,
            config.launch_latitude_deg,
        )
        previous_time_step = dt


        current_thrust = motor.thrust(state.time_s)
        if event_log.ignition_time_s is None and current_thrust > 0.0:
            event_log.ignition_time_s = state.time_s
        if event_log.burnout_time_s is None and previous_thrust > 0.0 and current_thrust <= 0.0:
            event_log.burnout_time_s = state.time_s

        if not rail_cleared:
            state, rail_cleared = _apply_launch_rail_constraint(state, launch_origin_world, launch_direction_world, config)
            state.on_rail = not rail_cleared
            if rail_cleared and event_log.rail_clear_time_s is None:
                event_log.rail_clear_time_s = state.time_s
        else:
            state.on_rail = False

        if state.position_world_m[2] < 0.0:
            state.position_world_m[2] = 0.0
            if lifted_off:
                state.velocity_world_m_s[2] = 0.0
            else:
                # Keep the rocket on the pad before liftoff.
                state.velocity_world_m_s = np.zeros(3, dtype=float)

        if not lifted_off and (state.position_world_m[2] > LIFTOFF_THRESHOLD_M or rail_cleared):
            lifted_off = True
            state.has_lifted_off = True

        current_altitude = float(state.position_world_m[2])
        if current_altitude > max_altitude_so_far:
            max_altitude_so_far = current_altitude

        # Apogee is detected as a 1 cm drop below the running-max altitude,
        # and the event is timestamped at the PREVIOUS step's time, not the
        # current step's -- a deliberately backdated approximation of the
        # true apogee instant, rather than an exact velocity-sign root-find.
        if lifted_off and event_log.apogee_time_s is None and current_altitude < max_altitude_so_far - APOGEE_HYSTERESIS_M:
            event_log.apogee_time_s = previous_time_s

        if should_deploy_recovery():
            state.recovery_deployed = True
            if event_log.recovery_deploy_time_s is None:
                event_log.recovery_deploy_time_s = state.time_s

        history["time_s"].append(state.time_s)
        history["position_world_m"].append(state.position_world_m.copy())
        history["velocity_world_m_s"].append(state.velocity_world_m_s.copy())
        history["orientation_wxyz"].append(state.orientation_body_to_world.as_array().copy())
        history["angular_velocity_world_rad_s"].append(state.angular_velocity_world_rad_s.copy())
        history["mass_kg"].append(state.mass_properties.mass_kg)
        history["propellant_mass_kg"].append(state.propellant_mass_kg)
        history["recovery_deployed"].append(state.recovery_deployed)
        if not lifted_off:
            history["phase"].append("pad")
        elif not rail_cleared:
            history["phase"].append("rail")
        elif state.recovery_deployed:
            history["phase"].append("recovery")
        else:
            history["phase"].append("coast" if current_thrust <= 0.0 else "boost")

        # Aerodynamic CP position and static stability margin at this time step.
        # These are diagnostic quantities logged in the history; they do not
        # affect the dynamics (forces are already computed inside _state_increment).
        cg_x = float(state.mass_properties.center_of_gravity_m[0])
        aero_bd = barrowman_aero_components(geometry, cg_x_m=cg_x)
        history["cp_x_m"].append(aero_bd.cp_x_total_m)
        history["static_margin_cal"].append(aero_bd.static_margin_cal)

        # Ground-hit is `position.Z < EPSILON` after liftoff -- no velocity
        # condition at all.
        if lifted_off and state.position_world_m[2] <= 0.0:
            event_log.touchdown_time_s = state.time_s
            break

        previous_thrust = current_thrust
        previous_time_s = state.time_s

    history["events"] = [
        {
            "ignition_time_s": event_log.ignition_time_s,
            "burnout_time_s": event_log.burnout_time_s,
            "rail_clear_time_s": event_log.rail_clear_time_s,
            "apogee_time_s": event_log.apogee_time_s,
            "recovery_deploy_time_s": event_log.recovery_deploy_time_s,
            "touchdown_time_s": event_log.touchdown_time_s,
        }
    ]

    return history
