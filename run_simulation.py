"""Run the full 6-DOF Green Egg simulation (boost, coast, apogee, recovery, landing).

Usage (from this folder, with the venv active):
    .venv\\Scripts\\python.exe run_simulation.py
"""

import csv
import math

import numpy as np

from numericalrocketry.rocket.rocket_config import green_egg_config
from numericalrocketry.simulation.integrator import run_6dof_rk4, SimulationConfig
from numericalrocketry.simulation.state import SimulationState


def main() -> None:
    cfg = green_egg_config()
    motor = cfg.build_motor()
    mass_model = cfg.build_component_mass_model(motor=motor)
    recovery = cfg.build_recovery()

    initial_state = SimulationState(
        time_s=0.0,
        position_world_m=np.zeros(3),
        velocity_world_m_s=np.array([0.0, 0.0, cfg.initial_velocity_m_s]),
        propellant_mass_kg=motor.propellant_mass_kg,
    )
    # Rail angle is measured from vertical, tilting the rod direction toward +X.
    rail_angle_rad = math.radians(cfg.rail_angle_deg)
    launch_direction_world = np.array(
        [math.sin(rail_angle_rad), 0.0, math.cos(rail_angle_rad)], dtype=float
    )
    sim_config = SimulationConfig(
        launch_rail_length_m=cfg.launch_rail_length_m,
        launch_direction_world=launch_direction_world,
        site_altitude_m=cfg.launch_altitude_m,
        surface_finish=cfg.surface_finish,
        launch_latitude_deg=cfg.launch_latitude_deg,
    )

    history = run_6dof_rk4(
        initial_state,
        cfg.geometry,
        motor,
        mass_model,
        dt_s=0.05,  # OpenRocket's own RECOMMENDED_TIME_STEP; adaptively refined internally.
        max_time_s=60.0,
        recovery_model=recovery,
        config=sim_config,
    )

    altitude_m = [float(p[2]) for p in history["position_world_m"]]
    vertical_velocity_m_s = [float(v[2]) for v in history["velocity_world_m_s"]]
    time_s = history["time_s"]
    vertical_acceleration_m_s2 = np.gradient(vertical_velocity_m_s, time_s)

    apogee_m = max(altitude_m)
    apogee_time_s = time_s[altitude_m.index(apogee_m)]
    events = history["events"][0]

    print(f"Apogee:          {apogee_m:.2f} m  at t={apogee_time_s:.2f} s")
    print(f"Flight time:     {time_s[-1]:.2f} s")
    print(f"Ignition:        {events['ignition_time_s']}")
    print(f"Burnout:         {events['burnout_time_s']}")
    print(f"Rail clear:      {events['rail_clear_time_s']}")
    print(f"Apogee event:    {events['apogee_time_s']}")
    print(f"Recovery deploy: {events['recovery_deploy_time_s']}")
    print(f"Touchdown:       {events['touchdown_time_s']}")

    out_path = "data/green_eggs_numericalrocketry_flight.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "phase", "altitude_m", "vertical_velocity_m_s", "vertical_acceleration_m_s2"])
        for row in zip(time_s, history["phase"], altitude_m, vertical_velocity_m_s, vertical_acceleration_m_s2):
            writer.writerow(row)
    print(f"Full time history written to {out_path}")


if __name__ == "__main__":
    main()
