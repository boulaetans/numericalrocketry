"""Animated 3D MP4 comparing three versions of the same Green Egg flight: this
project's own 6DOF simulation, OpenRocket's own simulated flight, and the real
logged flight-computer data from the actual launch. Time and altitude are real
axes; the third axis is categorical (which of the three this is), so the three
otherwise near-identical curves don't just sit on top of each other.

Rendered directly to MP4 via FFMpegWriter, which encodes each frame at an
exact, fixed interval, so the file's real-time playback speed always
matches what --fps and --playback-seconds actually specify, at a fraction
of the file size a comparable frame-by-frame image sequence would need.

Each track's own mission events (ignition, burnout, apogee, recovery deploy,
touchdown) are marked on that track as the animation reaches them. The
three tracks generally do NOT reach the same event at the same time, and watching
them diverge over the course of the flight (especially descent, where the real
flight's much longer hang time shows up clearly) is the point of this animation.

The real flight log's raw state machine (boost/coast/main/landed) doesn't line up
with true ignition or touchdown for this specific flight. data/derive_real_flight_csv.py
corrects both directly in the CSV before this script ever sees it (see that script's
docstring for the full account, including the ground-elevation and pressure-based
altitude fixes). Burnout still has no explicit state-machine marker, so it's
estimated from the (re-derived) acceleration trace and marked "(est.)"; recovery
deploy has no detectable signature at all in this sensor's data and is left
unmarked rather than guessed.

Usage (from the project root, with the venv active):
    .venv\\Scripts\\python.exe animate_flight_comparison.py [--output FILE.mp4] [--fps N]

Reads its reference data from data/Green Eggs OpenRocket Flight.csv and
data/Green Eggs Real Flight.csv, and writes to assets/flight_comparison.mp4 by default.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the 3d projection)

from numericalrocketry.simulation.integrator import SimulationConfig, run_6dof_rk4
from numericalrocketry.rocket.rocket_config import green_egg_config
from numericalrocketry.simulation.state import SimulationState

# Canonical mission-event keys shared across all three tracks (a track that can't
# determine a given event from its own data just leaves it as None).
EVENT_LABELS = {
    "ignition": "IGNITION",
    "burnout": "BURNOUT",
    "apogee": "APOGEE",
    "recovery_deploy": "DEPLOY",
    "touchdown": "TOUCHDOWN",
}
# Small z-nudge so apogee/recovery-deploy (this rocket ejects its chute right at
# apogee, so the two land very close together in time+altitude on the same track)
# don't render as fully overlapping text.
EVENT_EXTRA_DZ = {"apogee": 3.0, "recovery_deploy": -3.0}


def load_numericalrocketry_track() -> dict:
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
        dt_s=0.05,
        max_time_s=60.0,
        recovery_model=recovery,
        config=sim_config,
    )

    times = np.array(history["time_s"], dtype=float)
    altitudes = np.array([float(p[2]) for p in history["position_world_m"]], dtype=float)
    velocities = np.array([float(v[2]) for v in history["velocity_world_m_s"]], dtype=float)
    accelerations = np.gradient(velocities, times)  # deterministic ODE output, safe to differentiate
    raw_events = history["events"][0]
    events = {key: raw_events.get(f"{key}_time_s") for key in EVENT_LABELS}
    return {
        "name": "NumericalRocketry",
        "color": "#2a78d6",
        "times": times,
        "altitudes": altitudes,
        "velocities": velocities,
        "accelerations": accelerations,
        "events": events,
        "estimated_events": set(),
    }


def load_openrocket_track(csv_path: Path) -> dict:
    or_event_to_canonical = {
        "IGNITION": "ignition",
        "BURNOUT": "burnout",
        "APOGEE": "apogee",
        "RECOVERY_DEVICE_DEPLOYMENT": "recovery_deploy",
        "GROUND_HIT": "touchdown",
    }
    events: dict = {key: None for key in EVENT_LABELS}
    times: list[float] = []
    altitudes: list[float] = []
    velocities: list[float] = []
    accelerations: list[float] = []

    event_re = re.compile(r"^#\s*Event\s+(\w+)\s+occurred at t=([\d.eE+-]+)\s*seconds")
    with open(csv_path, newline="", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                match = event_re.match(line)
                if match:
                    name, t = match.group(1), float(match.group(2))
                    canonical = or_event_to_canonical.get(name)
                    if canonical is not None and events[canonical] is None:
                        events[canonical] = t
                continue
            row = next(csv.reader([line]))
            times.append(float(row[0]))
            altitudes.append(float(row[1]))
            velocities.append(float(row[3]))   # Vertical velocity (m/s)
            accelerations.append(float(row[5]))  # Vertical acceleration (m/s^2)

    return {
        "name": "OpenRocket (reference sim)",
        "color": "#eb6834",
        "times": np.array(times, dtype=float),
        "altitudes": np.array(altitudes, dtype=float),
        "velocities": np.array(velocities, dtype=float),
        "accelerations": np.array(accelerations, dtype=float),
        "events": events,
        "estimated_events": set(),
    }


def load_real_flight_track(csv_path: Path) -> dict:
    """Real flight-computer log for the actual launch, as rebuilt by
    data/derive_real_flight_csv.py. See that script's module docstring for
    the full account of what's been corrected in this CSV and why (duplicate
    rows, ground-elevation error, the missing pre-recording gap, and
    re-derived acceleration/speed) before it ever reaches this loader.

    Because that rebuild already grounds `time` at the estimated true
    liftoff (t=0) and relabels `state_name` at the points where the onboard
    flags didn't match physical reality (ignition, landed), event extraction
    here only needs to handle burnout and apogee directly:

    * ignition: exact, t=0 by construction of the rebuilt CSV.
    * burnout: first zero-crossing of (re-derived) acceleration after t=0,
      still an estimate, since the state machine's boost->coast transition
      doesn't track literal thrust cutoff for this flight.
    * apogee: the altitude maximum, unambiguous, no estimation needed.
    * recovery_deploy: NOT estimated. There's no detectable transient
      anywhere in the acceleration/speed data between "main" state's onset
      and apogee, consistent with this rocket's small/light chute opening
      too gently to leave a signature in this sensor's data.
    * touchdown: the first row relabeled state_name=="landed" in the rebuilt
      CSV (t=42.36s, where acceleration/speed actually settle), not the
      onboard flag's own much-delayed timestamp (t=54.99s).
    """
    times: list[float] = []
    altitudes: list[float] = []
    accel: list[float] = []
    speed: list[float] = []
    state_names: list[str] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time"]))
            altitudes.append(float(row["height"]))
            accel.append(float(row["acceleration"]))
            speed.append(float(row["speed"]))
            state_names.append(row["state_name"])

    t = np.array(times, dtype=float)
    alt = np.array(altitudes, dtype=float)
    a = np.array(accel, dtype=float)
    v = np.array(speed, dtype=float)

    events: dict = {key: None for key in EVENT_LABELS}
    estimated_events: set = set()

    # Ignition: exact. The rebuilt CSV's t=0 already is the estimated true liftoff.
    events["ignition"] = 0.0

    # Burnout: first acceleration zero-crossing after ignition.
    post_ignition = np.where(t > 0.0)[0]
    for i in post_ignition:
        if a[i] <= 0.0:
            i0 = i - 1
            frac = a[i0] / (a[i0] - a[i]) if a[i0] != a[i] else 0.0
            events["burnout"] = float(t[i0] + frac * (t[i] - t[i0]))
            estimated_events.add("burnout")
            break

    # Apogee: unambiguous, straight from the data.
    apogee_index = int(np.argmax(alt))
    events["apogee"] = float(t[apogee_index])

    # Touchdown: the rebuilt CSV's own corrected "landed" label.
    for i, name in enumerate(state_names):
        if name == "landed":
            events["touchdown"] = float(t[i])
            break

    return {
        "name": "Real Flight",
        "color": "#1baf7a",
        "times": t,
        "altitudes": alt,
        "velocities": v,
        "accelerations": a,
        "events": events,
        "estimated_events": estimated_events,
    }


def build_time_schedule(t_min: float, t_max: float, fps: int, playback_seconds: float, hold_seconds: float) -> np.ndarray:
    n_main = max(2, int(round(playback_seconds * fps)))
    main = np.linspace(t_min, t_max, n_main)
    n_hold = max(0, int(round(hold_seconds * fps)))
    hold = np.full(n_hold, t_max)
    return np.concatenate([main, hold])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="assets/flight_comparison.mp4")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--playback-seconds", type=float, default=None, help="Real-world seconds the main sweep takes to play (default: real time, i.e. one second of playback per second of flight)")
    parser.add_argument("--hold-seconds", type=float, default=3.0, help="Pause after the last touchdown before the animation loops/ends")
    args = parser.parse_args()

    tracks = [
        load_numericalrocketry_track(),
        load_openrocket_track(Path("data/Green Eggs OpenRocket Flight.csv")),
        load_real_flight_track(Path("data/Green Eggs Real Flight.csv")),
    ]
    lanes = {0: 0.0, 1: 1.0, 2: 2.0}

    # All three tracks ignite at (or very near) t=0. The real flight's
    # rebuilt CSV grounds its own t=0 at the estimated true liftoff, same as
    # NR/OR's simulated ignition.
    t_min = min(
        track["events"]["ignition"] if track["events"]["ignition"] is not None else track["times"][0]
        for track in tracks
    )
    # The sweep runs to the latest actual touchdown, not the latest raw data
    # sample. The real flight log's rebuilt CSV still keeps a short tail of
    # already-landed samples after touchdown, which would just add flat,
    # uninformative frames to the animation.
    t_max = max(track["events"]["touchdown"] or track["times"][-1] for track in tracks)
    z_max = max(track["altitudes"].max() for track in tracks)

    playback_seconds = args.playback_seconds if args.playback_seconds is not None else (t_max - t_min)
    t_schedule = build_time_schedule(t_min, t_max, args.fps, playback_seconds, args.hold_seconds)

    fig = plt.figure(figsize=(10, 7), dpi=130)
    ax = fig.add_subplot(111, projection="3d")
    try:
        ax.set_box_aspect((3.0, 1.0, 1.3))
    except AttributeError:
        pass  # older matplotlib without set_box_aspect, cosmetic only
    ax.set_xlim(t_min, t_max * 1.03)
    ax.set_ylim(-0.5, 2.5)
    ax.set_zlim(-10.0, z_max * 1.15)
    ax.set_xlabel("Time since ignition (s)")
    # Short lane codes only; the full names live in the fixed 2D legend below
    # instead, since text drawn in 3D space rotates with the camera and would
    # otherwise collide with whatever else happens to be nearby at a given angle.
    ax.set_yticks([lanes[0], lanes[1], lanes[2]])
    ax.set_yticklabels(["NR", "OR", "Real"], fontsize=8)
    ax.set_zlabel("Altitude AGL (m)")
    ax.set_title("Green Eggs: Simulated vs. Real Flight")

    lines = {}
    dots = {}
    for track in tracks:
        (line,) = ax.plot([], [], [], lw=1.8, color=track["color"], label=track["name"])
        (dot,) = ax.plot([], [], [], "o", color=track["color"], ms=7, mec="black", mew=0.6)
        lines[track["name"]] = line
        dots[track["name"]] = dot
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(0.0, 0.92))

    time_text = ax.text2D(0.02, 0.96, "", transform=ax.transAxes, fontsize=11, va="top")
    # (track_name, key) -> Text3D, created once and left on screen permanently.
    # The per-track lane separation already keeps different tracks' labels apart,
    # and EVENT_EXTRA_DZ keeps a single track's own close-together events apart.
    created_events: dict = {}

    def interpolated_altitude(track: dict, t: float):
        times = track["times"]
        if t < times[0]:
            return None
        if t >= times[-1]:
            return float(track["altitudes"][-1])
        return float(np.interp(t, times, track["altitudes"]))

    def update(frame_index: int):
        t = t_schedule[frame_index]
        artists = [time_text]
        for i, track in enumerate(tracks):
            lane = lanes[i]
            times = track["times"]
            mask = (times <= t) & (times >= t_min)
            line = lines[track["name"]]
            dot = dots[track["name"]]
            if mask.any():
                shown_t = times[mask]
                shown_z = track["altitudes"][mask]
                line.set_data(shown_t, np.full_like(shown_t, lane))
                line.set_3d_properties(shown_z)
                y = interpolated_altitude(track, t)
                if y is not None:
                    dot.set_data([t], [lane])
                    dot.set_3d_properties([y])
            artists.extend([line, dot])

            for key in EVENT_LABELS:
                event_time = track["events"][key]
                if event_time is None or t < event_time or event_time < t_min:
                    continue
                entry_key = (track["name"], key)
                if entry_key not in created_events:
                    z = interpolated_altitude(track, event_time)
                    if z is None:
                        z = 0.0
                    dz = EVENT_EXTRA_DZ.get(key, 0.0)
                    label = EVENT_LABELS[key]
                    if key in track["estimated_events"]:
                        label += " (est.)"
                    ax.scatter([event_time], [lane], [z], marker="x", s=45, color=track["color"], zorder=5)
                    text = ax.text(event_time, lane, z + dz, label, fontsize=7, color=track["color"],
                                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
                    created_events[entry_key] = text
                artists.append(created_events[entry_key])

        time_text.set_text(f"t = {t:5.2f} s")
        # Slow continuous rotation, timed to wrap exactly at the loop point so the
        # camera angle spins seamlessly if the video is looped.
        azim = -60.0 + 360.0 * (frame_index / len(t_schedule))
        ax.view_init(elev=18, azim=azim)
        return artists

    anim = FuncAnimation(fig, update, frames=len(t_schedule), interval=1000 / args.fps, blit=False)
    # crf 24 keeps the file under GitHub's undocumented inline-preview size
    # limit, well below the repo's general 100 MB storage cap.
    writer = FFMpegWriter(fps=args.fps, codec="libx264",
                           extra_args=["-pix_fmt", "yuv420p", "-crf", "24", "-preset", "slow",
                                        "-movflags", "+faststart"])
    anim.save(args.output, writer=writer)
    plt.close(fig)
    print(f"Wrote {args.output} ({len(t_schedule)} frames @ {args.fps} fps, "
          f"{len(t_schedule) / args.fps:.1f}s playback covering {t_max - t_min:.1f}s of flight)")


if __name__ == "__main__":
    main()
