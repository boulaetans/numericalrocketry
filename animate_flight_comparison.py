"""Animated 3D GIF comparing three versions of the same Green Egg flight: this
project's own 6DOF simulation, OpenRocket's own simulated flight, and the real
logged flight-computer data from the actual launch. Time and altitude are real
axes; the third axis is categorical (which of the three this is), so the three
otherwise near-identical curves don't just sit on top of each other.

Each track's own mission events (ignition, burnout, apogee, recovery deploy,
touchdown) are marked on that track as the animation reaches them -- the
three tracks generally do NOT reach the same event at the same time, and watching
them diverge over the course of the flight (especially descent, where the real
flight's much longer hang time shows up clearly) is the point of this animation.

The real flight log only records four coarse states (boost/coast/main/landed), and
checking them directly against the raw acceleration/speed samples shows they don't
line up with true motor burnout, chute deployment, or touchdown for this specific
log (see load_real_flight_track()'s docstring). Where a real physical event has a
detectable signature in the raw data, it's estimated directly from that data instead
(and marked "(est.)"); where it doesn't, it's left unmarked rather than guessed.

Usage (from the project root, with the venv active):
    .venv\\Scripts\\python.exe animate_flight_comparison.py [--output FILE.gif] [--fps N]

Reads its reference data from data/Green Eggs OR Flight.csv and
data/Green Eggs Real Flight.csv, and writes to assets/flight_comparison.gif by default.
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
from matplotlib.animation import FuncAnimation, PillowWriter
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
    raw_events = history["events"][0]
    events = {key: raw_events.get(f"{key}_time_s") for key in EVENT_LABELS}
    return {
        "name": "NumericalRocketry",
        "color": "#3b82f6",
        "times": times,
        "altitudes": altitudes,
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

    return {
        "name": "OpenRocket (reference sim)",
        "color": "#f59e0b",
        "times": np.array(times, dtype=float),
        "altitudes": np.array(altitudes, dtype=float),
        "events": events,
        "estimated_events": set(),
    }


def load_real_flight_track(csv_path: Path, launch_altitude_m: float) -> dict:
    """Real flight-computer log for the actual launch.

    The log's own state_name column (boost/coast/main/landed) does NOT correspond
    1:1 to OpenRocket-style events for this specific flight -- checked directly
    against the raw samples:

    * "boost" persists until t=2.21s, ~1.4s after the motor's real burn time
      (the state machine's boost->coast transition tracks something other than
      literal thrust cutoff).
    * "main" begins at t=3.75s, roughly a full second BEFORE the true altitude
      peak (t=4.76s) -- and the raw acceleration trace has no discontinuity there
      at all, so it isn't chute deployment either.
    * "landed" doesn't trigger until t=53.69s, but speed and altitude both settle
      to a dead stop around t=41s -- a ~12s confirmation/debounce lag.

    So instead of trusting the state labels, each event actually derivable from the
    raw data is estimated directly from it:

    * ignition: the log's earliest sample (t=-0.62s) is already only 0.66 m AGL
      and climbing at 15.77 m/s -- there's no pre-liftoff baseline recorded, but
      it's close enough to the ground that a full curve-fit extrapolation isn't
      needed. A single linear step back from that first sample, at its own
      recorded velocity (`t - altitude/speed`), gives ignition at t≈-0.66s -- a
      42ms extrapolation, not the far larger (and much less trustworthy) one a
      multi-sample curve fit across the whole early climb would need.
    * burnout: the first zero-crossing of raw acceleration after t=0 -- this marks
      when thrust drops below drag+weight, which for a motor's tail-off can (and
      here does) happen somewhat before the motor's total burn time.
    * apogee: the raw altitude maximum -- unambiguous, no estimation needed.
    * recovery_deploy: NOT estimated. There's no detectable transient anywhere in
      the raw acceleration/speed data between the "main" state's early onset and
      apogee -- the coast-to-descent transition is smooth throughout, consistent
      with this rocket's small/light chute opening too gently to leave a signature
      in this sensor's data. Left unmarked rather than guessed.
    * touchdown: the first point after apogee where speed stays under 0.2 m/s for
      the following 20 samples (a real, if soft, landing -- not the state
      machine's much-delayed "landed" flag).
    """
    times: list[float] = []
    altitudes: list[float] = []
    accel: list[float] = []
    speed: list[float] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time"]))
            altitudes.append(float(row["altitude"]) - launch_altitude_m)
            accel.append(float(row["acceleration"]))
            speed.append(float(row["speed"]))

    t = np.array(times, dtype=float)
    alt = np.array(altitudes, dtype=float)
    a = np.array(accel, dtype=float)
    v = np.array(speed, dtype=float)

    events: dict = {key: None for key in EVENT_LABELS}
    estimated_events: set = set()

    # Ignition: one linear step back from the very first recorded sample, at its
    # own recorded velocity -- see the docstring above for why this (rather than a
    # wider curve fit) is the extrapolation actually used.
    if v[0] > 0.0:
        events["ignition"] = float(t[0] - alt[0] / v[0])
        estimated_events.add("ignition")

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

    # Touchdown: first sustained (20-sample) settle to near-zero speed after apogee.
    threshold, hold = 0.2, 20
    for i in range(apogee_index, len(t) - hold):
        if np.all(np.abs(v[i:i + hold]) < threshold):
            events["touchdown"] = float(t[i])
            estimated_events.add("touchdown")
            break

    return {
        "name": "Real Flight",
        "color": "#22c55e",
        "times": t,
        "altitudes": alt,
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
    parser.add_argument("--output", default="assets/flight_comparison.gif")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--playback-seconds", type=float, default=16.0, help="Real-world seconds the main sweep takes to play")
    parser.add_argument("--hold-seconds", type=float, default=5.0, help="Pause after the last touchdown before the GIF loops")
    args = parser.parse_args()

    cfg = green_egg_config()
    tracks = [
        load_numericalrocketry_track(),
        load_openrocket_track(Path("data/Green Eggs OR Flight.csv")),
        load_real_flight_track(Path("data/Green Eggs Real Flight.csv"), cfg.launch_altitude_m),
    ]
    lanes = {0: 0.0, 1: 1.0, 2: 2.0}

    # Earliest ignition estimate across all three tracks (NR/OR both ignite at
    # ~t=0; the real flight's small negative extrapolation -- see
    # load_real_flight_track -- lets its own genuinely-recorded pre-t=0 samples
    # show too, instead of clipping them off).
    t_min = min(
        track["events"]["ignition"] if track["events"]["ignition"] is not None else track["times"][0]
        for track in tracks
    )
    # The sweep runs to the latest actual touchdown, not the latest raw data sample
    # -- the real flight log keeps recording for ~12s after it's already stopped
    # moving (see load_real_flight_track's docstring), and animating through that
    # flat "just sitting there" tail added nothing.
    t_max = max(track["events"]["touchdown"] or track["times"][-1] for track in tracks)
    z_max = max(track["altitudes"].max() for track in tracks)

    t_schedule = build_time_schedule(t_min, t_max, args.fps, args.playback_seconds, args.hold_seconds)

    fig = plt.figure(figsize=(10, 7), dpi=130)
    ax = fig.add_subplot(111, projection="3d")
    try:
        ax.set_box_aspect((3.0, 1.0, 1.3))
    except AttributeError:
        pass  # older matplotlib without set_box_aspect -- cosmetic only
    ax.set_xlim(t_min, t_max * 1.03)
    ax.set_ylim(-0.5, 2.5)
    ax.set_zlim(-10.0, z_max * 1.15)
    ax.set_xlabel("Time since ignition (s)")
    # Short lane codes only -- the full names live in the fixed 2D legend below
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
    # (track_name, key) -> Text3D, created once and left on screen permanently --
    # the per-track lane separation already keeps different tracks' labels apart,
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
        # GIF spins seamlessly when it repeats.
        azim = -60.0 + 360.0 * (frame_index / len(t_schedule))
        ax.view_init(elev=18, azim=azim)
        return artists

    anim = FuncAnimation(fig, update, frames=len(t_schedule), interval=1000 / args.fps, blit=False)
    anim.save(args.output, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"Wrote {args.output} ({len(t_schedule)} frames @ {args.fps} fps, "
          f"{len(t_schedule) / args.fps:.1f}s playback covering {t_max - t_min:.1f}s of flight)")


if __name__ == "__main__":
    main()
