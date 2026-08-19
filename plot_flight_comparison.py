"""Static altitude-vs-time comparison of the same three flights animate_flight_comparison.py
covers, for anyone who wants the shape of the result without waiting for the GIF to loop.

Usage (from the project root, with the venv active):
    .venv\\Scripts\\python.exe plot_flight_comparison.py [--output FILE.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from animate_flight_comparison import (
    load_numericalrocketry_track,
    load_openrocket_track,
    load_real_flight_track,
)
from numericalrocketry.rocket.rocket_config import green_egg_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="assets/flight_comparison.png")
    args = parser.parse_args()

    cfg = green_egg_config()
    tracks = [
        load_numericalrocketry_track(),
        load_openrocket_track(Path("data/Green Eggs OR Flight.csv")),
        load_real_flight_track(Path("data/Green Eggs Real Flight.csv"), cfg.launch_altitude_m),
    ]
    # NumericalRocketry and OpenRocket trace almost exactly on top of each other (that's the
    # whole point of the project), so one gets a dashed line or it would just look missing.
    line_styles = ["--", "-", "-"]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    for track, ls in zip(tracks, line_styles):
        ax.plot(track["times"], track["altitudes"], ls, lw=1.8, color=track["color"], label=track["name"])

    ax.set_xlim(0, max(t["times"][-1] for t in tracks) * 1.02)
    ax.set_ylim(0, max(t["altitudes"].max() for t in tracks) * 1.1)
    ax.set_xlabel("Time since ignition (s)")
    ax.set_ylabel("Altitude AGL (m)")
    ax.set_title("Green Eggs: Simulated vs. Real Flight")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(args.output)
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
