"""Static altitude/velocity/acceleration comparison of the same three flights
animate_flight_comparison.py covers, for anyone who wants the shape of the
result without waiting for the GIF to loop.

One plot, two y-axes, matching how OpenRocket's own plot view handles this
exact situation (confirmed against its docs, not assumed): "You can assign
multiple parameters to the Y-axis, and choose whether their scales appear on
the left, or the right side of the plot." Altitude (meters, 0-115 range this
flight) gets the left axis; velocity and acceleration (m/s and m/s^2, both
roughly the same -15..65 range this flight) share the right axis, the same
way OpenRocket caps out at two scales rather than one per quantity. Line
style tells the three quantities apart (solid/dashed/dotted); color tells
the three tracks apart. Two independent legends, stacked in one corner.

NumericalRocketry and OpenRocket trace almost exactly on top of each other
for most of the flight, since matching OpenRocket is this project's own
validation target. A fully opaque line drawn second would completely hide
the one drawn first wherever they coincide, so every line is drawn at
partial alpha (overlapping colors visibly blend instead of one erasing the
other) and NumericalRocketry is drawn slightly wider than OpenRocket, so a
thin OpenRocket line stays visible inside NumericalRocketry's wider halo
even at an exact overlap. Velocity and acceleration use bold dash/dot
patterns with round line caps rather than matplotlib's thin defaults, so
they read clearly against the solid altitude lines.

Each track's mission events (ignition, burnout, apogee, recovery deploy,
touchdown) are marked with an X and a label, in that track's own color,
matching the event markers already shown in the animated GIF.

Each track's plotted range stops just before its own touchdown, not at the
end of its raw data. This matters most for the NumericalRocketry track: the
6DOF sim hard-clamps velocity to exactly zero on ground contact, and
differentiating that step for the acceleration curve produces a fake ~90
m/s^2 spike in the final couple of samples that never physically happened;
trimming those samples removes the artifact rather than smoothing over it.
The overall x-axis is then capped at VIEW_PAD_S past the real flight's own
touchdown event (t=42.36s; see data/derive_real_flight_csv.py for how
that's determined from the re-derived speed settling) so the flat,
already-landed tail doesn't dominate the figure.

The real flight's lines are left exactly as re-derived (see
data/derive_real_flight_csv.py), including their natural roughness. That's
real recorded data, not a simulation, and the visible texture is the point.
This script always reads data/green_eggs_real_flight.csv specifically (never
data/green_eggs_real_flight_raw.csv, the raw pre-correction export that CSV
is rebuilt from; see derive_real_flight_csv.py for the raw source).

Usage (from the project root, with the venv active):
    .venv\\Scripts\\python.exe plot_flight_comparison.py [--output FILE.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from animate_flight_comparison import (
    EVENT_LABELS,
    load_numericalrocketry_track,
    load_openrocket_track,
    load_real_flight_track,
)

# altitude -> left axis; velocity/acceleration share the right axis. dashes/
# capstyle are set explicitly (rather than relying on matplotlib's default
# "--"/":" linestyles) so velocity/acceleration read as bold, clearly visible
# patterns instead of thin ticks.
LEFT_AXIS_QUANTITIES = {
    "altitude": {"key": "altitudes", "label": "Altitude (m)", "lw": 2.0,
                 "dashes": None, "capstyle": "butt"},
}
RIGHT_AXIS_QUANTITIES = {
    "velocity": {"key": "velocities", "label": "Velocity (m/s)", "lw": 2.1,
                 "dashes": (6, 2.5), "capstyle": "butt"},
    "acceleration": {"key": "accelerations", "label": "Acceleration (m/s²)", "lw": 2.8,
                      "dashes": (0.1, 2.4), "capstyle": "round"},
}

# NumericalRocketry and OpenRocket trace almost exactly on top of each other
# (see module docstring); NumericalRocketry is drawn wider so it forms a
# visible halo around OpenRocket's thinner line at an exact overlap.
TRACK_LINEWIDTH_BOOST = {"NumericalRocketry": 0.9, "OpenRocket (reference sim)": 0.0, "Real Flight": 0.0}
LINE_ALPHA = 0.78

TOUCHDOWN_TRIM_S = 0.15  # drops the ground-contact-clamp derivative artifact, see module docstring
VIEW_PAD_S = 3.0  # how far past real-flight touchdown the x-axis extends


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="assets/flight_comparison.png")
    args = parser.parse_args()

    tracks = [
        load_numericalrocketry_track(),
        load_openrocket_track(Path("data/green_eggs_openrocket_flight.csv")),
        load_real_flight_track(Path("data/green_eggs_real_flight.csv")),
    ]

    real_touchdown = next(t for t in tracks if t["name"] == "Real Flight")["events"]["touchdown"]
    x_max = real_touchdown + VIEW_PAD_S

    fig, ax_left = plt.subplots(figsize=(10, 6.5), dpi=150)
    ax_right = ax_left.twinx()

    def line_kwargs(spec: dict, color: str, lw_boost: float) -> dict:
        kwargs = dict(color=color, alpha=LINE_ALPHA, lw=spec["lw"] + lw_boost,
                      dash_capstyle=spec["capstyle"], solid_capstyle=spec["capstyle"])
        if spec["dashes"] is not None:
            kwargs["dashes"] = spec["dashes"]
        return kwargs

    for track in tracks:
        touchdown = track["events"]["touchdown"]
        end_t = touchdown - TOUCHDOWN_TRIM_S if touchdown is not None else track["times"][-1]
        mask = track["times"] <= min(end_t, x_max)
        t = track["times"][mask]
        lw_boost = TRACK_LINEWIDTH_BOOST.get(track["name"], 0.0)
        for spec in LEFT_AXIS_QUANTITIES.values():
            ax_left.plot(t, track[spec["key"]][mask], **line_kwargs(spec, track["color"], lw_boost))
        for spec in RIGHT_AXIS_QUANTITIES.values():
            ax_right.plot(t, track[spec["key"]][mask], **line_kwargs(spec, track["color"], lw_boost))

    # Event markers (ignition, burnout, apogee, recovery deploy, touchdown),
    # matching the ones already shown in the GIF. Ignition sits at the exact
    # same (0, 0) point for all three tracks, and several other events land
    # close together too, so each track gets its own small diagonal offset
    # (TRACK_LABEL_OFFSET) on top of a per-event base offset (EVENT_BASE_DY)
    # to keep same-event labels from stacking; the X marker itself always
    # sits at the true event point. Events a track can't determine (None)
    # are skipped; events found by estimation rather than a direct state-
    # machine marker are suffixed "(est.)", matching the GIF's convention.
    EVENT_BASE_DY = {"ignition": -14, "burnout": 12, "apogee": 12, "recovery_deploy": -18, "touchdown": 12}
    TRACK_LABEL_OFFSET = {"NumericalRocketry": 0, "OpenRocket (reference sim)": 1, "Real Flight": 2}
    for track in tracks:
        offset_idx = TRACK_LABEL_OFFSET.get(track["name"], 0)
        for key, label in EVENT_LABELS.items():
            event_t = track["events"].get(key)
            if event_t is None or event_t > x_max:
                continue
            event_alt = float(np.interp(event_t, track["times"], track["altitudes"]))
            text = label + (" (est.)" if key in track["estimated_events"] else "")
            dx = 6 + offset_idx * 34
            if event_t > x_max - 3.0:  # keep right-edge labels (real flight's touchdown) from clipping
                dx = -6 - offset_idx * 34
                ha = "right"
            else:
                ha = "left"
            dy = EVENT_BASE_DY.get(key, 10) + offset_idx * 4
            ax_left.scatter([event_t], [event_alt], marker="x", s=45, color=track["color"], zorder=5)
            ax_left.annotate(
                text, (event_t, event_alt), textcoords="offset points", ha=ha,
                xytext=(dx, dy), fontsize=7.5, color=track["color"],
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
            )

    ax_left.set_xlim(0, x_max)
    ax_left.set_ylim(top=max(track["altitudes"].max() for track in tracks) * 1.15)
    ax_left.set_xlabel("Time since ignition (s)")
    ax_left.set_ylabel("Altitude (m)")
    ax_right.set_ylabel("Velocity (m/s)  /  Acceleration (m/s²)")
    ax_left.set_title("Green Eggs: Simulated vs. Real Flight")
    ax_left.grid(alpha=0.25)
    ax_right.axhline(0, color="#c3c2b7", lw=0.8, zorder=0)

    # Two independent legends stacked in the same corner: color -> track (top),
    # line style -> quantity (below it).
    track_legend_handles = [
        Line2D([0], [0], color=track["color"], lw=2.2, label=track["name"]) for track in tracks
    ]
    all_quantities = {**LEFT_AXIS_QUANTITIES, **RIGHT_AXIS_QUANTITIES}
    quantity_legend_handles = [
        Line2D([0], [0], label=spec["label"], **line_kwargs(spec, "#52514e", 0.0))
        for spec in all_quantities.values()
    ]
    track_legend = ax_left.legend(handles=track_legend_handles, loc="upper right",
                                   bbox_to_anchor=(1.0, 1.0), fontsize=9, title="Track", title_fontsize=9)
    ax_left.add_artist(track_legend)
    ax_left.legend(handles=quantity_legend_handles, loc="upper right",
                    bbox_to_anchor=(1.0, 0.74), fontsize=9, title="Quantity", title_fontsize=9)

    fig.tight_layout()
    fig.savefig(args.output)
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
