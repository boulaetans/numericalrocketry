"""Interactive (zoomable, pannable, hoverable) version of plot_flight_comparison.py's
static PNG, for anyone who wants to actually explore the data instead of
looking at a fixed image, the same way OpenRocket's own plot viewer allows
zooming into a flight instead of only ever showing one fixed view of it.

Written with Plotly and saved as a single self-contained HTML file (the
plotly.js bundle is embedded directly in the file, not loaded from a CDN),
so it opens and works fully offline in any browser. GitHub's markdown
renderer only displays static images, not embedded JavaScript, so this
doesn't render inline on the README; the README links to it instead via
htmlpreview.github.io (renders a raw GitHub-hosted HTML file live, no
download needed) with a plain download link as the offline fallback.

Same design language as the static PNG: color identifies track, line style
identifies quantity (solid = altitude, dashed = velocity, dotted =
acceleration), altitude on the left axis and velocity/acceleration sharing
the right axis, and each track's mission events (ignition, burnout, apogee,
recovery deploy, touchdown) are marked with an X and a label in that
track's own color. Hover any point, marker included, for its exact value;
drag to zoom into a region, double-click to reset; click a legend entry to
toggle that line.

Usage (from the project root, with the venv active):
    .venv\\Scripts\\python.exe plot_flight_comparison_interactive.py [--output FILE.html]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from animate_flight_comparison import (
    EVENT_LABELS,
    load_numericalrocketry_track,
    load_openrocket_track,
    load_real_flight_track,
)

LEFT_AXIS_QUANTITIES = {"altitude": {"key": "altitudes", "dash": "solid", "label": "Altitude (m)"}}
RIGHT_AXIS_QUANTITIES = {
    "velocity": {"key": "velocities", "dash": "dash", "label": "Velocity (m/s)"},
    "acceleration": {"key": "accelerations", "dash": "dot", "label": "Acceleration (m/s²)"},
}

TOUCHDOWN_TRIM_S = 0.15  # drops the ground-contact-clamp derivative artifact, see module docstring
VIEW_PAD_S = 3.0  # how far past real-flight touchdown the default view extends


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="assets/flight_comparison_interactive.html")
    args = parser.parse_args()

    tracks = [
        load_numericalrocketry_track(),
        load_openrocket_track(Path("data/green_eggs_openrocket_flight.csv")),
        load_real_flight_track(Path("data/green_eggs_real_flight.csv")),
    ]
    real_touchdown = next(t for t in tracks if t["name"] == "Real Flight")["events"]["touchdown"]
    default_x_max = real_touchdown + VIEW_PAD_S

    fig = go.Figure()
    for track in tracks:
        touchdown = track["events"]["touchdown"]
        end_t = touchdown - TOUCHDOWN_TRIM_S if touchdown is not None else track["times"][-1]
        mask = track["times"] <= end_t
        t = track["times"][mask]

        for name, spec in LEFT_AXIS_QUANTITIES.items():
            fig.add_trace(go.Scatter(
                x=t, y=track[spec["key"]][mask], mode="lines", name=f"{track['name']} - {spec['label']}",
                legendgroup=track["name"], line=dict(color=track["color"], dash=spec["dash"], width=2),
                hovertemplate=f"{track['name']}<br>{spec['label']}: %{{y:.2f}}<br>t=%{{x:.2f}}s<extra></extra>",
            ))
        for name, spec in RIGHT_AXIS_QUANTITIES.items():
            fig.add_trace(go.Scatter(
                x=t, y=track[spec["key"]][mask], mode="lines", name=f"{track['name']} - {spec['label']}",
                legendgroup=track["name"], line=dict(color=track["color"], dash=spec["dash"], width=1.6),
                opacity=0.85, yaxis="y2",
                hovertemplate=f"{track['name']}<br>{spec['label']}: %{{y:.2f}}<br>t=%{{x:.2f}}s<extra></extra>",
            ))

    # Event markers (ignition, burnout, apogee, recovery deploy, touchdown),
    # matching the ones already shown in the GIF. Several events land close
    # together across tracks (ignition is the exact same (0, 0) point for
    # all three), so each track gets a different text position to reduce
    # overlap; hovering any marker shows its exact values regardless.
    TRACK_TEXT_POSITION = {"NumericalRocketry": "top left", "OpenRocket (reference sim)": "top right",
                            "Real Flight": "bottom right"}
    for track in tracks:
        for key, label in EVENT_LABELS.items():
            event_t = track["events"].get(key)
            if event_t is None:
                continue
            event_alt = float(np.interp(event_t, track["times"], track["altitudes"]))
            text = label + (" (est.)" if key in track["estimated_events"] else "")
            fig.add_trace(go.Scatter(
                x=[event_t], y=[event_alt], mode="markers+text", text=[text],
                textposition=TRACK_TEXT_POSITION.get(track["name"], "top right"),
                textfont=dict(color=track["color"], size=10),
                marker=dict(symbol="x", size=9, color=track["color"]),
                legendgroup=track["name"], showlegend=False,
                hovertemplate=f"{track['name']} {text}<br>alt: %{{y:.2f}} m<br>t=%{{x:.2f}}s<extra></extra>",
            ))

    fig.update_layout(
        title="Green Eggs: Simulated vs. Real Flight (drag to zoom, double-click to reset, click legend to toggle)",
        xaxis=dict(title="Time since ignition (s)", range=[0, default_x_max]),
        yaxis=dict(title="Altitude (m)"),
        yaxis2=dict(title="Velocity (m/s)  /  Acceleration (m/s²)", overlaying="y", side="right"),
        hovermode="closest",
        legend=dict(title="Track - Quantity (click to toggle)", font=dict(size=10)),
        template="plotly_white",
        width=1100, height=650,
    )

    fig.write_html(args.output, include_plotlyjs=True, full_html=True)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
