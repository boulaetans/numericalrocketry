"""Rebuilds Green Eggs Real Flight.csv from the raw EasyMini export
(Green Eggs Real Flight Raw.csv), documenting every correction applied to
the raw flight-computer log and making the whole thing reproducible from
scratch.

Why this exists: the raw log, straight off the EasyMini 2, has four real
problems that make it unusable as-is for comparison against a simulation:

1. Duplicate rows. 66 rows in the raw log are exact byte-for-byte repeats of
   their neighbor (64 of them stacked at a single `time=0.00` tick), a
   logging/export artifact rather than real samples. Removed.

2. Wrong ground elevation. The EasyMini has no accelerometer; it's a
   barometer-only board (Measurement Specialties MS5607). Its `altitude`
   column is computed from raw pressure via the standard ISA formula assuming
   a fixed 1013.25 hPa sea-level reference, not this day's actual pressure.
   That reads a ground elevation of 507.5 m for this flight; the site's real
   surveyed elevation is 532 m. The actual local reference pressure is solved
   by least-squares fitting the ISA formula against every row in the flight
   (using the AGL `height` column, which, being a pressure *difference*
   from the pad rather than an absolute reading, already cancels this bias
   almost exactly). Result: P0 = 1016.23 hPa, about 2.9 hPa above standard,
   an ordinary day and not an outlier. `altitude`/`height` are then
   recomputed directly from `pressure` with that P0.

3. Missing pre-recording gap. The flight computer doesn't start saving
   samples until it's confident a real launch is underway, so the first saved
   row is already ~0.68s into the flight (5+ m up, ~16 m/s). That gap is
   filled with a physically-motivated "starts from rest" cubic
   h(t) = k*(t-t0)^3, fit to the real early-flight trend (NOT a generic
   simulator run; an earlier attempt using this project's own 6DOF sim for
   the backfill found the sim's mass/drag assumptions don't match this
   specific flight's real dynamics closely enough to trust for this
   purpose). Solving h(t0)=0 and
   h'(t0)=0 against the real data's own local trend gives t0, the estimated
   true liftoff instant, which becomes the new t=0.

4. Untrustworthy onboard speed/acceleration. Because EasyMini has no
   accelerometer, its `acceleration`/`speed` columns are themselves estimates
   from a real-time Kalman filter running on pressure alone, and a real-time
   filter can't see the future, so it measurably lags during the fastest
   part of the flight (checked directly: integrating the onboard `speed`
   under-predicts the real height gained during boost by ~17%, and that gap
   stops growing once the boost phase ends). The onboard speed/acceleration
   are discarded and replaced with values from a forward Kalman filter and
   backward RTS smoother (a 3-state height/velocity/acceleration model, run
   OFFLINE across the whole flight so it isn't limited to past-only data the
   way the onboard filter is) applied to the corrected height signal, run as
   ONE continuous filter across the synthetic pre-recording segment and the
   real recorded segment together, so the two join without a seam.

   Known limitation (not flagged anywhere in the shipped CSV itself): the first
   ~0.3s of boost is the fastest-changing, highest-curvature part of the
   flight, and the sensor's per-sample noise there is close in size to the
   real motion. No amount of retuning fully separates the two (verified:
   loosening the filter chases noise, tightening it collapses toward a
   biased average; there is no clean middle that avoids both). Treat
   acceleration/speed values in that window as the roughest numbers in the
   file. Everything else (the rest of boost, all of coast, all of descent)
   sits on much firmer ground.

state/state_name are also touched up in two places where the onboard state
machine's flags don't reflect what's physically true: the very first row is
labeled "ignition" (state stays 3/boost) instead of just "boost", and every
row from t=42.36s onward (the first point where re-derived speed reads
exactly 0.00 and stays within sensor-noise range, under 0.7 m/s and 2.3
m/s^2, for the rest of the log) is relabeled state=8/"landed". The
onboard "landed" flag doesn't trigger until t=54.99s, a confirmation/debounce
lag of about 12.6s behind when the rocket actually stopped moving.
The output is then trimmed to POST_LANDING_PAD_S past LANDED_TIME_S rather
than keeping the onboard log's full ~13s tail of already-settled noise,
since that tail carries no new information once landing is established.

Usage (from the project root, with the venv active):
    .venv\\Scripts\\python.exe data/derive_real_flight_csv.py

Reads data/Green Eggs Real Flight Raw.csv, writes data/Green Eggs Real Flight.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

RAW_PATH = Path(__file__).parent / "Green Eggs Real Flight Raw.csv"
OUT_PATH = Path(__file__).parent / "Green Eggs Real Flight.csv"

TRUE_GROUND_ELEVATION_M = 532.0
LANDED_TIME_S = 42.36  # where acceleration/speed settle; see module docstring
POST_LANDING_PAD_S = 3.0  # confirmatory tail kept past LANDED_TIME_S
ASCENT_GRID_STEP_S = 0.01
BOOST_KF_UNCERTAIN_S = 0.30  # informational only, not written to the CSV


def isa_altitude_m(pressure_pa: np.ndarray, sea_level_pressure_pa: float) -> np.ndarray:
    """Standard ISA troposphere pressure-altitude formula (what EasyMini itself
    uses internally, just with the correct P0 instead of the fixed 101325 Pa
    standard reference)."""
    return 44330.0 * (1.0 - (pressure_pa / sea_level_pressure_pa) ** (1.0 / 5.255))


def load_and_dedup(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        rows = [[c.strip() for c in row] for row in reader if row]

    deduped: list[list[str]] = []
    prev: list[str] | None = None
    for row in rows:
        if row == prev:
            continue
        deduped.append(row)
        prev = row
    return header, deduped


def solve_reference_pressure(pressure_pa: np.ndarray, true_altitude_m: np.ndarray) -> float:
    """Least-squares fit for the day's actual sea-level-equivalent pressure,
    using every row in the flight simultaneously (far more robust than
    solving from a single ground-level sample)."""
    def cost(p0: float) -> float:
        return float(np.sum((isa_altitude_m(pressure_pa, p0) - true_altitude_m) ** 2))

    result = minimize_scalar(cost, bounds=(100_000, 103_000), method="bounded")
    return float(result.x)


def fit_rest_start_cubic(t: np.ndarray, height: np.ndarray, window_s: float = 0.15):
    """Fits h(t) = k*(t-t0)^3, the simplest curve consistent with 'starts
    from rest' (h(t0)=0 and h'(t0)=0 by construction), to the real early
    height trend, and solves for t0 (estimated true liftoff)."""
    mask = t <= (t[0] + window_s)
    quad_coeffs = np.polyfit(t[mask] - t[0], height[mask], 2)
    c2, c1, c0 = quad_coeffs
    run_up_s = 3 * c0 / c1
    t0 = t[0] - run_up_s
    k = c0 / run_up_s**3
    return t0, k


def kalman_rts_smooth(t: np.ndarray, z: np.ndarray, measurement_r: np.ndarray, jerk_q: float) -> np.ndarray:
    """Forward Kalman filter + backward RTS smoother over a 3-state
    [height, velocity, acceleration] constant-jerk model, using each sample's
    own irregular dt (no resampling needed, which is why it doesn't hit the
    seam artifacts a fixed-window filter, e.g. Savitzky-Golay, gets when the
    sample rate itself changes between flight phases)."""
    n = len(t)
    x = np.zeros((n, 3))
    P = np.zeros((n, 3, 3))
    x[0] = [z[0], 0.0, 0.0]
    P[0] = np.diag([measurement_r[0], 1e-8, 1e-6])  # v=a=0 pinned at t0: true by construction
    H = np.array([1.0, 0.0, 0.0])

    x_pred_hist = np.zeros((n, 3))
    P_pred_hist = np.zeros((n, 3, 3))
    x_pred_hist[0], P_pred_hist[0] = x[0], P[0]

    for i in range(1, n):
        dt = t[i] - t[i - 1]
        F = np.array([[1, dt, 0.5 * dt * dt], [0, 1, dt], [0, 0, 1]])
        Q = jerk_q * np.array([
            [dt**5 / 20, dt**4 / 8, dt**3 / 6],
            [dt**4 / 8, dt**3 / 3, dt**2 / 2],
            [dt**3 / 6, dt**2 / 2, dt],
        ])
        x_pred = F @ x[i - 1]
        P_pred = F @ P[i - 1] @ F.T + Q
        x_pred_hist[i], P_pred_hist[i] = x_pred, P_pred

        innovation = z[i] - H @ x_pred
        innovation_cov = H @ P_pred @ H.T + measurement_r[i]
        gain = (P_pred @ H) / innovation_cov
        x[i] = x_pred + gain * innovation
        P[i] = P_pred - np.outer(gain, H) @ P_pred

    smoothed = x.copy()
    for i in range(n - 2, -1, -1):
        dt = t[i + 1] - t[i]
        F = np.array([[1, dt, 0.5 * dt * dt], [0, 1, dt], [0, 0, 1]])
        gain = P[i] @ F.T @ np.linalg.inv(P_pred_hist[i + 1])
        smoothed[i] = x[i] + gain @ (smoothed[i + 1] - x_pred_hist[i + 1])
    return smoothed


def estimate_measurement_noise(t: np.ndarray, height: np.ndarray) -> float:
    """Robust measurement-noise variance, estimated from local residuals in a
    gentle (near-linear, low-real-jerk) stretch of descent, far from any
    dynamic transient, so the scatter there is essentially pure sensor noise."""
    gentle = (t > 20) & (t < 40)
    tg, hg = t[gentle], height[gentle]
    residuals = []
    for i in range(2, len(tg) - 2):
        local_fit = np.polyfit(tg[i - 2:i + 3] - tg[i], hg[i - 2:i + 3], 1)
        residuals.append(hg[i] - np.polyval(local_fit, 0.0))
    return float(np.var(residuals))


def fmt(x: float) -> str:
    x = round(float(x), 2)
    return f"{0.0 if x == 0 else x:.2f}"  # avoid printing "-0.00"


def main() -> None:
    header, rows = load_and_dedup(RAW_PATH)
    col = {name: header.index(name) for name in
           ["time", "state", "state_name", "pressure", "height"]}
    print(f"loaded {RAW_PATH.name}: deduped to {len(rows)} rows")

    t = np.array([float(r[col["time"]]) for r in rows])
    state = np.array([int(r[col["state"]]) for r in rows])
    state_name = [r[col["state_name"]] for r in rows]
    pressure = np.array([float(r[col["pressure"]]) for r in rows])
    height_onboard = np.array([float(r[col["height"]]) for r in rows])  # AGL, already ~unbiased

    # Altitude/height, properly derived from pressure.
    true_altitude_target = height_onboard + TRUE_GROUND_ELEVATION_M
    p0 = solve_reference_pressure(pressure, true_altitude_target)
    height = isa_altitude_m(pressure, p0) - TRUE_GROUND_ELEVATION_M
    altitude = height + TRUE_GROUND_ELEVATION_M
    print(f"solved reference pressure P0 = {p0:.2f} Pa ({p0/100:.2f} hPa)")

    # Pre-recording gap.
    t0, k = fit_rest_start_cubic(t, height)
    time_shift = -t0
    print(f"estimated run-up before first recorded sample: {t[0] - t0:.3f}s")

    synth_t = np.arange(t0, t[0], ASCENT_GRID_STEP_S)
    synth_h = k * (synth_t - t0) ** 3

    # One continuous Kalman filter and RTS smoother across synthetic and real.
    measurement_r_real = estimate_measurement_noise(t, height)
    measurement_r_synth = 1e-6  # the cubic segment is an exact closed-form curve, not sensor data
    jerk_q = 100.0

    t_all = np.concatenate([synth_t, t])
    h_all = np.concatenate([synth_h, height])
    r_all = np.concatenate([np.full(len(synth_t), measurement_r_synth),
                             np.full(len(t), measurement_r_real)])
    smoothed = kalman_rts_smooth(t_all, h_all, r_all, jerk_q)
    speed_all, accel_all = smoothed[:, 1], smoothed[:, 2]
    synth_v, synth_a = speed_all[:len(synth_t)], accel_all[:len(synth_t)]
    speed, accel = speed_all[len(synth_t):], accel_all[len(synth_t):]

    # State/state_name touch-ups.
    synth_state_name = ["ignition"] + ["boost"] * (len(synth_t) - 1) if len(synth_t) else []
    landed_mask = (t + time_shift) >= LANDED_TIME_S
    state_out = np.where(landed_mask, 8, state)
    state_name_out = ["landed" if landed else name for landed, name in zip(landed_mask, state_name)]

    # Write, trimmed to a short tail past landing rather than keeping the
    # onboard log's full ~13s of already-settled post-landing samples.
    view_end_s = LANDED_TIME_S + POST_LANDING_PAD_S
    fieldnames = ["time", "state", "state_name", "source", "pressure", "altitude", "height",
                  "acceleration", "speed"]
    n_synthetic = n_recorded = 0
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for i, (st, hh, sv, sa) in enumerate(zip(synth_t, synth_h, synth_v, synth_a)):
            name = "ignition" if i == 0 else "boost"
            writer.writerow([f"{st + time_shift:.2f}", 3, name, "synthetic",
                              "", fmt(hh + TRUE_GROUND_ELEVATION_M), fmt(hh), fmt(sa), fmt(sv)])
            n_synthetic += 1
        for i in range(len(t)):
            row_t = t[i] + time_shift
            if row_t > view_end_s:
                break
            writer.writerow([
                f"{row_t:.2f}", int(state_out[i]), state_name_out[i], "recorded",
                fmt(pressure[i]), fmt(altitude[i]), fmt(height[i]), fmt(accel[i]), fmt(speed[i]),
            ])
            n_recorded += 1

    print(f"wrote {OUT_PATH.name}: {n_synthetic + n_recorded} rows "
          f"({n_synthetic} synthetic, {n_recorded} recorded, trimmed to t<={view_end_s:.2f}s)")
    apogee_i = int(np.argmax(np.concatenate([synth_h, height])))
    all_t = np.concatenate([synth_t + time_shift, t + time_shift])
    print(f"apogee: {np.concatenate([synth_h, height])[apogee_i]:.2f} m AGL at t={all_t[apogee_i]:.2f}s")


if __name__ == "__main__":
    main()
