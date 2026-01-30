# Full results table

Supplemental to the main [README](../README.md) — every mission-event metric computed across
all three tracks (NumericalRocketry's own 6DOF simulation, OpenRocket's reference simulation,
and the real logged flight), not just the headline subset (apogee, max velocity, touchdown)
shown there. The README stays focused on what a reader actually cares about; this is the
complete data behind it, including the internal validation metrics (liftoff mass, CG) that
mattered a great deal during development but aren't something you'd ever observe watching the
rocket fly.

See the [legacy/](../legacy/) folder for how the project's original 1D model compares — a
separate table, since that comparison only covers ascent (the 1D model has no recovery system).

| Event / metric | NumericalRocketry (6DOF) | OpenRocket (reference sim) | Real Flight (logged) | NR vs. OR gap |
| --- | --- | --- | --- | --- |
| Ignition | t = 0.034 s | t = 0 s | t ≈ -0.66 s *(est.)* | — |
| Rail clear | t = 0.256 s | t = 0.256 s | not recoverable from this log | exact |
| Burnout | t = 0.81 s | t = 0.81 s | t ≈ 0.57 s *(est.)* | exact |
| Apogee altitude | 101.25 m | 101.29 m | 111.2 m | -0.04 m (0.04%) |
| Apogee time | 4.7663 s | 4.766 s | 4.76 s | ~1 ms |
| Max velocity | 45.53 m/s | 45.53 m/s | 35.4 m/s *(see README note)* | ~0.005 m/s (0.01%) |
| Recovery deploy | 5.8163 s | 5.811 s | not detectable in this log | ~5 ms |
| Touchdown | 27.116 s | 27.194 s | t ≈ 41.06 s *(est.)* | -0.08 s (0.29%) |
| Liftoff mass | 166.041 g | 166 g | not measured | +0.04 g (0.03%) |
| CG location | 35.52 cm | 35.7 cm | not measured | -0.18 cm (0.50%) |

**On the real-flight columns marked `(est.)` or "not detectable"/"not measured":** the flight
computer's own reported state (`boost`/`coast`/`main`/`landed`) doesn't match this flight's real
physical events (checked directly against the raw sensor samples — see the README's Results
section and `load_real_flight_track()` in [`animate_flight_comparison.py`](../animate_flight_comparison.py)
for the full account). Estimated values are derived directly from the raw data using the method
documented there; values with no real signature in the log (recovery deploy) or that were never
recorded by the flight computer at all (mass, CG) are left blank rather than guessed.

**On liftoff mass and CG:** these aren't flight-observable outcomes, but they're a meaningful
part of the validation story — every other row here depends on the mass model being right, and
getting CG within 0.5% required tracking down real shape-integration bugs in the nose cone and
fin set (see the git history / commit messages around `geometry.py`'s
`nose_shell_mass_cg_m()` and `fin_set_mass_cg_m()` for that investigation).
