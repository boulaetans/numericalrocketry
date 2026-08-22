# Full results table

Supplemental to the main [README](../README.md): every mission-event metric computed across
all three tracks (NumericalRocketry's own 6DOF simulation, OpenRocket's reference simulation,
and the real logged flight), not just the headline subset (apogee, max velocity, touchdown)
shown there. The README stays focused on what a reader actually cares about; this is the
complete data behind it, including the internal validation metrics (liftoff mass, CG) that
mattered a great deal during development but aren't something you'd ever observe watching the
rocket fly.

See the [legacy/](../legacy/) folder for how the project's original 1D model compares in a
separate table, since that comparison only covers ascent (the 1D model has no recovery system).

| Event / metric | NumericalRocketry (6DOF) | OpenRocket (reference sim) | Real Flight (logged) | NR vs. OR gap | NR vs. Real gap |
| --- | --- | --- | --- | --- | --- |
| Ignition | t = 0.034 s | t = 0 s | t = 0 s *(exact, see note)* | n/a | n/a |
| Rail clear | t = 0.256 s | t = 0.256 s | not recoverable from this log | exact | n/a |
| Burnout | t = 0.81 s | t = 0.81 s | t ≈ 1.42 s *(est.)* | exact | 42.92% |
| Apogee altitude | 101.30 m | 101.29 m | 115.67 m | +0.01 m (0.01%) | -14.37 m (12.42%) |
| Apogee time | 4.7662 s | 4.766 s | 6.06 s | ~0.2 ms | -1.294 s (21.35%) |
| Max velocity | 45.53 m/s | 45.53 m/s | 42.35 m/s *(re-derived, see README note)* | ~0.005 m/s (0.01%) | +3.18 m/s (7.51%) |
| Recovery deploy | 5.8163 s | 5.811 s | not detectable in this log | ~5 ms | n/a |
| Touchdown | 27.116 s | 27.194 s | t = 42.36 s *(corrected, see note)* | -0.08 s (0.29%) | -15.24 s (35.99%) |
| Liftoff mass | 166.041 g | 166 g | not measured | +0.04 g (0.03%) | n/a |
| CG location | 35.52 cm | 35.7 cm | not measured | -0.18 cm (0.50%) | n/a |

The **NR vs. Real gap** column is the "real flight vs. simulation" comparison discussed in the README's Results section. Apogee altitude's 12.42% gap is the number behind that discussion of dynamic-pressure/altimeter error, not a simulator inaccuracy (both NumericalRocketry and OpenRocket agree with each other to 0.01%, see above). Ignition, rail clear, recovery deploy, liftoff mass, and CG have no real-flight measurement to compare against, so there is no percentage to compute for those rows.

Burnout (42.92%) and touchdown (35.99%) look like much bigger misses than they actually are. Both real-flight values are `(est.)`/`(corrected)`, derived from a different definition than what NumericalRocketry/OpenRocket report: burnout as "thrust drops below drag+weight" versus total propellant consumption, and touchdown as "acceleration/speed settle near zero" versus the flight computer's own, much-delayed "landed" flag. These are related but not identical events, so a large gap reflects that definitional difference more than actual trajectory error, unlike apogee and max velocity, which are directly comparable across all three.

**On the real-flight columns marked `(exact)`, `(est.)`, `(corrected)`, "not detectable", or "not measured":** the flight computer's own raw log required real correction before any of these numbers were usable, not just estimation. The EasyMini board used for this flight is barometer-only (no accelerometer, so its own `acceleration`/`speed` are its real-time filter's estimate from pressure). Its raw ground elevation was off by 24.5 m, 66 rows were exact duplicate artifacts, and its `boost`/`coast`/`main`/`landed` state labels did not match this flight's real physical events. All of that is fixed directly in the CSV; see [`data/derive_real_flight_csv.py`](derive_real_flight_csv.py) for the full corrected pipeline: ground elevation solved by least-squares fit against the actual pressure data, duplicate rows removed, the missing pre-recording gap filled with a physically-motivated "starts from rest" curve fit to the real early data, speed/acceleration re-derived offline with a Kalman filter and RTS smoother instead of trusting the onboard real-time estimate, and `state_name` corrected at ignition and landed. Ignition is exact by construction (the rebuilt CSV's own t=0). Burnout is still `(est.)`: no state-machine marker exists for it, so it is found the same way NumericalRocketry/OpenRocket are, from the acceleration trace's zero-crossing, using the re-derived acceleration column. Touchdown is `(corrected)` rather than raw or purely estimated; it is baked into `state_name` in the rebuilt CSV at t=42.36s, the first point where re-derived speed reads exactly 0.00 and stays within sensor-noise range for the rest of the log, rather than the onboard flag's own delayed timestamp (t=54.99s, a confirmation/debounce lag of about 12.6s) or a separate speed-threshold heuristic. Values with no real signature in the log (recovery deploy) or that were never recorded by the flight computer at all (mass, CG) are left blank rather than guessed.

**On liftoff mass and CG:** these are not flight-observable outcomes, but they are a meaningful
part of the validation story. Every other row here depends on the mass model being correct, and
getting CG within 0.5% required tracking down real shape-integration bugs in the nose cone and
fin set (see the git history and commit messages around `geometry.py`'s
`nose_shell_mass_cg_m()` and `fin_set_mass_cg_m()` for that investigation).
