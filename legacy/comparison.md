# Comparison: original 1D model vs. this project

Supplemental to the main [README](../README.md) - how far the project's own predecessor, a
1D ascent-only numerical-methods comparison (`legacy_1d_simulator.py`, adapted here with Green
Eggs' real geometry and motor - see that file's docstring for exactly what changed), gets on
its own, next to the full 6DOF project it grew into and the two references the main README
validates against.

Ascent-only metrics, since the 1D model has no recovery system and free-falls under drag after
apogee - its descent isn't a fair comparison to anything else here, so it's left out (see
`legacy_1d_simulator.py`'s docstring).

| | Euler (1D) | ABM-2 (1D) | RK4 (1D) | NumericalRocketry (6DOF) | OpenRocket | Real Flight |
| --- | --- | --- | --- | --- | --- | --- |
| Apogee altitude | 99.18 m | 99.23 m | 99.27 m | 101.25 m | 101.29 m | 111.2 m |
| Time to apogee | 4.69 s | 4.68 s | 4.68 s | 4.77 s | 4.77 s | 4.76 s |
| Max velocity | 45.66 m/s | 45.63 m/s | 45.64 m/s | 45.53 m/s | 45.53 m/s | 35.4 m/s *(see README note)* |

A few things worth noticing:

* **The three 1D integrators land within 0.1% of each other** (99.18-99.27 m) - Euler furthest off, RK4 closest to the full 6DOF result, exactly the accuracy ordering the original comparison set out to demonstrate. The numerical-method choice matters far less here than the physics model underneath it.
* **The 1D model's RK4 apogee (99.27 m) is within 2.00% of OpenRocket's reference simulation (101.29 m)** despite having no attitude dynamics, no exact Barrowman CP/stability treatment, and a simplified drag model - a reasonable showing for a model this much simpler. The full 6DOF project (101.25 m) cuts that same gap to 0.04% (see the main README's head-to-head), on top of adding a stability margin, off-axis motion, and a modeled recovery/descent phase the 1D model has no way to represent.
* **Max velocity across all three simulated models (1D and 6DOF alike) agrees to within ~0.3%.** Peak velocity happens right at burnout, driven almost entirely by thrust and mass - the part of the flight where the extra fidelity matters least.

![Altitude comparison across the three integrators](altitude_comparison.png)
![Velocity comparison across the three integrators](velocity_comparison.png)
![Acceleration comparison across the three integrators](acceleration_comparison.png)

Regenerate the 1D numbers:
```sh
.venv\Scripts\python.exe legacy/legacy_1d_simulator.py
```
Writes `altitude_comparison.png`, `velocity_comparison.png`, and `acceleration_comparison.png` into `legacy/`, and prints the metrics table above.
