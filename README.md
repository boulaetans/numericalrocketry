# NumericalRocketry

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

A standalone Python 6DOF rocket flight simulator built to reproduce OpenRocket’s simulation results for one specific real rocket, “Green Eggs”, flying a real Estes C11-5 motor.

This isn't a general-purpose simulator: the geometry, mass table, and motor are hardcoded for this one design. The goal is an independent Python reimplementation of the rocket-flight physics used by OpenRocket, developed by studying its open-source implementation and validating the results against OpenRocket. It's a personal/research project and learning exercise.

## Table of contents

* [Background](#background)
* [Results](#results)
* [Getting started](#getting-started)
* [Usage](#usage)
* [Project structure](#project-structure)
* [License](#license)

## Background

This project started as an honors contract requirement for MAT-2 (Differential Equations). Since the dawn of the space age, numerical methods have played a defining role in rocket development by enabling engineers to model trajectories long before physical testing. Many governing equations in flight dynamics simply lack closed-form solutions, which is exactly the kind of case numerical methods exist for. This project applies those same numerical techniques to small-scale rocket flight. It extends a previously verified 1D ascent simulator, preserved and adapted with this rocket's real geometry in [`legacy/`](legacy/), into a full 3D Python-based flight simulation with explicit geometry, event-aware dynamics, and atmospheric modeling.

The rotational and translational equations of motion are integrated with a fixed-stage RK4 scheme (`k1`-`k4` per step) combined with adaptive step-size control. The step size shrinks around ignition, burnout, and other fast-changing-force events, then grows during smoother coasting instead of using a single fixed `dt` for the whole flight.

**How much did the full 3D rewrite actually buy over the original 1D model?** Run head-to-head on this same rocket and motor, the 1D model's own RK4 branch gets apogee within 2.00% of OpenRocket's reference simulation (99.27 m vs. 101.29 m), a reasonable result for a model with no attitude dynamics, no exact Barrowman CP/stability treatment, and a simplified drag model. This project's full 6DOF result closes about 98% of that remaining gap, landing within 0.04% of OpenRocket, roughly a 50× reduction in apogee error. Unlike the 1D model, it produces a stability margin, off-axis motion, and a modeled recovery/descent phase at all. Full breakdown, plots, and how each integrator does individually in [`legacy/COMPARISON.md`](legacy/COMPARISON.md).

The full project abstract is in [`docs/abstract.pdf`](docs/abstract.pdf).

## Results

Three versions of the same flight, compared on a shared timeline: this project's own simulation, OpenRocket's own simulation of the identical rocket/motor/conditions, and the real flight-computer log from the actual launch.

![Animated 3D comparison of all three flights](assets/flight_comparison.gif)

Quick look, no animation required:

![Static altitude-vs-time comparison of all three flights](assets/flight_comparison.png)

|              | NumericalRocketry (sim)  | OpenRocket (reference sim) | Real Flight (logged)    |
| ------------ | ------------------------ | -------------------------- | ----------------------- |
| Apogee       | t = 4.77 s, 101.25 m AGL | t = 4.77 s, 101.29 m AGL   | t = 4.76 s, 111.2 m AGL |
| Max velocity | 45.5 m/s                 | 45.5 m/s                   | 35.4 m/s *(see note)*   |
| Touchdown    | t = 27.12 s              | t = 27.19 s                | t ≈ 41.06 s *(est.)*    |

The real flight's touchdown time is *(est.)* rather than read straight from the log. That log's `boost`/`coast`/`main`/`landed` labels turned out not to match the real physical events for this flight (checked directly against the raw sensor samples: e.g. `landed` doesn't trigger until 53.69s even though speed and altitude both settle by ~41s). The value shown is instead the first point after apogee where speed sustainedly settles near zero. See `load_real_flight_track()` in [`animate_flight_comparison.py`](animate_flight_comparison.py) for the exact method.

The real flight's max velocity is notably *lower* than either simulation despite reaching a *higher* apogee. That does not make much physical sense for a launch on the same motor, so a sensor filtering or lag artifact is the most likely explanation (the same log shows a demonstrated multi-second lag around touchdown), not a genuine performance difference. Shown as reported, not corrected for.

**Why the real flight's apogee reads ~9% higher than either simulation.** A single real flight diverging from simulation by a few percent, sometimes more, is a common issue in amateur rocketry validation, and not a red flag by itself. Individual motors vary batch to batch within their certification tolerance, an as-built rocket rarely masses exactly what its design file says, and launch-day atmospheric conditions are never quite the standard atmosphere both simulations assume. One real flight is one noisy sample, not a controlled repeat measurement.

More specifically here, though: this project's own simulator and OpenRocket, two independently built physics engines, agree with each other to within 0.04%. For a shared modeling error to explain the real-flight gap, both would have to be wrong by ~9% in the same direction. It is much more likely that the gap comes from having only one real flight and one set of sensor data. That's supported directly by this log's own data, not just inferred: its reported flight state (`boost`/`coast`/`main`/`landed`) doesn't match the real physical events at all (see above), and its `height` and `altitude` columns, two onboard estimates of the same quantity, disagree with each other at the same instant. A barometric altimeter is also specifically prone to *dynamic-pressure error*: a fast-moving rocket creates a local low-pressure region right at the sensor's port (a Bernoulli effect), which a cheap altimeter can misread as extra altitude, especially at higher speeds, and in the same direction as what's shown here. There is nothing to fix in the simulator based on this result. It is an interesting real-world finding about the limits of a low-cost flight computer, not a simulator bug.

The most interesting real-world finding: **the real flight's descent took almost twice as long as either simulation** (~41 s vs. ~27 s), even though apogee time and altitude are all close across the three. This is worth looking into if descent and parachute-drag modeling become a focus later.

### Simulation accuracy vs. OpenRocket

These are the apogee, peak velocity, and touchdown results compared directly with OpenRocket's simulation of the same rocket, motor, and conditions.

| Metric          | NumericalRocketry | OpenRocket | Gap                |
| --------------- | ----------------- | ---------- | ------------------ |
| Apogee altitude | 101.25 m          | 101.29 m   | -0.04 m (0.04%)    |
| Apogee time     | 4.7663 s          | 4.766 s    | ~1 ms              |
| Max velocity    | 45.53 m/s         | 45.53 m/s  | ~0.005 m/s (0.01%) |
| Touchdown       | 27.116 s          | 27.194 s   | -0.08 s (0.29%)    |

Every event, including internal validation metrics such as rail clear, recovery deploy, liftoff mass, and CG, is in [`data/RESULTS.md`](data/RESULTS.md).

## Getting started

### Prerequisites

Python 3.10+.

### Installation

```sh
git clone https://github.com/boulaetans/numericalrocketry.git
cd numericalrocketry
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

## Usage

Run the simulation:

```sh
python run_simulation.py
```

Prints the mission-event timeline and writes `simulation_results.csv`.

Regenerate the 3D comparison animation:

```sh
python animate_flight_comparison.py
```

Writes `assets/flight_comparison.gif` by default.

| Flag                 | Default                        | Meaning                                                                             |
| -------------------- | ------------------------------ | ----------------------------------------------------------------------------------- |
| `--output`           | `assets/flight_comparison.gif` | Output file path                                                                    |
| `--fps`              | `20`                           | Frames per second                                                                   |
| `--playback-seconds` | real time                      | How long the main sweep takes to play (defaults to actual flight duration, ~41.7 s) |
| `--hold-seconds`     | `5.0`                          | Pause after the last touchdown before the GIF loops                                 |

Regenerate the static comparison plot:

```sh
python plot_flight_comparison.py
```

Writes `assets/flight_comparison.png` by default (`--output` to change it).

## Project structure

```text
numericalrocketry/
├── numericalrocketry/           # the simulator package
│   ├── constants.py                # shared physical constants
│   ├── rocket/                     # the vehicle definition
│   │   ├── rocket_config.py          # Green Eggs' geometry/mass table, motor path, recovery params
│   │   ├── geometry.py                # RocketGeometry, nose profile integration, fin geometry, shape-integrated CG
│   │   ├── mass_model.py              # ComponentMassModel (dynamic CG/inertia)
│   │   ├── propulsion.py              # .eng motor file loading, thrust curve, mass depletion
│   │   └── recovery.py                # Parachute deployment config
│   ├── physics/                    # the physics models
│   │   ├── atmosphere.py              # ISA atmosphere model
│   │   ├── gravity.py                 # WGS84 gravity model
│   │   ├── drag.py                    # Drag model (skin friction, pressure, base, wave)
│   │   ├── aerodynamics.py            # Barrowman CN/CP, damping moments, the core aero physics
│   │   ├── dynamics.py                # Rigid-body force/moment -> state derivative
│   │   └── quaternion.py              # Quaternion math (exponential-map integration)
│   ├── simulation/                 # the run loop
│   │   ├── integrator.py              # RK4 time-marching loop, adaptive timestep
│   │   └── state.py                   # Simulation state dataclasses
│   └── motors/
│       └── Estes_C11.eng             # RASP motor data (thrust curve, mass)
├── data/                         # reference/input data (.ork design file, both flight logs) + RESULTS.md (full data table)
├── assets/                       # the comparison GIF and static plot
├── docs/                         # the project abstract
├── legacy/                       # the original 1D numerical-methods simulator this project grew from
│   ├── legacy_1d_simulator.py      # Euler / ABM-2 / RK4 comparison, adapted with Green Eggs' real geometry
│   └── COMPARISON.md               # how the 1D model stacks up against the rest of the project
├── run_simulation.py             # entry point: runs the 6DOF simulation
├── animate_flight_comparison.py  # entry point: generates the 3-way comparison GIF
├── plot_flight_comparison.py     # entry point: generates the static comparison plot
└── requirements.txt
```

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
