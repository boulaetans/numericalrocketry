# numericalrocketry

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

A standalone Python 6DOF rocket flight simulator built to reproduce **OpenRocket's actual physics as closely as possible**, for one specific real rocket — **"Green Eggs"**, flying a real Estes C11-5 motor.

This isn't a general-purpose simulator: the geometry, mass table, and motor are hardcoded for this one design. The goal is a reproduction of OpenRocket's implementation. It's a personal/research project and learning exercise in understanding OpenRocket by rebuilding it.

## Table of contents

- [Background](#background)
- [Results](#results)
- [Getting started](#getting-started)
- [Usage](#usage)
- [Project structure](#project-structure)
- [License](#license)

## Background

This project started as a honors contract requirement for MAT-2 (Differential Equations), applying numerical methods for ODEs — Runge-Kutta integration in particular — to a real physical system with genuinely nonlinear, discontinuous forcing (thrust curves, aerodynamic stall, parachute deployment) rather than a textbook closed-form problem. Rigid-body flight dynamics has no closed-form solution, which is exactly the kind of case numerical ODE methods exist for. The rotational and translational equations of motion are integrated with a fixed-stage RK4 scheme (`k1`-`k4` per step) combined with adaptive step-size selection — the step size shrinks around ignition, burnout, and other fast-changing-force events, and grows during smoother coasting, rather than using a single fixed `dt` for the whole flight.

The full project abstract, written for that coursework, is in [`docs/abstract.pdf`](docs/abstract.pdf).

## Results

Three versions of the same flight, compared on a shared timeline: this project's own simulation, OpenRocket's own simulation of the identical rocket/motor/conditions, and the real flight-computer log from the actual launch.

![Animated 3D comparison of all three flights](assets/flight_comparison.gif)

| Event | numericalrocketry (sim) | OpenRocket (reference sim) | Real flight (logged) |
| --- | --- | --- | --- |
| Ignition | t = 0.03 s | t = 0 s | t ≈ -0.66 s *(est.)* |
| Burnout | t = 0.81 s | t = 0.81 s | t ≈ 0.57 s *(est.)* |
| Apogee | t = 4.77 s, 101.25 m AGL | t = 4.77 s, 101.29 m AGL | t = 4.76 s, 111.2 m AGL |
| Recovery deploy | t = 5.82 s | t = 5.81 s | not detectable in this log |
| Touchdown | t = 27.12 s | t = 27.19 s | t ≈ 41.06 s *(est.)* |

*(est.)* values for the real flight aren't from the flight computer's own reported state — that log's `boost`/`coast`/`main`/`landed` labels turned out not to match the real physical events for this flight (checked directly against the raw sensor samples: e.g. `boost` persists ~1.4s past the motor's actual burn time, and `landed` doesn't trigger until 53.69s even though speed/altitude both settle by ~41s). Instead each `(est.)` value is derived straight from the raw acceleration/speed/altitude data: ignition from a single linear step back from the earliest recorded sample at its own recorded velocity, burnout from an acceleration zero-crossing, touchdown from where speed sustainedly settles near zero. Recovery deploy is left blank rather than guessed — there's no detectable signature for it anywhere in this particular log. See `load_real_flight_track()` in [`animate_flight_comparison.py`](animate_flight_comparison.py) for the exact method.

The most interesting real-world finding: **the real flight's descent took almost twice as long as either simulation** (~41 s vs. ~27 s), even though apogee time and altitude are all close across the three — worth digging into if descent/parachute-drag modeling ever becomes a focus.

### Simulation accuracy vs. OpenRocket

| Metric | numericalrocketry | OpenRocket | Gap |
| --- | --- | --- | --- |
| Apogee altitude | 101.25 m | 101.29 m | -0.04 m (0.04%) |
| Apogee time | 4.7663 s | 4.766 s | ~1 ms |
| Rail clear | 0.256 s | 0.256 s | exact |
| Burnout | 0.81 s | 0.81 s | exact |
| Recovery deploy | 5.8163 s | 5.811 s | ~5 ms |
| Touchdown | 27.116 s | 27.194 s | -0.08 s (0.29%) |
| Liftoff mass | 166.041 g | 166 g | +0.04 g |
| CG location | 35.52 cm | 35.7 cm | -0.18 cm |

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
pip install -e .
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

| Flag | Default | Meaning |
| --- | --- | --- |
| `--output` | `assets/flight_comparison.gif` | Output file path |
| `--fps` | `20` | Frames per second |
| `--playback-seconds` | `16.0` | How long the main sweep takes to play |
| `--hold-seconds` | `5.0` | Pause after the last touchdown before the GIF loops |

## Project structure

```text
numericalrocketry/
├── src/numericalrocketry/       # the simulator package
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
│       └── estes_c11.eng             # RASP motor data (thrust curve, mass)
├── data/                         # reference/input data: the .ork design file + both reference flight logs
├── assets/                       # the comparison GIF
├── docs/                         # the project abstract
├── run_simulation.py             # entry point: runs the 6DOF simulation
├── animate_flight_comparison.py  # entry point: generates the 3-way comparison GIF
└── requirements.txt
```

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
