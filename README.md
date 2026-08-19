# NumericalRocketry

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

This project started as an honors contract requirement for MAT-2 (Differential Equations), applying numerical methods for ODEs — Runge-Kutta integration in particular — to a real physical system with genuinely nonlinear, discontinuous forcing (thrust curves, aerodynamic stall, parachute deployment) rather than a textbook closed-form problem. Rigid-body flight dynamics has no closed-form solution, which is exactly the kind of case numerical ODE methods exist for. The rotational and translational equations of motion are integrated with a fixed-stage RK4 scheme (`k1`-`k4` per step) combined with adaptive step-size selection — the step size shrinks around ignition, burnout, and other fast-changing-force events, and grows during smoother coasting, rather than using a single fixed `dt` for the whole flight.

The full project abstract, written for that coursework, is in [`docs/abstract.pdf`](docs/abstract.pdf).

## Results

Three versions of the same flight, compared on a shared timeline: this project's own simulation, OpenRocket's own simulation of the identical rocket/motor/conditions, and the real flight-computer log from the actual launch.

![Animated 3D comparison of all three flights](assets/flight_comparison.gif)

Quick look, no animation required:

![Static altitude-vs-time comparison of all three flights](assets/flight_comparison.png)

| | NumericalRocketry (sim) | OpenRocket (reference sim) | Real Flight (logged) |
| --- | --- | --- | --- |
| Apogee | t = 4.77 s, 101.25 m AGL | t = 4.77 s, 101.29 m AGL | t = 4.76 s, 111.2 m AGL |
| Max velocity | 45.5 m/s | 45.5 m/s | 35.4 m/s *(see note)* |
| Touchdown | t = 27.12 s | t = 27.19 s | t ≈ 41.06 s *(est.)* |

The real flight's touchdown time is *(est.)* rather than read straight from the log — that log's `boost`/`coast`/`main`/`landed` labels turned out not to match the real physical events for this flight (checked directly against the raw sensor samples: e.g. `landed` doesn't trigger until 53.69s even though speed and altitude both settle by ~41s). The value shown is instead the first point after apogee where speed sustainedly settles near zero. See `load_real_flight_track()` in [`animate_flight_comparison.py`](animate_flight_comparison.py) for the exact method.

The real flight's max velocity is notably *lower* than either simulation despite reaching a *higher* apogee, which doesn't add up physically for a launch on the same motor — most likely a sensor filtering/lag artifact (the same log shows a demonstrated multi-second lag around touchdown), not a genuine performance difference. Shown as reported, not corrected for.

The most interesting real-world finding: **the real flight's descent took almost twice as long as either simulation** (~41 s vs. ~27 s), even though apogee time and altitude are all close across the three — worth digging into if descent/parachute-drag modeling ever becomes a focus.

### Simulation accuracy vs. OpenRocket

The numbers behind the claim — apogee, peak velocity, and touchdown, compared directly against OpenRocket's own simulation of the identical rocket/motor/conditions.

| Metric | NumericalRocketry | OpenRocket | Gap |
| --- | --- | --- | --- |
| Apogee altitude | 101.25 m | 101.29 m | -0.04 m (0.04%) |
| Apogee time | 4.7663 s | 4.766 s | ~1 ms |
| Max velocity | 45.53 m/s | 45.53 m/s | ~0.005 m/s (0.01%) |
| Touchdown | 27.116 s | 27.194 s | -0.08 s (0.29%) |

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

| Flag | Default | Meaning |
| --- | --- | --- |
| `--output` | `assets/flight_comparison.gif` | Output file path |
| `--fps` | `20` | Frames per second |
| `--playback-seconds` | real time | How long the main sweep takes to play (defaults to actual flight duration, ~41.7 s) |
| `--hold-seconds` | `5.0` | Pause after the last touchdown before the GIF loops |

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
├── data/                         # reference/input data: the .ork design file + both reference flight logs
├── assets/                       # the comparison GIF and static plot
├── docs/                         # the project abstract
├── run_simulation.py             # entry point: runs the 6DOF simulation
├── animate_flight_comparison.py  # entry point: generates the 3-way comparison GIF
├── plot_flight_comparison.py     # entry point: generates the static comparison plot
└── requirements.txt
```

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
