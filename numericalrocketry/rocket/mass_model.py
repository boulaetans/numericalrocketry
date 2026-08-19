"""Mass-property models for dry structure and burning propellant."""

from dataclasses import dataclass

import numpy as np

from ..simulation.state import MassProperties


@dataclass(frozen=True)
class ComponentMass:
    name: str
    mass_kg: float
    x_cg_m: float
    length_m: float
    radius_m: float
    # 0.0 (default) = treat as a solid cylinder -- correct for point-mass-like
    # components (parachute, shock cord, avionics, motor casing+propellant).
    # >0.0 = treat as a hollow cylindrical tube of that wall thickness --
    # appropriate for genuine tube-shaped structural parts (body tubes,
    # couplers, inner tubes, launch lugs), where treating the tube as a solid
    # cylinder underestimates roll inertia by up to ~2x since real tube mass
    # sits at the rim rather than spread through the cross-section.
    wall_thickness_m: float = 0.0


@dataclass(frozen=True)
class ComponentMassModel:
    dry_components: list[ComponentMass]
    initial_propellant_mass_kg: float
    propellant_x_cg_m: float
    propellant_length_m: float
    propellant_radius_m: float

    @property
    def dry_mass_kg(self) -> float:
        return float(sum(component.mass_kg for component in self.dry_components))

    def _component_own_inertia(
        self, mass_kg: float, length_m: float, radius_m: float, wall_thickness_m: float
    ) -> tuple[float, float, float]:
        if mass_kg <= 0.0:
            return 0.0, 0.0, 0.0
        length = max(length_m, 0.0)
        radius_outer = max(radius_m, 0.0)

        if 0.0 < wall_thickness_m < radius_outer:
            # Hollow cylindrical tube: Ixx=(r_o^2+r_i^2)/2, transverse
            # I=(3(r_o^2+r_i^2)+L^2)/12, rather than the solid-cylinder
            # formula below.
            radius_inner = radius_outer - wall_thickness_m
            i_xx = 0.5 * mass_kg * (radius_outer**2 + radius_inner**2)
            i_transverse = (1.0 / 12.0) * mass_kg * (3.0 * (radius_outer**2 + radius_inner**2) + length**2)
            return i_xx, i_transverse, i_transverse

        i_xx = 0.5 * mass_kg * radius_outer**2
        i_transverse = (1.0 / 12.0) * mass_kg * (3.0 * radius_outer**2 + length**2)
        return i_xx, i_transverse, i_transverse

    def from_propellant_mass(self, propellant_mass_kg: float) -> MassProperties:
        # Clamp propellant mass to physically meaningful bounds.
        propellant = max(0.0, min(self.initial_propellant_mass_kg, propellant_mass_kg))

        masses: list[tuple[float, float, float, float, float]] = []
        for component in self.dry_components:
            if component.mass_kg <= 0.0:
                continue
            masses.append((
                component.mass_kg, component.x_cg_m, component.length_m,
                component.radius_m, component.wall_thickness_m,
            ))

        if propellant > 0.0:
            masses.append((propellant, self.propellant_x_cg_m, self.propellant_length_m, self.propellant_radius_m, 0.0))

        total_mass = sum(m for m, _, _, _, _ in masses)
        if total_mass <= 0.0:
            return MassProperties(
                mass_kg=0.0,
                center_of_gravity_m=np.zeros(3, dtype=float),
                inertia_body_kg_m2=np.eye(3, dtype=float) * 1e-9,
            )

        x_cg = sum(m * x for m, x, _, _, _ in masses) / total_mass

        i_xx = 0.0
        i_yy = 0.0
        i_zz = 0.0
        for mass, x, length, radius, wall_thickness in masses:
            own_i_xx, own_i_yy, own_i_zz = self._component_own_inertia(mass, length, radius, wall_thickness)
            dx = x - x_cg
            # Parallel-axis shift only along body x-axis.
            i_xx += own_i_xx
            i_yy += own_i_yy + mass * dx**2
            i_zz += own_i_zz + mass * dx**2

        return MassProperties(
            mass_kg=float(total_mass),
            center_of_gravity_m=np.array([x_cg, 0.0, 0.0], dtype=float),
            inertia_body_kg_m2=np.diag([i_xx, i_yy, i_zz]),
        )