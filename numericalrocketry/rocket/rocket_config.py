"""Fixed-rocket configuration for the Green Egg simulation.

Values below are extracted directly from the real "Green Eggs OR.ork" design
file (nose/body/fin/mass/motor/recovery), not placeholders. See the dry-mass
and recovery notes for how each was derived.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .geometry import RocketGeometry, fin_set_mass_cg_m, nose_shell_mass_cg_m
from .mass_model import ComponentMass, ComponentMassModel
from .propulsion import MotorModel, load_motor_from_eng_file
from .recovery import RecoveryModel


# Real dry components (name, mass_kg, x_start_m, length_m, radius_m, wall_thickness_m)
# from the .ork file. Axial positions were independently re-derived from the
# raw .ork XML's stored offsets using the file format's TOP/MIDDLE/BOTTOM
# position formulas and confirmed to match this table to the last digit.
# Wall thicknesses are the file's real per-component <thickness> values (or,
# for the two auto-sized centering rings, the real auto-fit inner/outer
# radii), not estimates.
#
# wall_thickness_m (6th field): 0.0 means "treat as a solid cylinder" --
# correct for point-mass-like components (payload/avionics/parachute/shock
# cord). A nonzero value means "treat as a hollow cylindrical tube of this
# wall thickness" -- for genuinely tube-shaped structural parts. The nose
# cone and fin set are also structurally non-solid but need a different
# (shell / flat-plate) formula, not a simple hollow-annulus one, so they
# still use the solid-cylinder approximation here.
_DRY_COMPONENTS_RAW = (
    ("Nose Cone", 0.021829133, 0.000000000, 0.136525000, 0.022809200, 0.0),
    ("Body Tube", 0.011339809, 0.136525000, 0.127000000, 0.022809200, 5.842e-4),
    ("Tube Coupler", 0.017009714, 0.238125000, 0.063500000, 0.022809200, 0.0013716),
    ("Apogee egg protector", 0.001000000, 0.170025000, 0.060000000, 0.020320000, 0.0),
    ("EasyMini 2", 0.006500000, 0.181025000, 0.038000000, 0.010000000, 0.0),
    ("Battery", 0.009000000, 0.181575000, 0.036900000, 0.013250000, 0.0),
    ("Body Tube", 0.000913656, 0.263525000, 0.003175000, 0.022809200, 0.002),
    ("Body Tube", 0.028916514, 0.266700000, 0.304800000, 0.022809200, 5.842e-4),
    ("Freeform Fin Set", 0.012473790, 0.471170000, 0.100330000, 0.022809200, 0.0),
    ("Inner Tube", 0.001585204, 0.501650000, 0.069850000, 0.012395200, 3.302e-4),
    ("engine hook", 0.001133981, 0.509270000, 0.072390000, 0.001524000, 0.0),
    # Motor-mount centering ring: explicit radii in the .ork file, not auto-fit.
    ("Centering Ring", 0.000748468, 0.501650000, 0.006350000, 0.012039600, 0.002667000),
    ("Inner Tube", 0.000232554, 0.535940000, 0.012700000, 0.013144500, 3.302e-4),
    ("Qualman Baffle", 0.001000000, 0.406400000, 0.063500000, 0.021590000, 0.0),
    # Canopy (pi*(0.4572/2)^2*0.04 Nylon, 0.006566929 kg) PLUS shroud-line mass:
    # line_count*line_length*line_material_density (6 lines * 0.4572 m *
    # 3.3e-4 kg/m Carpet Thread = 0.000905256 kg), which this table
    # previously omitted entirely.
    ("Parachute", 0.007472185, 0.309880000, 0.038100000, 0.022225000, 0.0),
    ("Shock Cord", 0.007937866, 0.353060000, 0.025000000, 0.021590000, 0.0),
    # Two "Centering Ring - 50-65" rings, auto-fit outer radius = main body
    # tube's inner radius, auto-fit inner radius = motor-mount tube's outer
    # radius (the design file only finds an overlapping inner-tube sibling
    # for the motor-mount tube at these two axial positions, not the aft
    # inner tube).
    ("Centering Ring - 50-65", 0.000738630, 0.517144000, 0.001016000, 0.022225000, 0.009829800),
    ("Centering Ring - 50-65", 0.000738630, 0.562864000, 0.001016000, 0.022225000, 0.009829800),
    ("Launch Lug", 0.000171034, 0.368300000, 0.050800000, 0.002882900, 2.159e-4),
)

# Motor mount inner tube: x_start=0.501650, length=0.06985. The motor itself
# (0.07 m long) is NOT centered in the mount tube: the .ork file's
# <motormount><overhang>0.00635</overhang> means the motor hangs 6.35mm past
# the tube's aft end (motor_x_position = mount.length - motor.length +
# overhang, relative to the mount's own start). RASP/.eng motor CG is always
# length/2 (this motor's digest confirms RASP loading, so there's no real
# internal CG offset being missed here) -- so
# motor x_cg = mount_x_start + motor_x_position + 0.5*motor_length.
_MOTOR_MOUNT_OVERHANG_M = 0.00635
_MOTOR_MOUNT_X_CG_M = 0.501650 + (0.069850 - 0.07 + _MOTOR_MOUNT_OVERHANG_M) + 0.5 * 0.07
_MOTOR_RADIUS_M = 0.012  # C11 diameter 0.024 m / 2
_MOTOR_LENGTH_M = 0.07

# Nose cone shell + rear-shoulder geometry, from the .ork file's <nosecone>
# element. Used only by nose_shell_mass_cg_m() below for the real mass-CG
# integration -- geometry.py's nose_profile_integration() deliberately treats
# the nose as a solid outer-profile revolution, which is correct for
# wetted-area/CP/planform but not for mass CG of the actual hollow shell
# (+ shoulder, which extends past the nose's own length into the body tube).
_NOSE_CONE_WALL_THICKNESS_M = 0.0010414
_NOSE_CONE_SHOULDER_LENGTH_M = 0.022224999999999998
_NOSE_CONE_SHOULDER_RADIUS_M = 0.021970999999999997
_NOSE_CONE_SHOULDER_THICKNESS_M = 0.001016
_NOSE_CONE_SHOULDER_CAPPED = True

# Freeform fin tab + fillet geometry/material, from the .ork file's
# <freeformfinset> element. Used only by fin_set_mass_cg_m() below for the
# real per-fin mass-CG; the aerodynamic fin calculations only need the
# exposed planform polygon (RocketGeometry.fin_points_m).
_FIN_THICKNESS_M = 0.0024638
_FIN_CROSS_SECTION_RELATIVE_VOLUME = 0.99  # ROUNDED cross section
_FIN_BULK_DENSITY_KG_M3 = 216.249255  # "Firm Balsa" (Custom material group)
_FIN_TAB_LENGTH_M = 0.043688
_FIN_TAB_HEIGHT_M = 0.009524999999999999
# <tabposition relativeto="absolute">0.02032</tabposition> is the second (and
# thus authoritative -- each <tabposition> tag is applied in file order, last
# one wins) of the two <tabposition> tags in the file; an "absolute" position
# is already the tab's front-edge offset from the fin's own leading edge, no
# further conversion needed.
_FIN_TAB_FRONT_OFFSET_M = 0.02032
_FIN_FILLET_RADIUS_M = 0.00127
_FIN_FILLET_DENSITY_KG_M3 = 170.0  # "Balsa" (Woods material group)


@dataclass(frozen=True)
class RocketConfig:
    geometry: RocketGeometry
    motor_path: str | Path
    dry_mass_kg: float = 0.1307
    motor_total_mass_kg: float | None = None
    motor_delay_s: float = 5.0
    parachute_cd: float = 0.8
    parachute_area_m2: float = 0.16419
    deploy_altitude_m: float = 0.0
    deploy_delay_s: float = 0.0
    launch_rail_length_m: float = 0.9144
    rail_angle_deg: float = 0.0
    launch_altitude_m: float = 512.0
    # From the .ork file's saved simulation conditions (<launchlatitude>),
    # rounded to ~1km precision (WGS84 gravity only needs ~0.01 deg precision,
    # so this doesn't affect the physics); feeds the WGS84 gravity model (gravity.py).
    launch_latitude_deg: float = 33.98
    initial_velocity_m_s: float = 0.0
    surface_finish: str = "normal"

    @classmethod
    def default_green_egg(cls) -> "RocketConfig":
        """Build the real Green Eggs configuration, values taken from the .ork file."""
        # Freeform fin polygon straight from the .ork file (local fin frame:
        # x = axial from root leading edge, y = spanwise from body surface).
        fin_points_m = (
            (0.0, 0.0),
            (0.0762, 0.06858),
            (0.10033, 0.06858),
            (0.10033, 0.04826),
            (0.07112, 0.0),
        )
        geometry = RocketGeometry(
            length_m=0.5715,
            diameter_m=0.0456184,
            nose_length_m=0.136525,
            fin_count=3,
            fin_thickness_m=0.0024638,
            fin_area_m2=0.0,  # derived exactly from fin_points_m; see geometry.fin_planform_area_m2
            fin_span_m=0.06858,
            fin_root_chord_m=0.07112,
            fin_tip_chord_m=0.02413,
            fin_sweep_length_m=0.0762,
            fin_leading_edge_x_m=0.47117,
            nose_type="elliptical",
            fin_cross_section="rounded",
            fin_points_m=fin_points_m,
        )
        return cls(
            geometry=geometry,
            motor_path=Path(__file__).parent.parent / "motors" / "Estes_C11.eng",
            # Dry mass = the reference simulation's recorded liftoff mass
            # (0.166 kg) minus the real C11 total motor mass (0.0353 kg);
            # includes the "EasyMini 2" flight computer + battery + egg
            # protector payload.
            dry_mass_kg=0.1307,
            motor_total_mass_kg=0.0353,
            # C11-5: 5 second ejection delay after burnout (from the .ork motor spec).
            motor_delay_s=5.0,
            # Auto-computed Cd for parachutes is 0.8; diameter 18 in (0.4572 m).
            parachute_cd=0.8,
            parachute_area_m2=0.16419,
            deploy_altitude_m=0.0,
            deploy_delay_s=0.0,
            launch_rail_length_m=0.9144,
            rail_angle_deg=0.0,
            launch_altitude_m=512.0,
            launch_latitude_deg=33.98,
            initial_velocity_m_s=0.0,
            surface_finish="normal",
        )

    def build_motor(self) -> MotorModel:
        motor = load_motor_from_eng_file(self.motor_path)
        if motor is None:
            raise FileNotFoundError(f"Could not load motor curve from {self.motor_path}")
        return motor

    def build_recovery(self) -> RecoveryModel:
        """Build the parachute deployment config for the 6DOF integrator."""
        return RecoveryModel(
            # Real design deploys via the motor's own ejection charge, not an
            # altitude or fixed-apogee-delay trigger.
            deploy_event="ejection",
            deploy_delay_s=self.deploy_delay_s,
            deploy_altitude_m=self.deploy_altitude_m,
            chute_cd=self.parachute_cd,
            chute_area_m2=self.parachute_area_m2,
            motor_ejection_delay_s=self.motor_delay_s,
        )

    def build_component_mass_model(self, motor: MotorModel | None = None) -> ComponentMassModel:
        """Build the real per-component mass model (accurate CG + inertia for 6-DOF)."""
        motor = motor or self.build_motor()

        # x_start + 0.5*length (used below as the default x_cg for every
        # other component) is only exact for uniform-cross-section
        # (cylindrical) parts. The nose cone (tapered shell + rear shoulder)
        # and the freeform fin set (non-rectangular planform + tab + fillet)
        # need a real shape-integrated mass-CG instead -- see
        # nose_shell_mass_cg_m()/fin_set_mass_cg_m() in geometry.py.
        nose_local_x_cg_m = nose_shell_mass_cg_m(
            self.geometry,
            wall_thickness_m=_NOSE_CONE_WALL_THICKNESS_M,
            shoulder_length_m=_NOSE_CONE_SHOULDER_LENGTH_M,
            shoulder_radius_m=_NOSE_CONE_SHOULDER_RADIUS_M,
            shoulder_thickness_m=_NOSE_CONE_SHOULDER_THICKNESS_M,
            shoulder_capped=_NOSE_CONE_SHOULDER_CAPPED,
        )
        fin_local_x_cg_m = fin_set_mass_cg_m(
            self.geometry,
            thickness_m=_FIN_THICKNESS_M,
            cross_section_relative_volume=_FIN_CROSS_SECTION_RELATIVE_VOLUME,
            bulk_density_kg_m3=_FIN_BULK_DENSITY_KG_M3,
            tab_length_m=_FIN_TAB_LENGTH_M,
            tab_height_m=_FIN_TAB_HEIGHT_M,
            tab_front_offset_m=_FIN_TAB_FRONT_OFFSET_M,
            fillet_radius_m=_FIN_FILLET_RADIUS_M,
            fillet_density_kg_m3=_FIN_FILLET_DENSITY_KG_M3,
            body_radius_m=self.geometry.diameter_m / 2.0,
        )
        # Nose Cone's x_start is 0 (nose tip), so its local and rocket-frame
        # x_cg coincide; the fin set's local x_cg is measured from the fin's
        # own leading edge (RocketGeometry.fin_leading_edge_x_m).
        x_cg_overrides_m = {
            "Nose Cone": nose_local_x_cg_m,
            "Freeform Fin Set": self.geometry.fin_leading_edge_x_m + fin_local_x_cg_m,
        }

        components = [
            ComponentMass(
                name=name, mass_kg=mass,
                x_cg_m=x_cg_overrides_m.get(name, x_start + 0.5 * length),
                length_m=length, radius_m=radius,
                wall_thickness_m=wall_thickness,
            )
            for name, mass, x_start, length, radius, wall_thickness in _DRY_COMPONENTS_RAW
        ]

        motor_dry_mass_kg = max((motor.total_mass_kg or 0.0) - motor.propellant_mass_kg, 0.0)
        if motor_dry_mass_kg > 0.0:
            components.append(
                ComponentMass(
                    name="Motor dry mass",
                    mass_kg=motor_dry_mass_kg,
                    x_cg_m=_MOTOR_MOUNT_X_CG_M,
                    length_m=_MOTOR_LENGTH_M,
                    radius_m=_MOTOR_RADIUS_M,
                )
            )

        return ComponentMassModel(
            dry_components=components,
            initial_propellant_mass_kg=motor.propellant_mass_kg,
            propellant_x_cg_m=_MOTOR_MOUNT_X_CG_M,
            propellant_length_m=_MOTOR_LENGTH_M,
            propellant_radius_m=_MOTOR_RADIUS_M,
        )


DEFAULT_GREEN_EGG_CONFIG = RocketConfig.default_green_egg()


def green_egg_config() -> RocketConfig:
    """Return the fixed-rocket configuration for the real Green Egg design."""
    return DEFAULT_GREEN_EGG_CONFIG
