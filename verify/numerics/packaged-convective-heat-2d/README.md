# Imported convective heat boundaries

This workflow imports `SteadyHeat2d`, `FixedTemperature2d`,
`OutwardHeatFlux2d` and `ConvectiveHeatTransfer2d` from the ordinary bundled
[Scalar package](../../../crates/eqiora-api/packages/Eqiora.Scalar/README.md).
The caller owns the smooth Kelvin temperature Field, the rectangle and its
four faces. Conductivity is shared with the volume heat law. The left face has
fixed temperature; the right face convects to a prescribed ambient temperature;
the horizontal faces are insulated. There is no volume heating.

The four fixed inputs come from the canonical
[Robin reference](../cartesian-robin-q1/reference/README.md): zero transfer,
two positive transfer coefficients, and a different rectangle and conductivity
with reversed temperature difference. Every input uses an actual 2 × 2 mesh on
its own rectangle. A source fixture with a hard-coded unit square cannot cover
these cases. The [native owner](../../../crates/eqiora/tests/standard_thermal_package/robin.rs)
checks ordinary bundled compilation and semantic unit, regularity, required-input
and exact-parent diagnostics. The maintained
[heated-body example](../../../examples/heated-body/README.md) preserves imported
and hand-written forms and offline package compilation.

## Independent evidence and composition

The canonical reference owns exact rational affine temperatures, three left-face
weak reactions and all numerical bounds. This case does not duplicate its oracle
or set tolerances from Eqiora output. The planned result-only check compares all
nine physical coefficients against its predeclared E, exact essential values,
reference free-row residual, three weak reactions against the reference residual
vector and rho, individual analytic reactions and their sum. Coordinates, units,
finite binary64 values and unique complete node coverage are checked as well.
Physical outward heat equals the negative weak reaction.

This higher-layer test follows imported compilation, resolution, ordinary public
execution, and Plan/Result save and reopening. Its result-only comparison does
not audit a same-run full matrix or load, or claim a direct reaction-to-exported-
physical-residual recovery gate. Those full operator, load and direct recovery
checks remain independently mandatory in the lower-layer
[Cartesian Robin case](../cartesian-robin-q1/README.md). The two cases compose:
the lower owner checks assembly and reaction recovery; this owner checks that
ordinary imports, typed bindings, geometry and artifact replay reach that path.
No fabricated matrix or load is supplied to the full comparator.

The result-only native and installed Python numerical checks are pending core
and reference integration and a review of the public boundary-observation API.
Scalar nodal weak reactions are currently private; this case will not expose
them solely for tests or claim their observation before a product path exists. Source checks alone do not establish numerical
correctness. The intended numerical scope is steady affine Cartesian 2D Q1 on
the serial CPU reference provider. TPFA rejection is checked by the core owner.
Transient convection, threaded CPU, MPI and GPU execution are not verified here.
