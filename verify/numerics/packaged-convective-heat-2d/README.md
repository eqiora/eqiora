# Imported convective heat boundaries

This workflow imports `SteadyHeat2d`, `FixedTemperature2d`,
`OutwardHeatFlux2d` and `ConvectiveHeatTransfer2d` from the ordinary bundled
[Scalar package](../../../crates/eqiora-api/packages/Eqiora.Scalar/README.md).
The caller owns the smooth Kelvin temperature Field, the rectangle and its
four faces. Conductivity is shared with the volume heat law. The left face has
fixed temperature; the right face convects to a prescribed ambient temperature;
the horizontal faces are insulated. There is no volume heating.

The four fixed inputs come from the canonical
[Robin reference](https://github.com/eqiora/eqiora/blob/c7add4f8be40f3dcd5333542c37bba99cdf743f1/verify/numerics/cartesian-robin-q1/reference/README.md): zero transfer,
two positive transfer coefficients, and a different rectangle and conductivity
with reversed temperature difference. Every input uses an actual 2 × 2 mesh on
its own rectangle. A source fixture with a hard-coded unit square cannot cover
these cases. The [native owner](../../../crates/eqiora/tests/standard_thermal_package/robin.rs)
checks ordinary bundled compilation and semantic unit, regularity, required-input
and exact-parent diagnostics. The maintained
[heated-body example](../../../examples/heated-body/README.md) preserves imported
and hand-written forms and offline package compilation.

## Independent evidence and composition

The canonical reference owns exact rational affine temperatures and numerical
bounds. This case does not duplicate its oracle or set tolerances from Eqiora
output. The planned result-only check compares all nine physical coefficients
against its predeclared E, exact essential values and reference free-row residual.
Coordinates, units, finite binary64 values and unique complete node coverage are
checked as well. Authored boundary integrals expose four physical outward heat
flows, the right-face temperature integral and the right-face convective heat
flow through general public Result observations with explicit facet quadrature.
The canonical reference supplies their predeclared Q1 functional bounds.

This higher-layer test follows imported compilation, resolution, ordinary public
execution, and Plan/Result save and reopening. Its result-only comparison does
not audit a same-run full matrix or load, or observe private nodal weak reactions.
Full operator, load and direct weak-reaction recovery checks remain independently
mandatory in the lower-layer [Cartesian Robin case](https://github.com/eqiora/eqiora/blob/a591b9f294310da859bebd75e107160d3990153e/verify/numerics/cartesian-robin-q1/README.md).
The two cases compose: the lower owner checks assembly and private reaction
recovery; this owner checks that ordinary imports, typed bindings, actual geometry,
public heat-flow observations and artifact replay reach that path. No fabricated
matrix or load is supplied to the full comparator. Physical outward heat and
private weak reactions have opposite sign; the package does not conflate them.

The result-only native and installed Python numerical checks are implemented
and awaiting execution after core and reference integration and source/build review. The observation path is the existing general
`CommonResult::observe` / Python `result.observe` API, without a new public API
or a private capture seam. Source checks alone do not establish numerical
correctness. The intended numerical scope is steady affine Cartesian 2D Q1 on
the serial CPU reference provider. TPFA rejection is checked by the core owner.
Transient convection, threaded CPU, MPI and GPU execution are not verified here.

## Retained producer artifacts

The native producer uses `EQIORA_ROBIN_PACKAGE_EVIDENCE` when supplied, otherwise
creates a fresh directory below the user's home cache. The installed Python
producer accepts `EQIORA_ROBIN_PACKAGE_PYTHON_EVIDENCE`; its test scratch must be
home-backed. Each case retains authored source, the exact Model, Mesh, Plan and
Result, the strict oracle input and report, and a producer receipt. The package
closure is saved as a native compilation record or a moved Python offline store.
The receipt binds actual Field and Observable identities, boundary Domain IDs,
accepted Result identities, units, facet quadrature and artifact hashes.

Both producers canonically reopen Model, Plan and Result bytes, verify all nine
replayed coefficients equal the exported coefficients, then evaluate all six
primary and replay observations through the general public observation owner.
The oracle checks each evaluation independently against its fixed coefficient-
functional and analytic-target bounds. Matching labels or replay values alone
are not treated as evidence of their origin. The native test and
[`test_convective_heat_observations.py`](../../../bindings/python/tests/test_convective_heat_observations.py)
do not access private operators, loads or nodal weak reactions.
