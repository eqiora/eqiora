# Exporting results and displaying quantities

The same native owners serve Rust and Python. Bulk visualization uses XDMF/HDF5;
selected typed observations use CSV. Both are projections of accepted results.
Keep the original Model, Plan and Result artifacts when you need to reopen or
continue the computation: neither export format is an Eqiora restart artifact.

## Display and copy a quantity

This complete example defines and solves a pressure relation. No example module
or prepared result is imported:

```python
import eqiora as q

model = q.compile(source="""model Stress() {
    variable stress:Pa;
    relation prescribed {stress=210000000000[Pa];}
    observable value:Pa=stress;
}""")
plan = q.resolve(model, solve=q.solve.Linear(
    algorithm=q.solve.LinearSolver.SparseLu,
    preconditioner=q.solve.Preconditioner.Identity,
    reduction=q.solve.Reduction.Fast,
    provider=q.solve.SolverProvider.faer(),
    relative_tolerance=1e-12, absolute_tolerance=1e-14,
    maximum_iterations=10,
))
result = q.run(plan, state=q.State.initial(plan))
observation = result.observe(model.observable("value"))
before = result.to_bytes()
gigapascal = q.units.Pa.prefixed("G")
presented = gigapascal.present(
    observation.value, observation.value_type.dimension, decimal_places=3,
)
assert presented.stored_value == 210000000000.
assert presented.value == 210.
assert presented.text == "210.000 GPa"
assert result.to_bytes() == before

# Use this same unit-bearing text for copy operations or accessibility labels.
print(presented.text)
q.data.export_observations_csv(
    "pressure.csv", [observation], unit=gigapascal, decimal_places=3,
)
```

`PresentedQuantity` retains the original coherent-SI value and Dimension, the
converted value and selected unit, formatted text, decimal places, locale and
rounding policy. Conversion treats the stored binary64 value exactly and rounds
once after applying the unit catalog's exact multiplicative scale. Text uses
nearest-even rounding with 0–17 fixed decimal places and explicit locale `"C"`.
An incompatible unit, unavailable locale, nonfinite value, overflow or nonzero
underflow rejects. These presentation settings never alter accepted identities.
Affine temperature conversion is not provided. Frequency/angular-frequency,
peak/RMS and spectral normalization are not inferred from display units.

## Selected observations as CSV

`q.data.export_observations_csv(path, observations, unit=...)` preserves input
order and duplicate occurrences. Each row selects one flattened, row-major
component (`component=0` by default). Its `projection` is one of `"real"`,
`"imaginary"`, `"magnitude"`, `"squared_magnitude"` or `"phase"`; these delegate
to the same mathematical projections as `Observation.project_component`.
Choose a unit compatible with the projected quantity, including squared units
for squared magnitude. Phase requires `q.units.one` and explicitly records
principal radians. At zero amplitude, phase has status `undefined` and empty
numeric/display cells; it is never reported as a measured zero angle.

Rows retain the selected original coherent real and imaginary components, exact
value type (shape, dimensions, frame and finite bases), projected SI value,
display value/unit/text, precision and rounding policy. They also retain Result
and Observable identities, effective quadrature, point support/coordinates and,
for retained temporal observations, output index, time in seconds and State
identity. Timeless observations have empty time and State cells. JSON cells
carry structured type, point and quadrature metadata. Standard CSV quoting
preserves commas, quotes and newlines; numeric cells use round-trip decimal text.

The profile admits primal observations, including point samples and retained
outputs. Directional derivative observations reject because this export does
not retain their complete tangent provenance. Other components, full mesh,
equations and numerical history are not included. The limits are 65,536 rows
and 16 MiB for the complete UTF-8 file, including escaped metadata. Validation
and staging finish before an existing destination is replaced.

## Spatial fields as XDMF/HDF5

After running a spatial Plan, call:

```python
result.export_xdmf("temperature.xdmf")
# For a completed transient result, the same retained frames are available via:
result.trajectory.export_xdmf("temperature-history.xdmf")
```

The profile admits complete real scalar or spatial-Cartesian vector Fields on
affine Tri3/Tet4 meshes, with P1 nodal values or constant cell values. Mesh entity
indices determine array association; coefficient ordering is not presumed to be
vertex ordering. Static Results have no fabricated time. Completed trajectories
export all requested retained outputs, with each accepted State's physical
coordinates and model time. Prescribed moving coordinates come from the Plan's
assembly map. Prescribed motion uses the admitted planar affine storage-map
profile; this adds no 3D moving-mesh execution. Internal unretained solver steps
are not exported.

The XDMF metadata names the geometry, topology and field datasets in one HDF5
companion. Values use coherent SI; `Information` entries retain dimensions,
coordinate frames, field association, units and Model/Plan/Result/Trajectory/
State lineage where applicable. XDMF does not carry Eqiora equations, complete
finite-element spaces or restart history. Complex coefficients, MINI bubbles,
higher-order or moment coefficients, partial supports and other cell families
reject rather than being relabeled as nodal values.

Python writes a content-addressed `eqiora-<digest>.h5` companion beside the
`.xdmf` file, then atomically publishes the metadata. Existing companions must
have identical bytes; collisions and symlinks reject. A failed publication
preserves the previous complete pair. A metadata-write failure may leave an
unreferenced immutable companion. Export does not delete companions that another
metadata file could use. Both metadata and bulk writes obey native byte limits;
this is a bounded serial export, not parallel production-scale I/O.

Rust callers use `export_common_result_xdmf_hdf5` or
`export_common_trajectory_xdmf_hdf5` with `XdmfHdf5ExportLimits`. These return
validated metadata and an HDF5 file image without opening a locator. Rust CSV
callers use `CommonObservation::export_csv`; presentation uses
`UnitCatalog::display_quantity` and `UnitCatalog::present_quantity`.
