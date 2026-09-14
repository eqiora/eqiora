use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyModule};

#[test]
fn python_transient_solver_inventory_executes_nonzero_flow_and_replays() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals.set_item("eqiora", public_module(py)?)?;
        locals.set_item(
            "mesh_bytes",
            pyo3::types::PyBytes::new(py, &affine_mesh_bytes()),
        )?;
        locals.set_item(
            "source",
            include_str!(
                "../../../verify/fluid/cell-centered-navier-stokes-fvm-2d/models/direct.eqi"
            ),
        )?;
        py.run(c_str!(r#"
import numpy as np

# Import an ordinary authenticated affine-triangle Mesh artifact.
mesh = eqiora.meshing.Mesh.from_bytes(mesh_bytes)
# Reuse the registered transient equations with a nonzero inlet and zero
# traction elsewhere. The independent oracle below is global incompressibility,
# not an assumed stationary profile at an open convective boundary.
source = source.replace("  relation force_definition", """
  variable inlet_profile: m / s on body;
  parameter inlet_speed: m / s = 1;
  relation inlet_definition on body { inlet_profile - inlet_speed = 0; }
  relation force_definition""")
source = source.replace("relation x_lower_value on x_lower { trace(velocity) = 0; }",
    "relation x_lower_value on x_lower { trace(velocity) + normal(isotropic_lift(inlet_profile)) = 0; }")
for boundary in ("x_upper", "y_lower", "y_upper"):
    source = source.replace(f"relation {boundary}_value on {boundary} {{ trace(velocity) = 0; }}",
        f"relation {boundary}_value on {boundary} {{ normal(2 * dynamic_viscosity * symmetric_part(grad(velocity)) - isotropic_lift(pressure)) = 0; }}")
model = eqiora.compile(source=source)

manual = eqiora.solve.Linear(
    algorithm=eqiora.solve.LinearSolver.SparseLu,
    preconditioner=eqiora.solve.Preconditioner.Identity,
    reduction=eqiora.solve.Reduction.Fast,
    provider=eqiora.solve.SolverProvider.faer(),
    relative_tolerance=1e-10, absolute_tolerance=1e-12, maximum_iterations=2000)
planned = eqiora.solve.Linear(objective=eqiora.solve.Robust,
    relative_tolerance=1e-10, absolute_tolerance=1e-12, maximum_iterations=2000)
outputs = []
for linear in (manual, planned):
    plan = eqiora.resolve(model, mesh=mesh, spatial=eqiora.fem.MiniP1(),
        solve=eqiora.solve.Newton(linear=linear),
        scaling=eqiora.fluid.IncompressibleScaling(length_m=1.0, velocity_m_per_s=1.0, pressure_pa=1.0),
        temporal=eqiora.time.BackwardEuler(step_s=0.01))
    replay = eqiora.Plan.from_bytes(plan.to_bytes())
    assert replay.identity == plan.identity
    v, p = replay.capability.velocity, replay.capability.pressure
    velocity = np.tile([1.0, 0.0], (mesh.vertex_count, 1))
    initial = eqiora.State.initial(replay, time_s=0.0, fields=(
        eqiora.InitialField(v, vertex_values=velocity, cell_values=np.zeros((mesh.cell_count,2))),
        # A deliberately nonzero algebraic pressure forces a Newton correction;
        # it is not a different velocity history or a pressure gauge.
        eqiora.InitialField(p, vertex_values=np.ones(mesh.vertex_count)),
    ))
    initial = eqiora.State.from_bytes(replay, initial.to_bytes())
    result = eqiora.run(replay, state=initial, until_s=0.02, output_times_s=(0.01, 0.02))
    restored = eqiora.Result.from_bytes(replay, result.to_bytes())
    assert restored.to_bytes() == result.to_bytes()
    final = restored.trajectory.states[-1]
    accepted_velocity = final.field(v).values("vertex")
    accepted_pressure = final.field(p).values("vertex")
    assert np.linalg.norm(accepted_velocity) > 0
    assert not np.array_equal(accepted_pressure, np.ones(mesh.vertex_count))
    # Integrate div(u) independently over affine triangles. Interior MINI
    # bubbles vanish on each cell boundary, so their net flux is exactly zero.
    flux = 0.0
    for triangle in mesh.cells:
        points = mesh.coordinates[triangle]
        coefficients = np.linalg.solve(np.column_stack((np.ones(3), points)), accepted_velocity[triangle])
        a, b = points[1]-points[0], points[2]-points[0]
        area = abs(a[0]*b[1]-a[1]*b[0])/2
        flux += area*(coefficients[1,0]+coefficients[2,1])
    assert abs(flux) < 1e-9
    outputs.append((accepted_velocity, accepted_pressure))
np.testing.assert_allclose(outputs[0][0], outputs[1][0], rtol=0, atol=1e-9)
np.testing.assert_allclose(outputs[0][1], outputs[1][1], rtol=0, atol=1e-9)
"#), Some(&locals), Some(&locals))
    })
}

fn affine_mesh_bytes() -> Vec<u8> {
    use eqiora::artifact::{
        AffineTriangleMeshCellsV1, GeometryMeshCorrespondenceEnvelopeV1,
        MeshProductionLineageEnvelopeV1,
    };
    use eqiora::geometry::GeometryGraph;
    use std::collections::BTreeMap;
    let graph = GeometryGraph::new();
    let rectangle = graph.rectangle([0.0, 1.0], [0.0, 1.0]).unwrap();
    let boundaries = rectangle.boundaries();
    let geometry = graph
        .build(
            &rectangle,
            &BTreeMap::from([
                ("body".to_owned(), vec![rectangle.region().into()]),
                ("x_lower".to_owned(), vec![boundaries[0].into()]),
                ("x_upper".to_owned(), vec![boundaries[1].into()]),
                ("y_lower".to_owned(), vec![boundaries[2].into()]),
                ("y_upper".to_owned(), vec![boundaries[3].into()]),
            ]),
        )
        .unwrap();
    let cells = AffineTriangleMeshCellsV1::new([2, 3]).unwrap();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_planar_rectangle_v2_affine_triangles(
            &geometry,
            cells.cells(),
        )
        .unwrap();
    let production = MeshProductionLineageEnvelopeV1::from_affine_triangle_rectangle_v1_resources(
        cells,
        &geometry,
        &mesh,
        &correspondence,
    )
    .unwrap();
    eqiora_numerics::AuthenticatedCommonMesh::affine_triangle_rectangle(
        geometry,
        mesh,
        correspondence,
        production,
    )
    .unwrap()
    .to_bytes()
    .unwrap()
}

fn public_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let native = pyo3::wrap_pymodule!(_eqiora::_eqiora)(py);
    let package_directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bindings/python/python/eqiora")
        .canonicalize()?;
    let locals = PyDict::new(py);
    locals.set_item("native", native.bind(py))?;
    locals.set_item("package_directory", package_directory.to_string_lossy())?;
    py.run(
        c_str!(
            r#"
import importlib.util
import pathlib
import sys

package_path = pathlib.Path(package_directory)
spec = importlib.util.spec_from_file_location(
    "eqiora",
    package_path / "__init__.py",
    submodule_search_locations=[str(package_path)],
)
package = importlib.util.module_from_spec(spec)
sys.modules["eqiora"] = package
sys.modules["eqiora._eqiora"] = native
spec.loader.exec_module(package)
"#
        ),
        None,
        Some(&locals),
    )?;
    Ok(locals
        .get_item("package")?
        .expect("public package must load")
        .cast_into::<PyModule>()?)
}
