#[path = "tests/transient.rs"]
mod transient;

use std::collections::BTreeMap;

use eqiora::api::ModelDocument;
use eqiora::geometry::{CanonicalGeometryV1, GeometryGraph, NamedEntitySet, PlanarTopologyHandle};
use eqiora::{DimExponents, DynQuantity};
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

use crate::geometry::PyGeometry;
use crate::model::PyModel;

const COMPONENT: &str = r#"
public component PoissonRectangle(
  support region: volume(ambient_dimension = 2),
  support left: boundary(parent = region),
  support right: boundary(parent = region),
  support bottom: boundary(parent = region),
  support top: boundary(parent = region),
  parameter wave_number: 1 / m,
  parameter source_scale: 1 / m ^ 2
) {

  variable potential: 1 on region;
  relation balance on region {
    -div(grad(potential))
      - source_scale * math.sin(wave_number * coordinate(0))
        * math.sin(wave_number * coordinate(1)) = 0;
  }
  relation left_value on left { trace(potential) = 0; }
  relation right_value on right { trace(potential) = 0; }
  relation bottom_value on bottom { trace(potential) = 0; }
  relation top_value on top { trace(potential) = 0; }
}
"#;

const STOKES_COMPONENT: &str =
    include_str!("../../../eqiora-api/src/steady_stokes/accepted_component.eqi");
const ELASTICITY_COMPONENT: &str = r#"
public component MixedBoundaryElasticity(
  support region: volume(ambient_dimension = 2),
  support left: boundary(parent = region),
  support right: boundary(parent = region),
  support bottom: boundary(parent = region),
  support top: boundary(parent = region),
  parameter mu: kg / (m * s ^ 2),
  parameter lambda: kg / (m * s ^ 2),
  parameter length_scale: m
) {

  variable displacement: vector<m, 2> on region;
  variable load_potential: kg / (m * s ^ 2) on region;
  relation load on region {
    load_potential - 2 * mu * coordinate(0) / length_scale = 0;
  }
  relation balance on region {
    -div(2 * mu * symmetric_part(grad(displacement))
      + lambda * isotropic_lift(div(displacement))) - grad(load_potential) = 0;
  }
  relation left_fixed on left { trace(displacement) = 0; }
  relation right_free on right {
    normal(2 * mu * symmetric_part(grad(displacement))
      + lambda * isotropic_lift(div(displacement))) = 0;
  }
  relation bottom_free on bottom {
    normal(2 * mu * symmetric_part(grad(displacement))
      + lambda * isotropic_lift(div(displacement))) = 0;
  }
  relation top_free on top {
    normal(2 * mu * symmetric_part(grad(displacement))
      + lambda * isotropic_lift(div(displacement))) = 0;
  }
}
"#;
const TRANSIENT_SOURCE: &str =
    include_str!("../../../../verify/fluid/cell-centered-navier-stokes-fvm-2d/models/direct.eqi");
const TRANSIENT_CYLINDER_COMPONENT: &str =
    include_str!("../../../../examples/transient-flow-past-cylinder.eqi");

type SupportBinding<'a> = (
    &'a str,
    &'a NamedEntitySet,
    Option<(&'a str, &'a NamedEntitySet)>,
);

fn rectangle_geometry(xmax: f64) -> CanonicalGeometryV1 {
    let graph = GeometryGraph::new();
    let rectangle = graph.rectangle([0.0, xmax], [0.0, 1.0]).unwrap();
    let edges = rectangle.boundaries();
    graph
        .build(
            &rectangle,
            &BTreeMap::from([
                ("region".to_owned(), vec![rectangle.region().into()]),
                (
                    "left".to_owned(),
                    vec![PlanarTopologyHandle::from(edges[0])],
                ),
                (
                    "right".to_owned(),
                    vec![PlanarTopologyHandle::from(edges[1])],
                ),
                (
                    "bottom".to_owned(),
                    vec![PlanarTopologyHandle::from(edges[2])],
                ),
                ("top".to_owned(), vec![PlanarTopologyHandle::from(edges[3])]),
            ]),
        )
        .unwrap()
}

fn scalar_document(geometry: &CanonicalGeometryV1, source_scale: f64) -> ModelDocument {
    scalar_document_from_source(geometry, source_scale, COMPONENT)
}

fn scalar_document_from_source(
    geometry: &CanonicalGeometryV1,
    source_scale: f64,
    component: &str,
) -> ModelDocument {
    let region = geometry.entity_set("region").unwrap();
    let supports: [SupportBinding<'_>; 5] = [
        ("region", region, None),
        (
            "left",
            geometry.entity_set("left").unwrap(),
            Some(("region", region)),
        ),
        (
            "right",
            geometry.entity_set("right").unwrap(),
            Some(("region", region)),
        ),
        (
            "bottom",
            geometry.entity_set("bottom").unwrap(),
            Some(("region", region)),
        ),
        (
            "top",
            geometry.entity_set("top").unwrap(),
            Some(("region", region)),
        ),
    ];
    let parameters = [
        (
            "wave_number",
            DynQuantity::new(
                std::f64::consts::PI,
                DimExponents::from_integers([0, -1, 0, 0, 0, 0, 0]).expect("bounded dimension"),
            ),
        ),
        (
            "source_scale",
            DynQuantity::new(
                source_scale,
                DimExponents::from_integers([0, -2, 0, 0, 0, 0, 0]).expect("bounded dimension"),
            ),
        ),
    ]
    .map(|(name, value)| (name, eqiora::ValueLiteral::try_from(value).unwrap()));
    {
        let expressions = parameters
            .iter()
            .map(|(name, value)| {
                (
                    *name,
                    eqiora::language::SourceAstFactory::value_literal(
                        value,
                        None,
                        eqiora::language::TextRange::default(),
                        |_| None,
                        |_| None,
                    )
                    .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let mut bindings = supports
            .iter()
            .map(|(name, selection, parent)| {
                (
                    *name,
                    eqiora::compiler::StaticBindingValue::GeometrySupport {
                        geometry,
                        selection,
                        parent: parent.map(|(_, selection)| selection),
                    },
                )
            })
            .collect::<Vec<_>>();
        bindings.extend(expressions.iter().map(|(name, value)| {
            (
                *name,
                eqiora::compiler::StaticBindingValue::Expression(value),
            )
        }));
        ModelDocument::compile_selected(
            "python-common-plan.eqi",
            component,
            "PoissonRectangle",
            &bindings,
        )
    }
    .unwrap()
}

fn stokes_document(geometry: &CanonicalGeometryV1) -> ModelDocument {
    stokes_document_with_speed(geometry, 0.3)
}

fn stokes_document_with_speed(geometry: &CanonicalGeometryV1, inlet_speed: f64) -> ModelDocument {
    let fluid = geometry.entity_set("fluid").unwrap();
    let supports: [SupportBinding<'_>; 5] = [
        ("fluid", fluid, None),
        (
            "inlet",
            geometry.entity_set("inlet").unwrap(),
            Some(("fluid", fluid)),
        ),
        (
            "outlet",
            geometry.entity_set("outlet").unwrap(),
            Some(("fluid", fluid)),
        ),
        (
            "walls",
            geometry.entity_set("walls").unwrap(),
            Some(("fluid", fluid)),
        ),
        (
            "cylinder",
            geometry.entity_set("cylinder").unwrap(),
            Some(("fluid", fluid)),
        ),
    ];
    let parameters = [
        (
            "dynamic_viscosity",
            DynQuantity::new(
                0.001,
                DimExponents::from_integers([1, -1, -1, 0, 0, 0, 0]).expect("bounded dimension"),
            ),
        ),
        (
            "zero_pressure",
            DynQuantity::new(
                0.0,
                DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension"),
            ),
        ),
        (
            "inlet_speed",
            DynQuantity::new(
                inlet_speed,
                DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension"),
            ),
        ),
        (
            "channel_height",
            DynQuantity::new(
                0.41,
                DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension"),
            ),
        ),
    ]
    .map(|(name, value)| (name, eqiora::ValueLiteral::try_from(value).unwrap()));
    {
        let expressions = parameters
            .iter()
            .map(|(name, value)| {
                (
                    *name,
                    eqiora::language::SourceAstFactory::value_literal(
                        value,
                        None,
                        eqiora::language::TextRange::default(),
                        |_| None,
                        |_| None,
                    )
                    .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let mut bindings = supports
            .iter()
            .map(|(name, selection, parent)| {
                (
                    *name,
                    eqiora::compiler::StaticBindingValue::GeometrySupport {
                        geometry,
                        selection,
                        parent: parent.map(|(_, selection)| selection),
                    },
                )
            })
            .collect::<Vec<_>>();
        bindings.extend(expressions.iter().map(|(name, value)| {
            (
                *name,
                eqiora::compiler::StaticBindingValue::Expression(value),
            )
        }));
        ModelDocument::compile_selected(
            "steady-flow-past-cylinder.eqi",
            STOKES_COMPONENT,
            "SteadyFlowPastCylinder",
            &bindings,
        )
    }
    .unwrap()
}

fn transient_cylinder_document(geometry: &CanonicalGeometryV1) -> ModelDocument {
    transient_cylinder_document_with_speed(geometry, 0.3)
}

fn transient_cylinder_document_with_speed(
    geometry: &CanonicalGeometryV1,
    inlet_speed: f64,
) -> ModelDocument {
    let fluid = geometry.entity_set("fluid").unwrap();
    let supports: [SupportBinding<'_>; 5] = [
        ("fluid", fluid, None),
        (
            "inlet",
            geometry.entity_set("inlet").unwrap(),
            Some(("fluid", fluid)),
        ),
        (
            "outlet",
            geometry.entity_set("outlet").unwrap(),
            Some(("fluid", fluid)),
        ),
        (
            "walls",
            geometry.entity_set("walls").unwrap(),
            Some(("fluid", fluid)),
        ),
        (
            "cylinder",
            geometry.entity_set("cylinder").unwrap(),
            Some(("fluid", fluid)),
        ),
    ];
    let parameters = [
        (
            "density",
            DynQuantity::new(
                1.0,
                DimExponents::from_integers([1, -3, 0, 0, 0, 0, 0]).expect("bounded dimension"),
            ),
        ),
        (
            "dynamic_viscosity",
            DynQuantity::new(
                0.001,
                DimExponents::from_integers([1, -1, -1, 0, 0, 0, 0]).expect("bounded dimension"),
            ),
        ),
        (
            "zero_pressure",
            DynQuantity::new(
                0.0,
                DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension"),
            ),
        ),
        (
            "inlet_speed",
            DynQuantity::new(
                inlet_speed,
                DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension"),
            ),
        ),
        (
            "channel_height",
            DynQuantity::new(
                0.41,
                DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension"),
            ),
        ),
    ]
    .map(|(name, value)| (name, eqiora::ValueLiteral::try_from(value).unwrap()));
    {
        let expressions = parameters
            .iter()
            .map(|(name, value)| {
                (
                    *name,
                    eqiora::language::SourceAstFactory::value_literal(
                        value,
                        None,
                        eqiora::language::TextRange::default(),
                        |_| None,
                        |_| None,
                    )
                    .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let mut bindings = supports
            .iter()
            .map(|(name, selection, parent)| {
                (
                    *name,
                    eqiora::compiler::StaticBindingValue::GeometrySupport {
                        geometry,
                        selection,
                        parent: parent.map(|(_, selection)| selection),
                    },
                )
            })
            .collect::<Vec<_>>();
        bindings.extend(expressions.iter().map(|(name, value)| {
            (
                *name,
                eqiora::compiler::StaticBindingValue::Expression(value),
            )
        }));
        ModelDocument::compile_selected(
            "transient-flow-past-cylinder.eqi",
            TRANSIENT_CYLINDER_COMPONENT,
            "TransientFlowPastCylinder",
            &bindings,
        )
    }
    .unwrap()
}

#[test]
fn installed_root_resolve_and_run_retain_exact_python_objects() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let native = pyo3::wrap_pymodule!(crate::_eqiora)(py);
        let package_directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../bindings/python/python/eqiora")
            .canonicalize()?;
        let locals = PyDict::new(py);
        locals.set_item("native", native.bind(py))?;
        locals.set_item("package_directory", package_directory.to_string_lossy())?;
        py.run(
                c_str!(r#"
import importlib.util, os, pathlib, sys, tempfile
package_path = pathlib.Path(package_directory)
spec = importlib.util.spec_from_file_location("eqiora", package_path / "__init__.py", submodule_search_locations=[str(package_path)])
package = importlib.util.module_from_spec(spec)
sys.modules["eqiora"] = package
sys.modules["eqiora._eqiora"] = native
spec.loader.exec_module(package)

graph = package.geometry.GeometryGraph()
rectangle = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
source = graph.build(rectangle, named_topology={
    "region": rectangle.region,
    "left": rectangle.boundaries[0],
    "right": rectangle.boundaries[1],
    "bottom": rectangle.boundaries[2],
    "top": rectangle.boundaries[3],
})
mesher = package.meshing.CartesianMesher(cells=(2, 3))
mesh_plan = package.meshing.resolve(source, mesher)
mesh = package.meshing.generate(mesh_plan)
"#),
                None,
                Some(&locals),
            )?;
        let geometry = locals
            .get_item("source")?
            .unwrap()
            .extract::<PyRef<'_, PyGeometry>>()?
            .geometry()
            .clone();
        let model = PyModel::from_document(
            py,
            scalar_document(&geometry, 2.0 * std::f64::consts::PI.powi(2)),
        )?;
        let fresh_model = PyModel::from_document(
            py,
            scalar_document(&geometry, 3.0 * std::f64::consts::PI.powi(2)),
        )?;
        let foreign_model = PyModel::from_document(
            py,
            scalar_document(&rectangle_geometry(2.0), 2.0 * std::f64::consts::PI.powi(2)),
        )?;
        let variable_coefficient = "(1 + wave_number * coordinate(0))";
        let variable_component = COMPONENT
            .replace(
                "-div(grad(potential))",
                &format!("-div({variable_coefficient} * grad(potential))"),
            )
            .replace(
                "relation top_value on top { trace(potential) = 0; }",
                &format!(
                    "relation top_value on top {{ normal({variable_coefficient} * grad(potential)) = wave_number; }}"
                ),
            );
        let variable_model = PyModel::from_document(
            py,
            scalar_document_from_source(
                &geometry,
                2.0 * std::f64::consts::PI.powi(2),
                &variable_component,
            ),
        )?;
        locals.set_item("model", Py::new(py, model)?)?;
        locals.set_item("fresh_model", Py::new(py, fresh_model)?)?;
        locals.set_item("foreign_model", Py::new(py, foreign_model)?)?;
        locals.set_item("variable_model", Py::new(py, variable_model)?)?;
        py.run(
                c_str!(r#"
linear = package.solve.Linear(
    algorithm=package.solve.LinearSolver.BiConjugateGradientStabilized,
    preconditioner=package.solve.Preconditioner.Identity,
    reduction=package.solve.Reduction.Reproducible,
    provider=package.solve.SolverProvider.reference(),
    relative_tolerance=1e-10,
    absolute_tolerance=1e-12,
    maximum_iterations=10000,
)
tpfa_linear = package.solve.Linear(
    relative_tolerance=1e-10,
    absolute_tolerance=1e-12,
    maximum_iterations=10000,
    algorithm=package.solve.LinearSolver.ConjugateGradient,
    preconditioner=package.solve.Preconditioner.Identity,
    reduction=package.solve.Reduction.Reproducible,
    provider=package.solve.SolverProvider.reference(),
)
q1 = package.resolve(model, mesh=mesh, spatial=package.fem.Q1(), solve=linear)
q1_repeat = package.resolve(model, mesh=mesh, spatial=package.fem.Q1(), solve=linear)
q1_exact = package.resolve(
    model, mesh=mesh, spatial=package.fem.Q1(),
    formulation=package.formulation.PrimalGalerkin, solve=linear,
)
tpfa = package.resolve(model, mesh=mesh, spatial=package.fvm.CellCenteredTpfa(), solve=tpfa_linear)
variable_q1 = package.resolve(variable_model, mesh=mesh, spatial=package.fem.Q1(), solve=linear)
variable_tpfa = package.resolve(variable_model, mesh=mesh, spatial=package.fvm.CellCenteredTpfa(), solve=tpfa_linear)
# Manual intent is complete and never acquires an invented ranking objective.
assert linear.objective is None and q1.solve.objective is None
assert linear.provider.id == "eqiora.reference"
assert linear.provider.implementation_version == q1.solve.backend_version
assert linear.provider.libraries == []
controls = dict(relative_tolerance=1e-10, absolute_tolerance=1e-12, maximum_iterations=10000)
for incomplete in ({}, {"algorithm": package.solve.LinearSolver.ConjugateGradient},
    {"objective": package.solve.Robust, "provider": package.solve.SolverProvider.reference()}):
    try:
        package.solve.Linear(**controls, **incomplete)
    except TypeError:
        pass
    else:
        raise AssertionError("incomplete or mixed solver intent must reject before resolution")
q1_bytes = q1.to_bytes()
portable_q1 = package.Plan.from_bytes(q1_bytes)
assert portable_q1.requested_solve == linear
assert portable_q1.requested_solve.provider == linear.provider
import copy, json
plan_payload = json.loads(q1_bytes)
assert plan_payload["schema"] == "eqiora.resolved-common-plan/v5"
assert plan_payload["solve"]["linear"]["intent"]["kind"] == "exact"
for mutation in ("version", "library", "missing-intent", "mixed-intent", "old-schema"):
    payload = copy.deepcopy(plan_payload)
    request = payload["solve"]["linear"]
    if mutation == "version":
        request["intent"]["provider"]["implementation_version"] = "stale-release"
    elif mutation == "library":
        request["intent"]["provider"]["libraries"] = [["invented", "1"]]
    elif mutation == "missing-intent":
        del request["intent"]
    elif mutation == "mixed-intent":
        request["intent"]["objective"] = "robust"
    else:
        payload["schema"] = "eqiora.resolved-common-plan/v2"
    try:
        package.Plan.from_bytes(json.dumps(payload, separators=(",", ":")).encode())
    except (package.CompatibilityError, package.ValidationError) as error:
        if mutation in ("version", "library"):
            assert "provider release or library inventory" in str(error)
    else:
        raise AssertionError(f"altered exact intent must reject: {mutation}")
plan_directory_owner = tempfile.TemporaryDirectory()
plan_directory = pathlib.Path(plan_directory_owner.name)
plan_path = plan_directory / "q1.eqplan"
plan_path.write_bytes(b"incomplete previous output")
q1.write(plan_path)
assert plan_path.read_bytes() == q1_bytes
assert not list(plan_directory.glob(".eqiora-plan-*.tmp"))
file_q1 = package.Plan.read(plan_path)
assert file_q1.identity == q1.identity
assert file_q1.to_bytes() == q1_bytes

for rejected_name, rejected_bytes in (
    ("truncated.eqplan", q1_bytes[:-1]),
    ("trailing.eqplan", q1_bytes + b"\n"),
    ("unknown-version.eqplan", q1_bytes.replace(b"resolved-common-plan/v5", b"resolved-common-plan/v9")),
):
    rejected_path = plan_directory / rejected_name
    rejected_path.write_bytes(rejected_bytes)
    try:
        package.Plan.read(rejected_path)
    except package.CompatibilityError:
        pass
    else:
        raise AssertionError(f"hostile Plan file must reject: {rejected_name}")

wrong_suffix = plan_directory / "q1.json"
try:
    q1.write(wrong_suffix)
except package.CompatibilityError:
    pass
else:
    raise AssertionError("Plan file paths require the exact .eqplan suffix")
try:
    package.Plan.read(wrong_suffix)
except package.CompatibilityError:
    pass
else:
    raise AssertionError("Plan file paths require the exact .eqplan suffix")
assert not wrong_suffix.exists()

directory_path = plan_directory / "directory.eqplan"
directory_path.mkdir()
try:
    q1.write(directory_path)
except package.CompatibilityError:
    pass
else:
    raise AssertionError("Plan file I/O must reject non-regular paths")
try:
    package.Plan.read(directory_path)
except package.CompatibilityError:
    pass
else:
    raise AssertionError("Plan file I/O must reject non-regular paths")

if os.name != "nt":
    symlink_target = plan_directory / "target.eqplan"
    symlink_target.write_bytes(q1_bytes)
    symlink_path = plan_directory / "symlink.eqplan"
    symlink_path.symlink_to(symlink_target)
    try:
        q1.write(symlink_path)
    except package.CompatibilityError:
        pass
    else:
        raise AssertionError("Plan file I/O must reject symlinks")
    try:
        package.Plan.read(symlink_path)
    except package.CompatibilityError:
        pass
    else:
        raise AssertionError("Plan file I/O must reject symlinks")
    assert symlink_target.read_bytes() == q1_bytes

oversized_path = plan_directory / "oversized.eqplan"
with oversized_path.open("wb") as oversized:
    oversized.truncate(256 * 1024 * 1024 + 1)
try:
    package.Plan.read(oversized_path)
except package.CompatibilityError:
    pass
else:
    raise AssertionError("Plan file I/O must reject input above the decoder bound")

missing_parent_path = plan_directory / "missing" / "q1.eqplan"
try:
    q1.write(missing_parent_path)
except package.CompatibilityError:
    pass
else:
    raise AssertionError("Plan publication must reject a missing staging directory")
assert not missing_parent_path.exists()
assert not list(plan_directory.glob(".eqiora-plan-*.tmp"))
replayed = package.Model.from_bytes(model.to_bytes())
replayed_plan = package.resolve(replayed, mesh=mesh, spatial=package.fem.Q1(), solve=linear)
mesh_bytes = mesh.to_bytes()
replayed_mesh = package.meshing.Mesh.from_bytes(mesh_bytes)
replayed_mesh_plan = package.resolve(model, mesh=replayed_mesh, spatial=package.fem.Q1(), solve=linear)
fresh_plan = package.resolve(fresh_model, mesh=mesh, spatial=package.fem.Q1(), solve=linear)
assert q1.model is model
assert q1.mesh is mesh
assert q1.mesh_digest == mesh.digest
assert q1.identity == q1_repeat.identity
assert portable_q1.identity == q1.identity
assert portable_q1.to_bytes() == q1_bytes
assert portable_q1.model.to_bytes() == model.to_bytes()
assert portable_q1.mesh.to_bytes() == mesh.to_bytes()
assert portable_q1.spatial == package.fem.Q1()
assert portable_q1.requested_solve.relative_tolerance == linear.relative_tolerance
assert portable_q1.requested_solve.absolute_tolerance == linear.absolute_tolerance
assert portable_q1.requested_solve.maximum_iterations == linear.maximum_iterations
assert portable_q1.solve.algorithm == q1.solve.algorithm
assert replayed_plan.identity == q1.identity
assert replayed_mesh.digest == mesh.digest
assert replayed_mesh.correspondence_digest == mesh.correspondence_digest
assert replayed_mesh.production_lineage_digest == mesh.production_lineage_digest
assert replayed_mesh.to_bytes() == mesh_bytes
assert replayed_mesh_plan.identity == q1.identity
try:
    package.meshing.Mesh.from_bytes(mesh_bytes + b"\n")
except package.ValidationError:
    pass
else:
    raise AssertionError("noncanonical Mesh bytes must reject")
try:
    package.Plan.from_bytes(q1_bytes + b"\n")
except package.ValidationError:
    pass
else:
    raise AssertionError("noncanonical Plan bytes must reject")
assert fresh_model.digest != model.digest
assert fresh_plan.model_digest == fresh_model.digest
assert fresh_plan.identity != q1.identity
assert q1.identity != tpfa.identity
assert len(q1.realization_digest) == 64 and len(tpfa.realization_digest) == 64
assert q1.realization_digest == replayed_plan.realization_digest
assert q1.realization_digest != tpfa.realization_digest
assert q1.mesh.cells.shape == (6, 4)
assert isinstance(q1.capability, package.ScalarPlanView)
assert q1.capability.kind == "scalar"
assert isinstance(q1.capability.fields, tuple)
assert q1.capability.fields == q1.fields
assert len(q1.capability.fields) == 1
assert not hasattr(q1.capability, "field")
assert "ScalarPlanView(fields=" in repr(q1.capability)
assert replayed_plan.capability.fields == q1.capability.fields
assert tpfa.capability.fields == tpfa.fields
assert q1.capability.coefficient_sampling == "quadrature-point"
assert q1.capability.face_coefficient_policy == "not-applicable"
assert tpfa.capability.coefficient_sampling == "facet-centroid"
assert tpfa.capability.face_coefficient_policy == "direct-centroid-evaluation"
assert isinstance(q1.formulation, package.FormulationView)
assert q1.formulation.requested is package.FormulationSelectionMode.Automatic
assert q1.formulation.effective is package.formulation.PrimalGalerkin
assert q1.formulation.boundary_treatment == "complete-essential"
assert q1.formulation.rule_ids == [
    "fem.derive.v1.test-pairing",
    "fem.derive.v1.divergence-by-parts",
    "fem.derive.v2.boundary-discharge.zero-test-trace",
    "fem.derive.v1.source-pairing",
]
assert q1.formulation.selection_reason_codes == [
    "eqiora.formulation.auto.primal-galerkin-for-q1/v1",
]
assert q1_exact.formulation.requested is package.FormulationSelectionMode.Exact
assert q1_exact.formulation.effective is q1.formulation.effective
assert q1_exact.identity != q1.identity
assert q1_exact.realization_digest == q1.realization_digest
assert tpfa.formulation is None
assert variable_q1.formulation is None
for label, wrong_spatial, wrong_formulation in (
    ("Q1 mixed", package.fem.Q1(), package.formulation.MixedGalerkin),
    ("TPFA primal", package.fvm.CellCenteredTpfa(), package.formulation.PrimalGalerkin),
):
    try:
        package.resolve(
            model, mesh=mesh, spatial=wrong_spatial,
            formulation=wrong_formulation, solve=linear,
        )
    except package.ValidationError:
        pass
    else:
        raise AssertionError(f"incompatible scalar Formulation must reject: {label}")
try:
    package.resolve(
        variable_model, mesh=mesh, spatial=package.fem.Q1(),
        formulation=package.formulation.PrimalGalerkin, solve=linear,
    )
except package.ValidationError:
    pass
else:
    raise AssertionError("exact primal Formulation must reject an unproved natural boundary")
assert not hasattr(q1.capability, "scaling")
assert q1.requested_solve is linear
assert q1.solve.algorithm == "bicgstab"
assert tpfa.solve.algorithm == "conjugate-gradient"
q1_result = package.run(q1)
q1_result_bytes = q1_result.to_bytes()
replayed_q1_result = package.Result.from_bytes(q1, q1_result_bytes)
assert replayed_q1_result.to_bytes() == q1_result_bytes
assert replayed_q1_result.output(q1.capability.fields[0]).values("vertex").numpy().tolist() == q1_result.output(q1.capability.fields[0]).values("vertex").numpy().tolist()
portable_q1_result = package.run(portable_q1)
file_q1_result = package.run(file_q1)
tpfa_result = package.run(tpfa)
variable_q1_result = package.run(variable_q1)
variable_tpfa_result = package.run(variable_tpfa)
assert variable_q1_result.output(variable_q1.capability.fields[0]).coefficient_count("vertex") == mesh.vertex_count
assert variable_tpfa_result.output(variable_tpfa.capability.fields[0]).coefficient_count("cell") == mesh.cell_count
assert q1_result.model_digest == q1.model_digest
assert q1_result.plan_key == q1.identity
assert q1_result.mesh(q1.capability.fields[0]) is mesh
q1_output = q1_result.output(q1.capability.fields[0])
assert q1_output.associations == ("vertex",)
assert q1_output.logical_shape("vertex") == (3, 4)
assert len(q1_output.values("vertex")) == 12
assert portable_q1_result.output(portable_q1.capability.fields[0]).logical_shape("vertex") == (3, 4)
assert file_q1_result.output(file_q1.capability.fields[0]).logical_shape("vertex") == (3, 4)
plan_directory_owner.cleanup()
assert not hasattr(q1_result, "run_manifest")
tpfa_output = tpfa_result.output(tpfa.capability.fields[0])
assert tpfa_output.associations == ("cell",)
assert tpfa_output.logical_shape("cell") == (2, 3)
assert len(tpfa_output.values("cell")) == 6
try:
    package.resolve(foreign_model, mesh=mesh, spatial=package.fem.Q1(), solve=linear)
except package.ValidationError:
    pass
else:
    raise AssertionError("foreign Model was admitted against the caller Mesh")
try:
    package.resolve(model, mesh=mesh, spatial=package.fem.MiniP1(), solve=linear)
except package.ValidationError:
    pass
else:
    raise AssertionError("MINI/P1 selected Stokes physics for a scalar Model")
try:
    package.resolve(model, mesh=mesh, spatial=package.fem.Q1(), solve=linear,
                    scaling=package.fluid.IncompressibleScaling())
except package.ValidationError:
    pass
else:
    raise AssertionError("flow scaling was admitted for scalar Model mathematics")

foreign_rectangle = graph.rectangle(x_bounds=(0.0, 2.0), y_bounds=(0.0, 1.0))
foreign_source = graph.build(foreign_rectangle, named_topology={
    "region": foreign_rectangle.region,
    "left": foreign_rectangle.boundaries[0],
    "right": foreign_rectangle.boundaries[1],
    "bottom": foreign_rectangle.boundaries[2],
    "top": foreign_rectangle.boundaries[3],
})
foreign_plan = package.meshing.resolve(foreign_source, mesher)
foreign_mesh = package.meshing.generate(foreign_plan)
try:
    package.resolve(model, mesh=foreign_mesh, spatial=package.fem.Q1(), solve=linear)
except package.ValidationError:
    pass
else:
    raise AssertionError("same-cardinality foreign Mesh was admitted")
try:
    package.run(model)
except TypeError:
    pass
else:
    raise AssertionError("root run retained a Model-shaped compatibility path")
"#),
                None,
                Some(&locals),
            )
    })
}

#[test]
fn root_plan_resolves_model_owned_steady_stokes_with_automatic_scaling() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let native = pyo3::wrap_pymodule!(crate::_eqiora)(py);
        let package_directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../bindings/python/python/eqiora")
            .canonicalize()?;
        let locals = PyDict::new(py);
        locals.set_item("native", native.bind(py))?;
        locals.set_item("package_directory", package_directory.to_string_lossy())?;
        py.run(
                c_str!(r#"
import importlib.util, pathlib, sys
package_path = pathlib.Path(package_directory)
spec = importlib.util.spec_from_file_location("eqiora", package_path / "__init__.py", submodule_search_locations=[str(package_path)])
package = importlib.util.module_from_spec(spec)
sys.modules["eqiora"] = package
sys.modules["eqiora._eqiora"] = native
spec.loader.exec_module(package)

graph = package.geometry.GeometryGraph()
rectangle = graph.rectangle(x_bounds=(0.0, 2.2), y_bounds=(0.0, 0.41))
circle = graph.circle(center=(0.2, 0.2), radius=0.05)
fluid = graph.subtract(rectangle, circle)
source = graph.build(fluid, named_topology={
    "fluid": fluid.region,
    "inlet": rectangle.boundaries[0],
    "outlet": rectangle.boundaries[1],
    "walls": rectangle.boundaries[2:],
    "cylinder": circle.boundaries[0],
})
mesher = package.meshing.GmshMesher(maximum_boundary_error=1e-4, minimum_mean_ratio=1e-5, maximum_boundary_facets=50)
mesh_plan = package.meshing.resolve(source, mesher)
mesh = package.meshing.generate(mesh_plan)
"#),
                None,
                Some(&locals),
            )?;
        let geometry = locals
            .get_item("source")?
            .unwrap()
            .extract::<PyRef<'_, PyGeometry>>()?
            .geometry()
            .clone();
        let model = PyModel::from_document(py, stokes_document(&geometry))?;
        let fresh_model = PyModel::from_document(py, stokes_document(&geometry))?;
        let foreign_model = PyModel::from_document(
            py,
            scalar_document(&rectangle_geometry(1.0), 2.0 * std::f64::consts::PI.powi(2)),
        )?;
        locals.set_item("model", Py::new(py, model)?)?;
        locals.set_item("fresh_model", Py::new(py, fresh_model)?)?;
        locals.set_item("foreign_model", Py::new(py, foreign_model)?)?;
        py.run(
                c_str!(r#"
import math
linear = package.solve.Linear(
    algorithm=package.solve.LinearSolver.SparseLu,
    preconditioner=package.solve.Preconditioner.Identity,
    reduction=package.solve.Reduction.Fast,
    provider=package.solve.SolverProvider.faer(),
    relative_tolerance=1e-6,
    absolute_tolerance=1e-13,
    maximum_iterations=10000,
)
plan = package.resolve(model, mesh=mesh, spatial=package.fem.MiniP1(), solve=linear)
explicit_none = package.resolve(model, mesh=mesh, spatial=package.fem.MiniP1(), solve=linear, scaling=None)
all_auto_request = package.fluid.IncompressibleScaling()
all_auto = package.resolve(model, mesh=mesh, spatial=package.fem.MiniP1(), solve=linear, scaling=all_auto_request)
partial_equal = package.resolve(
    model, mesh=mesh, spatial=package.fem.MiniP1(), solve=linear,
    scaling=package.fluid.IncompressibleScaling(length_m=0.41),
)
partial_changed = package.resolve(
    model, mesh=mesh, spatial=package.fem.MiniP1(), solve=linear,
    scaling=package.fluid.IncompressibleScaling(length_m=0.82),
)
manual_equal = package.resolve(
    model, mesh=mesh, spatial=package.fem.MiniP1(), solve=linear,
    scaling=package.fluid.IncompressibleScaling(
        length_m=0.41,
        velocity_m_per_s=0.3,
        pressure_pa=0.001 * 0.3 / 0.41,
    ),
)
replayed = package.Model.from_bytes(model.to_bytes())
replayed_plan = package.resolve(replayed, mesh=mesh, spatial=package.fem.MiniP1(), solve=linear)
replayed_mesh = package.meshing.Mesh.from_bytes(mesh.to_bytes())
replayed_mesh_plan = package.resolve(model, mesh=replayed_mesh, spatial=package.fem.MiniP1(), solve=linear)
fresh_plan = package.resolve(fresh_model, mesh=mesh, spatial=package.fem.MiniP1(), solve=linear)
assert plan.identity == explicit_none.identity == all_auto.identity == replayed_plan.identity == fresh_plan.identity
assert replayed_mesh_plan.identity == plan.identity
assert replayed_mesh.to_bytes() == mesh.to_bytes()
assert all_auto_request == package.fluid.IncompressibleScaling()
assert all_auto_request.length_m is None
assert all_auto_request.velocity_m_per_s is None
assert all_auto_request.pressure_pa is None
assert plan.model is model
assert plan.mesh is mesh
assert plan.mesh_digest == mesh.digest
assert plan.realization_digest is not None
assert plan.realization_digest == partial_equal.realization_digest == manual_equal.realization_digest
assert partial_changed.realization_digest != plan.realization_digest
assert partial_equal.identity != plan.identity
assert manual_equal.identity != plan.identity
assert partial_equal.identity != manual_equal.identity
assert plan.capability.scaling.length_m == 0.41
assert plan.capability.scaling.velocity_m_per_s == 0.3
assert plan.capability.scaling.pressure_pa == 0.001 * 0.3 / 0.41
assert partial_changed.capability.scaling.length_m == 0.82
assert partial_changed.capability.scaling.velocity_m_per_s == 0.3
assert partial_changed.capability.scaling.pressure_pa == 0.001 * 0.3 / 0.82
assert partial_equal.capability.scaling.pressure_pa == plan.capability.scaling.pressure_pa
assert manual_equal.capability.scaling.pressure_pa == plan.capability.scaling.pressure_pa
receipt = plan.capability.scaling_receipt
partial_receipt = partial_equal.capability.scaling_receipt
manual_receipt = manual_equal.capability.scaling_receipt
assert receipt.model_digest == plan.model_digest
assert receipt.geometry_digest == plan.geometry_digest
assert receipt.correspondence_digest == plan.correspondence_digest
assert receipt.mesh_digest == plan.mesh_digest
assert len(receipt.components) == 5
component = package.fluid.IncompressibleScalingComponent2d
mode = package.fluid.IncompressibleScalingMode
rule = package.fluid.IncompressibleScalingRule2d
authority = package.fluid.IncompressibleScalingAuthorityKind
assert tuple(record.component for record in receipt.components) == (
    component.Length, component.Velocity, component.Pressure,
    component.Gauge, component.WeakFunctional,
)
assert receipt.length.dimension == (0, 1, 0, 0, 0, 0, 0)
assert receipt.velocity.dimension == (0, 1, -1, 0, 0, 0, 0)
assert receipt.pressure.dimension == (1, -1, -2, 0, 0, 0, 0)
assert receipt.gauge.dimension == (0, 0, -1, 0, 0, 0, 0)
assert receipt.weak_functional.dimension == (1, 1, -3, 0, 0, 0, 0)
assert (receipt.length.mode, receipt.velocity.mode, receipt.pressure.mode,
        receipt.gauge.mode, receipt.weak_functional.mode) == (
    mode.Automatic, mode.Automatic, mode.Derived, mode.Derived, mode.Derived,
)
assert receipt.length.rule == rule.ExactChannelHeightV1
assert receipt.velocity.rule == rule.ExactInletMaximumV1
assert receipt.pressure.rule == rule.ViscousStokesPressureV1
assert receipt.gauge.rule == rule.GaugeRateV1
assert receipt.weak_functional.rule == rule.WeakFunctionalV1
assert receipt.length.dependencies == ()
assert receipt.velocity.dependencies == ()
assert receipt.pressure.dependencies == (component.Length, component.Velocity)
assert receipt.gauge.dependencies == (component.Velocity, component.Length)
assert receipt.weak_functional.dependencies == (component.Pressure, component.Velocity, component.Length)
assert receipt.length.authorities[0].kind == authority.ExactGeometrySpan
assert receipt.length.authorities[0].axis == 1
assert receipt.length.authorities[0].bounds_m == (0.0, 0.41)
assert receipt.velocity.authorities[1].kind == authority.ModelInletMaximum
assert receipt.pressure.authorities[0].kind == authority.ModelDynamicViscosity
assert receipt.gauge.authorities == ()
assert receipt.weak_functional.authorities == ()
assert partial_receipt.length.mode == mode.Manual
assert partial_receipt.velocity.mode == mode.Automatic
assert partial_receipt.pressure.mode == mode.Derived
assert partial_receipt.length.authorities[0].kind == authority.ManualRequest
assert manual_receipt.length.mode == mode.Manual
assert manual_receipt.velocity.mode == mode.Manual
assert manual_receipt.pressure.mode == mode.Manual
assert receipt.provenance_digest != partial_receipt.provenance_digest
assert receipt.provenance_digest != manual_receipt.provenance_digest
assert partial_receipt.provenance_digest != manual_receipt.provenance_digest
try:
    plan.capability.scaling.length_m = 1.0
except AttributeError:
    pass
else:
    raise AssertionError("effective scaling was mutable")
try:
    receipt.components += (receipt.length,)
except AttributeError:
    pass
else:
    raise AssertionError("scaling receipt was mutable")
assert plan.spatial.method == "mini-p1"
assert plan.solve.backend == "eqiora.faer"
assert plan.requested_solve is linear
assert plan.solve.algorithm == "sparse-lu"
assert plan.solve.operator == "symmetric-indefinite"
assert plan.solve.preconditioner == "identity"
assert plan.solve.reduction == "fast"
assert plan.solve.backend == "eqiora.faer"
result = package.run(plan)
assert isinstance(result, package.Result)
stokes_evidence = package.fluid.steady_stokes_evidence(result)
assert isinstance(stokes_evidence, package.fluid.SteadyStokesEvidence)
assert stokes_evidence.plan_key == result.plan_key
assert stokes_evidence.exact_bounds == ((0.0, 2.2), (0.0, 0.41))
for objective, algorithm, provider, reduction in (
    (package.solve.Robust, package.solve.LinearSolver.MinimumResidual, package.solve.SolverProvider.reference(), package.solve.Reduction.Reproducible),
    (package.solve.Fast, package.solve.LinearSolver.SparseLu, package.solve.SolverProvider.faer(), package.solve.Reduction.Fast),
):
    controls = dict(relative_tolerance=1e-6, absolute_tolerance=1e-13, maximum_iterations=10000)
    ranked_request = package.solve.Linear(**controls, objective=objective)
    exact_request = package.solve.Linear(**controls, algorithm=algorithm,
        preconditioner=package.solve.Preconditioner.Identity, reduction=reduction, provider=provider)
    ranked_plan = package.resolve(model, mesh=mesh, spatial=package.fem.MiniP1(), solve=ranked_request)
    exact_plan = package.resolve(model, mesh=mesh, spatial=package.fem.MiniP1(), solve=exact_request)
    assert ranked_plan.solve.operator == "symmetric-indefinite"
    assert ranked_plan.solve.provider == provider
    assert ranked_plan.solve.objective is objective
    assert exact_plan.solve.objective is None
    assert ranked_plan.solve.algorithm == exact_plan.solve.algorithm
    ranked_plan = package.Plan.from_bytes(ranked_plan.to_bytes())
    assert ranked_plan.requested_solve == ranked_request
    ranked_result = package.run(ranked_plan)
    exact_result = package.run(exact_plan)
    for ranked_field, exact_field in zip(ranked_plan.fields, exact_plan.fields):
        ranked_output = ranked_result.output(ranked_field)
        exact_output = exact_result.output(exact_field)
        for association in ranked_output.associations:
            assert ranked_output.values(association).numpy().tolist() == exact_output.values(association).numpy().tolist()
    observation = package.fluid.steady_stokes_evidence(ranked_result)
    assert observation.solve.true_residual_norm <= observation.solve.residual_target

try:
    package.State.zero(plan)
except ValueError:
    pass
else:
    raise AssertionError("State.zero admitted a steady Plan")
assert result.model_digest == plan.model_digest
assert result.plan_key == plan.identity
assert package.run(partial_equal).plan_key == partial_equal.identity
assert package.run(manual_equal).plan_key == manual_equal.identity
assert len(plan.fields) == 2
assert plan.fields == (plan.capability.velocity, plan.capability.pressure)
assert isinstance(plan.capability, package.fluid.IncompressibleFlowPlanView)
assert not hasattr(result, "outputs")
velocity = result.output(plan.capability.velocity)
pressure = result.output(plan.capability.pressure)
assert result.mesh(plan.capability.velocity) is mesh
assert result.mesh(plan.capability.pressure) is mesh
assert velocity.field == plan.capability.velocity
assert pressure.field == plan.capability.pressure
assert velocity.mesh is pressure.mesh is mesh
assert velocity.dimension == (0, 1, -1, 0, 0, 0, 0)
assert velocity.value_shape == (2,)
assert velocity.associations == ("vertex", "cell-bubble")
assert velocity.coefficient_count("vertex") == mesh.vertex_count
assert len(velocity.values("vertex")) == mesh.vertex_count * 2
assert velocity.coefficient_count("cell-bubble") == mesh.cell_count
assert len(velocity.values("cell-bubble")) == mesh.cell_count * 2
assert pressure.dimension == (1, -1, -2, 0, 0, 0, 0)
assert pressure.value_shape == ()
assert pressure.associations == ("vertex",)
assert pressure.coefficient_count("vertex") == mesh.vertex_count
assert len(pressure.values("vertex")) == mesh.vertex_count
cylinder = source.selection("cylinder")
inlet = source.selection("inlet")
outlet = source.selection("outlet")
cylinder_force = result.boundary_force(cylinder)
inlet_flux = result.boundary_flux(inlet)
outlet_flux = result.boundary_flux(outlet)
assert cylinder_force.selection == cylinder
assert cylinder_force.source_digest == result.plan_key
assert cylinder_force.source_kind == "result"
assert all(map(math.isfinite, cylinder_force.on_domain))
assert inlet_flux.selection == inlet and outlet_flux.selection == outlet
assert inlet_flux.value + outlet_flux.value == stokes_evidence.net_flux
assert result.solve.algorithm == "sparse-lu"
foreign_field = foreign_model.field(foreign_model.field_ids[0])
try:
    result.output(foreign_field)
except ValueError:
    pass
else:
    raise AssertionError("foreign exact FieldRef selected a common output")
try:
    result.output("pressure")
except TypeError:
    pass
else:
    raise AssertionError("string/name lookup selected a common output")
try:
    package.resolve(model, mesh=mesh, spatial=package.fem.Q1(), solve=linear)
except package.ValidationError:
    pass
else:
    raise AssertionError("Stokes Model selected scalar physics through a spatial policy")
try:
    package.resolve(model, mesh=mesh, spatial=package.fem.MiniP1(), solve=linear, scaling=object())
except TypeError:
    pass
else:
    raise AssertionError("untyped scaling reached Plan publication")
try:
    package.resolve(
        model, mesh=mesh, spatial=package.fem.MiniP1(), solve=linear,
        temporal=package.time.BackwardEuler(0.01),
    )
except package.ValidationError:
    pass
else:
    raise AssertionError("steady Stokes admitted a temporal policy")
for invalid in (True, False, 1, 0.0, -0.0, -1.0, float("nan"), float("inf"), -float("inf"), "auto", {}):
    try:
        package.fluid.IncompressibleScaling(length_m=invalid)
    except (TypeError, package.ValidationError):
        pass
    else:
        raise AssertionError(f"invalid scaling component was admitted: {invalid!r}")
assert not hasattr(package, "AUTO")
assert not hasattr(package.fluid, "AUTO")
"#),
                None,
                Some(&locals),
            )
    })
}

#[test]
fn root_plan_runs_geometry_backed_transient_cylinder_with_explicit_state() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let native = pyo3::wrap_pymodule!(crate::_eqiora)(py);
        let package_directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../bindings/python/python/eqiora")
            .canonicalize()?;
        let locals = PyDict::new(py);
        locals.set_item("native", native.bind(py))?;
        locals.set_item("package_directory", package_directory.to_string_lossy())?;
        py.run(
            c_str!(r#"
import importlib.util, pathlib, sys
package_path = pathlib.Path(package_directory)
spec = importlib.util.spec_from_file_location("eqiora", package_path / "__init__.py", submodule_search_locations=[str(package_path)])
package = importlib.util.module_from_spec(spec)
sys.modules["eqiora"] = package
sys.modules["eqiora._eqiora"] = native
spec.loader.exec_module(package)

graph = package.geometry.GeometryGraph()
rectangle = graph.rectangle(x_bounds=(0.0, 2.2), y_bounds=(0.0, 0.41))
circle = graph.circle(center=(0.2, 0.2), radius=0.05)
fluid = graph.subtract(rectangle, circle)
source = graph.build(fluid, named_topology={
    "fluid": fluid.region,
    "inlet": rectangle.boundaries[0],
    "outlet": rectangle.boundaries[1],
    "walls": rectangle.boundaries[2:],
    "cylinder": circle.boundaries[0],
})
"#),
            None,
            Some(&locals),
        )?;
        let geometry = locals
            .get_item("source")?
            .unwrap()
            .extract::<PyRef<'_, PyGeometry>>()?
            .geometry()
            .clone();
        let model = PyModel::from_document(py, transient_cylinder_document(&geometry))?;
        let steady_model = PyModel::from_document(py, stokes_document_with_speed(&geometry, 0.3))?;
        locals.set_item("model", Py::new(py, model)?)?;
        locals.set_item("steady_model", Py::new(py, steady_model)?)?;
        py.run(
            c_str!(r#"
mesher = package.meshing.GmshMesher(maximum_boundary_error=1e-4, minimum_mean_ratio=1e-5, maximum_boundary_facets=50)
mesh = package.meshing.generate(package.meshing.resolve(source, mesher))
import numpy as np
linear = package.solve.Linear(
    algorithm=package.solve.LinearSolver.SparseLu,
    preconditioner=package.solve.Preconditioner.Identity,
    reduction=package.solve.Reduction.Fast,
    provider=package.solve.SolverProvider.faer(),
    relative_tolerance=1e-6,
    absolute_tolerance=1e-12,
    maximum_iterations=20000,
)
steady_plan = package.resolve(
    steady_model,
    mesh=mesh,
    spatial=package.fem.MiniP1(),
    solve=linear,
    scaling=package.fluid.IncompressibleScaling(
        length_m=0.41,
        velocity_m_per_s=0.3,
        pressure_pa=0.001 * 0.3 / 0.41,
    ),
)
steady_result = package.run(steady_plan)
steady_velocity = steady_result.output(steady_plan.capability.velocity)
steady_pressure = steady_result.output(steady_plan.capability.pressure)
plan = package.resolve(
    model,
    mesh=mesh,
    spatial=package.fem.MiniP1(),
    solve=package.solve.Newton(linear=linear),
    temporal=package.time.BackwardEuler(0.0001),
    scaling=package.fluid.IncompressibleScaling(
        length_m=0.41,
        velocity_m_per_s=0.3,
        pressure_pa=1.0 * 0.3 * 0.3,
    ),
)
state = package.State.initial(
    plan,
    time_s=0.0,
    fields=(
        package.InitialField(
            plan.capability.velocity,
            vertex_values=np.asarray(steady_velocity.values("vertex")).reshape(mesh.vertex_count, 2),
            cell_values=np.asarray(steady_velocity.values("cell-bubble")).reshape(mesh.cell_count, 2),
        ),
        package.InitialField(
            plan.capability.pressure,
            vertex_values=np.asarray(steady_pressure.values("vertex")),
        ),
    ),
)
assert plan.model is model and plan.mesh is mesh
assert plan.capability.pressure_gauge is package.fluid.PressureGauge2d.BoundaryTraction
assert np.max(np.abs(np.asarray(steady_velocity.values("vertex")))) > 0.0
assert state.field(plan.capability.velocity).associations == ("vertex", "cell")
result = package.run(plan, state=state, steps=1, output_steps=(1,))
result_bytes = result.to_bytes()
replayed_result = package.Result.from_bytes(plan, result_bytes)
assert replayed_result.to_bytes() == result_bytes
assert replayed_result.trajectory.states[0].digest == result.trajectory.states[0].digest
assert len(result.trajectory.states) == 1
wake_state = result.trajectory.states[0]
assert wake_state.time_s == 0.0001
assert np.max(np.abs(wake_state.field(plan.capability.velocity).values("vertex"))) > 0.0
vorticity = wake_state.curl(plan.capability.velocity)
assert vorticity.operator == "curl"
assert vorticity.source_state_digest == wake_state.digest
assert vorticity.source_field == plan.capability.velocity
assert vorticity.mesh_digest == mesh.digest
assert vorticity.support_domain_id == wake_state.field(plan.capability.velocity).support_domain_id
assert vorticity.dimension == (0, 0, -1, 0, 0, 0, 0)
assert vorticity.value_shape == ()
assert vorticity.frame == "spatial-axial"
assert vorticity.associations == ("cell",)
assert np.array_equal(vorticity.support_indices("cell"), np.arange(mesh.cell_count, dtype=np.uint32))
assert np.all(np.isfinite(vorticity.values("cell")))
assert np.max(np.abs(vorticity.values("cell"))) > 0.0
assert not vorticity.values("cell").flags.writeable
assert wake_state.curl(plan.capability.velocity) == vorticity
front_pressure = wake_state.sample(plan.capability.pressure, at=(0.15, 0.2))
rear_pressure = wake_state.sample(plan.capability.pressure, at=(0.25, 0.2))
assert front_pressure.source_state_digest == wake_state.digest
assert front_pressure.field == plan.capability.pressure
assert front_pressure.mesh_digest == mesh.digest
assert front_pressure.support_domain_id == wake_state.field(plan.capability.pressure).support_domain_id
assert front_pressure.point_m == (0.15, 0.2)
assert front_pressure.dimension == (1, -1, -2, 0, 0, 0, 0)
assert front_pressure.frame == "invariant"
assert np.isfinite(front_pressure.value) and np.isfinite(rear_pressure.value)
assert wake_state.sample(plan.capability.pressure, at=(0.15, 0.2)) == front_pressure
cylinder = source.selection("cylinder")
cylinder_force = wake_state.boundary_force(cylinder)
assert cylinder_force.source_digest == wake_state.digest
assert cylinder_force.source_kind == "state"
assert cylinder_force.selection == cylinder
assert cylinder_force.geometry_digest == source.digest
assert cylinder_force.mesh_digest == mesh.digest
assert cylinder_force.dimension == (1, 0, -2, 0, 0, 0, 0)
assert cylinder_force.frame == "spatial-cartesian"
assert np.all(np.isfinite(cylinder_force.on_domain))
assert cylinder_force.on_selection == tuple(-value for value in cylinder_force.on_domain)
assert wake_state.boundary_force(cylinder) == cylinder_force
try:
    wake_state.curl(plan.capability.pressure)
except ValueError as error:
    assert "velocity FieldRef" in str(error)
else:
    raise AssertionError("curl accepted the pressure Field")
for invalid_sample in range(3):
    try:
        if invalid_sample == 0:
            wake_state.sample(plan.capability.velocity, at=(0.15, 0.2))
        elif invalid_sample == 1:
            wake_state.sample(plan.capability.pressure, at=(3.0, 3.0))
        else:
            wake_state.sample(plan.capability.pressure, at=(True, 0.2))
    except (ValueError, package.ValidationError):
        pass
    else:
        raise AssertionError("invalid point sample was admitted")
try:
    state.boundary_force(cylinder)
except KeyError:
    pass
else:
    raise AssertionError("initial State fabricated an accepted boundary force")
try:
    wake_state.boundary_force(source.selection("fluid"))
except ValueError:
    pass
else:
    raise AssertionError("boundary force accepted a region selection")
"#),
            None,
            Some(&locals),
        )
    })
}

#[test]
fn installed_root_common_elasticity_uses_exact_caller_mesh_and_field() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let native = pyo3::wrap_pymodule!(crate::_eqiora)(py);
        let package_directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../bindings/python/python/eqiora")
            .canonicalize()?;
        let locals = PyDict::new(py);
        locals.set_item("native", native.bind(py))?;
        locals.set_item("package_directory", package_directory.to_string_lossy())?;
        locals.set_item("elasticity_source", ELASTICITY_COMPONENT)?;
        py.run(
                c_str!(r#"
import importlib.util, pathlib, sys
package_path = pathlib.Path(package_directory)
spec = importlib.util.spec_from_file_location("eqiora", package_path / "__init__.py", submodule_search_locations=[str(package_path)])
package = importlib.util.module_from_spec(spec)
sys.modules["eqiora"] = package
sys.modules["eqiora._eqiora"] = native
spec.loader.exec_module(package)

graph = package.geometry.GeometryGraph()
rectangle = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
geometry = graph.build(rectangle, named_topology={
    "region": rectangle.region,
    "left": rectangle.boundaries[0],
    "right": rectangle.boundaries[1],
    "bottom": rectangle.boundaries[2],
    "top": rectangle.boundaries[3],
})
mesher = package.meshing.CartesianMesher(cells=(2, 3))
mesh_plan = package.meshing.resolve(geometry, mesher)
mesh = package.meshing.generate(mesh_plan)
model = package.compile(source=elasticity_source, filename='elasticity.eqi', geometry=geometry, entry='MixedBoundaryElasticity', bindings={'region': geometry.selection('region'), 'left': (geometry.selection('left'), geometry.selection('region')), 'right': (geometry.selection('right'), geometry.selection('region')), 'bottom': (geometry.selection('bottom'), geometry.selection('region')), 'top': (geometry.selection('top'), geometry.selection('region')), **{'mu': 3.0, 'lambda': 0.0, 'length_scale': 1.0}})
replayed = package.Model.from_bytes(model.to_bytes())
alternate = package.compile(source=elasticity_source, filename='alternate-elasticity.eqi', geometry=geometry, entry='MixedBoundaryElasticity', bindings={'region': geometry.selection('region'), 'left': (geometry.selection('left'), geometry.selection('region')), 'right': (geometry.selection('right'), geometry.selection('region')), 'bottom': (geometry.selection('bottom'), geometry.selection('region')), 'top': (geometry.selection('top'), geometry.selection('region')), **{'mu': 4.0, 'lambda': 0.0, 'length_scale': 1.0}})
linear = package.solve.Linear(
    algorithm=package.solve.LinearSolver.ConjugateGradient,
    preconditioner=package.solve.Preconditioner.Identity,
    reduction=package.solve.Reduction.Reproducible,
    provider=package.solve.SolverProvider.reference(),
    relative_tolerance=1e-10,
    absolute_tolerance=1e-12,
    maximum_iterations=10000,
)
plan = package.resolve(model, mesh=mesh, spatial=package.fem.Q1(), solve=linear)
replayed_plan = package.resolve(replayed, mesh=mesh, spatial=package.fem.Q1(), solve=linear)
alternate_plan = package.resolve(alternate, mesh=mesh, spatial=package.fem.Q1(), solve=linear)
assert plan.identity == replayed_plan.identity
assert alternate_plan.identity != plan.identity
assert alternate_plan.model_digest == alternate.digest
assert len(plan.realization_digest) == 64
assert plan.realization_digest == replayed_plan.realization_digest
assert alternate_plan.realization_digest == plan.realization_digest
assert plan.model is model
assert plan.mesh is mesh
assert isinstance(plan.capability, package.solid.ElasticityPlanView)
assert plan.capability.displacement == model.field(plan.capability.displacement.id)
assert plan.fields == (plan.capability.displacement,)
assert plan.mesh.cells.shape == (6, 4)
assert plan.spatial == package.fem.Q1()
assert plan.requested_solve is linear
assert plan.solve.algorithm == "conjugate-gradient"
assert not hasattr(plan.capability, "scaling")
assert plan.temporal is None
assert plan.solve.preconditioner == "identity"
assert plan.solve.reduction == "reproducible"
assert plan.execution.placement == "host-serial" and plan.execution.workers == 1

result = package.run(plan)
result_bytes = result.to_bytes()
replayed_result = package.Result.from_bytes(plan, result_bytes)
assert replayed_result.to_bytes() == result_bytes
assert replayed_result.output(plan.capability.displacement).values("vertex").numpy().tolist() == result.output(plan.capability.displacement).values("vertex").numpy().tolist()
elasticity_evidence = package.solid.linear_elasticity_evidence(result)
assert isinstance(elasticity_evidence, package.solid.LinearElasticityEvidence)
assert elasticity_evidence.plan_key == result.plan_key
assert elasticity_evidence.exact_bounds == ((0.0, 1.0), (0.0, 1.0))
output = result.output(plan.capability.displacement)
assert output.field == plan.capability.displacement
assert output.mesh is mesh
assert output.value_shape == (2,)
assert output.coefficient_count("vertex") == 12
assert len(output.values("vertex")) == 24
assert output.dimension == (0, 1, 0, 0, 0, 0, 0)
assert output.associations == ("vertex",)
assert result.mesh(plan.capability.displacement) is mesh
assert package.submit(plan).result().output(plan.capability.displacement).coefficient_count("vertex") == 12
# The known SPD class with no diagonal claim admits reference CG/Identity and
# identity LU. Exact provider selection, manual parity, and replay are observed
# through ordinary Linear authoring, rather than a Newton-only adapter.
for objective, algorithm, provider, reduction in (
    (package.solve.Robust, package.solve.LinearSolver.ConjugateGradient, package.solve.SolverProvider.reference(), package.solve.Reduction.Reproducible),
    (package.solve.Fast, package.solve.LinearSolver.SparseLu, package.solve.SolverProvider.faer(), package.solve.Reduction.Fast),
    (package.solve.LowMemory, package.solve.LinearSolver.ConjugateGradient, package.solve.SolverProvider.reference(), package.solve.Reduction.Reproducible),
):
    common_controls = dict(relative_tolerance=1e-10, absolute_tolerance=1e-12, maximum_iterations=10000)
    ranked_request = package.solve.Linear(**common_controls, objective=objective)
    manual_request = package.solve.Linear(**common_controls, algorithm=algorithm,
        preconditioner=package.solve.Preconditioner.Identity, reduction=reduction, provider=provider)
    ranked_plan = package.resolve(model, mesh=mesh, spatial=package.fem.Q1(), solve=ranked_request)
    manual_plan = package.resolve(model, mesh=mesh, spatial=package.fem.Q1(), solve=manual_request)
    assert ranked_plan.solve.operator == "symmetric-positive-definite"
    assert ranked_plan.solve.provider == provider
    assert ranked_plan.solve.backend == provider.id
    assert ranked_plan.solve.backend_version == provider.implementation_version
    assert ranked_plan.solve.objective is objective
    assert manual_plan.solve.objective is None
    assert ranked_plan.solve.algorithm == manual_plan.solve.algorithm
    assert ranked_plan.solve.preconditioner == manual_plan.solve.preconditioner == "identity"
    ranked_plan = package.Plan.from_bytes(ranked_plan.to_bytes())
    assert ranked_plan.requested_solve == ranked_request
    ranked_result = package.run(ranked_plan)
    manual_result = package.run(manual_plan)
    assert ranked_result.output(ranked_plan.capability.displacement).values("vertex").numpy().tolist() == manual_result.output(manual_plan.capability.displacement).values("vertex").numpy().tolist()
    observation = package.solid.linear_elasticity_evidence(ranked_result)
    assert observation.solve.true_residual_norm <= observation.solve.residual_target


load_potential_id = model.field_ids[0]
if load_potential_id == plan.capability.displacement.id:
    load_potential_id = model.field_ids[1]
load_potential = model.field(load_potential_id)
try:
    result.output(load_potential)
except KeyError:
    pass
else:
    raise AssertionError("load potential escaped as an elasticity Result Field")
try:
    result.output("displacement")
except TypeError:
    pass
else:
    raise AssertionError("string Field lookup was admitted")
try:
    result.output(alternate_plan.capability.displacement)
except ValueError:
    pass
else:
    raise AssertionError("foreign exact FieldRef was admitted")

foreign_rectangle = graph.rectangle(x_bounds=(0.0, 2.0), y_bounds=(0.0, 1.0))
foreign_geometry = graph.build(foreign_rectangle, named_topology={
    "region": foreign_rectangle.region,
    "left": foreign_rectangle.boundaries[0],
    "right": foreign_rectangle.boundaries[1],
    "bottom": foreign_rectangle.boundaries[2],
    "top": foreign_rectangle.boundaries[3],
})
foreign_mesh_plan = package.meshing.resolve(foreign_geometry, mesher)
foreign_mesh = package.meshing.generate(foreign_mesh_plan)
triangle_plan = package.meshing.resolve(
    geometry,
    package.meshing.AffineTriangleMesher(cells=(2, 3)),
)
triangle_mesh = package.meshing.generate(triangle_plan)
for kwargs in (
    dict(mesh=mesh, spatial=package.fvm.CellCenteredTpfa(), solve=linear),
    dict(mesh=triangle_mesh, spatial=package.fem.Q1(), solve=linear),
    dict(mesh=foreign_mesh, spatial=package.fem.Q1(), solve=linear),
    dict(mesh=mesh, spatial=package.fem.Q1(), solve=package.solve.Newton(linear=linear)),
    dict(mesh=mesh, spatial=package.fem.Q1(), solve=linear, temporal=package.time.BackwardEuler(0.1)),
    dict(mesh=mesh, spatial=package.fem.Q1(), solve=linear, scaling=package.fluid.IncompressibleScaling()),
):
    try:
        package.resolve(model, **kwargs)
    except package.ValidationError:
        pass
    else:
        raise AssertionError(f"elasticity cross-wire was admitted: {kwargs}")
"#),
                None,
                Some(&locals),
            )
    })
}
