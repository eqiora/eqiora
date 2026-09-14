use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyModule};

const SOURCE: &str = r#"
public component AuthoredPoisson(
  support square: volume(ambient_dimension = 2),
  support x_lower: boundary(parent = square),
  support x_upper: boundary(parent = square),
  support y_lower: boundary(parent = square),
  support y_upper: boundary(parent = square),
  parameter diffusion: 1,
  parameter other_diffusion: 1,
  parameter source_scale: 1 / m ^ 2,
  parameter other_source: 1 / m ^ 2
) {

  variable potential: 1 on square;
  law balance on square { flux -diffusion * grad(potential); source source_scale; }
  relation x_lower_value on x_lower { trace(potential) = 0; }
  relation x_upper_value on x_upper { trace(potential) = 0; }
  relation y_lower_value on y_lower { trace(potential) = 0; }
  relation y_upper_value on y_upper { trace(potential) = 0; }
  form weak for balance { test w: 1 for potential zero_on x_lower, x_upper, y_lower, y_upper;
    integrate(square, dot(grad(w), diffusion * grad(potential)))
      = integrate(square, w * source_scale);
  }
}
"#;

#[test]
fn python_authored_scalar_form_closes_compile_resolve_run_and_plan_replay() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let module = public_module(py)?;
        let locals = PyDict::new(py);
        locals.set_item("eqiora", module)?;
        locals.set_item("source", SOURCE)?;
        py.run(
            c_str!(r#"
graph = eqiora.geometry.GeometryGraph()
rectangle = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
geometry = graph.build(rectangle, named_topology={
    "square": rectangle.region,
    "x_lower": rectangle.boundaries[0],
    "x_upper": rectangle.boundaries[1],
    "y_lower": rectangle.boundaries[2],
    "y_upper": rectangle.boundaries[3],
})
parameters = {"diffusion": 1.0, "other_diffusion": 2.0, "source_scale": 1.0, "other_source": 3.0}
model = eqiora.compile(source=source, geometry=geometry, entry='AuthoredPoisson', bindings={'square': geometry.selection('square'), 'x_lower': (geometry.selection('x_lower'), geometry.selection('square')), 'x_upper': (geometry.selection('x_upper'), geometry.selection('square')), 'y_lower': (geometry.selection('y_lower'), geometry.selection('square')), 'y_upper': (geometry.selection('y_upper'), geometry.selection('square')), **parameters})
mesh_plan = eqiora.meshing.resolve(geometry, eqiora.meshing.CartesianMesher(cells=(2, 2)))
mesh = eqiora.meshing.generate(mesh_plan)
linear = eqiora.solve.Linear(
    algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
    preconditioner=eqiora.solve.Preconditioner.Identity,
    reduction=eqiora.solve.Reduction.Reproducible,
    provider=eqiora.solve.SolverProvider.reference(),
    relative_tolerance=1e-10,
    absolute_tolerance=1e-12,
    maximum_iterations=1000,
)
plan = eqiora.resolve(model, mesh=mesh, spatial=eqiora.fem.Q1(), solve=linear)
assert model.authored_formulations[0].name == 'weak'
assert model.authored_formulations[0].test_restrictions[0][0] == 'w'
assert model.authored_formulations[0].implication == 'strong-implies-weak'
assert model.authored_formulations[0].assumptions == ['fixed-domain', 'classical-divergence-and-boundary-trace', 'admissible-h1-test-with-zero-essential-trace']
assert len(model.authored_formulations[0].test_restrictions[0][2]) == 4
assert plan.formulation.requested == eqiora.FormulationSelectionMode.Authored
assert plan.formulation.requested_source_identity == model.authored_formulations[0].source_identity
result = eqiora.run(plan)
assert result.plan_key == plan.identity
# Four h=1/2 Q1 squares leave one free central hat. Its assembled stiffness is
# 4*(2/3)=8/3 and its load integral is 1/4, hence u_center=3/32.
# The requested residual bound divided by 8/3 is below 1e-10 in these SI coordinates.
def check_analytic_coefficients(model, accepted):
    field = model.field(model.authored_formulations[0].trial_field_ids[0])
    values = sorted(accepted.output(field).values("vertex").numpy().reshape(-1))
    assert len(values) == 9
    assert all(abs(value) <= 1e-10 for value in values[:-1])
    assert abs(values[-1] - 3/32) <= 1e-10
check_analytic_coefficients(model, result)
# Program-controlled and manual solver choices preserve the same exact scalar structure.
planned = eqiora.resolve(model, mesh=mesh, spatial=eqiora.fem.Q1(),
    solve=eqiora.solve.Linear(objective=eqiora.solve.Robust,
        relative_tolerance=1e-10, absolute_tolerance=1e-12, maximum_iterations=1000))
check_analytic_coefficients(model, eqiora.run(eqiora.Plan.from_bytes(planned.to_bytes())))


def check_nonzero_temperature(model, accepted):
    field = model.field(model.authored_formulations[0].trial_field_ids[0])
    values = sorted(accepted.output(field).values("vertex").numpy().reshape(-1))
    assert len(values) == 9
    assert all(abs(value - 300.0) <= 1e-10 for value in values[:-1])
    assert abs(values[-1] - (300.0 + 3/32)) <= 1e-10

plan_bytes = plan.to_bytes()
replayed = eqiora.Plan.from_bytes(plan_bytes)
assert replayed.to_bytes() == plan_bytes
assert replayed.identity == plan.identity
assert replayed.formulation.requested == eqiora.FormulationSelectionMode.Authored
assert replayed.formulation.requested_source_identity == plan.formulation.requested_source_identity
assert eqiora.run(replayed).plan_key == plan.identity
try:
    eqiora.Plan.from_bytes(plan_bytes.replace(b"resolved-common-plan/v5", b"resolved-common-plan/v1"))
except eqiora.ValidationError:
    pass
else:
    raise AssertionError("superseded Plan v1 schema was accepted")

plain = eqiora.Model.from_bytes(model.to_bytes())
plain_plan = eqiora.resolve(plain, mesh=mesh, spatial=eqiora.fem.Q1(), solve=linear)
assert plain_plan.formulation.requested == eqiora.FormulationSelectionMode.Automatic
assert plain_plan.identity != plan.identity

# The same retained Law/form vocabulary supports independently dimensioned heat and mass.
for field_type, diffusion_type, source_type in (
    ("K", "W / (m * K)", "W / m^3"),
    ("kg / m^3", "m^2 / s", "kg / (m^3 * s)"),
):
    dimensional_source = source.replace("potential: 1", "potential: " + field_type)
    dimensional_source = dimensional_source.replace("diffusion: 1", "diffusion: " + diffusion_type)
    dimensional_source = dimensional_source.replace("source_scale: 1 / m ^ 2", "source_scale: " + source_type)
    dimensional_source = dimensional_source.replace("other_source: 1 / m ^ 2", "other_source: " + source_type)
    dimensional_source = dimensional_source.replace("trace(potential) = 0", "trace(potential) = 0[" + field_type + "]")
    dimensional_model = eqiora.compile(source=dimensional_source, geometry=geometry, entry='AuthoredPoisson', bindings={'square': geometry.selection('square'), 'x_lower': (geometry.selection('x_lower'), geometry.selection('square')), 'x_upper': (geometry.selection('x_upper'), geometry.selection('square')), 'y_lower': (geometry.selection('y_lower'), geometry.selection('square')), 'y_upper': (geometry.selection('y_upper'), geometry.selection('square')), **parameters})
    dimensional_plan = eqiora.resolve(dimensional_model, mesh=mesh, spatial=eqiora.fem.Q1(), solve=linear)
    check_analytic_coefficients(dimensional_model, eqiora.run(eqiora.Plan.from_bytes(dimensional_plan.to_bytes())))

# Python authoring and source formatting retain one exact complete-exterior restriction.
q = eqiora.lang
module = eqiora.Module("main")
component = module.component("Diffusion")
body = component.volume("body", dimensions=2)
surface = component.complete_exterior("surface", parent=body)
u = component.field("u", value_type=eqiora.ValueType.real(eqiora.Dimension(temperature=1)), role=eqiora.FieldRole.Variable, on=body)
k = component.parameter("k", value_type=eqiora.ValueType.real(eqiora.Dimension(mass=1, length=1, time=-3, temperature=-1)))
f = component.parameter("f", value_type=eqiora.ValueType.real(eqiora.Dimension(mass=1, length=-1, time=-3)))
heat = component.law("heat", on=body, flux=-k*q.grad(u), source=f)
face = surface.member("face")
component.relation("essential", q.equation(q.trace(u), q.quantity(300, eqiora.units.K)), on=face)
w = component.test("w", for_=u, zero_on=surface)
component.weak_form("weak_heat", [heat], equations=[(q.integrate(body, q.dot(q.grad(w), k*q.grad(u))), q.integrate(body, w*f))])
source_bindings = {"body": geometry.selection("square"), "surface": (tuple(geometry.selection(name) for name in ("x_lower", "x_upper", "y_lower", "y_upper")), geometry.selection("square")), "k": 1.0, "f": 1.0}
python_model = eqiora.compile(source=module, geometry=geometry, entry="Diffusion", bindings=source_bindings)
assert "test w: 1 for u zero_on surface;" in module.to_eqi()
emitted_model = eqiora.compile(source=module.to_eqi(), geometry=geometry, entry="Diffusion", bindings=source_bindings)
assert python_model.digest == emitted_model.digest
assert python_model.authored_formulations[0].test_restrictions[0][2] == emitted_model.authored_formulations[0].test_restrictions[0][2]
python_plan = eqiora.resolve(python_model, mesh=mesh, spatial=eqiora.fem.Q1(), solve=linear)
assert python_plan.formulation.boundary_treatment == "complete-essential"
assert "fem.derive.v2.boundary-discharge.zero-test-trace" in python_plan.formulation.rule_ids
check_nonzero_temperature(python_model, eqiora.run(eqiora.Plan.from_bytes(python_plan.to_bytes())))

for changed, expected in (
    (source.replace("zero_on x_lower, x_upper, y_lower, y_upper", "zero_on x_lower"), "zero_on"),
    (source.replace("diffusion * grad(potential)))", "other_diffusion * grad(potential)))"), "coefficient"),
    (source.replace("w * source_scale", "w * other_source"), "source"),
    (source.replace("w * source_scale", "-w * source_scale"), "source term"),
    (source.replace("trace(potential) = 0;", "trace(potential) = trace(potential);", 1), "unmatched signed leaves"),
):
    mismatched = eqiora.compile(source=changed, geometry=geometry, entry='AuthoredPoisson', bindings={'square': geometry.selection('square'), 'x_lower': (geometry.selection('x_lower'), geometry.selection('square')), 'x_upper': (geometry.selection('x_upper'), geometry.selection('square')), 'y_lower': (geometry.selection('y_lower'), geometry.selection('square')), 'y_upper': (geometry.selection('y_upper'), geometry.selection('square')), **parameters})
    try:
        eqiora.resolve(mismatched, mesh=mesh, spatial=eqiora.fem.Q1(), solve=linear)
    except eqiora.ValidationError as error:
        assert expected in str(error), str(error)
    else:
        raise AssertionError(f"mismatched authored Formulation was accepted: {expected}")
"#),
            Some(&locals),
            Some(&locals),
        )
    })
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
