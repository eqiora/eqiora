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
# One retained energy supplies both the generated weak form and ordinary observation.
energy_source = source.replace("  form weak for balance", "  observable energy:1=integral(diffusion*contract(grad(potential),grad(potential),axes=((0,0),))/2-source_scale*potential,measure(square));\n  form weak for balance")
energy_source = energy_source.replace("integrate(square, dot(grad(w), diffusion * grad(potential)))\n      = integrate(square, w * source_scale)", "variation(energy,wrt=potential,direction=w,holding=(diffusion,source_scale))=0")
def compile_energy(energy_source):
    return eqiora.compile(source=energy_source, geometry=geometry, entry='AuthoredPoisson', bindings={'square': geometry.selection('square'), 'x_lower': (geometry.selection('x_lower'), geometry.selection('square')), 'x_upper': (geometry.selection('x_upper'), geometry.selection('square')), 'y_lower': (geometry.selection('y_lower'), geometry.selection('square')), 'y_upper': (geometry.selection('y_upper'), geometry.selection('square')), **parameters})
energy_model = compile_energy(energy_source)
energy_plan = eqiora.resolve(energy_model, mesh=mesh, spatial=eqiora.fem.Q1(), solve=linear)
energy_plan = eqiora.Plan.from_bytes(energy_plan.to_bytes())
energy_result = eqiora.run(energy_plan)
check_analytic_coefficients(energy_model, energy_result)
energy_value = energy_result.observe(energy_model.observable("definition.energy"), quadrature_points=2)
# Independently: K=8/3, b=1/4, u=3/32, so F=u*K*u/2-b*u=-3/256.
assert abs(energy_value.value + 3/256) <= 1e-10
# Integral sums keep each retained functional lineage and the same exact weak law.
first = "variation(energy,wrt=potential,direction=w,holding=(diffusion,source_scale))"
summed_model = compile_energy(energy_source.replace(first, f"({first}+{first})-{first}"))
summed_plan = eqiora.resolve(summed_model, mesh=mesh, spatial=eqiora.fem.Q1(), solve=linear)
summed_result = eqiora.run(eqiora.Plan.from_bytes(summed_plan.to_bytes()))
check_analytic_coefficients(summed_model, summed_result)
assert abs(summed_result.observe(summed_model.observable("definition.energy"), quadrature_points=2).value + 3/256) <= 1e-10
try:
    doubled_model = compile_energy(energy_source.replace(first, f"{first}+{first}"))
    eqiora.resolve(doubled_model, mesh=mesh, spatial=eqiora.fem.Q1(), solve=linear)
except eqiora.ValidationError as error:
    assert "weak residual" in str(error), str(error)
else:
    raise AssertionError("doubled functional derivative was accepted as the exact strong law")
# The right and horizontal sides carry natural flux laws; the direction is
# constrained only on the essential left side. No extra zero trace is invented.
natural_source = energy_source.replace("zero_on x_lower, x_upper, y_lower, y_upper", "zero_on x_lower")
for name in ("x_upper", "y_lower", "y_upper"):
    natural_source = natural_source.replace(
        f"relation {name}_value on {name} {{ trace(potential) = 0; }}",
        f"relation {name}_value on {name} {{ normal(diffusion * grad(potential)) = 0; }}",
    )
natural_model = compile_energy(natural_source)
natural_plan = eqiora.resolve(natural_model, mesh=mesh, spatial=eqiora.fem.Q1(), solve=linear)
natural_plan = eqiora.Plan.from_bytes(natural_plan.to_bytes())
natural_result = eqiora.run(natural_plan)
natural_field = natural_model.field(natural_model.authored_formulations[0].trial_field_ids[0])
values = natural_result.output(natural_field).values("vertex").numpy().reshape(-1)
# The exact nodal Q1 solution interpolates x-x*x/2, constant in y.
assert len(values) == 9
assert all(abs(value - (x-x*x/2)) <= 1e-9 for value,(x,y) in zip(values, mesh.coordinates))
# Slopes 3/4 and 1/4 give internal energy 5/32 and load pairing 5/16.
assert abs(natural_result.observe(natural_model.observable("definition.energy"), quadrature_points=2).value + 5/32) <= 1e-9
# Extra zero traces on natural sides and missing essential traces are both false claims.
for restriction in ("zero_on x_lower, x_upper", "zero_on y_lower"):
    mutant = compile_energy(natural_source.replace("zero_on x_lower", restriction))
    try:
        eqiora.resolve(mutant, mesh=mesh, spatial=eqiora.fem.Q1(), solve=linear)
    except eqiora.ValidationError as error:
        assert "zero_on restriction" in str(error), str(error)
    else:
        raise AssertionError("incorrect essential/natural direction restriction was accepted")
# A surface load needs its own functional term; it cannot be discarded as zero flux.
loaded = compile_energy(natural_source.replace("normal(diffusion * grad(potential)) = 0", "normal(diffusion * grad(potential)) = 1[1/m]"))
try:
    eqiora.resolve(loaded, mesh=mesh, spatial=eqiora.fem.Q1(), solve=linear)
except eqiora.ValidationError as error:
    assert "weak residual" in str(error), str(error)
else:
    raise AssertionError("nonzero natural work was silently discharged")
# A prescribed right flux contributes actual surface work. For -u_xx=1,
# u(0)=0, u_x(1)=1, the Q1 nodal values interpolate 2*x-x*x/2.
loaded_source = natural_source.replace(
    "relation x_upper_value on x_upper { normal(diffusion * grad(potential)) = 0; }",
    "relation x_upper_value on x_upper { normal(diffusion * grad(potential)) = source_scale*coordinate(0); }",
)
loaded_source = loaded_source.replace("  form weak for balance",
    "  observable surface:1=integral(-source_scale*coordinate(0)*trace(potential),measure(x_upper));\n  form weak for balance")
surface_variation = "variation(surface,wrt=potential,direction=w,holding=(source_scale,))"
loaded_source = loaded_source.replace(first, f"{first}+{surface_variation}")
loaded_model = compile_energy(loaded_source)
loaded_plan = eqiora.resolve(loaded_model, mesh=mesh, spatial=eqiora.fem.Q1(), solve=linear)
loaded_result = eqiora.run(eqiora.Plan.from_bytes(loaded_plan.to_bytes()))
loaded_field = loaded_model.field(loaded_model.authored_formulations[0].trial_field_ids[0])
loaded_values = loaded_result.output(loaded_field).values("vertex").numpy().reshape(-1)
assert all(abs(value-(2*x-x*x/2)) <= 1e-9 for value,(x,y) in zip(loaded_values,mesh.coordinates))
# Independently, element slopes 7/4 and 5/4 give internal energy 37/32,
# volume load work 13/16, and right surface work 3/2. Independent rational
# assembly gives ||K^-1||_inf=234/49 and ||b||_2²=87/128, so the stated
# relative residual bound gives coefficient error <4e-10 (test bound 1e-9).
assert abs(loaded_result.observe(loaded_model.observable("definition.energy"), quadrature_points=2).value-11/32) <= 1e-9
assert abs(loaded_result.observe(loaded_model.observable("definition.surface"), quadrature_points=2).value+3/2) <= 1e-9
for mutant_source in (
    loaded_source.replace(f"+{surface_variation}", ""),
    loaded_source.replace("integral(-source_scale*coordinate(0)*trace(potential),measure(x_upper))", "integral(source_scale*coordinate(0)*trace(potential),measure(x_upper))"),
    loaded_source.replace("trace(potential),measure(x_upper)", "trace(potential),measure(y_upper)"),
):
    try:
        eqiora.resolve(compile_energy(mutant_source), mesh=mesh, spatial=eqiora.fem.Q1(), solve=linear)
    except eqiora.ValidationError as error:
        assert "weak residual" in str(error), str(error)
    else:
        raise AssertionError("missing, reversed or misplaced surface work was accepted")
# With no volume load the same prescribed surface work gives u=x,
# volume energy 1/2 and boundary energy -1. The zero source remains explicit.
zero_volume_source = loaded_source.replace("source source_scale;", "source 0[1/m^2];")
zero_volume_source = zero_volume_source.replace("/2-source_scale*potential", "/2")
zero_volume_source = zero_volume_source.replace(first, "variation(energy,wrt=potential,direction=w,holding=(diffusion,))")
zero_volume_model = compile_energy(zero_volume_source)
zero_volume_plan = eqiora.resolve(zero_volume_model, mesh=mesh, spatial=eqiora.fem.Q1(), solve=linear)
zero_volume_result = eqiora.run(eqiora.Plan.from_bytes(zero_volume_plan.to_bytes()))
zero_volume_field = zero_volume_model.field(zero_volume_model.authored_formulations[0].trial_field_ids[0])
zero_volume_values = zero_volume_result.output(zero_volume_field).values("vertex").numpy().reshape(-1)
assert all(abs(value-x) <= 1e-9 for value,(x,y) in zip(zero_volume_values,mesh.coordinates))
assert abs(zero_volume_result.observe(zero_volume_model.observable("definition.energy"), quadrature_points=2).value-1/2) <= 1e-9
assert abs(zero_volume_result.observe(zero_volume_model.observable("definition.surface"), quadrature_points=2).value+1) <= 1e-9
# The variation direction carries the Field unit; here both are m and F is J.
physical_energy = energy_source.replace("potential: 1", "potential: m").replace("w: 1", "w: m")
physical_energy = physical_energy.replace("diffusion: 1", "diffusion: J/m^2").replace("1 / m ^ 2", "J/m^3").replace("energy:1", "energy:J")
physical_model = compile_energy(physical_energy)
physical_plan = eqiora.resolve(physical_model, mesh=mesh, spatial=eqiora.fem.Q1(), solve=linear)
physical_plan = eqiora.Plan.from_bytes(physical_plan.to_bytes())
physical_result = eqiora.run(physical_plan)
check_analytic_coefficients(physical_model, physical_result)
physical_value = physical_result.observe(physical_model.observable("definition.energy"), quadrature_points=2)
assert abs(physical_value.value + 3/256) <= 1e-10
assert physical_value.value_type == eqiora.ValueType.real(eqiora.Dimension(mass=1, length=2, time=-2))
try:
    compile_energy(physical_energy.replace("w: m", "w: 1"))
except eqiora.ValidationError as error:
    assert "Field identity and dimension" in str(error), str(error)
else:
    raise AssertionError("dimensionless direction of a length Field was accepted")
# A numerically unit-valued dimensional multiplier is not an exact weak-law pairing.
unit_scaled_energy = energy_source.replace("observable energy:1=integral(", "observable energy:J=integral(1[J]*(").replace(",measure(square));", "),measure(square));")
unit_scaled_model = compile_energy(unit_scaled_energy)
try:
    eqiora.resolve(unit_scaled_model, mesh=mesh, spatial=eqiora.fem.Q1(), solve=linear)
except eqiora.ValidationError as error:
    assert "variation dimension differs from the strong-law test pairing" in str(error), str(error)
else:
    raise AssertionError("dimensionally different energy was admitted by numerical coefficient equality")
for changed in (
    energy_source.replace("diffusion*contract", "2*diffusion*contract"),
    energy_source.replace("-source_scale*potential", "+source_scale*potential"),
    energy_source.replace("-source_scale*potential", "-other_source*potential").replace("holding=(diffusion,source_scale)", "holding=(diffusion,other_source)"),
):
    changed_model = compile_energy(changed)
    try:
        eqiora.resolve(changed_model, mesh=mesh, spatial=eqiora.fem.Q1(), solve=linear)
    except eqiora.ValidationError as error:
        assert "strong-law weak residual" in str(error), str(error)
    else:
        raise AssertionError("energy/strong-law mismatch was admitted")
# Public Python authoring emits the same retained energy and variation syntax.
q = eqiora.lang
energy_module = eqiora.Module("energy")
component = energy_module.component("Energy")
body = component.volume("body", dimensions=2)
surface = component.complete_exterior("surface", parent=body)
u = component.field("u", value_type=eqiora.ValueType.real(eqiora.Dimension(length=1)), role=eqiora.FieldRole.Variable, on=body)
k = component.parameter("k", value_type=eqiora.ValueType.real(eqiora.Dimension(mass=1, time=-2)))
f = component.parameter("f", value_type=eqiora.ValueType.real(eqiora.Dimension(mass=1, length=-1, time=-2)))
balance = component.law("balance", on=body, flux=-k*q.grad(u), source=f)
face = surface.member("face")
component.relation("fixed", q.equation(q.trace(u), q.quantity(0, eqiora.units.m)), on=face)
energy = component.observable("energy", k*q.contract(q.grad(u), q.grad(u), axes=((0,0),))/2-f*u,
                              value_type=eqiora.ValueType.real(eqiora.Dimension(mass=1, length=2, time=-2)), on=body)
w = component.test("w", for_=u, dimension=eqiora.Dimension(length=1), zero_on=surface)
first = q.variation(energy, wrt=u, direction=w, holding=(k,f))
component.weak_form("stationary", [balance], equations=[(first,0)])
emitted = energy_module.to_eqi()
assert "variation(energy" in emitted and "measure(body)" in emitted
energy_bindings = {"body": geometry.selection("square"), "surface": (tuple(geometry.selection(name) for name in ("x_lower", "x_upper", "y_lower", "y_upper")), geometry.selection("square")), "k": 1.0, "f": 1.0}
for authored in (energy_module, emitted):
    authored_model = eqiora.compile(source=authored, geometry=geometry, entry="Energy", bindings=energy_bindings)
    authored_plan = eqiora.resolve(authored_model, mesh=mesh, spatial=eqiora.fem.Q1(), solve=linear)
    authored_result = eqiora.run(eqiora.Plan.from_bytes(authored_plan.to_bytes()))
    check_analytic_coefficients(authored_model, authored_result)
    assert abs(authored_result.observe(authored_model.observable("definition.energy"), quadrature_points=2).value + 3/256) <= 1e-10
try:
    q.equation(energy, 0)
except TypeError:
    pass
else:
    raise AssertionError("Observable entered ordinary expression algebra")
foreign_module = eqiora.Module("foreign")
foreign_component = foreign_module.component("Foreign")
foreign = foreign_component.parameter("foreign", value_type=eqiora.ValueType.real())
try:
    q.variation(energy, wrt=foreign, direction=w, holding=(k,f))
except q.ModuleError:
    pass
else:
    raise AssertionError("foreign variation binding was accepted")
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
    eqiora.Plan.from_bytes(plan_bytes.replace(b"resolved-common-plan/v6", b"resolved-common-plan/v1"))
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
    (source.replace("zero_on x_lower, x_upper, y_lower, y_upper", ""), "zero_on"),
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
