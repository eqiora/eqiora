use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyModule};

const SHIPPED_SOURCE: &str = include_str!("../../../examples/steady-flow-past-cylinder.eqi");

#[test]
fn python_occurrence_labels_use_the_full_native_model_catalog() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let module = public_module(py)?;
        let locals = PyDict::new(py);
        locals.set_item("eqiora", module)?;
        py.run(c_str!(r"
source = eqiora.Module('main')
model = source.model('Main')
left = model.field('left', role=eqiora.FieldRole.Variable, value_type=eqiora.ValueType.real())
right = model.field('right', role=eqiora.FieldRole.Variable, value_type=eqiora.ValueType.real())
model.relation('left_law', eqiora.lang.equation(left, 0))
model.relation('right_law', eqiora.lang.equation(right, 0))
model.set_notation('left', eqiora.lang.Notation(r'@{x_i}'))
model.set_notation('right', eqiora.lang.Notation(r'@{\mathbf{x_i}}'))
compiled = eqiora.compile(source=source)
full = compiled.notation_labels()
assert len(full) == 2
assert all(isinstance(entry, eqiora.QuantityLabel) for entry in full)
for profile in ['latex', 'mathml', 'unicode', 'plain', 'speech']:
    entries = compiled.notation_labels(profile)
    assert len({entry.label for entry in entries}) == 2
    view = compiled.notation_labels(profile, identities=[full[0].identity]*2)
    assert len(view) == 1
    assert view[0].label == next(entry.label for entry in entries if entry.identity == full[0].identity)
reopened = eqiora.Model.from_bytes(compiled.to_bytes())
assert reopened.digest == compiled.digest
assert len(reopened.notation_labels()) == 2
assert all(entry.definition_span is None for entry in reopened.notation_labels())
"), Some(&locals), Some(&locals))?;
        Ok(())
    })
}

#[test]
fn python_notation_uses_the_typed_native_parser_and_preserves_compiled_identity() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let module = public_module(py)?;
        let locals = PyDict::new(py);
        locals.set_item("eqiora", module)?;
        py.run(
            c_str!(
                r"
q = eqiora.lang
def make(decorated):
    source = eqiora.Module('main')
    model = source.model('Main', doc='Physical model.')
    value = model.field('value', role=eqiora.FieldRole.Variable, value_type=eqiora.ValueType.real())
    model.relation('law', eqiora.lang.equation(value, 0))
    if decorated:
        source.set_notation('Main', q.Notation(r'@{\mathcal{M}}'))
        model.set_notation('value', q.Notation(r'@{\hat{x}_{ij}}'))
    return source
plain, annotated = make(False), make(True)
assert r'value @{\hat{x}_{i j}}: 1' in annotated.to_eqi()
assert 'value = 0;' in annotated.to_eqi()
assert eqiora.compile(source=plain).digest == eqiora.compile(source=annotated).digest
for island in [r'@{\input{x}}', r'@{\text{words}}', '@{$x$}', '@{x_i_i}']:
    try:
        q.Notation(island)
    except ValueError:
        pass
    else:
        raise AssertionError(island)
"
            ),
            Some(&locals),
            Some(&locals),
        )?;
        let text: String = py
            .eval(c_str!("annotated.to_eqi()"), None, Some(&locals))?
            .extract()?;
        let parsed = eqiora::language::parse("python-notation.eqi", &text)
            .into_document()
            .unwrap();
        assert_eq!(parsed.notations().len(), 2);
        assert_eq!(
            eqiora::language::format(
                &eqiora::language::parse("formatted.eqi", &eqiora::language::format(&parsed))
                    .into_document()
                    .unwrap()
            ),
            eqiora::language::format(&parsed)
        );
        Ok(())
    })
}

#[test]
fn python_language_source_round_trips_through_the_existing_compiler() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let module = public_module(py)?;
        let locals = PyDict::new(py);
        locals.set_item("eqiora", module)?;
        locals.set_item("shipped_source", SHIPPED_SOURCE)?;
        py.run(
            c_str!(
                r#"
import pathlib
import tempfile

q = eqiora.lang
u = eqiora.units

namespace_probe = eqiora.Module('main')
probe_component = namespace_probe.component("ScalarMath")
probe_body = probe_component.volume("body", dimensions=1)
probe_value = probe_component.field("value", role=eqiora.FieldRole.Variable, on=probe_body, value_type=eqiora.ValueType.real())
probe_component.relation("law", eqiora.lang.equation(probe_value - q.math.sin(q.math.pi), 0), on=probe_body)
assert "math.sin(math.pi)" in namespace_probe.to_eqi()
assert not hasattr(q, "sin")


def cylinder_source(*, doc="Equations-only steady incompressible flow component.", velocity_extent=2):
    source = eqiora.Module('main')
    stokes = source.component("SteadyFlowPastCylinder", doc=doc)
    fluid = stokes.volume("fluid", dimensions=2)
    inlet = stokes.boundary("inlet", parent=fluid)
    outlet = stokes.boundary("outlet", parent=fluid)
    walls = stokes.boundary("walls", parent=fluid)
    cylinder = stokes.boundary("cylinder", parent=fluid)

    dynamic_viscosity = stokes.parameter("dynamic_viscosity", value_type=eqiora.ValueType.real(eqiora.Dimension(mass=1, length=-1, time=-1)))
    zero_pressure = stokes.parameter("zero_pressure", value_type=eqiora.ValueType.real(eqiora.Dimension(mass=1, length=-1, time=-2)))
    inlet_speed = stokes.parameter("inlet_speed", value_type=eqiora.ValueType.real(eqiora.Dimension(length=1, time=-1)))
    channel_height = stokes.parameter("channel_height", value_type=eqiora.ValueType.real(eqiora.Dimension(length=1)))

    velocity = stokes.field(
        "velocity", role=eqiora.FieldRole.Variable, on=fluid, value_type=(
            eqiora.ValueType.vector(eqiora.ValueType.real(eqiora.Dimension(length=1, time=-1)), velocity_extent)
            if velocity_extent is not None else eqiora.ValueType.real(eqiora.Dimension(length=1, time=-1))
        )
    )
    pressure = stokes.field(
        "pressure", role=eqiora.FieldRole.Variable, on=fluid, value_type=eqiora.ValueType.real(eqiora.Dimension(mass=1, length=-1, time=-2))
    )
    force_potential = stokes.field(
        "force_potential", role=eqiora.FieldRole.Variable, on=fluid, value_type=eqiora.ValueType.real(eqiora.Dimension(mass=1, length=-1, time=-2))
    )
    inlet_profile = stokes.field(
        "inlet_profile", role=eqiora.FieldRole.Variable, on=fluid, value_type=eqiora.ValueType.real(eqiora.Dimension(length=1, time=-1))
    )

    stokes.relation("force_definition", eqiora.lang.equation(force_potential - zero_pressure, 0), on=fluid)
    stokes.relation("inlet_profile_definition", eqiora.lang.equation(inlet_profile
            - 4
            * inlet_speed
            * q.coordinate(1)
            * (channel_height - q.coordinate(1))
            / channel_height**2, 0), on=fluid)
    stress = (
        2 * dynamic_viscosity * q.symmetric_part(q.grad(velocity))
        - q.isotropic_lift(pressure)
    )
    stokes.relation("momentum", eqiora.lang.equation(-q.div(stress) - q.grad(force_potential), 0), on=fluid, doc="Steady Stokes momentum balance.")
    stokes.relation("incompressibility", eqiora.lang.equation(q.div(velocity), 0), on=fluid)
    stokes.relation("inlet_velocity", eqiora.lang.equation(q.trace(velocity) + q.normal(q.isotropic_lift(inlet_profile)), 0), on=inlet)
    stokes.relation("outlet_traction", eqiora.lang.equation(q.normal(stress), 0), on=outlet)
    stokes.relation("wall_velocity", eqiora.lang.equation(q.trace(velocity), 0), on=walls)
    stokes.relation("cylinder_velocity", eqiora.lang.equation(q.trace(velocity), 0), on=cylinder)
    return source


first = cylinder_source()
second = cylinder_source()
assert first.to_eqi() == second.to_eqi()
assert "/// Equations-only steady incompressible flow component." in first.to_eqi()
assert "relation momentum on fluid" in first.to_eqi()

graph = eqiora.geometry.GeometryGraph()
rectangle = graph.rectangle(x_bounds=(0.0, 2.2), y_bounds=(0.0, 0.41))
circle = graph.circle(center=(0.2, 0.2), radius=0.05)
fluid = graph.subtract(rectangle, circle)
geometry = graph.build(fluid, named_topology={
    "fluid": fluid.region,
    "inlet": rectangle.boundaries[0],
    "outlet": rectangle.boundaries[1],
    "walls": rectangle.boundaries[2:],
    "cylinder": circle.boundaries[0],
})
parameters = {
    "dynamic_viscosity": 0.001,
    "zero_pressure": 0.0,
    "inlet_speed": 0.3,
    "channel_height": 0.41,
}

direct_source = cylinder_source()
direct = eqiora.compile(source=direct_source, geometry=geometry, entry='SteadyFlowPastCylinder', bindings={'fluid': geometry.selection('fluid'), 'inlet': (geometry.selection('inlet'), geometry.selection('fluid')), 'outlet': (geometry.selection('outlet'), geometry.selection('fluid')), 'walls': (geometry.selection('walls'), geometry.selection('fluid')), 'cylinder': (geometry.selection('cylinder'), geometry.selection('fluid')), **parameters})
with tempfile.TemporaryDirectory() as directory:
    path = pathlib.Path(directory) / "steady-flow-past-cylinder.eqi"
    direct_source.write_eqi(path)
    assert path.read_text(encoding="utf-8") == direct_source.to_eqi()
    emitted = eqiora.compile(path=path, geometry=geometry, entry='SteadyFlowPastCylinder', bindings={'fluid': geometry.selection('fluid'), 'inlet': (geometry.selection('inlet'), geometry.selection('fluid')), 'outlet': (geometry.selection('outlet'), geometry.selection('fluid')), 'walls': (geometry.selection('walls'), geometry.selection('fluid')), 'cylinder': (geometry.selection('cylinder'), geometry.selection('fluid')), **parameters})
assert direct.structural_fingerprint == emitted.structural_fingerprint
shipped = eqiora.compile(source=shipped_source, filename='steady-flow-past-cylinder.eqi', geometry=geometry, entry='SteadyFlowPastCylinder', bindings={'fluid': geometry.selection('fluid'), 'inlet': (geometry.selection('inlet'), geometry.selection('fluid')), 'outlet': (geometry.selection('outlet'), geometry.selection('fluid')), 'walls': (geometry.selection('walls'), geometry.selection('fluid')), 'cylinder': (geometry.selection('cylinder'), geometry.selection('fluid')), **parameters})
assert direct.structural_fingerprint == shipped.structural_fingerprint

other_comments = eqiora.compile(source=cylinder_source(doc='Different presentation-only documentation.'), geometry=geometry, entry='SteadyFlowPastCylinder', bindings={'fluid': geometry.selection('fluid'), 'inlet': (geometry.selection('inlet'), geometry.selection('fluid')), 'outlet': (geometry.selection('outlet'), geometry.selection('fluid')), 'walls': (geometry.selection('walls'), geometry.selection('fluid')), 'cylinder': (geometry.selection('cylinder'), geometry.selection('fluid')), **parameters})
assert direct.digest == other_comments.digest

try:
    direct_source.component("Second")
except q.ModuleError:
    pass
else:
    raise AssertionError("an emitted Module remained mutable")

left = eqiora.Module('main')
left_component = left.component("Left")
left_volume = left_component.volume("left", dimensions=2)
left_value = left_component.field("value", role=eqiora.FieldRole.Variable, on=left_volume, value_type=eqiora.ValueType.real(eqiora.Dimension(length=1)))
right = eqiora.Module('main')
right_component = right.component("Right")
right_volume = right_component.volume("right", dimensions=2)
right_value = right_component.field("value", role=eqiora.FieldRole.Variable, on=right_volume, value_type=eqiora.ValueType.real(eqiora.Dimension(length=1)))
for invalid in (
    lambda: left_value + right_value,
    lambda: left_component.boundary("foreign_parent", parent=right_volume),
    lambda: left_component.field("wrong_support", role=eqiora.FieldRole.Variable, on=right_volume, value_type=eqiora.ValueType.real(eqiora.Dimension(length=1))),
    lambda: left_component.field("value", role=eqiora.FieldRole.Variable, on=left_volume, value_type=eqiora.ValueType.real(eqiora.Dimension(length=1))),
    lambda: left_value + float("nan"),
):
    try:
        invalid()
    except q.ModuleError:
        pass
    else:
        raise AssertionError("invalid Module authoring input was accepted")

try:
    eqiora.Module('main').component("not-valid")
except q.ModuleError:
    pass
else:
    raise AssertionError("an invalid declaration name was accepted")

deep = q.coordinate(0)
try:
    for _ in range(100):
        deep = q.grad(deep)
except q.ModuleError:
    pass
else:
    raise AssertionError("an excessive expression depth was accepted")

with tempfile.TemporaryDirectory() as directory:
    target = pathlib.Path(directory) / "target.eqi"
    target.write_text("preserved", encoding="utf-8")
    link = pathlib.Path(directory) / "link.eqi"
    link.symlink_to(target)
    try:
        cylinder_source().write_eqi(link)
    except ValueError:
        pass
    else:
        raise AssertionError("a symlink output path was accepted")
    assert target.read_text(encoding="utf-8") == "preserved"

try:
    eqiora.compile(source=cylinder_source(velocity_extent=None), geometry=geometry, entry='SteadyFlowPastCylinder', bindings={'fluid': geometry.selection('fluid'), 'inlet': (geometry.selection('inlet'), geometry.selection('fluid')), 'outlet': (geometry.selection('outlet'), geometry.selection('fluid')), 'walls': (geometry.selection('walls'), geometry.selection('fluid')), 'cylinder': (geometry.selection('cylinder'), geometry.selection('fluid')), **parameters})
except eqiora.ValidationError as error:
    assert error.diagnostics
    assert any(
        diagnostic.source_span is None
        and diagnostic.graph_path
        for diagnostic in error.diagnostics
    )
else:
    raise AssertionError("a shape-invalid generated source passed the canonical compiler")
"#
            ),
            Some(&locals),
            Some(&locals),
        )
    })
}

#[test]
fn python_sampled_signature_round_trips_through_the_actual_parser() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals.set_item("eqiora", public_module(py)?)?;
        py.run(c_str!(r#"
q = eqiora.lang
source = eqiora.Module('main')
model = source.model("Sampled")
tick = model.clock_requirement("tick")
drive = model.input("drive", value_type=eqiora.ValueType.real(), at=tick)
observed = model.output("observed", value_type=eqiora.ValueType.real(), at=tick)
memory = model.field("memory", value_type=eqiora.ValueType.real(), role=eqiora.FieldRole.State, at=tick)
model.initial((q.pre(memory), 0))
model.relation("update", eqiora.lang.equation(q.next(memory), q.pre(memory) + drive), at=tick)
model.relation("observe", eqiora.lang.equation(observed, q.pre(memory)), at=tick)
text = source.to_eqi()
"#), Some(&locals), Some(&locals))?;
        let text: String = locals
            .get_item("text")?
            .expect("authored Module")
            .extract()?;
        let parsed = eqiora::language::parse("sampled.eqi", &text);
        assert!(
            parsed.diagnostics().is_empty(),
            "{:?}",
            parsed.diagnostics()
        );
        let formatted = eqiora::language::format(parsed.document().expect("parsed Module"));
        let reparsed = eqiora::language::parse("formatted.eqi", &formatted);
        assert!(
            reparsed.diagnostics().is_empty(),
            "{:?}",
            reparsed.diagnostics()
        );
        assert_eq!(
            formatted,
            eqiora::language::format(reparsed.document().unwrap())
        );
        Ok(())
    })
}

#[test]
fn explicit_boundary_selectors_use_shared_compilation_and_lexical_ownership() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals.set_item("eqiora", public_module(py)?)?;
        py.run(c_str!(r#"
q = eqiora.lang
graph = eqiora.geometry.GeometryGraph()
rectangle = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
geometry = graph.build(rectangle, named_topology={"body": rectangle.region, "face": rectangle.boundaries[0], "rest": rectangle.boundaries[1:]})
for operator in (q.trace, q.normal, q.tangential_trace):
    source = eqiora.Module('main')
    component = source.component('BoundaryOps')
    body = component.volume('body', dimensions=2)
    face = component.boundary('face', parent=body)
    value = component.field('u', role=eqiora.FieldRole.Variable, on=body,
                            value_type=eqiora.ValueType.vector(eqiora.ValueType.real(), 2))
    component.relation('bulk', q.equation(value, value), on=body)
    component.relation('surface', q.equation(operator(value, on=face, from_=body), 0), on=face)
    foreign = eqiora.Module('foreign').component('Foreign')
    foreign_body = foreign.volume('body', dimensions=2)
    foreign_face = foreign.boundary('face', parent=foreign_body)
    for selector in ({'on': foreign_face}, {'on': face, 'from_': foreign_body}):
        try:
            operator(value, **selector)
        except q.ModuleError:
            pass
        else:
            raise AssertionError('foreign support was accepted')
    try:
        operator(value, on='face')
    except TypeError:
        pass
    else:
        raise AssertionError('string support was accepted')
    text = source.to_eqi()
    assert 'on = face' in text and 'from = body' in text, text
    compiled = eqiora.compile(source=source, geometry=geometry, entry='BoundaryOps', bindings={
        'body': geometry.selection('body'),
        'face': (geometry.selection('face'), geometry.selection('body')),
    })
    replay = eqiora.Model.from_bytes(compiled.to_bytes())
    assert replay.digest == compiled.digest
"#), Some(&locals), Some(&locals))?;
        Ok(())
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

#[test]
fn python_sine_operator_emission_and_artifact_keep_shared_identity() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let locals = PyDict::new(py);
        locals.set_item("eqiora", public_module(py)?)?;
        py.run(c_str!(r"
q = eqiora.lang
module = eqiora.Module('main')
wave = module.operator('wave', inputs={'x': eqiora.ValueType.real()}, result_type=eqiora.ValueType.real(), body=lambda x: q.math.sin(x))
model = module.model('Wave')
x = model.parameter('x', value_type=eqiora.ValueType.real())
model.set_default(x, 0)
y = model.field('y', role=eqiora.FieldRole.Variable, value_type=eqiora.ValueType.real())
model.relation('value', q.equation(y, wave(x=x)))
native = eqiora.compile(source=module)
parsed = eqiora.compile(source=module.to_eqi())
assert native.digest == parsed.digest
reopened = eqiora.Model.from_bytes(native.to_bytes())
assert reopened.digest == native.digest
"), Some(&locals), Some(&locals))?;
        Ok(())
    })
}
