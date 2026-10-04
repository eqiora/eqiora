use eqiora::EntityKind;
use eqiora::kernel::KernelNode;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyModule};

#[test]
fn python_rational_dimensions_preserve_exact_equality_and_native_authoring() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let native = public_module(py)?;
        let locals = PyDict::new(py);
        locals.set_item("eqiora", &native)?;
        py.run(
            c_str!(
                r#"
from fractions import Fraction
Dimension = eqiora.Dimension
wave = Dimension(length=Fraction(-1, 2))
assert wave == Dimension(length=Fraction(-2, 4))
assert wave != Dimension(length=-1)
assert len({wave, Dimension(length=Fraction(1, -2))}) == 1
assert wave.exponents == (0, Fraction(-1, 2), 0, 0, 0, 0, 0)
assert all(isinstance(value, Fraction) for value in wave.exponents)
assert eval(repr(wave)) == wave
assert Dimension().exponents == (0, 0, 0, 0, 0, 0, 0)
assert Dimension(length=2147483647).exponents[1] == 2147483647
for invalid in [0.5, 1.0, True, (1, 2), None, 2147483648, -2147483648,
                Fraction(1, 2147483648)]:
    try:
        Dimension(length=invalid)
    except (TypeError, ValueError, OverflowError):
        pass
    else:
        raise AssertionError(f"accepted invalid exponent: {invalid!r}")
field = eqiora.Field("psi", role=eqiora.FieldRole.Variable, value_type=eqiora.ValueType.real(wave))
assert field.dimension == wave
balance = eqiora.Relation("balance", equations=[(field, 0)])
model = eqiora.compile(source=eqiora.Module("wave", field, balance))
"#
            ),
            Some(&locals),
            None,
        )
    })
}

#[test]
fn python_native_parameters_lower_complex_scalars_and_shaped_zero() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let native = public_module(py)?;
        let locals = PyDict::new(py);
        locals.set_item("eqiora", &native)?;
        py.run(
            c_str!(
                r#"
scalar = eqiora.ValueType.complex(eqiora.Dimension(length=1))
for value_type, value in [(scalar, 2.0), (eqiora.ValueType.array(scalar, 3), 0.0)]:
    coefficient = eqiora.Parameter("coefficient", value_type=value_type, value=value)
    field = eqiora.Field("state", role=eqiora.FieldRole.Variable, value_type=value_type)
    relation = eqiora.Relation("balance", equations=[(field - coefficient, 0)])
    model = eqiora.compile(source=eqiora.Module("typed_parameter", coefficient, field, relation))
    assert coefficient.value_type == value_type
    assert eqiora.Model.from_bytes(model.to_bytes()).digest == model.digest
"#
            ),
            Some(&locals),
            None,
        )
    })
}

#[test]
fn python_physical_domains_preserve_complete_scalar_types() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let native = public_module(py)?;
        let locals = PyDict::new(py);
        locals.set_item("eqiora", &native)?;
        py.run(c_str!(r#"
voltage = eqiora.ValueType.complex(eqiora.Dimension(mass=1, length=2, time=-3, current=-1))
current = eqiora.ValueType.complex(eqiora.Dimension(current=1))
domain = eqiora.PhysicalDomain("electrical", across_name="voltage", across_type=voltage, through_name="current", through_type=current)
assert domain.across_type == voltage
assert domain.through_type == current
left = eqiora.ConservingPort("left", domain=domain)
right = eqiora.ConservingPort("right", domain=domain)
relation = eqiora.Relation("balance", equations=[(residual, 0) for residual in ([eqiora.across(left) - eqiora.across(right), eqiora.through(left) + eqiora.through(right)])])
model = eqiora.compile(source=eqiora.Module("complex_physical", domain, left, right, relation, eqiora.connect(left, right)))
assert eqiora.Model.from_bytes(model.to_bytes()).digest == model.digest
"#), Some(&locals), None)
    })
}

#[test]
fn python_native_modeling_crosses_only_shared_rust_contracts() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let native = public_module(py)?;
        let module = &native;
        let locals = PyDict::new(py);
        locals.set_item("eqiora", module)?;

        py.run(
            c_str!(
                r#"
x = eqiora.Field("x", role=eqiora.FieldRole.State)
rate = eqiora.Parameter(
    "rate",
    value_type=eqiora.ValueType.real(eqiora.Dimension(time=-1)),
    value=1.0,
)
flow = eqiora.Relation(
    "flow",
    equations=[(eqiora.derivative(x) + rate * x, 0)],
)
scalar_model = eqiora.compile(source=eqiora.Module("decay", x, rate, flow, eqiora.Initial((x, 1.0))))

voltage = eqiora.Dimension(mass=1, length=2, time=-3, current=-1)
current = eqiora.Dimension(current=1)
electrical = eqiora.PhysicalDomain(
    "electrical",
    across_type=eqiora.ValueType.real(voltage),
    across_name="voltage",
    through_name="current",
    through_type=eqiora.ValueType.real(current),
)
left = eqiora.ConservingPort("left", domain=electrical)
right = eqiora.ConservingPort("right", domain=electrical)
tap = eqiora.ConservingPort("tap", domain=electrical)
component = eqiora.Relation(
    "component",
    equations=[(residual, 0) for residual in ([
        eqiora.across(left) - eqiora.across(tap),
        eqiora.through(right) + eqiora.through(tap),
    ])],
)
connection = eqiora.connect(left, right, tap)
physical_model = eqiora.compile(source=eqiora.Module(
    "physical_pair",
    electrical,
    left,
    right,
    tap,
    component,
    connection,
))

interval = eqiora.Domain.box("interval", (0.0, 1.0))
lower_end = interval.boundary(
    "lower_end",
    axis=0,
    side=eqiora.BoundarySide.Lower,
)
upper_end = interval.boundary(
    "upper_end",
    axis=0,
    side=eqiora.BoundarySide.Upper,
)

potential = eqiora.Field(
    "potential",
        role=eqiora.FieldRole.Variable,
    domain=interval,

)
source_scale = eqiora.Parameter(
    "source_scale",
    value_type=eqiora.ValueType.real(eqiora.Dimension(length=-2)),
    value=1.0,
)
spatial_model = eqiora.compile(source=eqiora.Module(
    "native_poisson",
    source_scale,
    upper_end,
    interval,
    potential,

    lower_end,
    eqiora.Relation(
        "upper_value",
        domain=upper_end,
        equations=[(eqiora.trace(potential), 0)],
    ),
    eqiora.Relation(
        "balance",
        domain=interval,
        equations=[(-eqiora.div(eqiora.grad(potential)) - source_scale, 0)],
    ),
    eqiora.Relation(
        "lower_value",
        domain=lower_end,
        equations=[(eqiora.trace(potential), 0)],
    ),
))
"#
            ),
            None,
            Some(&locals),
        )?;

        let scalar = replay_python_model(&locals, "scalar_model");
        let source_scalar = eqiora::api::ModelDocument::compile(
            "source-decay.eqi",
            r#"
model source_decay() {
  parameter coefficient: 1 / s = 1;
  state state: 1;
  initial { state = 1; }
  relation balance { derivative(state) + coefficient * state = 0; }
}
"#,
        )
        .unwrap();
        assert_ne!(scalar.digest().unwrap(), source_scalar.digest().unwrap());
        assert_eq!(
            scalar.structural_fingerprint().unwrap(),
            source_scalar.structural_fingerprint().unwrap()
        );
        assert!(scalar.structurally_equivalent(&source_scalar).unwrap());
        let physical = replay_python_model(&locals, "physical_model");
        let source_physical = eqiora::api::ModelDocument::compile(
            "source-physical.eqi",
            r#"
model source_physical() {
  domain pin = scalar_physical(across voltage: kg * m ^ 2 / (s ^ 3 * A), through current: A);
  port a: pin;
  port b: pin;
  port c: pin;
  relation law {
    a.voltage - c.voltage = 0;
    b.current + c.current = 0;
  }
  connect a, b, c;
}
"#,
        )
        .unwrap();
        assert_ne!(
            physical.digest().unwrap(),
            source_physical.digest().unwrap()
        );
        assert_eq!(
            physical.structural_fingerprint().unwrap(),
            source_physical.structural_fingerprint().unwrap()
        );
        assert!(physical.structurally_equivalent(&source_physical).unwrap());
        let program = physical.program();
        for (kind, expected) in [
            (EntityKind::Domain, 1),
            (EntityKind::Port, 3),
            (EntityKind::Relation, 1),
            (EntityKind::Connection, 1),
        ] {
            assert_eq!(
                program
                    .nodes()
                    .filter(|node| node.id().kind() == kind)
                    .count(),
                expected,
                "Python physical authoring lost a {kind:?} declaration"
            );
        }
        let relation = program
            .nodes()
            .find_map(|node| match node {
                KernelNode::Relation(relation) => Some(relation),
                _ => None,
            })
            .expect("the replayed physical Model must retain its Relation");
        assert_eq!(
            relation.equation_sides().len(),
            2,
            "ordered equation meaning was lost during Python artifact replay"
        );

        let spatial = replay_python_model(&locals, "spatial_model");
        let source_spatial = eqiora::api::ModelDocument::compile(
            "python-native-poisson.eqi",
            include_str!("../../../verify/interfaces/python-native-modeling/models/poisson.eqi"),
        )
        .unwrap();
        assert_ne!(spatial.digest().unwrap(), source_spatial.digest().unwrap());
        assert_eq!(
            spatial.structural_fingerprint().unwrap(),
            source_spatial.structural_fingerprint().unwrap()
        );
        assert_eq!(
            locals
                .get_item("component")?
                .unwrap()
                .getattr("equations")?
                .len()?,
            2
        );
        assert_eq!(
            locals.get_item("flow")?.unwrap().repr()?.to_str()?,
            "Relation(\"flow\", activation='continuous')"
        );

        assert_rejected_without_model(
            py,
            module,
            c_str!(
                r#"
included = eqiora.Field("x", role=eqiora.FieldRole.Variable)
same_named_foreign = eqiora.Field("x", role=eqiora.FieldRole.Variable)
relation = eqiora.Relation("flow", equations=[(same_named_foreign, 0)])
rejected_model = eqiora.compile(source=eqiora.Module("foreign_symbol", included, relation))
"#
            ),
            "EQ0603",
            &["foreign_symbol", "flow"],
            None,
        )?;

        assert_rejected_without_model(
            py,
            module,
            c_str!(
                r#"
temperature = eqiora.Field(
    "temperature",
        role=eqiora.FieldRole.Variable,
    value_type=eqiora.ValueType.real(eqiora.Dimension(temperature=1)),

)
duration = eqiora.Parameter(
    "duration",
    value_type=eqiora.ValueType.real(eqiora.Dimension(time=1)),
    value=1.0,
)
invalid = eqiora.Relation("invalid", equations=[(temperature + duration, 0)])
rejected_model = eqiora.compile(source=eqiora.Module("dimension_mismatch", temperature, duration, invalid))
"#
            ),
            "EQ0603",
            &["dimension_mismatch", "invalid"],
            None,
        )?;

        assert_rejected_without_model(
            py,
            module,
            c_str!(
                r#"
voltage = eqiora.Dimension(mass=1, length=2, time=-3, current=-1)
current = eqiora.Dimension(current=1)
left_domain = eqiora.PhysicalDomain(
    "electrical_left",
    across_name="voltage",
    through_name="current",
    across_type=eqiora.ValueType.real(voltage),
    through_type=eqiora.ValueType.real(current),
)
equal_but_foreign = eqiora.PhysicalDomain(
    "electrical_foreign",
    across_name="voltage",
    through_name="current",
    across_type=eqiora.ValueType.real(voltage),
    through_type=eqiora.ValueType.real(current),
)
left = eqiora.ConservingPort("left", domain=left_domain)
foreign = eqiora.ConservingPort("foreign", domain=equal_but_foreign)
bad_connection = eqiora.connect(left, foreign)
rejected_model = eqiora.compile(source=eqiora.Module(
    "nominal_domain_mismatch",
    left_domain,
    equal_but_foreign,
    left,
    foreign,
    eqiora.Relation("left_owner", equations=[(eqiora.across(left), 0)]),
    eqiora.Relation("foreign_owner", equations=[(eqiora.across(foreign), 0)]),
    bad_connection,
))
"#
            ),
            "EQ0603",
            &["nominal_domain_mismatch"],
            Some("exact same"),
        )?;

        assert_rejected_without_model(
            py,
            module,
            c_str!(
                r#"
voltage = eqiora.Dimension(mass=1, length=2, time=-3, current=-1)
current = eqiora.Dimension(current=1)
electrical = eqiora.PhysicalDomain(
    "electrical",
    across_name="voltage",
    through_name="current",
    across_type=eqiora.ValueType.real(voltage),
    through_type=eqiora.ValueType.real(current),
)
left = eqiora.ConservingPort("left", domain=electrical)
omitted = eqiora.ConservingPort("omitted", domain=electrical)
bad_connection = eqiora.connect(left, omitted)
rejected_model = eqiora.compile(source=eqiora.Module(
    "omitted_connection_member",
    electrical,
    left,
    eqiora.Relation("left_owner", equations=[(eqiora.across(left), 0)]),
    bad_connection,
))
"#
            ),
            "EQ0603",
            &["omitted_connection_member"],
            None,
        )?;

        assert_rejected_without_model(
            py,
            module,
            c_str!(
                r#"
included = eqiora.Domain.box("interval", (0.0, 1.0))
same_named_foreign = eqiora.Domain.box("interval", (0.0, 1.0))

field = eqiora.Field(
    "u",
        role=eqiora.FieldRole.Variable,
    domain=same_named_foreign,

)
rejected_model = eqiora.compile(source=eqiora.Module("foreign_domain", included, field))
"#
            ),
            "EQ0603",
            &["foreign_domain", "u"],
            Some("foreign or omitted Domain"),
        )?;

        assert_rejected_without_model(
            py,
            module,
            c_str!(
                r#"
included = eqiora.Domain.box("interval", (0.0, 1.0))
same_named_foreign = eqiora.Domain.box("interval", (0.0, 1.0))
relation = eqiora.Relation(
    "balance",
    domain=same_named_foreign,
    equations=[(1.0, 0)],
)
rejected_model = eqiora.compile(source=eqiora.Module("foreign_relation_domain", included, relation))
"#
            ),
            "EQ0603",
            &["foreign_relation_domain", "balance"],
            Some("foreign or omitted Domain"),
        )?;

        assert_rejected_without_model(
            py,
            module,
            c_str!(
                r#"
included = eqiora.Domain.box("interval", (0.0, 1.0))
same_named_foreign = eqiora.Domain.box("interval", (0.0, 1.0))
lower = same_named_foreign.boundary(
    "lower",
    axis=0,
    side=eqiora.BoundarySide.Lower,
)
rejected_model = eqiora.compile(source=eqiora.Module("foreign_parent", included, lower))
"#
            ),
            "EQ0603",
            &["foreign_parent", "lower"],
            Some("foreign or omitted parent Domain"),
        )?;

        assert_rejected_without_model(
            py,
            module,
            c_str!(
                r#"
interval = eqiora.Domain.box("interval", (0.0, 1.0))

field = eqiora.Field("u", role=eqiora.FieldRole.Variable, domain=interval)
invalid = eqiora.Relation(
    "invalid",
    domain=interval,
    equations=[(eqiora.trace(field), 0)],
)
rejected_model = eqiora.compile(source=eqiora.Module(
    "support_mismatch",
    interval,

    field,
    invalid,
))
"#
            ),
            "EQ0603",
            &["support_mismatch", "invalid"],
            Some("trace/normal operator requires an AppliesOn boundary Domain"),
        )?;

        Ok(())
    })
}

fn replay_python_model(locals: &Bound<'_, PyDict>, name: &str) -> eqiora::api::ModelDocument {
    let bytes: Vec<u8> = locals
        .get_item(name)
        .unwrap()
        .unwrap()
        .call_method0("to_bytes")
        .unwrap()
        .extract()
        .unwrap();
    let document = eqiora::api::ModelDocument::replay(&bytes).unwrap();
    assert_eq!(document.canonical_json().unwrap(), bytes);
    document
}

fn assert_rejected_without_model(
    py: Python<'_>,
    module: &Bound<'_, PyModule>,
    code: &std::ffi::CStr,
    expected_code: &str,
    expected_path_prefix: &[&str],
    expected_message_fragment: Option<&str>,
) -> PyResult<()> {
    let locals = PyDict::new(py);
    locals.set_item("eqiora", module)?;
    let error = py
        .run(code, None, Some(&locals))
        .expect_err("the Python native definition must fail closed");
    assert!(error.is_instance(py, &module.getattr("ValidationError")?));
    assert!(
        !locals.contains("rejected_model")?,
        "a rejected definition exposed a partial Model"
    );
    let diagnostics = error.value(py).getattr("diagnostics")?;
    assert!(diagnostics.len()? > 0);
    let diagnostic = diagnostics.get_item(0)?;
    assert_eq!(
        diagnostic.getattr("code")?.extract::<String>()?,
        expected_code
    );
    let graph_path = diagnostic
        .getattr("graph_path")?
        .extract::<Option<Vec<String>>>()?
        .expect("native construction diagnostics must retain a declaration path");
    assert!(
        graph_path.len() >= expected_path_prefix.len()
            && graph_path
                .iter()
                .zip(expected_path_prefix)
                .all(|(actual, expected)| actual == expected),
        "unexpected graph path {graph_path:?}"
    );
    if let Some(expected) = expected_message_fragment {
        let message = diagnostic.getattr("message")?.extract::<String>()?;
        assert!(
            message.contains(expected),
            "diagnostic {message:?} did not prove the intended falsifier"
        );
    }
    assert!(diagnostic.getattr("source_span")?.is_none());
    Ok(())
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
fn python_observables_preserve_typed_source_identity_and_exact_references() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let native = public_module(py)?;
        let locals = PyDict::new(py);
        locals.set_item("eqiora", &native)?;
        py.run(
            c_str!(
                r#"
kind = eqiora.ValueType.real(eqiora.Dimension())
x = eqiora.Field("x", value_type=kind, role=eqiora.FieldRole.Variable)
output = eqiora.Observable("double", value_type=kind, expression=x + x)
assert output.name == "double"
assert not hasattr(eqiora, "boundary_integral")
body = eqiora.Domain.box("body", (0.0, 2.0))
measure = eqiora.measure(body)
length = eqiora.ValueType.real(eqiora.Dimension(length=1))
spatial = eqiora.Observable("total", value_type=length, expression=eqiora.integral(3, measure))
spatial_model = eqiora.compile(source=eqiora.Module("Spatial", x, body,
    eqiora.Relation("law", equations=[(x, 2)]), spatial))
foreign_body = eqiora.Domain.box("body", (0.0, 2.0))
try:
    eqiora.Module("Spatial", x, body, eqiora.Observable("total", value_type=length,
        expression=eqiora.integral(3, eqiora.measure(foreign_body))))
except eqiora.ValidationError as error:
    assert "foreign or omitted Domain" in str(error)
else:
    raise AssertionError("accepted same-name foreign measure Domain")

authored = eqiora.Module("authored")
component = authored.model("M")
field = component.field("x", value_type=kind, role=eqiora.FieldRole.Variable)
component.relation("law", eqiora.lang.equation(field, 2))
assert isinstance(component.observable("double", field + field, value_type=kind), eqiora.lang.Observable)
builder_model = eqiora.compile(source=authored, entry="M")
model = eqiora.compile(source=eqiora.Module("M", x,
    eqiora.Relation("law", equations=[(x, 2)]), output))
foreign = eqiora.Field("x", value_type=kind, role=eqiora.FieldRole.Variable)
try:
    eqiora.Module("M", x, eqiora.Observable("double", value_type=kind, expression=foreign))
except eqiora.ValidationError as error:
    assert "foreign or omitted Field" in str(error)
else:
    raise AssertionError("accepted same-name foreign Field")
try:
    x + output
except TypeError:
    pass
else:
    raise AssertionError("Observable became an expression symbol")
"#
            ),
            Some(&locals),
            None,
        )?;
        let native = replay_python_model(&locals, "model");
        let builder = replay_python_model(&locals, "builder_model");
        assert_eq!(
            native.structural_fingerprint().unwrap(),
            builder.structural_fingerprint().unwrap()
        );
        let source = eqiora::api::ModelDocument::compile(
            "observable.eqi",
            r#"
model M() {
    variable x: 1;
    relation law { x = 2; }
    observable double: 1 = x + x;
}
"#,
        )
        .unwrap();
        assert_eq!(
            native.structural_fingerprint().unwrap(),
            source.structural_fingerprint().unwrap()
        );
        Ok(())
    })
}
