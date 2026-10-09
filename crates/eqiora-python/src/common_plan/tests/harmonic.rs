use super::*;

#[test]
fn python_harmonic_rc_and_wave_run_replay_and_reconstruct_original_fields() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let native = pyo3::wrap_pymodule!(crate::_eqiora)(py);
        let package_directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../bindings/python/python/eqiora")
            .canonicalize()?;
        let source = include_str!("../../../../../docs/language/harmonic-rc.md")
            .split_once("```eqiora\n")
            .unwrap()
            .1
            .split_once("```")
            .unwrap()
            .0;
        let locals = PyDict::new(py);
        locals.set_item("native", native.bind(py))?;
        locals.set_item("package_directory", package_directory.to_string_lossy())?;
        locals.set_item("source", source)?;
        py.run(c_str!(r#"
import importlib.util, math, pathlib, sys
package_path = pathlib.Path(package_directory)
spec = importlib.util.spec_from_file_location("eqiora", package_path / "__init__.py", submodule_search_locations=[str(package_path)])
eqiora = importlib.util.module_from_spec(spec)
sys.modules["eqiora"] = eqiora
sys.modules["eqiora._eqiora"] = native
spec.loader.exec_module(eqiora)

model = eqiora.compile(source=source, entry="RC")
before = model.to_bytes()
linear = eqiora.solve.Linear(
    algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
    preconditioner=eqiora.solve.Preconditioner.Identity,
    reduction=eqiora.solve.Reduction.Reproducible,
    provider=eqiora.solve.SolverProvider.reference(),
    relative_tolerance=1e-14, absolute_tolerance=1e-16, maximum_iterations=32,
)
plan = eqiora.resolve(model, solve=linear)
for plan in (plan, eqiora.Plan.from_bytes(plan.to_bytes())):
    assert plan.model_digest != model.digest
    assert plan.model.digest == plan.model_digest
    assert plan.harmonic_original_model.digest == model.digest
    assert plan.harmonic_angular_frequency == 1000.
    assert plan.formulation.requested == eqiora.FormulationSelectionMode.Authored
    assert plan.formulation.effective == eqiora.FormulationKind.HarmonicResponse
    assert plan.formulation.requested_source_identity is not None
    mappings = {name: (original, amplitude) for name, original, amplitude in plan.harmonic_amplitudes}
    assert set(mappings) == {"voltage_hat", "current_hat"}
    assert mappings["voltage_hat"][0] == model.field("voltage")
    assert all(amplitude in plan.fields for original, amplitude in mappings.values())
    assert all(original.model_digest == model.digest and amplitude.model_digest == plan.model_digest
               for original, amplitude in mappings.values())
    state = eqiora.State.initial(plan)
    state = eqiora.State.from_bytes(plan, state.to_bytes())
    result = eqiora.run(plan, state=state)
    result = eqiora.Result.from_bytes(plan, result.to_bytes())
    # R*C=0.001 s; Vhat=1/(1-i)=0.5+0.5i, Ihat=-i*omega*C*Vhat.
    for time, voltage, current in ((0., .5, .0005), (math.pi/2000., .5, -.0005),
                                    (math.pi/1000., -.5, -.0005)):
        fields = {field.id: (value, kind) for field, value, kind
                  in result.reconstruct_harmonic_fields(time_seconds=time)}
        for name, expected in (("voltage_hat", voltage), ("current_hat", current)):
            value, kind = fields[mappings[name][0].id]
            assert type(value) is float
            assert abs(value-expected) < 1e-10, (name, time, value)
    for time in (float("nan"), float("inf")):
        try:
            result.reconstruct_harmonic_fields(time_seconds=time)
        except eqiora.ValidationError as error:
            assert "time in seconds" in str(error)
        else:
            raise AssertionError("nonfinite time admitted")
assert model.to_bytes() == before

q, units = eqiora.lang, eqiora.units
volts = eqiora.Dimension(mass=1, length=2, time=-3, current=-1)
amps = eqiora.Dimension(current=1)
for owner_kind in ("model", "component"):
    module = eqiora.Module("main")
    owner = getattr(module, owner_kind)("RC")
    def parameter(name, dimension, value, unit):
        item = owner.parameter(name, value_type=eqiora.ValueType.real(dimension))
        owner.set_default(item, q.quantity(value, unit))
        return item
    resistance = parameter("resistance", eqiora.Dimension(mass=1, length=2, time=-3, current=-2), 1000, units.Ohm)
    capacitance = parameter("capacitance", eqiora.Dimension(mass=-1, length=-2, time=4, current=2), 1e-6, units.F)
    omega = parameter("omega", eqiora.Dimension(time=-1), 1000, units.s**-1)
    source_input = owner.input("source", value_type=eqiora.ValueType.real(volts))
    voltage = owner.field("voltage", value_type=eqiora.ValueType.real(volts), role=eqiora.FieldRole.State)
    current = owner.field("current", value_type=eqiora.ValueType.real(amps), role=eqiora.FieldRole.Variable)
    owner.initial((voltage, q.quantity(0, units.V)))
    network = owner.relation("network", q.equation(source_input-voltage, resistance*current),
                             q.equation(current, capacitance*q.derivative(voltage)))
    for convention, normalization in (("positive_exponential", "peak"), ("negative_exponential", "rms")):
        try:
            owner.harmonic_form("response", [network], angular_frequency=omega,
                convention=convention, normalization=normalization, excitations=[], amplitudes=[])
        except q.ModuleError as error:
            assert "negative_exponential" in str(error) and "peak" in str(error)
        else:
            raise AssertionError("unsupported convention admitted")
    owner.harmonic_form("response", [network], angular_frequency=omega,
        convention="negative_exponential", normalization="peak",
        excitations=[(source_input, q.math.complex(q.quantity(1, units.V), q.quantity(0, units.V)))],
        amplitudes=[("voltage_hat", voltage, eqiora.ValueType.complex(volts)),
                    ("current_hat", current, eqiora.ValueType.complex(amps))])
    models = [eqiora.compile(source=candidate, entry="RC") for candidate in (module, module.to_eqi())]
    assert models[0].digest == models[1].digest
    plans = [eqiora.resolve(candidate, solve=linear) for candidate in models]
    assert plans[0].to_bytes() == plans[1].to_bytes()
    for plan in plans:
        result = eqiora.run(plan, state=eqiora.State.initial(plan))
        original = dict((name, field) for name, field, _ in plan.harmonic_amplitudes)["voltage_hat"]
        fields = {field.id: value for field, value, kind
                  in result.reconstruct_harmonic_fields(time_seconds=math.pi/2000.)}
        assert abs(fields[original.id]-.5) < 1e-10

wave_source = """public component Wave(
 support body:volume(ambient_dimension=1),
 support left:boundary(parent=body), support right:boundary(parent=body),
 input force:1/s^2, input boundary_value:1
) {
 state u:1 on body in smooth;
 initial {u=0; derivative(u)=0[1/s];}
 relation balance on body {
  derivative(derivative(u))+1[1/s]*derivative(u)-div(1[m^2/s^2]*grad(u))=force;
 }
 relation fixed on left {trace(u)=boundary_value;}
 relation flux on right {normal(1[m^2/s^2]*grad(u))=0[m/s^2];}
 form response for balance, fixed, flux {
  harmonic(angular_frequency=1[1/s],convention=negative_exponential,normalization=peak);
  excitation force=math.complex(-1[1/s^2],-3[1/s^2]);
  excitation boundary_value=math.complex(2,1);
  amplitude u_hat:complex<1> on body for u;
 }
}"""
graph = eqiora.geometry.GeometryGraph()
interval = graph.interval(bounds=(0., 1.))
geometry = graph.build(interval, named_topology={"body": interval.region,
    "left": interval.boundaries[0], "right": interval.boundaries[1]})
bindings = {"body": geometry.selection("body"),
    "left": (geometry.selection("left"), geometry.selection("body")),
    "right": (geometry.selection("right"), geometry.selection("body"))}
model = eqiora.compile(source=wave_source, entry="Wave", geometry=geometry, bindings=bindings)
before = model.to_bytes()
mesh = eqiora.meshing.generate(eqiora.meshing.resolve(geometry,
    eqiora.meshing.CartesianMesher(cells=(2,))))
plan = eqiora.resolve(model, mesh=mesh, spatial=eqiora.fem.Q1(), solve=linear)
assert plan.mesh is mesh
for plan in (plan, eqiora.Plan.from_bytes(plan.to_bytes())):
    assert plan.model.digest == plan.model_digest != model.digest
    assert plan.harmonic_original_model.digest == model.digest
    assert plan.harmonic_angular_frequency == 1.
    assert plan.formulation.effective == eqiora.FormulationKind.HarmonicResponse
    ((name, original, amplitude),) = plan.harmonic_amplitudes
    assert name == "u_hat" and original == model.field(original.id)
    assert amplitude in plan.fields
    result = eqiora.run(plan)
    result = eqiora.Result.from_bytes(plan, result.to_bytes())
    # U=2+i, (-omega²-i*omega)U=-1-3i; both boundaries meet the constant solution.
    values = result.output(amplitude).values("vertex")
    assert len(values) == 3 and all(abs(value-(2+1j)) < 1e-10 for value in values)
    for time, expected in ((0., 2.), (math.pi/2., 1.), (math.pi, -2.)):
        values = result.reconstruct_harmonic_field_block(original, time_seconds=time)
        assert len(values) == 3 and all(abs(value-expected) < 1e-10 for value in values)
    try:
        result.reconstruct_harmonic_field_block(amplitude, time_seconds=0.)
    except ValueError as error:
        assert "original Model artifact" in str(error)
    else:
        raise AssertionError("amplitude identity accepted as original Field")
assert model.to_bytes() == before
"#), Some(&locals), None)
    })
}
