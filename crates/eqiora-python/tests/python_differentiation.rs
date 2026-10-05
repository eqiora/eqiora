use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

const POISSON: &str = r#"
public component PythonDifferentiatedPoisson(
  support square: volume(ambient_dimension = 2),
  support x_lower: boundary(parent = square),
  support x_upper: boundary(parent = square),
  support y_lower: boundary(parent = square),
  support y_upper: boundary(parent = square),
  parameter diffusion: 1,
  parameter wave_number: 1 / m,
  parameter source_scale: 1 / m ^ 2,
  parameter boundary_offset: 1
) {

  variable potential: 1 on square;
  relation balance on square {
    -div(diffusion * grad(potential))
      - source_scale * math.sin(wave_number * coordinate(0))
        * math.sin(wave_number * coordinate(1)) = 0;
  }
  relation x_lower_value on x_lower { trace(potential) - boundary_offset = 0; }
  relation x_upper_value on x_upper { trace(potential) - boundary_offset = 0; }
  relation y_lower_value on y_lower { trace(potential) - boundary_offset = 0; }
  relation y_upper_value on y_upper { trace(potential) - boundary_offset = 0; }
}
"#;

#[test]
fn python_differentiable_program_is_exact_paired_and_fail_closed() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let native = pyo3::wrap_pymodule!(_eqiora::_eqiora)(py);
        let package_directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../bindings/python/python/eqiora")
            .canonicalize()?;
        let locals = PyDict::new(py);
        locals.set_item("native", native.bind(py))?;
        locals.set_item("package_directory", package_directory.to_string_lossy())?;
        locals.set_item("source", POISSON)?;
        locals.set_item(
            "complex_tests",
            include_str!("../../../bindings/python/tests/test_complex_differentiation.py"),
        )?;

        py.run(
            c_str!(
                r#"
import importlib.util, pathlib, sys
import numpy as np

package_path = pathlib.Path(package_directory)
spec = importlib.util.spec_from_file_location("eqiora", package_path / "__init__.py", submodule_search_locations=[str(package_path)])
eqiora = importlib.util.module_from_spec(spec)
sys.modules["eqiora"] = eqiora
sys.modules["eqiora._eqiora"] = native
spec.loader.exec_module(eqiora)
exec(complex_tests, globals())
test_common_complex_sensitivity_uses_real_pairing_at_each_parameter_point()

graph = eqiora.geometry.GeometryGraph()
rectangle = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
geometry = graph.build(rectangle, named_topology={
    "square": rectangle.region,
    "x_lower": rectangle.boundaries[0],
    "x_upper": rectangle.boundaries[1],
    "y_lower": rectangle.boundaries[2],
    "y_upper": rectangle.boundaries[3],
})
mesh_request = eqiora.meshing.CartesianMesher(cells=(6, 6))
mesh_plan = eqiora.meshing.resolve(geometry, mesh_request)
mesh = eqiora.meshing.generate(mesh_plan)

direction = np.array([0.7, -0.2, 0.3], dtype=np.float64)
step = 1.0e-5

class DLPackProducer:
    def __init__(self, values, device=(1, 0)):
        self.values = values
        self.device = device
        self.device_queries = 0
        self.exports = []

    def __dlpack_device__(self):
        self.device_queries += 1
        return self.device

    def __dlpack__(self, **kwargs):
        self.exports.append(dict(kwargs))
        return self.values.__dlpack__(**kwargs)

def make_model(values):
    return eqiora.compile(source=source, geometry=geometry, entry='PythonDifferentiatedPoisson', bindings={'square': geometry.selection('square'), 'x_lower': (geometry.selection('x_lower'), geometry.selection('square')), 'x_upper': (geometry.selection('x_upper'), geometry.selection('square')), 'y_lower': (geometry.selection('y_lower'), geometry.selection('square')), 'y_upper': (geometry.selection('y_upper'), geometry.selection('square')), **dict(zip(('source_scale', 'diffusion', 'boundary_offset'), values)) | {'wave_number': np.pi}})

def make_plan(model, method):
    spatial = (
        eqiora.fem.Q1()
        if method == eqiora.fem.Q1()
        else eqiora.fvm.CellCenteredTpfa()
    )
    return eqiora.resolve(
        model,
        mesh=mesh,
        spatial=spatial,
        solve=eqiora.solve.Linear(
            algorithm=(eqiora.solve.LinearSolver.ConjugateGradient if spatial == eqiora.fvm.CellCenteredTpfa() else eqiora.solve.LinearSolver.BiConjugateGradientStabilized),
            preconditioner=eqiora.solve.Preconditioner.Identity,
            reduction=eqiora.solve.Reduction.Reproducible,
            provider=eqiora.solve.SolverProvider.reference(),
            relative_tolerance=1e-12,
            absolute_tolerance=1e-14,
            maximum_iterations=10000,
        ),
    )

def compile_program(model, method):
    plan = make_plan(model, method)
    inputs = [
        model.parameter("source_scale"),
        model.parameter("diffusion"),
        model.parameter("boundary_offset"),
    ]
    output = plan.capability.fields[0]
    return eqiora.diff.compile(
        plan,
        inputs=inputs,
        output=output,
    )

def at(values, method):
    return compile_program(make_model(values), method).primal().output.numpy()

nominal = np.array([19.739208802178716, 1.0, 0.0], dtype=np.float64)
for method in (
    eqiora.fem.Q1(),
    eqiora.fvm.CellCenteredTpfa(),
):
    model = make_model(nominal)
    program = compile_program(model, method)
    assert program.model_digest == model.digest
    assert program.input_shape == (3,)
    assert program.output_shape == (
        49 if method == eqiora.fem.Q1() else 36,
    )
    assert program.dtype == "float64"
    assert program.device == "cpu:0"
    assert program.derivative_contract == "implicit-first-order"
    assert program.input_ids == [
        model.parameter("source_scale").id,
        model.parameter("diffusion").id,
        model.parameter("boundary_offset").id,
    ]

    alternate_point = np.array(
        [nominal[0] * 0.8, 1.2, 0.15],
        dtype=np.float64,
    )
    expected_point = alternate_point.copy()
    point_producer = DLPackProducer(alternate_point)
    evaluation = program.evaluate(point_producer)
    alternate_point[:] = -123.0
    assert isinstance(evaluation, eqiora.DifferentiableEvaluation)
    assert point_producer.device_queries == 1
    assert len(point_producer.exports) == 1
    assert point_producer.exports[0]["copy"] is False
    np.testing.assert_array_equal(evaluation.point.numpy(), expected_point)
    assert not evaluation.point.numpy().flags.writeable

    evaluated_primal = evaluation.primal()
    evaluated_jvp = evaluation.jvp(direction)
    evaluated_finite = (
        at(expected_point + step * direction, method)
        - at(expected_point - step * direction, method)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        evaluated_primal.output.numpy(),
        at(expected_point, method),
        rtol=0,
        atol=2.0e-13,
    )
    np.testing.assert_array_equal(
        evaluated_jvp.output.numpy(), evaluated_primal.output.numpy()
    )
    np.testing.assert_allclose(
        evaluated_jvp.tangent.numpy(),
        evaluated_finite,
        rtol=7.0e-7,
        atol=5.0e-10,
    )
    evaluated_cotangent = np.ones(program.output_shape, dtype=np.float64)
    evaluated_vjp = evaluation.vjp(evaluated_cotangent)
    np.testing.assert_allclose(
        np.dot(evaluated_jvp.tangent.numpy(), evaluated_cotangent),
        np.dot(direction, evaluated_vjp.input_cotangent.numpy()),
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    np.testing.assert_array_equal(
        program.evaluate(nominal).primal().output.numpy(),
        program.primal().output.numpy(),
    )

    primal = program.primal()
    jvp = program.jvp(direction)
    dlpack_direction = DLPackProducer(direction)
    dlpack_jvp = program.jvp(dlpack_direction)
    np.testing.assert_array_equal(
        dlpack_jvp.tangent.numpy(), jvp.tangent.numpy()
    )
    assert dlpack_direction.device_queries == 1
    assert len(dlpack_direction.exports) == 1
    assert dlpack_direction.exports[0]["dl_device"] == (1, 0)
    assert dlpack_direction.exports[0]["copy"] is False
    assert dlpack_direction.exports[0]["max_version"][0] == 1
    assert dlpack_direction.exports[0].get("stream") is None
    finite = (at(nominal + step * direction, method) - at(
        nominal - step * direction, method
    )) / (2.0 * step)
    np.testing.assert_allclose(jvp.output.numpy(), primal.output.numpy(), rtol=0, atol=0)
    np.testing.assert_allclose(jvp.tangent.numpy(), finite, rtol=7.0e-7, atol=5.0e-10)
    if method == eqiora.fem.Q1():
        boundary = np.concatenate(
            [
                np.arange(7),
                np.arange(42, 49),
                np.arange(0, 49, 7),
                np.arange(6, 49, 7),
            ]
        )
        np.testing.assert_allclose(jvp.tangent.numpy()[boundary], 0.3, rtol=0, atol=2.0e-12)

    cotangent = np.arange(1, program.output_shape[0] + 1, dtype=np.float64)
    cotangent /= program.output_shape[0]
    vjp = program.vjp(cotangent)
    dlpack_cotangent = DLPackProducer(cotangent)
    dlpack_vjp = program.vjp(dlpack_cotangent)
    np.testing.assert_array_equal(
        dlpack_vjp.input_cotangent.numpy(), vjp.input_cotangent.numpy()
    )
    assert dlpack_cotangent.device_queries == 1
    assert len(dlpack_cotangent.exports) == 1
    assert dlpack_cotangent.exports[0]["copy"] is False
    array_tangent = program.jvp(vjp.input_cotangent)
    assert array_tangent.tangent.shape == program.output_shape
    finite_gradient = np.empty(3, dtype=np.float64)
    for coordinate in range(3):
        basis = np.zeros(3, dtype=np.float64)
        basis[coordinate] = 1.0
        finite_gradient[coordinate] = np.dot(
            cotangent,
            (at(nominal + step * basis, method) - at(
                nominal - step * basis, method
            )) / (2.0 * step),
        )
    np.testing.assert_allclose(
        vjp.input_cotangent.numpy(), finite_gradient, rtol=9.0e-7, atol=1.0e-9
    )
    np.testing.assert_allclose(vjp.output.numpy(), primal.output.numpy(), rtol=0, atol=0)
    np.testing.assert_allclose(
        np.dot(jvp.tangent.numpy(), cotangent),
        np.dot(direction, vjp.input_cotangent.numpy()),
        rtol=2.0e-10,
        atol=2.0e-11,
    )

    assert primal.evidence.mode == eqiora.DifferentiationMode.Primal
    assert jvp.evidence.mode == eqiora.DifferentiationMode.Jvp
    assert vjp.evidence.mode == eqiora.DifferentiationMode.Vjp
    assert jvp.evidence.implementation == eqiora.DerivativeImplementation.AnalyticAssembled
    assert jvp.evidence.linearization_state == eqiora.LinearizationState.Reused
    assert primal.evidence.linearization_state == eqiora.LinearizationState.Established
    assert primal.evidence.derivative_solve is None
    assert jvp.evidence.primal_solve.orientation == "normal"
    assert jvp.evidence.derivative_solve.orientation == "normal"
    assert vjp.evidence.derivative_solve.orientation == "transposed"
    expected_algorithm = "bicgstab" if method == eqiora.fem.Q1() else "conjugate-gradient"
    assert jvp.evidence.derivative_solve.algorithm == expected_algorithm
    assert jvp.evidence.derivative_solve.preconditioner == "identity"
    assert jvp.evidence.derivative_solve.reduction == "reproducible"
    assert len(jvp.evidence.state_system_fingerprint) == 64
    assert jvp.evidence.primal_residual_norm <= jvp.evidence.residual_tolerance

    recomputed = compile_program(model, method)
    assert recomputed.model_digest == program.model_digest
    assert recomputed.plan_identity == program.plan_identity
    assert (
        recomputed.primal().evidence.state_system_fingerprint
        == primal.evidence.state_system_fingerprint
    )

    for bad in (
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        np.array([1.0, np.nan, 3.0], dtype=np.float64),
        np.array([1.0, 2.0, 3.0], dtype=np.dtype(">f8" if np.little_endian else "<f8")),
        np.ndarray((3,), dtype=np.float64, buffer=bytearray(25), offset=1),
        np.arange(6, dtype=np.float64)[::2],
        [1.0, 2.0, 3.0],
    ):
        try:
            program.jvp(bad)
        except (BufferError, TypeError):
            pass
        else:
            raise AssertionError("an inadmissible tangent must fail before execution")

    for bad in (
        DLPackProducer(np.array([1.0, 2.0, 3.0], dtype=np.float32)),
        DLPackProducer(np.array([[1.0, 2.0, 3.0]], dtype=np.float64)),
        DLPackProducer(np.arange(6, dtype=np.float64)[::2]),
        DLPackProducer(np.array([1.0, np.nan, 3.0], dtype=np.float64)),
    ):
        try:
            program.jvp(bad)
        except BufferError:
            pass
        else:
            raise AssertionError("an inadmissible DLPack tangent must fail before execution")

    foreign_device = DLPackProducer(direction, device=(2, 0))
    try:
        program.jvp(foreign_device)
    except BufferError:
        pass
    else:
        raise AssertionError("a non-CPU DLPack input must fail before export")
    assert foreign_device.exports == []

    foreign_ordinal = DLPackProducer(direction, device=(1, 1))
    try:
        program.jvp(foreign_ordinal)
    except BufferError:
        pass
    else:
        raise AssertionError("a foreign CPU ordinal must fail before export")
    assert foreign_ordinal.exports == []

    class IncompleteDLPackProducer:
        def __dlpack__(self, **kwargs):
            return direction.__dlpack__(**kwargs)

    try:
        program.jvp(IncompleteDLPackProducer())
    except BufferError:
        pass
    else:
        raise AssertionError("an incomplete DLPack protocol must fail before export")

    try:
        program.vjp(np.zeros(program.output_shape[0] - 1, dtype=np.float64))
    except BufferError:
        pass
    else:
        raise AssertionError("a mismatched cotangent must fail before execution")

    try:
        program.evaluate(np.array([nominal[0], -1.0, nominal[2]], dtype=np.float64))
    except eqiora.ValidationError:
        pass
    else:
        raise AssertionError("an inadmissible Parameter point must fail closed")

original = make_model(nominal)
original_plan = make_plan(original, eqiora.fem.Q1())
foreign = make_model((nominal[0], 2.0, nominal[2]))
foreign_plan = make_plan(foreign, eqiora.fem.Q1())
try:
    eqiora.diff.compile(
        foreign_plan,
        inputs=[original.parameter("source_scale")],
        output=original_plan.capability.fields[0],
    )
except eqiora.ValidationError:
    pass
else:
    raise AssertionError("foreign semantic roles must fail closed")

try:
    eqiora.diff.compile(
        foreign_plan,
        inputs=[original.parameter("source_scale")],
        output=original_plan.capability.fields[0],
    )
except eqiora.ValidationError:
    pass
else:
    raise AssertionError("foreign Plan must fail closed")

try:
    eqiora.diff.compile(
        original,
        original_plan,
        inputs=[original.parameter("source_scale")],
        output=original_plan.capability.fields[0],
    )
except TypeError:
    pass
else:
    raise AssertionError("the displaced two-object signature must be absent")

for constructor in (
    eqiora.ParameterRef,
    eqiora.FieldRef,
    eqiora.DifferentiableEvaluation,
    eqiora.DifferentiableProgram,
    eqiora.DifferentiablePrimal,
    eqiora.DifferentiableJvp,
    eqiora.DifferentiableVjp,
    eqiora.DifferentiationEvidence,
):
    try:
        constructor()
    except TypeError:
        pass
    else:
        raise AssertionError("accepted differentiation values must be opaque")
"#
            ),
            Some(&locals),
            Some(&locals),
        )?;
        Ok(())
    })
}
