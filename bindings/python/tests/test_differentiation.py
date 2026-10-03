from __future__ import annotations

import numpy as np
import pytest

import eqiora
from _signature_bindings import support_bindings


POISSON = """
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
"""

ELASTICITY = """
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
"""


def model_and_plan(method, *, diffusion: float = 1.0):
    graph = eqiora.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    geometry = graph.build(
        rectangle,
        named_topology={
            "square": rectangle.region,
            "x_lower": rectangle.boundaries[0],
            "x_upper": rectangle.boundaries[1],
            "y_lower": rectangle.boundaries[2],
            "y_upper": rectangle.boundaries[3],
        },
    )
    request = eqiora.meshing.CartesianMesher(cells=(4, 4))
    mesh_plan = eqiora.meshing.resolve(geometry, request)
    mesh = eqiora.meshing.generate(mesh_plan)
    model = eqiora.compile(source=POISSON, geometry=geometry, entry='PythonDifferentiatedPoisson', bindings={**support_bindings(geometry, ['square'], [('x_lower', 'square'), ('x_upper', 'square'), ('y_lower', 'square'), ('y_upper', 'square')]), **{'diffusion': diffusion, 'wave_number': np.pi, 'source_scale': 2.0 * np.pi ** 2, 'boundary_offset': 0.0}})
    spatial = (
        eqiora.fem.Q1()
        if method == eqiora.fem.Q1()
        else eqiora.fvm.CellCenteredTpfa()
    )
    plan = eqiora.resolve(
        model,
        mesh=mesh,
        spatial=spatial,
        solve=eqiora.solve.Linear(
            algorithm=(eqiora.solve.LinearSolver.ConjugateGradient if spatial == eqiora.fvm.CellCenteredTpfa() else eqiora.solve.LinearSolver.BiConjugateGradientStabilized),
            preconditioner=eqiora.solve.Preconditioner.Identity,
            reduction=eqiora.solve.Reduction.Reproducible,
            provider=eqiora.solve.SolverProvider.reference(),
            relative_tolerance=1.0e-10,
            absolute_tolerance=1.0e-12,
            maximum_iterations=10_000,
        ),
    )
    return model, plan


def test_numerical_method_changes_plan_but_not_model_identity() -> None:
    model, fem_plan = model_and_plan(eqiora.fem.Q1())
    fingerprint = model.structural_fingerprint
    model_bytes = model.to_bytes()
    fvm_plan = eqiora.resolve(
        model,
        mesh=fem_plan.mesh,
        spatial=eqiora.fvm.CellCenteredTpfa(),
        solve=eqiora.solve.Linear(
            algorithm=eqiora.solve.LinearSolver.ConjugateGradient,
            preconditioner=eqiora.solve.Preconditioner.Identity,
            reduction=eqiora.solve.Reduction.Reproducible,
            provider=eqiora.solve.SolverProvider.reference(),
            relative_tolerance=1.0e-10,
            absolute_tolerance=1.0e-12,
            maximum_iterations=10_000,
        ),
    )
    assert fvm_plan.identity != fem_plan.identity
    assert fvm_plan.model is fem_plan.model is model
    assert fvm_plan.model_digest == fem_plan.model_digest == model.digest
    assert model.structural_fingerprint == fingerprint
    assert model.to_bytes() == model_bytes


def elasticity_model_and_plan():
    graph = eqiora.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    geometry = graph.build(
        rectangle,
        named_topology={
            "region": rectangle.region,
            "left": rectangle.boundaries[0],
            "right": rectangle.boundaries[1],
            "bottom": rectangle.boundaries[2],
            "top": rectangle.boundaries[3],
        },
    )
    mesh_request = eqiora.meshing.CartesianMesher(cells=(4, 4))
    mesh_plan = eqiora.meshing.resolve(geometry, mesh_request)
    mesh = eqiora.meshing.generate(mesh_plan)
    model = eqiora.compile(source=ELASTICITY, geometry=geometry, entry='MixedBoundaryElasticity', bindings={**support_bindings(geometry, ['region'], [('left', 'region'), ('right', 'region'), ('bottom', 'region'), ('top', 'region')]), **{'mu': 2.0, 'lambda': 3.0, 'length_scale': 1.0}})
    plan = eqiora.resolve(
        model,
        mesh=mesh,
        spatial=eqiora.fem.Q1(),
        solve=eqiora.solve.Linear(
            algorithm=eqiora.solve.LinearSolver.ConjugateGradient,
            preconditioner=eqiora.solve.Preconditioner.Identity,
            reduction=eqiora.solve.Reduction.Reproducible,
            provider=eqiora.solve.SolverProvider.reference(),
            relative_tolerance=1.0e-10,
            absolute_tolerance=1.0e-12,
            maximum_iterations=10_000,
        ),
    )
    return model, plan


class DLPackProducer:
    def __init__(self, values: np.ndarray, *, device: tuple[int, int] = (1, 0)):
        self.values = values
        self.device = device
        self.device_queries = 0
        self.exports: list[dict[str, object]] = []

    def __dlpack_device__(self) -> tuple[int, int]:
        self.device_queries += 1
        return self.device

    def __dlpack__(self, **kwargs):
        self.exports.append(dict(kwargs))
        return self.values.__dlpack__(**kwargs)


@pytest.mark.parametrize(
    "method",
    [
        eqiora.fem.Q1(),
        eqiora.fvm.CellCenteredTpfa(),
    ],
)
def test_public_diff_module_exposes_paired_complete_field_actions(method) -> None:
    model, plan = model_and_plan(method)
    inputs = (
        model.parameter("source_scale"),
        model.parameter("diffusion"),
        model.parameter("boundary_offset"),
    )
    program = eqiora.diff.compile(
        plan,
        inputs=inputs,
        output=plan.capability.fields[0],
    )

    direction = np.array([0.7, -0.2, 0.3], dtype=np.float64)
    primal = program.primal()
    jvp = program.jvp(direction)
    cotangent = np.ones(program.output_shape, dtype=np.float64)
    vjp = program.vjp(cotangent)

    assert isinstance(program, eqiora.DifferentiableProgram)
    assert isinstance(primal.output, eqiora.Array)
    assert primal.output.shape == program.output_shape
    np.testing.assert_array_equal(jvp.output.numpy(), primal.output.numpy())
    np.testing.assert_array_equal(vjp.output.numpy(), primal.output.numpy())
    np.testing.assert_allclose(
        np.dot(jvp.tangent.numpy(), cotangent),
        np.dot(direction, vjp.input_cotangent.numpy()),
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    assert primal.evidence.linearization_state == eqiora.LinearizationState.Established
    assert primal.evidence.derivative_solve is None
    assert jvp.evidence.primal_solve.orientation == "normal"
    assert jvp.evidence.derivative_solve.orientation == "normal"
    assert vjp.evidence.derivative_solve.orientation == "transposed"
    assert program.input_ids == [reference.id for reference in inputs]

    point = np.array([17.0, 1.2, 0.1], dtype=np.float64)
    evaluation = program.evaluate(point)
    point[:] = -1.0
    assert isinstance(evaluation, eqiora.DifferentiableEvaluation)
    np.testing.assert_array_equal(
        evaluation.point.numpy(copy=False),
        np.array([17.0, 1.2, 0.1], dtype=np.float64),
    )
    assert not evaluation.point.numpy(copy=False).flags.writeable
    evaluated_jvp = evaluation.jvp(direction)
    evaluated_vjp = evaluation.vjp(cotangent)
    np.testing.assert_array_equal(
        evaluated_jvp.output.numpy(copy=False),
        evaluation.primal().output.numpy(copy=False),
    )
    np.testing.assert_allclose(
        np.dot(evaluated_jvp.tangent.numpy(copy=False), cotangent),
        np.dot(direction, evaluated_vjp.input_cotangent.numpy(copy=False)),
        rtol=2.0e-10,
        atol=2.0e-11,
    )


def test_diff_input_admission_is_explicit_and_model_bound() -> None:
    model, plan = model_and_plan(eqiora.fem.Q1())
    program = eqiora.diff.compile(
        plan,
        inputs=(model.parameter("source_scale"),),
        output=plan.capability.fields[0],
    )

    direction = np.array([1.0], dtype=np.float64)
    producer = DLPackProducer(direction)
    protocol_result = program.jvp(producer)
    numpy_result = program.jvp(direction)
    np.testing.assert_array_equal(
        protocol_result.tangent.numpy(copy=False),
        numpy_result.tangent.numpy(copy=False),
    )
    assert producer.device_queries == 1
    assert len(producer.exports) == 1
    export = producer.exports[0]
    assert export["dl_device"] == (1, 0)
    assert export["copy"] is False
    assert export["max_version"][0] == 1
    assert export.get("stream") is None

    cotangent = np.arange(program.output_shape[0], dtype=np.float64)
    cotangent_producer = DLPackProducer(cotangent)
    protocol_vjp = program.vjp(cotangent_producer)
    numpy_vjp = program.vjp(cotangent)
    np.testing.assert_array_equal(
        protocol_vjp.input_cotangent.numpy(copy=False),
        numpy_vjp.input_cotangent.numpy(copy=False),
    )
    assert cotangent_producer.device_queries == 1
    assert len(cotangent_producer.exports) == 1
    assert cotangent_producer.exports[0]["copy"] is False

    foreign_device = DLPackProducer(direction, device=(2, 0))
    with pytest.raises(BufferError, match="already reside on CPU device 0"):
        program.jvp(foreign_device)
    assert foreign_device.exports == []

    foreign_ordinal = DLPackProducer(direction, device=(1, 1))
    with pytest.raises(BufferError, match="already reside on CPU device 0"):
        program.jvp(foreign_ordinal)
    assert foreign_ordinal.exports == []

    for rejected in (
        DLPackProducer(np.array([1.0], dtype=np.float32)),
        DLPackProducer(np.array([[1.0]], dtype=np.float64)),
        DLPackProducer(np.array([1.0, 2.0], dtype=np.float64)),
        DLPackProducer(np.array([np.nan], dtype=np.float64)),
    ):
        with pytest.raises(BufferError):
            program.jvp(rejected)

    strided_cotangent = DLPackProducer(
        np.arange(2 * program.output_shape[0], dtype=np.float64)[::2]
    )
    with pytest.raises(BufferError, match="C-contiguous"):
        program.vjp(strided_cotangent)

    class IncompleteDLPackProducer:
        def __dlpack__(self, **kwargs):
            return direction.__dlpack__(**kwargs)

    with pytest.raises(BufferError, match="complete DLPack producer"):
        program.jvp(IncompleteDLPackProducer())

    with pytest.raises(BufferError):
        program.jvp(np.array([1.0], dtype=np.float32))
    with pytest.raises(BufferError):
        program.jvp(np.array([np.nan], dtype=np.float64))
    with pytest.raises(BufferError):
        program.jvp(
            np.array(
                [1.0],
                dtype=np.dtype(">f8" if np.little_endian else "<f8"),
            )
        )
    with pytest.raises(BufferError):
        program.vjp(np.zeros(program.output_shape[0] - 1, dtype=np.float64))
    with pytest.raises(BufferError):
        program.evaluate(np.zeros(2, dtype=np.float64))
    with pytest.raises(eqiora.ValidationError):
        eqiora.diff.compile(
            plan,
            inputs=iter((model.parameter("source_scale"),)),
            output=plan.capability.fields[0],
        )

    foreign, foreign_plan = model_and_plan(
        eqiora.fem.Q1(), diffusion=2.0
    )
    with pytest.raises(eqiora.ValidationError):
        eqiora.diff.compile(
            foreign_plan,
            inputs=(model.parameter("source_scale"),),
            output=plan.capability.fields[0],
        )
    with pytest.raises(eqiora.ValidationError, match="duplicate"):
        eqiora.diff.compile(
            plan,
            inputs=(
                model.parameter("source_scale"),
                model.parameter("source_scale"),
            ),
            output=plan.capability.fields[0],
        )
    with pytest.raises(TypeError):
        eqiora.diff.compile(
            model,
            plan,
            inputs=(model.parameter("source_scale"),),
            output=plan.capability.fields[0],
        )
    elasticity_model, elasticity_plan = elasticity_model_and_plan()
    with pytest.raises(eqiora.ValidationError, match="does not admit implicit output differentiation"):
        eqiora.diff.compile(
            elasticity_plan,
            inputs=(elasticity_model.parameter("mu"),),
            output=elasticity_plan.capability.displacement,
        )


def test_finite_program_preserves_partial_and_reduced_actions_and_seed_ownership():
    from test_constraints import nonlinear_root

    model, plan = nonlinear_root()
    seed = eqiora.State.initial(plan, fields=(eqiora.InitialField(model.field("w"), scalar_value=2.0),))
    program = eqiora.diff.compile(plan, inputs=(model.parameter("p"),),
                                 output=model.observable("output"), state=seed)
    one = np.array([1.0], dtype=np.float64)
    zero = np.array([0.0], dtype=np.float64)
    default = program.evaluate(np.array([4.0], dtype=np.float64))
    assert isinstance(default.primal().evidence.primal_solve, eqiora.NonlinearSolveSummary)
    assert default.primal().evidence.primal_solve.completed_iterations == 0
    assert default.primal().evidence.derivative_solve is None
    for p, w, reduced in ((4.0, 2.0, 1.25), (9.0, 3.0, 7.0 / 6.0)):
        point = program.evaluate(np.array([p], dtype=np.float64))
        np.testing.assert_allclose(point.accepted_unknowns.numpy(), [w], rtol=0, atol=1e-12)
        np.testing.assert_allclose(point.primal().output.numpy(), [w+p], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(point.residual_jvp(zero, one).numpy(), [-1.0])
        np.testing.assert_allclose(point.residual_jvp(one, zero).numpy(), [2*w], rtol=0, atol=2e-12)
        rw, rp = point.residual_vjp(one)
        np.testing.assert_allclose(rw.numpy(), [2*w], rtol=0, atol=2e-12)
        np.testing.assert_array_equal(rp.numpy(), [-1.0])
        np.testing.assert_array_equal(point.output_partial_jvp(zero, one).numpy(), [1.0])
        ow, op = point.output_partial_vjp(one)
        np.testing.assert_array_equal(ow.numpy(), [1.0])
        np.testing.assert_array_equal(op.numpy(), [1.0])
        np.testing.assert_allclose(point.jvp(one).tangent.numpy(), [reduced], rtol=0, atol=1e-12)
        reverse = point.vjp(one)
        np.testing.assert_allclose(reverse.input_cotangent.numpy(), [reduced], rtol=0, atol=1e-12)
        assert reverse.evidence.implementation == eqiora.DerivativeImplementation.OperatorIr
        assert reverse.evidence.initial_state_identity == seed.digest
        assert reverse.evidence.derivative_solve.orientation == "transposed"
        with pytest.raises(BufferError, match="shape"):
            point.residual_jvp(np.array([], dtype=np.float64), one)
    np.testing.assert_array_equal(default.vjp(one).input_cotangent.numpy(), [1.25])
    with pytest.raises((eqiora.ValidationError, eqiora.ExecutionError)):
        program.evaluate(zero)
    with pytest.raises((eqiora.ValidationError, eqiora.ExecutionError)):
        eqiora.diff.compile(plan, inputs=(model.parameter("p"),), output=model.observable("output"))
    foreign, foreign_plan = nonlinear_root(p=9)
    foreign_seed = eqiora.State.initial(foreign_plan, fields=(eqiora.InitialField(foreign.field("w"), scalar_value=2.0),))
    for inputs, output, state in (
        ((foreign.parameter("p"),), model.observable("output"), seed),
        ((model.parameter("p"),), foreign.observable("output"), seed),
        ((model.parameter("p"),), model.observable("output"), foreign_seed),
    ):
        with pytest.raises((eqiora.ValidationError, eqiora.ExecutionError)):
            eqiora.diff.compile(plan, inputs=inputs, output=output, state=state)


def test_finite_parameter_roles_do_not_alias_equal_values_or_frozen_coordinates():
    from test_constraints import nonlinear_root

    _, controls = nonlinear_root()

    def compile_root(equality):
        model = eqiora.compile(source=f"""
model OrderedRoot() {{
  parameter p: 1 = 4; parameter q: 1 = 4; variable w: 1;
  relation root {{ {equality}; inequality(p>=0); inequality(w>=0); }}
  observable output: 1 = w+p+2*q;
}}
""", entry="OrderedRoot")
        margins = tuple(eqiora.solve.ConstraintTolerance.inequality(
            model.constraint("root", ordinal), 1e-8, eqiora.Dimension()) for ordinal in (1, 2))
        plan = eqiora.resolve(model, solve=controls.requested_solve,
                              enforcement=eqiora.solve.StrictInterior(margins=margins))
        state = eqiora.State.initial(plan, fields=(eqiora.InitialField(model.field("w"), scalar_value=2.0),))
        return model, plan, state

    model, plan, state = compile_root("w*w=p")
    for names, values, expected_output, partial, total in (
        (("p", "q"), [4.0, 4.0], 14.0, [1.0, 2.0], [1.25, 2.0]),
        (("q", "p"), [4.0, 4.0], 14.0, [2.0, 1.0], [2.0, 1.25]),
        (("q", "p"), [4.0, 9.0], 20.0, [2.0, 1.0], [2.0, 7.0/6.0]),
        (("p",), [9.0], 20.0, [1.0], [7.0/6.0]),  # q stays at 4
        (("q",), [9.0], 24.0, [2.0], [2.0]),  # p and w stay at 4 and 2
    ):
        inputs = tuple(model.parameter(name) for name in names)
        program = eqiora.diff.compile(plan, inputs=inputs, output=model.observable("output"), state=state)
        point = program.evaluate(np.array(values, dtype=np.float64))
        assert program.input_ids == [reference.id for reference in inputs]
        assert point.primal().evidence.input_ids == program.input_ids
        np.testing.assert_allclose(point.primal().output.numpy(), [expected_output], rtol=0, atol=1e-12)
        _, op = point.output_partial_vjp(np.ones(1, dtype=np.float64))
        np.testing.assert_array_equal(op.numpy(), partial)
        np.testing.assert_allclose(point.vjp(np.ones(1, dtype=np.float64)).input_cotangent.numpy(), total, rtol=0, atol=1e-12)
    # The original equation is exactly satisfied, but R_w=2*(w-2)=0.
    singular, singular_plan, singular_state = compile_root("(w-2)*(w-2)=p-4")
    with pytest.raises((eqiora.ValidationError, eqiora.ExecutionError), match="regular|singular|rank"):
        eqiora.diff.compile(singular_plan, inputs=(singular.parameter("p"),),
                            output=singular.observable("output"), state=singular_state)
