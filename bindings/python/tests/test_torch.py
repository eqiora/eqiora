from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import eqiora
import eqiora.torch as eqtorch
from test_differentiation import branch_programs


@pytest.fixture(autouse=True)
def isolated_compiler_cache():
    # Each test owns different immutable native programs. Do not let unrelated
    # cases exhaust Dynamo's per-code-object specialization cache.
    torch.compiler.reset()
    yield
    torch.compiler.reset()


POISSON = """
public component PytorchDifferentiatedPoisson(
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

  variable potential: 1 on square in h1;
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


def test_base_import_remains_torch_free() -> None:
    script = """
import sys
import eqiora

assert "torch" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=True,
        text=True,
        capture_output=True,
    )


def differentiable_program(method) -> eqiora.DifferentiableProgram:
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
    mesh_plan = eqiora.meshing.resolve(
        geometry,
        eqiora.meshing.CartesianMesher(cells=(4, 4)),
    )
    mesh = eqiora.meshing.generate(mesh_plan)
    model = eqiora.compile(
        source=POISSON,
        geometry=geometry,
        entry='PytorchDifferentiatedPoisson',
        bindings={
            'square': geometry.selection('square'),
            'x_lower': (geometry.selection('x_lower'), geometry.selection('square')),
            'x_upper': (geometry.selection('x_upper'), geometry.selection('square')),
            'y_lower': (geometry.selection('y_lower'), geometry.selection('square')),
            'y_upper': (geometry.selection('y_upper'), geometry.selection('square')),
            'diffusion': 1.0,
            'wave_number': np.pi,
            'source_scale': 2.0 * np.pi ** 2,
            'boundary_offset': 0.0,
        },
    )
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
    return eqiora.diff.compile(
        plan,
        inputs=(
            model.parameter("source_scale"),
            model.parameter("diffusion"),
            model.parameter("boundary_offset"),
        ),
        outputs=(plan.capability.fields[0],),
    )


@pytest.mark.parametrize(
    "method",
    [
        eqiora.fem.Q1(),
        eqiora.fvm.CellCenteredTpfa(),
    ],
)
def test_functional_operator_matches_native_primal_and_vjp(method) -> None:
    program = differentiable_program(method)
    operator = eqtorch.bind(program)
    parameters = torch.tensor(
        [17.0, 1.2, 0.1],
        dtype=torch.float64,
        requires_grad=True,
    )
    cotangent = torch.linspace(
        0.25,
        1.25,
        operator.outputs[0].shape[0],
        dtype=torch.float64,
    )

    output = operator(tuple(parameters.unbind()))[0]
    native = program.evaluate(tuple(parameters.detach().unbind()))
    np.testing.assert_allclose(
        output.detach().numpy(),
        native.primal().output[0],
        rtol=0.0,
        atol=0.0,
    )
    assert output.dtype is torch.float64
    assert output.device == torch.device("cpu")
    assert output.shape == operator.outputs[0].shape
    assert output.data_ptr() != parameters.data_ptr()

    output.backward(cotangent)
    np.testing.assert_allclose(
        parameters.grad.numpy(),
        native.vjp((cotangent,)).input_cotangent,
        rtol=2.0e-11,
        atol=2.0e-12,
    )

    with torch.no_grad():
        output[0] = -123.0
    assert operator(tuple(parameters.detach().unbind()))[0][0].item() != -123.0


def test_autograd_is_repeatable_and_zero_cotangent_is_exact() -> None:
    operator = eqtorch.bind(
        differentiable_program(eqiora.fem.Q1())
    )
    parameters = torch.tensor(
        [17.0, 1.2, 0.1],
        dtype=torch.float64,
        requires_grad=True,
    )

    output = operator(tuple(parameters.unbind()))[0]
    output.sum().backward(retain_graph=True)
    first = parameters.grad.detach().clone()
    parameters.grad = None
    output.sum().backward()
    torch.testing.assert_close(parameters.grad, first, rtol=0.0, atol=0.0)

    zero = torch.autograd.grad(
        operator(tuple(parameters.unbind()))[0],
        parameters,
        grad_outputs=torch.zeros(operator.outputs[0].shape, dtype=torch.float64),
    )[0]
    torch.testing.assert_close(zero, torch.zeros_like(parameters), rtol=0.0, atol=0.0)
    assert not operator(tuple(parameters.detach().unbind()))[0].requires_grad


def test_double_backward_is_an_explicit_nonclaim() -> None:
    operator = eqtorch.bind(
        differentiable_program(eqiora.fem.Q1())
    )
    parameters = torch.tensor(
        [17.0, 1.2, 0.1],
        dtype=torch.float64,
        requires_grad=True,
    )
    first = torch.autograd.grad(
        operator(tuple(parameters.unbind()))[0].sum(),
        parameters,
        create_graph=True,
    )[0]
    with pytest.raises(RuntimeError):
        torch.autograd.grad(first.sum(), parameters)


@pytest.mark.parametrize(
    "method",
    [
        eqiora.fem.Q1(),
        eqiora.fvm.CellCenteredTpfa(),
    ],
)
def test_registration_passes_opcheck_gradcheck_and_fullgraph_compile(method) -> None:
    operator = eqtorch.bind(differentiable_program(method))
    parameters = torch.tensor(
        [17.0, 1.2, 0.1],
        dtype=torch.float64,
        requires_grad=True,
    )

    torch.library.opcheck(
        eqtorch._solve,
        (
            list(parameters.detach().unbind()),
            operator._token,
            [],
        ),
    )
    assert torch.autograd.gradcheck(
        lambda values: operator(tuple(values.unbind())),
        (parameters,),
        eps=1.0e-6,
        atol=2.0e-6,
        rtol=2.0e-5,
    )

    def objective(values):
        return operator(tuple(values.unbind()))[0].square().sum()

    eager_value = objective(parameters)
    eager = torch.autograd.grad(eager_value, parameters)[0]
    compiled = torch.compile(objective, fullgraph=True)
    compiled_parameters = parameters.detach().clone().requires_grad_(True)
    compiled_value = compiled(compiled_parameters)
    torch.testing.assert_close(
        compiled_value.detach(),
        eager_value.detach(),
        rtol=2.0e-13,
        atol=2.0e-14,
    )
    actual = torch.autograd.grad(compiled_value, compiled_parameters)[0]
    torch.testing.assert_close(actual, eager, rtol=2.0e-11, atol=2.0e-12)


@pytest.mark.parametrize(
    ("parameters", "error"),
    [
        (torch.ones(3, dtype=torch.float64), TypeError),
        ((torch.tensor(17., dtype=torch.float32),) * 3, TypeError),
        ((torch.ones(1, dtype=torch.float64),) * 3, ValueError),
        ((torch.tensor(17., dtype=torch.float64),) * 2, ValueError),
        ((torch.tensor(v, dtype=torch.float64) for v in (17., 1.2, .1)), TypeError),
        ((torch.tensor(17., dtype=torch.float64), torch.tensor(float("nan"), dtype=torch.float64),
          torch.tensor(.1, dtype=torch.float64)), BufferError),
    ],
)
def test_inputs_fail_closed_before_or_at_native_admission(parameters, error) -> None:
    operator = eqtorch.bind(differentiable_program(eqiora.fem.Q1()))
    with pytest.raises(error):
        operator(parameters)


def test_registry_is_identity_deduplicated_and_retained_for_backward() -> None:
    program = differentiable_program(eqiora.fem.Q1())
    first = eqtorch.bind(program)
    second = eqtorch.bind(program)
    assert first._token == second._token

    parameters = torch.tensor(
        [17.0, 1.2, 0.1],
        dtype=torch.float64,
        requires_grad=True,
    )
    output = eqtorch.bind(program)(tuple(parameters.unbind()))[0]
    output.sum().backward()
    assert parameters.grad is not None


def test_unknown_or_mismatched_tokens_fail_closed() -> None:
    parameters = torch.tensor([17.0, 1.2, 0.1], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="not registered"):
        eqtorch._solve(list(parameters.unbind()), "not-a-token", [])

    operator = eqtorch.bind(
        differentiable_program(eqiora.fem.Q1())
    )
    with pytest.raises(ValueError, match="leaves"):
        eqtorch._solve(list(parameters[:2].unbind()), operator._token, [])


def test_supported_torch_series_is_exact() -> None:
    assert eqtorch._torch_series(torch.__version__) == (2, 14)
    if expected := os.environ.get("EQIORA_TEST_TORCH_VERSION"):
        assert torch.__version__.split("+", maxsplit=1)[0] == expected


def test_seed_identity_keeps_nonlinear_branches_and_derivatives_distinct():
    programs = branch_programs()
    operators = [eqtorch.bind(program) for program in programs]
    assert operators[0]._token != operators[1]._token
    assert operators[0]._token == operators[2]._token
    for operator, sign in zip(operators, (1., -1., 1.), strict=True):
        for operation in (operator, torch.compile(operator, backend="eager", fullgraph=True)):
            for p in (4., 9.):
                point = torch.tensor([p], dtype=torch.float64, requires_grad=True)
                value, = operation(tuple(point.unbind()))
                assert tuple(value.shape) == operator.outputs[0].shape == operator.program.outputs[0].shape
                # w=sign*sqrt(p); O=w+p and dO/dp=1+sign/(2*sqrt(p)).
                np.testing.assert_allclose(value.detach().numpy(), p+sign*np.sqrt(p), rtol=0, atol=1e-12)
                value.sum().backward()
                np.testing.assert_allclose(point.grad.numpy(), [1+sign/(2*np.sqrt(p))], rtol=0, atol=1e-12)


@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize("advanced", [False, True])
@pytest.mark.parametrize("complex_field", [False, True])
def test_fixed_history_spatial_step_composes_with_backward_and_compile(nonlinear, advanced, complex_field):
    from test_spatial_step_program import held_step_program

    program, state = held_step_program(nonlinear, complex_field, advanced)
    before = state.to_bytes()
    solve = eqtorch.bind(program)
    assert solve.outputs[0].shape == (1, 3)
    phase = 1+2j if complex_field else 1.
    previous = float(advanced)
    # v-v_previous+v^3=p or 2*v-v_previous=p, with v_previous held fixed.
    value, derivative = (10.-previous, 1/13) if nonlinear else (4.-previous, .5)
    for operation in (solve, torch.compile(solve, backend="aot_eager", fullgraph=True)):
        point = torch.tensor([value], dtype=torch.float64, requires_grad=True)
        output, = operation(tuple(point.unbind()))
        np.testing.assert_allclose(output.detach().numpy(), [[2*phase]*3], rtol=0, atol=1e-10)
        gradient, = torch.autograd.grad(output.real.sum(), point)
        np.testing.assert_allclose(gradient.detach().numpy(), [3*derivative],
                                   rtol=0, atol=1e-10)
    assert state.to_bytes() == before


@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize("complex_field", [False, True])
@pytest.mark.parametrize("observable_output", [None, "total", "amount"])
def test_spatial_trajectory_composes_with_backward_and_compile(complex_field, nonlinear, observable_output):
    from test_spatial_step_program import trajectory_program, trajectory_reference

    program = trajectory_program(nonlinear, complex_field, observable_output)
    solve = eqtorch.bind(program)
    assert solve.outputs[0].shape == ((2, 3) if observable_output is None else (2,))
    expected, derivative = trajectory_reference(4., nonlinear, complex_field, observable_output)
    for operation in (solve, torch.compile(solve, backend="aot_eager", fullgraph=True)):
        point = torch.tensor([4.], dtype=torch.float64, requires_grad=True)
        output, = operation(tuple(point.unbind()))
        np.testing.assert_allclose(output.detach().numpy(), expected, rtol=0, atol=1e-10)
        gradient, = torch.autograd.grad(output.real.sum(), point)
        np.testing.assert_allclose(gradient.detach().numpy(), [derivative.real.sum()], rtol=0, atol=1e-10)

    # Physical point axes precede retained time/Field axes; gradient sums also
    # include the unobserved second step through the native history products.
    def loss(p):
        return solve((p,))[0].real.sum()

    points = torch.tensor([[2., 4., 4.], [4., 2., 4.]], dtype=torch.float64).T
    expected_gradients = np.array([[trajectory_reference(float(p), nonlinear, complex_field, observable_output)[1].real.sum()
                                   for p in row] for row in points])
    operation = torch.vmap(torch.vmap(torch.func.grad(loss)))
    for mapped in (operation, torch.compile(operation, backend="aot_eager", fullgraph=True)):
        np.testing.assert_allclose(mapped(points).numpy(), expected_gradients, rtol=0, atol=1e-10)
    shared_gradient = torch.func.grad(lambda p: torch.vmap(torch.vmap(loss))(p + points - 4).sum())
    for operation in (shared_gradient, torch.compile(shared_gradient, backend="aot_eager", fullgraph=True)):
        np.testing.assert_allclose(operation(torch.tensor(4., dtype=torch.float64)).numpy(),
                                   expected_gradients.sum(), rtol=0, atol=1e-10)


def test_structured_complex_leaves_preserve_shapes_and_real_pairing():
    from test_complex_differentiation import structured_map_program

    _, program = structured_map_program()
    solve = eqtorch.bind(program)
    assert solve.inputs == program.inputs
    assert solve.outputs == program.outputs
    assert not hasattr(solve, "input_shape")
    assert not hasattr(solve, "output_shape")
    point = (torch.tensor(2., dtype=torch.float64, requires_grad=True),
             torch.tensor([1+2j, 3-4j], dtype=torch.complex128, requires_grad=True))
    torch.library.opcheck(eqtorch._solve, ([v.detach() for v in point], solve._token, []))
    assert torch.autograd.gradcheck(lambda p, c: solve((p, c)), point)
    for operation in (solve, torch.compile(solve, fullgraph=True)):
        output = operation(point)
        assert tuple(value.shape for value in output) == ((2,), (), ())
        assert tuple(value.dtype for value in output) == (torch.complex128, torch.float64, torch.float64)
        # z = p*c + conj(c), loss = sum|z|^2 + p, margin = p-1.
        torch.testing.assert_close(output[0], torch.tensor([3+2j, 9-4j], dtype=torch.complex128))
        torch.testing.assert_close(output[1], torch.tensor(112., dtype=torch.float64))
        torch.testing.assert_close(output[2], torch.tensor(1., dtype=torch.float64))
        gradient = torch.autograd.grad(output[1], point, retain_graph=True)
        torch.testing.assert_close(gradient[0], torch.tensor(101., dtype=torch.float64))
        torch.testing.assert_close(gradient[1], torch.tensor([18+4j, 54-8j], dtype=torch.complex128))
        # Native and Torch use Re(conj(cotangent)*dy), including non-holomorphic z.
        # dz = c*dp + p*dc + conj(dc); weight z by 2+3i, loss by 1/2, margin by -2.
        cotangent = (torch.full((2,), 2+3j, dtype=torch.complex128),
                     torch.tensor(.5, dtype=torch.float64), torch.tensor(-2., dtype=torch.float64))
        gradient = torch.autograd.grad(output, point, grad_outputs=cotangent)
        torch.testing.assert_close(gradient[0], torch.tensor(50.5, dtype=torch.float64))
        torch.testing.assert_close(gradient[1], torch.tensor([15+5j, 33-1j], dtype=torch.complex128))
    # Lazy conjugate views are resolved without mutating the underlying tensor.
    before = point[1].detach().clone()
    output = solve((point[0], point[1].conj()))
    torch.testing.assert_close(output[0], torch.tensor([3-2j, 9+4j], dtype=torch.complex128))
    torch.testing.assert_close(point[1], before)
    gradient, = torch.autograd.grad(output[1], (point[1],))
    torch.testing.assert_close(gradient, torch.tensor([18+4j, 54-8j], dtype=torch.complex128))
    for name in ("_program", "_token", "_input_metadata"):
        with pytest.raises(AttributeError, match="immutable"):
            setattr(solve, name, None)
        with pytest.raises(AttributeError, match="immutable"):
            delattr(solve, name)


def test_structured_tensor_metadata_is_admitted_per_leaf():
    from test_complex_differentiation import structured_map_program

    _, program = structured_map_program()
    solve = eqtorch.bind(program)
    p = torch.tensor(2., dtype=torch.float64)
    for value, error in (
        (torch.ones(2, dtype=torch.float64), TypeError),
        (torch.ones((1, 2), dtype=torch.complex128), ValueError),
        (torch.ones(2, dtype=torch.complex128, device="meta"), ValueError),
        (torch.sparse_coo_tensor(torch.tensor([[0, 1]]),
                                torch.ones(2, dtype=torch.complex128), (2,), check_invariants=True), TypeError),
    ):
        with pytest.raises(error):
            solve((p, value))


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("kind", ["grad", "shared_grad", "jvp", "jacrev", "jacfwd"])
def test_native_point_and_seed_axes_compose(kind, compiled):
    from test_complex_differentiation import structured_map_program

    _, program = structured_map_program()
    solve = eqtorch.bind(program)
    p = torch.tensor([[2., 3., 2.], [1., 4., 2.]], dtype=torch.float64)
    c = torch.tensor([1+2j, 3-4j], dtype=torch.complex128)
    z = p[..., None]*c+c.conj()
    dp = 2*(z.conj()*c).sum(-1).real+1
    dc = 2*(p[..., None]*z+z.conj())

    def call(p, c): return solve((p, c))
    def loss(p, c): return call(p, c)[1]
    def nest(fn): return torch.vmap(torch.vmap(fn, in_dims=(0, None)), in_dims=(0, None))

    if kind == "grad":
        operation = nest(torch.func.grad(loss, argnums=(0, 1)))
        expected = dp, dc
    elif kind == "shared_grad":
        operation = torch.func.grad(lambda p, c: nest(loss)(p, c).sum(), argnums=(0, 1))
        expected = dp, dc.sum((0, 1))
    elif kind == "jvp":
        def directional(p, c):
            return torch.func.jvp(call, (p, c), (torch.ones_like(p)*.5, torch.ones_like(c)*(1-2j)))[1]
        operation = nest(directional)
        dz = .5*c+p[..., None]*(1-2j)+(1+2j)
        expected = dz, 2*(z.conj()*dz).sum(-1).real+.5, torch.ones_like(p)*.5
    else:
        # Torch Jacobian transforms require real inputs, even for an unselected
        # complex argument. Expose its real coordinates in this consumer only.
        def outputs(p, cr, ci):
            values = call(p, torch.complex(cr, ci))
            return torch.stack((values[1], values[2]))
        derivative = torch.func.jacrev if kind == "jacrev" else torch.func.jacfwd
        jacobian = torch.vmap(torch.vmap(derivative(outputs), in_dims=(0, None, None)),
                              in_dims=(0, None, None))
        operation = lambda p, c: jacobian(p, c.real, c.imag)
        expected = torch.stack((dp, torch.ones_like(p)), dim=-1)
    if compiled:
        operation = torch.compile(operation, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(operation(p, c), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize("compiled", [False, True])
def test_mapped_grad_handles_nonleading_empty_and_singleton_axes(count, compiled):
    from test_complex_differentiation import structured_map_program

    _, program = structured_map_program()
    solve = eqtorch.bind(program)
    p = torch.tensor([[2., 3., 2.]], dtype=torch.float64)[:, :count].T
    c = torch.tensor([1+2j, 3-4j], dtype=torch.complex128)
    gradient = torch.func.grad(lambda p, c: solve((p, c))[1], argnums=(0, 1))
    operation = torch.vmap(torch.vmap(gradient, in_dims=(0, None)),
                           in_dims=(1, None), out_dims=1)
    if compiled:
        operation = torch.compile(operation, backend="aot_eager", fullgraph=True)
    z = p[..., None]*c+c.conj()
    expected = 2*(z.conj()*c).sum(-1).real+1, 2*(p[..., None]*z+z.conj())
    torch.testing.assert_close(operation(p, c), expected, rtol=0, atol=1e-12)


def test_transform_inputs_preserve_views_versions_and_failure():
    from test_complex_differentiation import structured_map_program

    _, program = structured_map_program()
    solve = eqtorch.bind(program)
    p = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    storage = torch.tensor([1+2j, 99., 3-4j, 99.], dtype=torch.complex128, requires_grad=True)
    view = storage[::2]
    output = solve((p, view))
    torch.testing.assert_close(output[1], torch.tensor(112., dtype=torch.float64))
    gradient = torch.autograd.grad(output[1], (p, storage), retain_graph=True)
    torch.testing.assert_close(gradient, (torch.tensor(101., dtype=torch.float64),
        torch.tensor([18+4j, 0., 54-8j, 0.], dtype=torch.complex128)))
    with torch.no_grad():
        storage[0] = 5.
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(output[1], (p, storage))

    branch = eqtorch.bind(branch_programs()[0])
    with pytest.raises(RuntimeError, match="Eqiora evaluation stopped at member 1"):
        torch.vmap(lambda p: branch((p,)))(torch.tensor([4., -1., 9.], dtype=torch.float64))


def test_higher_products_are_rejected_instead_of_zero_gradients():
    from test_complex_differentiation import structured_map_program

    _, program = structured_map_program()
    solve = eqtorch.bind(program)
    c = torch.tensor([1+2j, 3-4j], dtype=torch.complex128)
    loss = lambda p: solve((p, c))[1]
    p = torch.tensor(2., dtype=torch.float64)
    with pytest.raises(RuntimeError):
        torch.func.grad(torch.func.grad(loss))(p)
    with pytest.raises(RuntimeError):
        torch.func.jvp(torch.func.grad(loss), (p,), (torch.ones_like(p),))
    # AOT may request a derivative of the returned gradient when its inputs
    # require grad. This remains an explicit missing native higher-order path.
    compiled = torch.compile(torch.func.grad(loss), backend="aot_eager", fullgraph=True)
    with pytest.raises(RuntimeError):
        compiled(p.requires_grad_())
