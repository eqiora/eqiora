from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import eqiora
import eqiora.torch as eqtorch


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
        output=plan.capability.fields[0],
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
        operator.output_shape[0],
        dtype=torch.float64,
    )

    output = operator(parameters)
    native = program.evaluate(parameters.detach())
    np.testing.assert_allclose(
        output.detach().numpy(),
        native.primal().output.numpy(copy=False),
        rtol=0.0,
        atol=0.0,
    )
    assert output.dtype is torch.float64
    assert output.device == torch.device("cpu")
    assert output.shape == operator.output_shape
    assert output.data_ptr() != parameters.data_ptr()

    output.backward(cotangent)
    np.testing.assert_allclose(
        parameters.grad.numpy(),
        native.vjp(cotangent).input_cotangent.numpy(copy=False),
        rtol=2.0e-11,
        atol=2.0e-12,
    )

    with torch.no_grad():
        output[0] = -123.0
    assert operator(parameters.detach())[0].item() != -123.0


def test_autograd_is_repeatable_and_zero_cotangent_is_exact() -> None:
    operator = eqtorch.bind(
        differentiable_program(eqiora.fem.Q1())
    )
    parameters = torch.tensor(
        [17.0, 1.2, 0.1],
        dtype=torch.float64,
        requires_grad=True,
    )

    output = operator(parameters)
    output.sum().backward(retain_graph=True)
    first = parameters.grad.detach().clone()
    parameters.grad = None
    output.sum().backward()
    torch.testing.assert_close(parameters.grad, first, rtol=0.0, atol=0.0)

    zero = torch.autograd.grad(
        operator(parameters),
        parameters,
        grad_outputs=torch.zeros(operator.output_shape, dtype=torch.float64),
    )[0]
    torch.testing.assert_close(zero, torch.zeros_like(parameters), rtol=0.0, atol=0.0)
    assert not operator(parameters.detach()).requires_grad


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
        operator(parameters).sum(),
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
            parameters,
            operator._token,
            operator.input_shape[0],
            operator.output_shape[0],
        ),
    )
    assert torch.autograd.gradcheck(
        operator,
        (parameters,),
        eps=1.0e-6,
        atol=2.0e-6,
        rtol=2.0e-5,
    )

    def objective(values):
        return operator(values).square().sum()

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
        (torch.ones(3, dtype=torch.float32), TypeError),
        (torch.ones((1, 3), dtype=torch.float64), ValueError),
        (torch.arange(6, dtype=torch.float64)[::2], ValueError),
        (
            torch.sparse_coo_tensor(
                torch.tensor([[0, 2]]),
                torch.ones(2, dtype=torch.float64),
                (3,),
                check_invariants=False,
            ),
            TypeError,
        ),
        (torch.ones(2, dtype=torch.float64), ValueError),
        (torch.tensor([17.0, float("nan"), 0.1], dtype=torch.float64), BufferError),
    ],
)
def test_inputs_fail_closed_before_or_at_native_admission(parameters, error) -> None:
    operator = eqtorch.bind(
        differentiable_program(eqiora.fem.Q1())
    )
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
    output = eqtorch.bind(program)(parameters)
    output.sum().backward()
    assert parameters.grad is not None


def test_unknown_or_mismatched_tokens_fail_closed() -> None:
    parameters = torch.tensor([17.0, 1.2, 0.1], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="not available"):
        eqtorch._solve(parameters, "not-a-token", 3, 25)

    operator = eqtorch.bind(
        differentiable_program(eqiora.fem.Q1())
    )
    with pytest.raises(RuntimeError, match="metadata"):
        eqtorch._solve(parameters, operator._token, 3, operator.output_shape[0] + 1)


def test_supported_torch_series_is_exact() -> None:
    assert eqtorch._torch_series(torch.__version__) == (2, 14)
    if expected := os.environ.get("EQIORA_TEST_TORCH_VERSION"):
        assert torch.__version__.split("+", maxsplit=1)[0] == expected
