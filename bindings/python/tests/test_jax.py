from __future__ import annotations

import gc
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxlib

import eqiora
import eqiora.jax as eqjax
from eqiora import _eqiora
from test_complex_differentiation import structured_map_program


POISSON = """
public component JaxDifferentiatedPoisson(
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


def scalar_leaves(values):
    """Columns of this scalar Poisson fixture, retaining dtype and device."""
    return tuple(values[..., i] for i in range(values.shape[-1]))


def test_base_import_remains_jax_free() -> None:
    script = """
import sys
import eqiora

assert "jax" not in sys.modules
assert "jaxlib" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=True,
        text=True,
        capture_output=True,
    )


def differentiable_program(
    method,
    *,
    include_wave_number: bool = False,
) -> eqiora.DifferentiableProgram:
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
        entry='JaxDifferentiatedPoisson',
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
    inputs = [
        model.parameter("source_scale"),
        model.parameter("diffusion"),
        model.parameter("boundary_offset"),
    ]
    if include_wave_number:
        inputs.append(model.parameter("wave_number"))
    return eqiora.diff.compile(
        plan,
        inputs=inputs,
        outputs=(plan.capability.fields[0],),
    )


@pytest.mark.parametrize(
    "method",
    [
        eqiora.fem.Q1(),
        eqiora.fvm.CellCenteredTpfa(),
    ],
)
def test_eager_primal_jvp_and_vjp_match_framework_neutral_actions(method) -> None:
    program = differentiable_program(method)
    solve = eqjax.bind(program)
    parameters = scalar_leaves(jnp.array([17.0, 1.2, 0.1], dtype=jnp.float64))
    tangent = scalar_leaves(jnp.array([0.3, -0.2, 0.4], dtype=jnp.float64))
    cotangent = (jnp.linspace(0.25, 1.25, solve.outputs[0].shape[0], dtype=jnp.float64),)
    output, = jax.block_until_ready(solve(parameters))
    native = program.evaluate(tuple(np.asarray(value) for value in parameters))
    np.testing.assert_array_equal(output, native.primal().output[0])
    assert output.dtype == jnp.float64
    assert output.shape == solve.outputs[0].shape
    assert output.devices() == {jax.devices("cpu")[0]}
    assert solve.inputs == program.inputs and solve.outputs == program.outputs
    (jvp_output,), (output_tangent,) = jax.jvp(solve, (parameters,), (tangent,))
    native_jvp = native.jvp(tuple(np.asarray(value) for value in tangent))
    np.testing.assert_array_equal(jvp_output, native_jvp.output[0])
    assert_product(output_tangent, native_jvp.tangent[0])
    (vjp_output,), pullback = jax.vjp(solve, parameters)
    input_cotangent = pullback(cotangent)[0]
    native_vjp = native.vjp(tuple(np.asarray(value) for value in cotangent))
    np.testing.assert_array_equal(vjp_output, native_vjp.output[0])
    assert_product(input_cotangent, native_vjp.input_cotangent)


@pytest.mark.parametrize(
    "method",
    [
        eqiora.fem.Q1(),
        eqiora.fvm.CellCenteredTpfa(),
    ],
)
def test_jitted_primal_gradient_and_jvp_use_typed_custom_calls(method) -> None:
    program = differentiable_program(method)
    solve = eqjax.bind(program)
    parameters = scalar_leaves(jnp.array([17.0, 1.2, 0.1], dtype=jnp.float64))
    tangent = scalar_leaves(jnp.array([0.3, -0.2, 0.4], dtype=jnp.float64))
    cotangent = jnp.linspace(0.25, 1.25, solve.outputs[0].shape[0], dtype=jnp.float64)
    jitted_solve = jax.jit(solve)
    eager_output = solve(parameters)
    compiled_output = jitted_solve(parameters)
    np.testing.assert_array_equal(compiled_output[0], eager_output[0])
    def objective(values):
        return jnp.vdot(solve(values)[0], cotangent)
    eager_gradient = jax.grad(objective)(parameters)
    compiled_gradient = jax.jit(jax.grad(objective))(parameters)
    native = program.evaluate(tuple(np.asarray(value) for value in parameters))
    expected_gradient = native.vjp((np.asarray(cotangent),)).input_cotangent
    assert_product(eager_gradient, expected_gradient)
    assert_product(compiled_gradient, eager_gradient)
    def apply_jvp(values, direction):
        return jax.jvp(solve, (values,), (direction,))
    eager_primal, eager_tangent = apply_jvp(parameters, tangent)
    compiled_primal, compiled_tangent = jax.jit(apply_jvp)(parameters, tangent)
    np.testing.assert_array_equal(compiled_primal[0], eager_primal[0])
    assert_product(compiled_tangent[0], eager_tangent[0])
    primal_ir = str(jitted_solve.lower(parameters).compiler_ir())
    gradient_ir = str(jax.jit(jax.grad(objective)).lower(parameters).compiler_ir())
    jvp_ir = str(jax.jit(apply_jvp).lower(parameters, tangent).compiler_ir())
    assert "eqiora_differentiable_primal_v2" in primal_ir
    assert "eqiora_differentiable_vjp_v2" in gradient_ir
    assert "eqiora_differentiable_jvp_v2" in jvp_ir
    for lowered in (primal_ir, gradient_ir, jvp_ir):
        assert "stablehlo.custom_call" in lowered
        assert "xla_python_cpu_callback" not in lowered
        assert "pure_callback" not in lowered


def test_zero_actions_and_compiled_executable_lifetime_are_safe() -> None:
    program = differentiable_program(eqiora.fem.Q1())
    solve = eqjax.bind(program)
    parameters = scalar_leaves(jnp.array([17.0, 1.2, 0.1], dtype=jnp.float64))
    zero_tangent = tuple(jnp.zeros(leaf.shape, dtype=jnp.float64) for leaf in solve.inputs)
    zero_cotangent = tuple(jnp.zeros(leaf.shape, dtype=jnp.float64) for leaf in solve.outputs)
    _, tangent = jax.jvp(solve, (parameters,), (zero_tangent,))
    _, pullback = jax.vjp(solve, parameters)
    input_cotangent = pullback(zero_cotangent)[0]
    np.testing.assert_allclose(np.asarray(tangent[0]), 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(input_cotangent), 0.0, rtol=0.0, atol=0.0)
    compiled = jax.jit(solve).lower(parameters).compile()
    pending = compiled(parameters)
    del solve
    del program
    gc.collect()
    jax.block_until_ready(pending)
    replay = jax.block_until_ready(compiled(parameters))
    np.testing.assert_array_equal(replay[0], pending[0])


@pytest.mark.parametrize(
    ("parameters", "error"),
    [
        (scalar_leaves(jnp.ones(3, dtype=jnp.float32)), TypeError),
        (scalar_leaves(jnp.ones((1, 3), dtype=jnp.float64)), ValueError),
        (scalar_leaves(jnp.ones(2, dtype=jnp.float64)), ValueError),
        (scalar_leaves(np.ones(3, dtype=np.float64)), TypeError),
        (jnp.ones(3, dtype=jnp.float64), TypeError),
        ((jnp.asarray(1.),) * 3, TypeError),
        (1.0, TypeError),
    ],
)
def test_abstract_inputs_fail_before_native_execution(parameters, error) -> None:
    solve = eqjax.bind(
        differentiable_program(eqiora.fem.Q1())
    )
    with pytest.raises(error):
        solve(parameters)


def test_nonfinite_and_unknown_program_identity_fail_closed() -> None:
    solve = eqjax.bind(differentiable_program(eqiora.fem.Q1()))
    nonfinite = scalar_leaves(jnp.array([17.0, jnp.nan, 0.1], dtype=jnp.float64))
    with pytest.raises(jax.errors.JaxRuntimeError, match="finite"):
        jax.block_until_ready(solve(nonfinite))
    parameters = scalar_leaves(jnp.array([17.0, 1.2, 0.1], dtype=jnp.float64))
    nonfinite_tangent = scalar_leaves(jnp.array([0.3, jnp.inf, 0.4], dtype=jnp.float64))
    with pytest.raises(jax.errors.JaxRuntimeError, match="finite"):
        jax.block_until_ready(jax.jvp(solve, (parameters,), (nonfinite_tangent,))[1])
    _, pullback = jax.vjp(solve, parameters)
    nonfinite_cotangent = (jnp.full(solve.outputs[0].shape, jnp.nan, dtype=jnp.float64),)
    with pytest.raises(jax.errors.JaxRuntimeError, match="finite"):
        jax.block_until_ready(pullback(nonfinite_cotangent)[0])
    # Forge only the private FFI key after constructing an otherwise valid call.
    coordinates = jnp.stack(parameters)
    input_aval = jax.typeof(coordinates)
    output_aval = input_aval.update(shape=solve.outputs[0].shape, weak_type=False)
    forged = dict(solve._params)
    forged["program_key"] = "0" * 64
    with pytest.raises(jax.errors.JaxRuntimeError, match="not registered"):
        eqjax._Solve(input_aval, output_aval, forged)(coordinates).block_until_ready()


def test_sharded_input_is_rejected_without_implicit_gather() -> None:
    if len(jax.devices("cpu")) < 2:
        pytest.skip("the JAX evidence gate supplies two host devices")
    _, program = structured_map_program()
    solve = eqjax.bind(program)
    mesh = jax.make_mesh((2,), ("partition",), devices=jax.devices("cpu")[:2])
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("partition"))
    shaped = jax.device_put(jnp.array([1+2j, 3-4j], dtype=jnp.complex128), sharding)
    parameters = (jnp.asarray(2., dtype=jnp.float64), shaped)
    with pytest.raises(ValueError, match="unsharded"):
        solve(parameters)
    with pytest.raises(NotImplementedError, match="sharding"):
        jax.block_until_ready(jax.jit(solve)(parameters))


def test_one_host_cpu_ordinal_is_preserved_without_transfer() -> None:
    if len(jax.devices("cpu")) < 2:
        pytest.skip("the JAX evidence gate supplies two host devices")
    solve = eqjax.bind(differentiable_program(eqiora.fem.Q1()))
    device = jax.devices("cpu")[1]
    parameters = scalar_leaves(jax.device_put(jnp.array([17.0, 1.2, 0.1], dtype=jnp.float64), device))
    eager, = jax.block_until_ready(solve(parameters))
    compiled, = jax.block_until_ready(jax.jit(solve)(parameters))
    assert eager.devices() == compiled.devices() == {device}
    np.testing.assert_array_equal(compiled, eager)
    batch = tuple(jax.device_put(jnp.stack((leaf, leaf)), device) for leaf in parameters)
    mapped, = jax.block_until_ready(jax.jit(jax.vmap(solve))(batch))
    assert mapped.devices() == {device}
    np.testing.assert_array_equal(mapped, np.stack((np.asarray(eager),) * 2))


def test_unsupported_transformations_fail_explicitly() -> None:
    solve = eqjax.bind(differentiable_program(eqiora.fem.Q1()))
    parameters = scalar_leaves(jnp.array([17.0, 1.2, 0.1], dtype=jnp.float64))
    tangent = scalar_leaves(jnp.array([0.3, -0.2, 0.4], dtype=jnp.float64))
    batch = tuple(jnp.stack((leaf, leaf)) for leaf in parameters)
    def gradient(values):
        return jax.grad(lambda point: jnp.sum(solve(point)[0]))(values)
    with pytest.raises(NotImplementedError, match="first-order|higher-order"):
        jax.jacfwd(gradient)(parameters)
    def forward(values):
        return jax.jvp(solve, (values,), (tangent,))[1]
    with pytest.raises(NotImplementedError, match="first-order|higher-order"):
        jax.jvp(forward, (parameters,), (tangent,))
    with pytest.raises(NotImplementedError):
        jax.linearize(solve, parameters)
    device_batch = tuple(jnp.stack([leaf] * jax.local_device_count()) for leaf in parameters)
    with pytest.raises(NotImplementedError, match="pmap"):
        jax.pmap(solve)(device_batch)
    with pytest.raises(NotImplementedError, match="named axes|collectives"):
        jax.vmap(lambda point: jax.lax.psum(solve(point)[0], "samples"), axis_name="samples")(batch)


def test_registration_identity_is_deterministic_and_deduplicated() -> None:
    program = differentiable_program(eqiora.fem.Q1())
    first = eqjax.bind(program)
    second = eqjax.bind(program)
    assert first._program_key == second._program_key
    assert len(first._program_key) == 64
    assert set(first._program_key) <= set("0123456789abcdef")


def test_bound_program_configuration_is_immutable() -> None:
    solve = eqjax.bind(
        differentiable_program(eqiora.fem.Q1())
    )
    with pytest.raises(AttributeError, match="immutable"):
        solve._program_key = "0" * 64
    with pytest.raises(TypeError):
        solve._params["program_key"] = "0" * 64
    with pytest.raises(AttributeError, match="immutable"):
        del solve._params


def test_exact_jaxlib_header_matches_allowlisted_native_abi(tmp_path: Path) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        if os.environ.get("EQIORA_REQUIRE_JAX_ABI_PROBE") == "1":
            pytest.fail("the exact JAX ABI gate requires a C compiler")
        pytest.skip("a C compiler is unavailable")
    include = Path(jaxlib.__file__).resolve().parent / "include"
    source = tmp_path / "xla_ffi_layout.c"
    executable = tmp_path / "xla_ffi_layout"
    source.write_text(
        r"""
#include <stddef.h>
#include <stdio.h>
#include "xla/ffi/api/c_api.h"

int main(void) {
  printf("api_major=%d\n", XLA_FFI_API_MAJOR);
  printf("api_minor=%d\n", XLA_FFI_API_MINOR);
  printf("extension_metadata=%d\n", XLA_FFI_Extension_Metadata);
  printf("execution_stage_execute=%d\n", XLA_FFI_ExecutionStage_EXECUTE);
  printf("arg_type_buffer=%d\n", XLA_FFI_ArgType_BUFFER);
  printf("attr_type_string=%d\n", XLA_FFI_AttrType_STRING);
  printf("data_type_f64=%d\n", XLA_FFI_DataType_F64);
  printf("error_invalid_argument=%d\n", XLA_FFI_Error_Code_INVALID_ARGUMENT);
  printf("error_not_found=%d\n", XLA_FFI_Error_Code_NOT_FOUND);
  printf("error_failed_precondition=%d\n",
         XLA_FFI_Error_Code_FAILED_PRECONDITION);
  printf("error_internal=%d\n", XLA_FFI_Error_Code_INTERNAL);
  printf("error_data_loss=%d\n", XLA_FFI_Error_Code_DATA_LOSS);
  printf("extension_base_size=%zu\n", sizeof(XLA_FFI_Extension_Base));
  printf("api_version_size=%zu\n", sizeof(XLA_FFI_Api_Version));
  printf("error_create_args_size=%zu\n", sizeof(XLA_FFI_Error_Create_Args));
  printf("buffer_size=%zu\n", sizeof(XLA_FFI_Buffer));
  printf("args_size=%zu\n", sizeof(XLA_FFI_Args));
  printf("rets_size=%zu\n", sizeof(XLA_FFI_Rets));
  printf("byte_span_size=%zu\n", sizeof(XLA_FFI_ByteSpan));
  printf("attrs_size=%zu\n", sizeof(XLA_FFI_Attrs));
  printf("call_frame_size=%zu\n", sizeof(XLA_FFI_CallFrame));
  printf("call_frame_attrs_offset=%zu\n", offsetof(XLA_FFI_CallFrame, attrs));
  printf("call_frame_future_offset=%zu\n", offsetof(XLA_FFI_CallFrame, future));
  printf("call_frame_required_size=%zu\n", (size_t)XLA_FFI_CallFrame_STRUCT_SIZE);
  printf("metadata_size=%zu\n", sizeof(XLA_FFI_Metadata));
  printf("metadata_traits_offset=%zu\n", offsetof(XLA_FFI_Metadata, traits));
  printf("metadata_state_type_id_offset=%zu\n",
         offsetof(XLA_FFI_Metadata, state_type_id));
  printf("metadata_required_size=%zu\n", (size_t)XLA_FFI_Metadata_STRUCT_SIZE);
  printf("metadata_extension_size=%zu\n", sizeof(XLA_FFI_Metadata_Extension));
  printf("api_error_create_offset=%zu\n",
         offsetof(XLA_FFI_Api, XLA_FFI_Error_Create));
  printf("api_error_destroy_offset=%zu\n",
         offsetof(XLA_FFI_Api, XLA_FFI_Error_Destroy));
  printf("api_device_ordinal_offset=%zu\n",
         offsetof(XLA_FFI_Api, XLA_FFI_DeviceOrdinal_Get));
  printf("api_device_ordinal_required_size=%zu\n", (size_t)XLA_FFI_Api_STRUCT_SIZE);
  printf("device_ordinal_args_size=%zu\n", sizeof(XLA_FFI_DeviceOrdinal_Get_Args));
  printf("device_ordinal_args_required_size=%zu\n",
         (size_t)XLA_FFI_DeviceOrdinal_Get_Args_STRUCT_SIZE);
  printf("error_destroy_args_size=%zu\n", sizeof(XLA_FFI_Error_Destroy_Args));
  return 0;
}
""",
        encoding="utf-8",
    )
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-I",
            str(include),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    output = subprocess.run(
        [str(executable)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    observed = {
        key: int(value)
        for line in output.splitlines()
        for key, value in [line.split("=", maxsplit=1)]
    }
    assert observed == dict(_eqiora._jax_ffi_abi_layout())


def test_supported_versions_and_evidence_interpreter_are_exact() -> None:
    assert jax.__version__ == "0.11.0"
    assert jaxlib.__version__ == "0.11.0"
    if expected := os.environ.get("EQIORA_TEST_JAX_VERSION"):
        assert jax.__version__ == expected
        assert jaxlib.__version__ == expected
    if expected_python := os.environ.get("EQIORA_TEST_PYTHON_VERSION"):
        assert f"{sys.version_info.major}.{sys.version_info.minor}" == expected_python


def assert_product(actual, expected) -> None:
    # Reuse the already admitted pointwise product precision, not fitted draws.
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2.0e-11, atol=2.0e-12)


@pytest.mark.parametrize("method", [eqiora.fem.Q1(), eqiora.fvm.CellCenteredTpfa()])
def test_nested_maps_and_first_order_products_preserve_every_occurrence(method) -> None:
    program = differentiable_program(method)
    solve = eqjax.bind(program)
    points = np.array([[[3.0, 2.0, 0.1], [1.0, 1.0, -0.2], [3.0, 2.0, 0.1]],
                       [[2.0, 0.5, 0.3], [3.0, 2.0, 0.1], [1.0, 1.0, -0.2]]])
    accepted = program.map(tuple(np.ascontiguousarray(leaf) for leaf in scalar_leaves(points)),
                           point_shape=points.shape[:-1]).execute()
    reference = accepted.primal()[0]
    individually_accepted = [program.evaluate(tuple(point)) for point in points.reshape(-1, 3)]
    np.testing.assert_array_equal(reference.reshape(-1, *solve.outputs[0].shape),
                                  [member.primal().output[0] for member in individually_accepted])
    for index, point in enumerate(points.reshape(-1, 3)):
        np.testing.assert_array_equal(accepted[index].point, point)
    mapped = jax.vmap(jax.vmap(solve))
    values = scalar_leaves(jnp.asarray(points))
    for operation in [mapped, jax.jit(mapped)]:
        np.testing.assert_array_equal(operation(values)[0], reference)
    directions = np.arange(points.size, dtype=np.float64).reshape(points.shape) / 8 - 1
    expected_jvp = accepted.jvp(tuple(np.ascontiguousarray(leaf) for leaf in scalar_leaves(directions))).tangent[0]
    for operation in [lambda p, t: jax.jvp(mapped, (p,), (t,)),
                      jax.jit(lambda p, t: jax.jvp(mapped, (p,), (t,)))]:
        (primal,), (tangent,) = operation(values, scalar_leaves(jnp.asarray(directions)))
        np.testing.assert_array_equal(primal, reference)
        assert_product(tangent, expected_jvp)
    cotangents = np.linspace(-0.5, 1.0, reference.size).reshape(reference.shape)
    expected_vjp = np.stack(accepted.vjp((cotangents,)).mapped_cotangents, axis=-1)
    reverse = lambda p, c: jax.vjp(mapped, p)[1](c)[0]
    assert_product(np.stack(reverse(values, (jnp.asarray(cotangents),)), axis=-1), expected_vjp)
    assert_product(np.stack(jax.jit(reverse)(values, (jnp.asarray(cotangents),)), axis=-1), expected_vjp)
    gradients = jax.vmap(jax.vmap(jax.grad(lambda p, c: jnp.vdot(solve(p)[0], c))))
    assert_product(np.stack(gradients(values, jnp.asarray(cotangents)), axis=-1), expected_vjp)
    assert_product(np.stack(jax.jit(gradients)(values, jnp.asarray(cotangents)), axis=-1), expected_vjp)
    # A non-leading outer point axis and a last inner output axis remain explicit.
    permuted = jax.vmap(jax.vmap(solve, in_axes=0, out_axes=-1), in_axes=1, out_axes=0)
    assert_product(jax.jit(permuted)(tuple(leaf.T for leaf in values))[0], np.moveaxis(reference, 1, -1))
    flat = tuple(leaf.reshape(-1) for leaf in values)
    shared_direction = scalar_leaves(jnp.array([0.5, -0.25, 0.125]))
    forward = jax.vmap(lambda p, t: jax.jvp(solve, (p,), (t,))[1], in_axes=(0, None), out_axes=1)
    expected = program.map(tuple(np.asarray(leaf) for leaf in flat), point_shape=(len(flat[0]),)).execute().jvp(
        tuple(np.full(len(flat[0]), float(leaf)) for leaf in shared_direction)).tangent[0]
    assert_product(jax.jit(forward)(flat, shared_direction)[0], expected.T)
    reverse_nonleading = jax.vmap(lambda p, c: jax.vjp(solve, p)[1]((c,))[0], in_axes=(0, 1), out_axes=-1)
    assert_product(jax.jit(reverse_nonleading)(flat, jnp.asarray(cotangents.reshape(-1, solve.outputs[0].shape[0]).T)),
                   expected_vjp.reshape(-1, 3).T)


@pytest.mark.parametrize("method", [eqiora.fem.Q1(), eqiora.fvm.CellCenteredTpfa()])
def test_shared_input_gradients_have_the_independent_sum_and_mean(method) -> None:
    program = differentiable_program(method)
    solve = eqjax.bind(program)
    sources = jnp.array([[2.0, 3.0, 2.0], [1.0, 4.0, 1.0]])
    offsets = jnp.array([-0.1, 0.2])
    # Linearity of the admitted elliptic operator gives u(s,k,b)=s*q/k+b.
    # q is the separately accepted unit-source response, not a batched gradient.
    q = program.evaluate(tuple(np.array([1.0, 1.0, 0.0]))).primal().output[0]
    q_mean = float(np.mean(q))

    def losses(diffusion, boundary, source):
        return jax.vmap(lambda b, row: jax.vmap(
            lambda s: jnp.mean(solve((s, diffusion, b))[0]))(row))(boundary, source)

    # k is shared across both axes; b is shared only within each row. The
    # native global-shared partition cannot be substituted for this association.
    total = lambda k, b, s: jnp.sum(losses(k, b, s))
    average = lambda k, b, s: jnp.mean(losses(k, b, s))
    expected_k = -13.0 * q_mean / 4.0  # sum(s)=13, k=2
    expected_b = np.array([3.0, 3.0])
    expected_s = np.full((2, 3), q_mean / 2.0)
    for operation in [total, average]:
        scale = 1.0 if operation is total else 1.0 / 6.0
        for differentiated in [jax.grad(operation, argnums=(0, 1, 2)),
                               jax.jit(jax.grad(operation, argnums=(0, 1, 2)))]:
            dk, db, ds = differentiated(jnp.asarray(2.0, dtype=jnp.float64), offsets, sources)
            assert_product(dk, scale * expected_k)
            assert_product(db, scale * expected_b)
            assert_product(ds, scale * expected_s)
    expected_value = 13.0 * q_mean / 2.0 + 3 * (-0.1 + 0.2)
    assert_product(total(jnp.asarray(2.0, dtype=jnp.float64), offsets, sources), expected_value)


@pytest.mark.parametrize("method", [eqiora.fem.Q1(), eqiora.fvm.CellCenteredTpfa()])
def test_basis_batches_reuse_points_and_distinguish_seed_axes(method) -> None:
    program = differentiable_program(method)
    solve = eqjax.bind(program)
    point = scalar_leaves(jnp.array([3.0, 2.0, 0.1]))
    accepted = program.evaluate(tuple(np.asarray(leaf) for leaf in point))
    basis = jnp.eye(3, dtype=jnp.float64)
    expected = np.array([accepted.jvp(tuple(row)).tangent[0] for row in np.asarray(basis)]).T
    for jacobian in [jax.jacfwd(solve), jax.jacrev(solve)]:
        assert_product(np.stack(jacobian(point)[0], axis=-1), expected)
        assert_product(np.stack(jax.jit(jacobian)(point)[0], axis=-1), expected)
    apply = lambda tangent: jax.jvp(solve, (point,), (tangent,))
    (primal,), (products,) = jax.jit(jax.vmap(apply, out_axes=(None, 0)))(scalar_leaves(basis))
    assert_product(primal, accepted.primal().output[0])
    assert_product(products, expected.T)
    nested_basis = tuple(jnp.stack((leaf, 2 * leaf)) for leaf in scalar_leaves(basis))
    nested = jax.vmap(jax.vmap(lambda tangent: apply(tangent)[1]))
    assert_product(jax.jit(nested)(nested_basis)[0], np.stack((expected.T, 2 * expected.T)))
    points = tuple(jnp.stack((leaf, jnp.asarray(1., dtype=jnp.float64) if i == 0 else leaf))
                   for i, leaf in enumerate(point))
    mapped = jax.vmap(solve)
    expected_full = np.zeros((2, solve.outputs[0].shape[0], 2, 3))
    for index, values in enumerate(zip(*(np.asarray(leaf) for leaf in points), strict=True)):
        evaluation = program.evaluate(values)
        expected_full[index, :, index, :] = np.array([
            evaluation.jvp(tuple(row)).tangent[0] for row in np.asarray(basis)]).T
    assert_product(np.stack(jax.jit(jax.jacfwd(mapped))(points)[0], axis=-1), expected_full)
    assert_product(np.stack(jax.jit(jax.jacrev(mapped))(points)[0], axis=-1), expected_full)
    diagonal = np.stack((expected_full[0, :, 0, :], expected_full[1, :, 1, :]))
    # Reverse transform order uses point-major p,s rather than seed-major s,p.
    assert_product(np.stack(jax.jit(jax.vmap(jax.jacfwd(solve)))(points)[0], axis=-1), diagonal)
    assert_product(np.stack(jax.jit(jax.vmap(jax.jacrev(solve)))(points)[0], axis=-1), diagonal)
    lowered = str(jax.jit(jax.jacfwd(solve)).lower(point).compiler_ir())
    assert 'batch = "s3"' in lowered
    assert "tensor<3xf64>, tensor<3x3xf64>" in lowered


def test_empty_singleton_batches_and_failed_dense_operations() -> None:
    program = differentiable_program(eqiora.fem.Q1())
    solve = eqjax.bind(program)
    mapped = jax.vmap(solve)
    for points in [jnp.empty((0, 3)), jnp.array([[1.0, 2.0, 0.0]])]:
        leaves = scalar_leaves(points)
        expected = program.map(tuple(np.ascontiguousarray(leaf) for leaf in leaves), point_shape=(len(points),)).execute()
        assert_product(jax.jit(mapped)(leaves)[0], expected.primal()[0])
        assert_product(jax.jvp(mapped, (leaves,), (tuple(jnp.zeros_like(leaf) for leaf in leaves),))[1][0],
                       np.zeros((len(points), *solve.outputs[0].shape)))
        reverse = jax.vjp(mapped, leaves)[1]((jnp.zeros((len(points), *solve.outputs[0].shape)),))[0]
        assert_product(np.stack(reverse, axis=-1), np.zeros(points.shape))
    points = scalar_leaves(jnp.empty((2, 0, 3)))
    assert jax.jit(jax.vmap(mapped))(points)[0].shape == (2, 0, *solve.outputs[0].shape)
    point = scalar_leaves(jnp.array([1.0, 2.0, 0.0]))
    empty_seed = jax.jit(jax.vmap(lambda t: jax.jvp(solve, (point,), (t,))[1]))
    assert empty_seed(scalar_leaves(jnp.empty((0, 3))))[0].shape == (0, *solve.outputs[0].shape)
    failed = jnp.array([[1.0, 2.0, 0.0], [1.0, -1.0, 0.0], [3.0, 2.0, 0.0]])
    for operation in [mapped, jax.grad(lambda p: jnp.sum(mapped(p)[0]))]:
        with pytest.raises(jax.errors.JaxRuntimeError, match="occurrence 1 failed"):
            jax.block_until_ready(jax.jit(operation)(scalar_leaves(failed)))
    valid = failed.at[1, 1].set(2.0)
    directions = jnp.zeros_like(valid).at[1].set(jnp.array([sys.float_info.max, 0.0, sys.float_info.max]))
    cotangents = jnp.zeros((3, *solve.outputs[0].shape)).at[1].set(sys.float_info.max)
    forward = jax.jit(lambda p, t: jax.jvp(mapped, (p,), (t,))[1])
    reverse = jax.jit(lambda p, c: jax.vjp(mapped, p)[1](c)[0])
    for operation, direction, role in [(forward, scalar_leaves(directions), "JVP"), (reverse, (cotangents,), "VJP")]:
        with pytest.raises(jax.errors.JaxRuntimeError, match=f"mapped {role} point occurrence 1.*seed occurrence 0"):
            jax.block_until_ready(operation(scalar_leaves(valid), direction))


def test_static_lowering_does_not_unroll_points_or_inspect_numeric_values() -> None:
    solve = eqjax.bind(differentiable_program(eqiora.fem.Q1()))
    mapped = jax.vmap(solve)
    primitive_counts = []
    for count in [1, 3, 7]:
        abstract = tuple(jax.ShapeDtypeStruct((count,), jnp.float64) for _ in solve.inputs)
        assert jax.eval_shape(mapped, abstract)[0].shape == (count, *solve.outputs[0].shape)
        primitive_counts.append(len(jax.make_jaxpr(mapped)(abstract).jaxpr.eqns))
        lowered = str(jax.jit(mapped).lower(abstract).compiler_ir())
        assert lowered.count("stablehlo.custom_call") == 1
        assert "eqiora_differentiable_primal_v2" in lowered
        for forbidden in ["callback", "stablehlo.while", "all_gather", "all_reduce"]:
            assert forbidden not in lowered
    # Leaf packing adds static shape operations; no work is unrolled per point.
    assert len(set(primitive_counts)) == 1
    traces = []
    def traced(points):
        traces.append(True)
        return mapped(points)
    compiled = jax.jit(traced)
    jax.block_until_ready(compiled(scalar_leaves(jnp.array([[1.0, 2.0, 0.0], [3.0, 1.0, 0.2]]))))
    jax.block_until_ready(compiled(scalar_leaves(jnp.array([[3.0, 1.0, 0.2], [1.0, 2.0, 0.0]]))))
    assert len(traces) == 1
    with pytest.raises(ValueError, match="bound|limit|addressable"):
        jax.eval_shape(mapped, tuple(jax.ShapeDtypeStruct((sys.maxsize,), jnp.float64) for _ in solve.inputs))
    too_deep = solve
    for _ in range(33):
        too_deep = jax.vmap(too_deep)
    with pytest.raises(ValueError, match="32|rank"):
        jax.eval_shape(too_deep, tuple(jax.ShapeDtypeStruct(tuple([1] * 33), jnp.float64) for _ in solve.inputs))


def test_structured_complex_values_use_native_signatures_and_real_linear_products():
    _, program = structured_map_program()
    solve = eqjax.bind(program)
    assert solve.inputs == program.inputs
    assert solve.outputs == program.outputs
    point = (jnp.asarray(2., dtype=jnp.float64), jnp.array([1+2j, 3-4j], dtype=jnp.complex128))
    direction = (jnp.asarray(.5, dtype=jnp.float64), jnp.array([1-2j, -3+4j], dtype=jnp.complex128))
    native = program.evaluate(tuple(np.asarray(leaf) for leaf in point))
    for operation in (solve, jax.jit(solve)):
        output = operation(point)
        assert tuple(leaf.shape for leaf in output) == ((2,), (), ())
        assert tuple(leaf.dtype for leaf in output) == (jnp.complex128, jnp.float64, jnp.float64)
        for actual, expected in zip(output, native.primal().output, strict=True):
            assert_product(actual, expected)
        # z=p*c+conj(c), loss=sum|z|²+p and margin=p-1, independently evaluated.
        assert_product(output[0], [3+2j, 9-4j])
        assert_product(output[1:], [112., 1.])
    # dz=c*dp+p*dc+conj(dc), dloss=2Re(conj(z)*dz)+dp.
    dz = np.array([3.5-1j, -7.5+2j])
    dloss = 2*np.vdot(np.array([3+2j, 9-4j]), dz).real + .5
    for operation in (lambda p, t: jax.jvp(solve, (p,), (t,)),
                      jax.jit(lambda p, t: jax.jvp(solve, (p,), (t,)))):
        _, tangent = operation(point, direction)
        for actual, expected in zip(tangent, (dz, dloss, .5), strict=True):
            assert_product(actual, expected)
    # JAX cotangents pair as Re(sum(w*dz)); native ones pair as Re(sum(conj(w)*dz)).
    # Packing/unpacking must implement this boundary exactly once on each side.
    cotangent = (jnp.full(2, 2+3j, dtype=jnp.complex128),
                 jnp.asarray(.5, dtype=jnp.float64), jnp.asarray(-2., dtype=jnp.float64))
    native_reverse = native.vjp(tuple(np.asarray(leaf).conj() for leaf in cotangent))
    for operation in (lambda p, c: jax.vjp(solve, p)[1](c)[0],
                      jax.jit(lambda p, c: jax.vjp(solve, p)[1](c)[0])):
        reverse = operation(point, cotangent)
        for actual, expected in zip(reverse, native_reverse.input_cotangent, strict=True):
            assert_product(actual, np.asarray(expected).conj())
        left = sum(np.sum(np.asarray(w)*np.asarray(t)).real
                   for w, t in zip(cotangent, (dz, dloss, .5), strict=True))
        right = sum(np.sum(np.asarray(w)*np.asarray(t)).real
                    for w, t in zip(reverse, direction, strict=True))
        assert_product(left, right)
    for operation in (jax.grad(lambda p: solve(p)[1]), jax.jit(jax.grad(lambda p: solve(p)[1]))):
        gradient = operation(point)
        assert_product(gradient[0], 101.)
        assert_product(gradient[1], [18-4j, 54+8j])
    for bad, error in [
        (jnp.ones(5, dtype=jnp.float64), TypeError),
        ((point[0],), ValueError),
        ((point[0], jnp.ones(2, dtype=jnp.float64)), TypeError),
        ((point[0], jnp.ones((1, 2), dtype=jnp.complex128)), ValueError),
    ]:
        with pytest.raises(error):
            solve(bad)
    for name in ("input_shape", "output_shape"):
        assert not hasattr(solve, name)


def test_structured_nested_maps_preserve_shared_complex_inputs_and_empty_leaves():
    _, program = structured_map_program()
    solve = eqjax.bind(program)
    p = jnp.array([[2., 3.], [1., 2.]], dtype=jnp.float64)
    c = jnp.array([1+2j, 3-4j], dtype=jnp.complex128)
    mapped = jax.vmap(jax.vmap(solve, in_axes=((0, None),)), in_axes=((0, None),))
    z = np.asarray(p)[..., None]*np.asarray(c)+np.asarray(c).conj()
    expected = (z, np.sum(np.abs(z)**2, axis=-1)+np.asarray(p), np.asarray(p)-1)
    for operation in (mapped, jax.jit(mapped)):
        for actual, reference in zip(operation((p, c)), expected, strict=True):
            assert_product(actual, reference)
    gradient = jax.jit(jax.grad(lambda shared: jnp.sum(mapped((p, shared))[1])))(c)
    # d/dc sum(loss) has native covector sum(2*(p*z+conj(z))); JAX returns its conjugate.
    assert_product(gradient, np.sum(2*(np.asarray(p)[..., None]*z+z.conj()), axis=(0, 1)).conj())
    empty = jax.jit(mapped)((jnp.empty((2, 0), dtype=jnp.float64), c))
    assert tuple(leaf.shape for leaf in empty) == ((2, 0, 2), (2, 0), (2, 0))
    assert tuple(leaf.dtype for leaf in empty) == (jnp.complex128, jnp.float64, jnp.float64)
    points = jnp.array([2., 3.], dtype=jnp.float64)
    # The real leaf is intrinsically scalar: use a non-leading axis only on the shaped leaf.
    nonleading = jax.vmap(solve, in_axes=((0, 1),), out_axes=-1)
    values = nonleading((points, jnp.stack((c, 2*c), axis=1)))
    for index in range(2):
        scalar = solve((points[index], c*(index+1)))
        for actual, reference in zip(values, scalar, strict=True):
            assert_product(actual[..., index], reference)
    if len(jax.devices("cpu")) >= 2:
        other = jax.device_put(c, jax.devices("cpu")[1])
        with pytest.raises(ValueError, match="same CPU device ordinal"):
            solve((jnp.asarray(2., dtype=jnp.float64), other))
