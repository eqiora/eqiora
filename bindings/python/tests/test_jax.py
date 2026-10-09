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
        output=plan.capability.fields[0],
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
    parameters = jnp.array([17.0, 1.2, 0.1], dtype=jnp.float64)
    tangent = jnp.array([0.3, -0.2, 0.4], dtype=jnp.float64)
    cotangent = jnp.linspace(
        0.25,
        1.25,
        solve.output_shape[0],
        dtype=jnp.float64,
    )

    output = solve(parameters)
    output.block_until_ready()
    native = program.evaluate(np.asarray(parameters))
    np.testing.assert_allclose(
        np.asarray(output),
        native.primal().output.numpy(copy=False),
        rtol=0.0,
        atol=0.0,
    )
    assert output.dtype == jnp.float64
    assert output.shape == solve.output_shape
    assert output.devices() == {jax.devices("cpu")[0]}

    jvp_output, output_tangent = jax.jvp(
        solve,
        (parameters,),
        (tangent,),
    )
    native_jvp = native.jvp(np.asarray(tangent))
    np.testing.assert_allclose(
        np.asarray(jvp_output),
        native_jvp.output.numpy(copy=False),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(output_tangent),
        native_jvp.tangent.numpy(copy=False),
        rtol=2.0e-11,
        atol=2.0e-12,
    )

    vjp_output, pullback = jax.vjp(solve, parameters)
    input_cotangent = pullback(cotangent)[0]
    native_vjp = native.vjp(np.asarray(cotangent))
    np.testing.assert_allclose(
        np.asarray(vjp_output),
        native_vjp.output.numpy(copy=False),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(input_cotangent),
        native_vjp.input_cotangent.numpy(copy=False),
        rtol=2.0e-11,
        atol=2.0e-12,
    )


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
    parameters = jnp.array([17.0, 1.2, 0.1], dtype=jnp.float64)
    tangent = jnp.array([0.3, -0.2, 0.4], dtype=jnp.float64)
    cotangent = jnp.linspace(
        0.25,
        1.25,
        solve.output_shape[0],
        dtype=jnp.float64,
    )

    jitted_solve = jax.jit(solve)
    eager_output = solve(parameters)
    compiled_output = jitted_solve(parameters)
    np.testing.assert_allclose(
        np.asarray(compiled_output),
        np.asarray(eager_output),
        rtol=0.0,
        atol=0.0,
    )

    def objective(values):
        return jnp.vdot(solve(values), cotangent)

    eager_gradient = jax.grad(objective)(parameters)
    compiled_gradient = jax.jit(jax.grad(objective))(parameters)
    native = program.evaluate(np.asarray(parameters))
    expected_gradient = native.vjp(np.asarray(cotangent)).input_cotangent
    np.testing.assert_allclose(
        np.asarray(eager_gradient),
        expected_gradient.numpy(copy=False),
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(compiled_gradient),
        np.asarray(eager_gradient),
        rtol=2.0e-11,
        atol=2.0e-12,
    )

    def apply_jvp(values, direction):
        return jax.jvp(solve, (values,), (direction,))

    eager_primal, eager_tangent = apply_jvp(parameters, tangent)
    compiled_primal, compiled_tangent = jax.jit(apply_jvp)(parameters, tangent)
    np.testing.assert_allclose(
        np.asarray(compiled_primal),
        np.asarray(eager_primal),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(compiled_tangent),
        np.asarray(eager_tangent),
        rtol=2.0e-11,
        atol=2.0e-12,
    )

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
    parameters = jnp.array([17.0, 1.2, 0.1], dtype=jnp.float64)
    zero_tangent = jnp.zeros(solve.input_shape, dtype=jnp.float64)
    zero_cotangent = jnp.zeros(solve.output_shape, dtype=jnp.float64)

    _, tangent = jax.jvp(solve, (parameters,), (zero_tangent,))
    _, pullback = jax.vjp(solve, parameters)
    input_cotangent = pullback(zero_cotangent)[0]
    np.testing.assert_allclose(np.asarray(tangent), 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(input_cotangent),
        0.0,
        rtol=0.0,
        atol=0.0,
    )

    compiled = jax.jit(solve).lower(parameters).compile()
    pending = compiled(parameters)
    del solve
    del program
    gc.collect()
    pending.block_until_ready()
    replay = compiled(parameters)
    replay.block_until_ready()
    np.testing.assert_allclose(
        np.asarray(replay),
        np.asarray(pending),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("parameters", "error"),
    [
        (jnp.ones(3, dtype=jnp.float32), TypeError),
        (jnp.ones((1, 3), dtype=jnp.float64), ValueError),
        (jnp.ones(2, dtype=jnp.float64), ValueError),
        (np.ones(3, dtype=np.float64), TypeError),
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
    solve = eqjax.bind(
        differentiable_program(eqiora.fem.Q1())
    )
    nonfinite = jnp.array([17.0, jnp.nan, 0.1], dtype=jnp.float64)
    with pytest.raises(jax.errors.JaxRuntimeError, match="finite"):
        solve(nonfinite).block_until_ready()

    parameters = jnp.array([17.0, 1.2, 0.1], dtype=jnp.float64)
    nonfinite_tangent = jnp.array([0.3, jnp.inf, 0.4], dtype=jnp.float64)
    with pytest.raises(jax.errors.JaxRuntimeError, match="finite"):
        jax.jvp(solve, (parameters,), (nonfinite_tangent,))[1].block_until_ready()
    _, pullback = jax.vjp(solve, parameters)
    nonfinite_cotangent = jnp.full(
        solve.output_shape,
        jnp.nan,
        dtype=jnp.float64,
    )
    with pytest.raises(jax.errors.JaxRuntimeError, match="finite"):
        pullback(nonfinite_cotangent)[0].block_until_ready()

    input_aval = jax.typeof(parameters)
    output_aval = input_aval.update(shape=solve.output_shape, weak_type=False)
    forged = dict(solve._params)
    forged["program_key"] = "0" * 64
    with pytest.raises(jax.errors.JaxRuntimeError, match="not registered"):
        eqjax._Solve(input_aval, output_aval, forged)(parameters).block_until_ready()


def test_sharded_input_is_rejected_without_implicit_gather() -> None:
    if len(jax.devices("cpu")) < 2:
        pytest.skip("the JAX evidence gate supplies two host devices")
    solve = eqjax.bind(
        differentiable_program(
            eqiora.fem.Q1(),
            include_wave_number=True,
        )
    )
    mesh = jax.make_mesh((2,), ("partition",), devices=jax.devices("cpu")[:2])
    sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec("partition"),
    )
    parameters = jax.device_put(
        jnp.array([17.0, 1.2, 0.1, np.pi], dtype=jnp.float64),
        sharding,
    )
    with pytest.raises(ValueError, match="unsharded"):
        solve(parameters)
    with pytest.raises(NotImplementedError, match="sharding"):
        jax.jit(solve)(parameters).block_until_ready()


def test_one_host_cpu_ordinal_is_preserved_without_transfer() -> None:
    if len(jax.devices("cpu")) < 2:
        pytest.skip("the JAX evidence gate supplies two host devices")
    solve = eqjax.bind(
        differentiable_program(eqiora.fem.Q1())
    )
    device = jax.devices("cpu")[1]
    parameters = jax.device_put(
        jnp.array([17.0, 1.2, 0.1], dtype=jnp.float64),
        device,
    )

    eager = solve(parameters)
    compiled = jax.jit(solve)(parameters)
    eager.block_until_ready()
    compiled.block_until_ready()
    assert eager.devices() == {device}
    assert compiled.devices() == {device}
    np.testing.assert_allclose(
        np.asarray(compiled),
        np.asarray(eager),
        rtol=0.0,
        atol=0.0,
    )
    batch = jax.device_put(jnp.stack((parameters, parameters)), device)
    mapped = jax.jit(jax.vmap(solve))(batch)
    mapped.block_until_ready()
    assert mapped.devices() == {device}
    np.testing.assert_array_equal(np.asarray(mapped), np.stack((np.asarray(eager),) * 2))


def test_unsupported_transformations_fail_explicitly() -> None:
    solve = eqjax.bind(
        differentiable_program(eqiora.fem.Q1())
    )
    parameters = jnp.array([17.0, 1.2, 0.1], dtype=jnp.float64)
    tangent = jnp.array([0.3, -0.2, 0.4], dtype=jnp.float64)
    batch = jnp.stack((parameters, parameters))

    def gradient(values):
        return jax.grad(lambda point: jnp.sum(solve(point)))(values)

    with pytest.raises(NotImplementedError, match="first-order|higher-order"):
        jax.jacfwd(gradient)(parameters)

    def forward(values):
        return jax.jvp(solve, (values,), (tangent,))[1]

    with pytest.raises(NotImplementedError, match="first-order|higher-order"):
        jax.jvp(forward, (parameters,), (tangent,))
    with pytest.raises(NotImplementedError):
        jax.linearize(solve, parameters)
    # Reach Eqiora's unsupported-pmap gate with a valid local device axis.
    device_batch = jnp.stack([parameters] * jax.local_device_count())
    with pytest.raises(NotImplementedError, match="pmap"):
        jax.pmap(solve)(device_batch)
    with pytest.raises(NotImplementedError, match="named axes|collectives"):
        jax.vmap(lambda point: jax.lax.psum(solve(point), "samples"), axis_name="samples")(batch)


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
    accepted = program.map(points).execute()
    reference = accepted.primal()
    individually_accepted = [program.evaluate(point) for point in points.reshape(-1, 3)]
    np.testing.assert_array_equal(reference.reshape(-1, *solve.output_shape),
                                  [member.primal().output.numpy() for member in individually_accepted])
    for index, point in enumerate(points.reshape(-1, 3)):
        np.testing.assert_array_equal(accepted[index].point.numpy(), point)
    mapped = jax.vmap(jax.vmap(solve))
    values = jnp.asarray(points)
    for operation in [mapped, jax.jit(mapped)]:
        np.testing.assert_array_equal(np.asarray(operation(values)), reference)
    directions = np.arange(points.size, dtype=np.float64).reshape(points.shape) / 8 - 1
    expected_jvp = accepted.jvp(directions).tangent
    for operation in [lambda p, t: jax.jvp(mapped, (p,), (t,)),
                      jax.jit(lambda p, t: jax.jvp(mapped, (p,), (t,)))]:
        primal, tangent = operation(values, jnp.asarray(directions))
        np.testing.assert_array_equal(np.asarray(primal), reference)
        assert_product(tangent, expected_jvp)
    cotangents = np.linspace(-0.5, 1.0, reference.size).reshape(reference.shape)
    expected_vjp = accepted.vjp(cotangents).mapped_cotangents
    reverse = lambda p, c: jax.vjp(mapped, p)[1](c)[0]
    assert_product(reverse(values, jnp.asarray(cotangents)), expected_vjp)
    assert_product(jax.jit(reverse)(values, jnp.asarray(cotangents)), expected_vjp)
    gradients = jax.vmap(jax.vmap(jax.grad(lambda p, c: jnp.vdot(solve(p), c))))
    assert_product(gradients(values, jnp.asarray(cotangents)), expected_vjp)
    assert_product(jax.jit(gradients)(values, jnp.asarray(cotangents)), expected_vjp)
    # Both input axes are non-leading; the inner output axis is placed last.
    permuted = jax.vmap(jax.vmap(solve, in_axes=1, out_axes=-1), in_axes=1, out_axes=0)
    assert_product(jax.jit(permuted)(jnp.transpose(values, (2, 0, 1))),
                   np.moveaxis(reference, 1, -1))
    flat = values.reshape(-1, 3)
    shared_direction = jnp.array([0.5, -0.25, 0.125])
    forward = jax.vmap(lambda p, t: jax.jvp(solve, (p,), (t,))[1],
                       in_axes=(1, None), out_axes=1)
    expected = program.map(np.asarray(flat)).execute().jvp(
        np.broadcast_to(np.asarray(shared_direction), flat.shape).copy()).tangent
    assert_product(jax.jit(forward)(flat.T, shared_direction), expected.T)
    reverse_nonleading = jax.vmap(lambda p, c: jax.vjp(solve, p)[1](c)[0],
                                  in_axes=(1, 1), out_axes=-1)
    assert_product(jax.jit(reverse_nonleading)(flat.T, jnp.asarray(cotangents.reshape(-1, solve.output_shape[0]).T)),
                   expected_vjp.reshape(-1, 3).T)


@pytest.mark.parametrize("method", [eqiora.fem.Q1(), eqiora.fvm.CellCenteredTpfa()])
def test_shared_input_gradients_have_the_independent_sum_and_mean(method) -> None:
    program = differentiable_program(method)
    solve = eqjax.bind(program)
    sources = jnp.array([[2.0, 3.0, 2.0], [1.0, 4.0, 1.0]])
    offsets = jnp.array([-0.1, 0.2])
    # Linearity of the admitted elliptic operator gives u(s,k,b)=s*q/k+b.
    # q is the separately accepted unit-source response, not a batched gradient.
    q = program.evaluate(np.array([1.0, 1.0, 0.0])).primal().output.numpy()
    q_mean = float(np.mean(q))

    def losses(diffusion, boundary, source):
        return jax.vmap(lambda b, row: jax.vmap(
            lambda s: jnp.mean(solve(jnp.stack((s, diffusion, b)))))(row))(boundary, source)

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
            dk, db, ds = differentiated(jnp.array(2.0), offsets, sources)
            assert_product(dk, scale * expected_k)
            assert_product(db, scale * expected_b)
            assert_product(ds, scale * expected_s)
    expected_value = 13.0 * q_mean / 2.0 + 3 * (-0.1 + 0.2)
    assert_product(total(jnp.array(2.0), offsets, sources), expected_value)


@pytest.mark.parametrize("method", [eqiora.fem.Q1(), eqiora.fvm.CellCenteredTpfa()])
def test_basis_batches_reuse_points_and_distinguish_seed_axes(method) -> None:
    program = differentiable_program(method)
    solve = eqjax.bind(program)
    point = jnp.array([3.0, 2.0, 0.1])
    accepted = program.evaluate(np.asarray(point))
    basis = jnp.eye(3, dtype=jnp.float64)
    expected = np.array([accepted.jvp(row).tangent.numpy() for row in np.asarray(basis)]).T
    for jacobian in [jax.jacfwd(solve), jax.jacrev(solve)]:
        assert_product(jacobian(point), expected)
        assert_product(jax.jit(jacobian)(point), expected)
    apply = lambda tangent: jax.jvp(solve, (point,), (tangent,))
    primal, products = jax.jit(jax.vmap(apply, out_axes=(None, 0)))(basis)
    assert_product(primal, accepted.primal().output.numpy())
    assert_product(products, expected.T)
    nested_basis = jnp.stack((basis, 2 * basis))
    nested = jax.vmap(jax.vmap(lambda tangent: apply(tangent)[1]))
    assert_product(jax.jit(nested)(nested_basis), np.stack((expected.T, 2 * expected.T)))
    points = jnp.stack((point, point.at[0].set(1.0)))
    mapped = jax.vmap(solve)
    expected_full = np.zeros((2, solve.output_shape[0], 2, 3))
    for index, values in enumerate(np.asarray(points)):
        evaluation = program.evaluate(values)
        expected_full[index, :, index, :] = np.array([
            evaluation.jvp(row).tangent.numpy() for row in np.asarray(basis)]).T
    assert_product(jax.jit(jax.jacfwd(mapped))(points), expected_full)
    assert_product(jax.jit(jax.jacrev(mapped))(points), expected_full)
    diagonal = np.stack((expected_full[0, :, 0, :], expected_full[1, :, 1, :]))
    # Reverse transform order uses point-major p,s rather than seed-major s,p.
    assert_product(jax.jit(jax.vmap(jax.jacfwd(solve)))(points), diagonal)
    assert_product(jax.jit(jax.vmap(jax.jacrev(solve)))(points), diagonal)
    lowered = str(jax.jit(jax.jacfwd(solve)).lower(point).compiler_ir())
    assert 'batch = "s3"' in lowered
    assert "tensor<3xf64>, tensor<3x3xf64>" in lowered


def test_empty_singleton_batches_and_failed_dense_operations() -> None:
    program = differentiable_program(eqiora.fem.Q1())
    solve = eqjax.bind(program)
    mapped = jax.vmap(solve)
    for points in [jnp.empty((0, 3)), jnp.array([[1.0, 2.0, 0.0]])]:
        expected = program.map(np.asarray(points)).execute()
        assert_product(jax.jit(mapped)(points), expected.primal())
        assert_product(jax.jvp(mapped, (points,), (jnp.zeros_like(points),))[1],
                       np.zeros((len(points), *solve.output_shape)))
        assert_product(jax.vjp(mapped, points)[1](jnp.zeros((len(points), *solve.output_shape)))[0],
                       np.zeros(points.shape))
    points = jnp.empty((2, 0, 3))
    assert jax.jit(jax.vmap(mapped))(points).shape == (2, 0, *solve.output_shape)
    point = jnp.array([1.0, 2.0, 0.0])
    empty_seed = jax.jit(jax.vmap(lambda t: jax.jvp(solve, (point,), (t,))[1]))
    assert empty_seed(jnp.empty((0, 3))).shape == (0, *solve.output_shape)
    failed = jnp.array([[1.0, 2.0, 0.0], [1.0, -1.0, 0.0], [3.0, 2.0, 0.0]])
    for operation in [mapped, jax.grad(lambda p: jnp.sum(mapped(p)))]:
        with pytest.raises(jax.errors.JaxRuntimeError, match="occurrence 1 failed"):
            jax.jit(operation)(failed).block_until_ready()
    valid = failed.at[1, 1].set(2.0)
    directions = jnp.zeros_like(valid).at[1].set(jnp.array([sys.float_info.max, 0.0, sys.float_info.max]))
    cotangents = jnp.zeros((3, *solve.output_shape)).at[1].set(sys.float_info.max)
    forward = jax.jit(lambda p, t: jax.jvp(mapped, (p,), (t,))[1])
    reverse = jax.jit(lambda p, c: jax.vjp(mapped, p)[1](c)[0])
    for operation, direction, role in [(forward, directions, "JVP"), (reverse, cotangents, "VJP")]:
        with pytest.raises(jax.errors.JaxRuntimeError, match=f"mapped {role} point occurrence 1.*seed occurrence 0"):
            operation(valid, direction).block_until_ready()


def test_static_lowering_does_not_unroll_points_or_inspect_numeric_values() -> None:
    solve = eqjax.bind(differentiable_program(eqiora.fem.Q1()))
    mapped = jax.vmap(solve)
    primitive_counts = []
    for count in [1, 3, 7]:
        abstract = jax.ShapeDtypeStruct((count, 3), jnp.float64)
        assert jax.eval_shape(mapped, abstract).shape == (count, *solve.output_shape)
        primitive_counts.append(len(jax.make_jaxpr(mapped)(abstract).jaxpr.eqns))
        lowered = str(jax.jit(mapped).lower(abstract).compiler_ir())
        assert lowered.count("stablehlo.custom_call") == 1
        assert "eqiora_differentiable_primal_v2" in lowered
        for forbidden in ["callback", "stablehlo.while", "all_gather", "all_reduce"]:
            assert forbidden not in lowered
    assert primitive_counts == [1, 1, 1]
    traces = []
    def traced(points):
        traces.append(True)
        return mapped(points)
    compiled = jax.jit(traced)
    compiled(jnp.array([[1.0, 2.0, 0.0], [3.0, 1.0, 0.2]])).block_until_ready()
    compiled(jnp.array([[3.0, 1.0, 0.2], [1.0, 2.0, 0.0]])).block_until_ready()
    assert len(traces) == 1
    with pytest.raises(ValueError, match="bound|limit|addressable"):
        jax.eval_shape(mapped, jax.ShapeDtypeStruct((sys.maxsize, 3), jnp.float64))
    too_deep = solve
    for _ in range(33):
        too_deep = jax.vmap(too_deep)
    with pytest.raises(ValueError, match="32|rank"):
        jax.eval_shape(too_deep, jax.ShapeDtypeStruct((*([1] * 33), 3), jnp.float64))
