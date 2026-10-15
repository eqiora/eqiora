"""Use Eqiora differentiable programs from JAX.

This optional module registers Eqiora programs with JAX through the typed XLA
FFI. Primal and derivative evaluations call native Rust directly, without
Python host callbacks.
"""

from __future__ import annotations

import math
import sys
from types import MappingProxyType
from typing import Any

import numpy as np

from . import DifferentiableProgram
from . import _eqiora

if sys.version_info < (3, 12):  # pragma: no cover - JAX cannot be installed
    raise ImportError("eqiora.jax requires Python 3.12 or newer")

try:
    import jax
    import jax.numpy as jnp
    import jaxlib
    # AxisData uses this version-local sentinel; its old jax.core export was
    # removed in the exact 0.11.0 release. Do not guess anonymity from its type.
    from jax._src.core import no_axis_name
    from jax.experimental.hijax import VJPHiPrimitive
except ImportError as error:  # pragma: no cover - depends on optional install
    raise ImportError(
        "eqiora.jax requires the optional 'jax' dependency; install eqiora[jax]"
    ) from error


_SUPPORTED_VERSION = "0.11.0"
_INTERFACE_VERSION = 2
_NUMERICAL_BYTES_LIMIT = 67_108_864

if (
    jax.__version__ != _SUPPORTED_VERSION
    or jaxlib.__version__ != _SUPPORTED_VERSION
):
    raise ImportError(
        "eqiora.jax requires the exact JAX/JAXLIB 0.11.0 pair; "
        f"found JAX {jax.__version__} and JAXLIB {jaxlib.__version__}"
    )


_TARGET_CAPSULES = _eqiora._jax_ffi_targets()
for _target_name in sorted(_TARGET_CAPSULES):
    jax.ffi.register_ffi_target(
        _target_name,
        _TARGET_CAPSULES[_target_name],
        platform="cpu",
        api_version=1,
    )

_PRIMAL_TARGET = "eqiora_differentiable_primal_v2"
_JVP_TARGET = "eqiora_differentiable_jvp_v2"
_VJP_TARGET = "eqiora_differentiable_vjp_v2"


def _validate_aval(aval: Any, shape: tuple[int, ...], *, role: str, dtype=np.float64) -> None:
    if tuple(aval.shape) != shape:
        raise ValueError(f"{role} must have exact static shape {shape}")
    if any(type(extent) is not int or extent < 0 for extent in shape):
        raise ValueError(f"{role} requires nonnegative static integer dimensions")
    dtype = np.dtype(dtype)
    # Check nonzero dimensions as well: an empty axis cannot hide bad strides.
    if math.prod(max(extent, 1) for extent in shape) > sys.maxsize // dtype.itemsize:
        raise ValueError(f"{role} shape/byte product is not addressable")
    if math.prod(shape) * dtype.itemsize > _NUMERICAL_BYTES_LIMIT:
        raise ValueError(f"{role} exceeds the JAX numerical buffer limit")
    if np.dtype(aval.dtype) != dtype:
        raise TypeError(f"{role} must have dtype {dtype.name}")
    if getattr(aval, "weak_type", False):
        raise TypeError(f"{role} must not have a weak dtype")
    sharding = getattr(aval, "sharding", None)
    mesh = getattr(sharding, "mesh", None)
    if mesh is not None and not mesh.empty:
        raise NotImplementedError(
            "Eqiora JAX explicit sharding and pmap are not supported"
        )


def _validate_concrete_device(value: Any, *, role: str):
    if isinstance(value, jax.core.Tracer):
        return
    if not isinstance(value, jax.Array):
        raise TypeError(f"{role} must be an already placed JAX array")
    devices = getattr(value, "devices", None)
    if devices is None:  # pragma: no cover - every admitted JAX Array has this
        raise TypeError(f"{role} does not expose JAX device placement")
    selected = devices()
    if len(selected) != 1:
        raise ValueError(f"{role} must be an unsharded single-device array")
    (device,) = tuple(selected)
    if device.platform != "cpu":
        raise ValueError(f"{role} must reside on one host CPU device")
    return device


def _leaf_dtype(signature):
    # Scalar domain and intrinsic shape are native signature projections.
    domain = signature.value_type.scalar_domain
    if domain == "real":
        return np.dtype(np.float64)
    if domain == "complex":
        return np.dtype(np.complex128)
    raise TypeError(f"unsupported native differentiable scalar domain {domain}")


def _pack(parameters, signatures):
    if not isinstance(parameters, (tuple, list)):
        raise TypeError("parameters must be an ordered tuple or list of value leaves")
    if len(parameters) != len(signatures):
        raise ValueError(f"parameters require {len(signatures)} ordered value leaves")
    coordinates = []
    device = None
    for index, (value, signature) in enumerate(zip(parameters, signatures, strict=True)):
        role = f"parameter leaf {index}"
        selected = _validate_concrete_device(value, role=role)
        if selected is not None:
            if device is not None and device != selected:
                raise ValueError("parameter leaves must reside on the same CPU device ordinal")
            device = selected
        dtype = _leaf_dtype(signature)
        _validate_aval(jax.typeof(value), signature.shape, role=role, dtype=dtype)
        parts = (jnp.stack((value.real, value.imag), axis=-1)
                 if dtype == np.dtype(np.complex128) else value)
        coordinates.append(parts.reshape(-1))
    # Native Program admission requires a nonempty Parameter selection.
    return jnp.concatenate(coordinates)


def _unpack(coordinates, signatures):
    leaves = []
    offset = 0
    for signature in signatures:
        complex_leaf = _leaf_dtype(signature) == np.dtype(np.complex128)
        count = math.prod(signature.shape) * (2 if complex_leaf else 1)
        parts = coordinates[offset:offset+count]
        if complex_leaf:
            parts = parts.reshape((*signature.shape, 2))
            leaf = jax.lax.complex(parts[..., 0], parts[..., 1])
        else:
            leaf = parts.reshape(signature.shape)
        leaves.append(leaf)
        offset += count
    if offset != coordinates.shape[0]:
        raise ValueError("native output signatures differ from the FFI coordinate shape")
    return tuple(leaves)


def _static_params(
    program_key: str,
    input_size: int,
    output_size: int,
) -> dict[str, Any]:
    return {
        "program_key": program_key,
        "input_size": input_size,
        "output_size": output_size,
        "dtype": "float64",
        "device": "host-cpu",
        "interface_version": _INTERFACE_VERSION,
        "grid_shape": (),
        "point_axes": (),
    }


def _ffi_call(
    target: str,
    outputs: Any,
    *inputs: Any,
    program_key: str,
    grid_shape: tuple[int, ...],
    point_axes: tuple[int, ...],
) -> Any:
    # FFI layouts use major-to-minor order; all buffers are dense row-major.
    input_layouts = [tuple(range(value.ndim)) for value in inputs]
    output_layouts = (
        [tuple(range(len(aval.shape))) for aval in outputs]
        if isinstance(outputs, tuple)
        else tuple(range(len(outputs.shape)))
    )
    batch = ",".join(
        f"{'p' if axis in point_axes else 's'}{extent}"
        for axis, extent in enumerate(grid_shape)
    )
    try:
        return jax.ffi.ffi_call(
            target,
            outputs,
            has_side_effect=False,
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            input_output_aliases={},
            custom_call_api_version=4,
        )(*inputs, program_key=program_key, batch=batch)
    except ValueError as error:
        message = str(error)
        if any(
            message.startswith(f"{status}:")
            for status in (
                "INVALID_ARGUMENT",
                "NOT_FOUND",
                "FAILED_PRECONDITION",
                "INTERNAL",
                "DATA_LOSS",
            )
        ):
            raise jax.errors.JaxRuntimeError(message) from error
        raise


class _Batching:
    """One static axis projection, not a numerical or per-point executor."""

    def _initialize(self, input_aval, output_aval, params):
        self.params = params
        grid = params["grid_shape"]
        axes = params["point_axes"]
        if len(grid) > 32 or axes != tuple(sorted(set(axes))):
            raise ValueError("Eqiora JAX point/seed axes exceed the admitted rank")
        if any(axis < 0 or axis >= len(grid) for axis in axes):
            raise ValueError("Eqiora JAX point axis is out of bounds")
        point_shape = tuple(grid[axis] for axis in axes)
        _validate_aval(
            input_aval, (*point_shape, params["input_size"]), role="parameters"
        )
        _validate_aval(
            output_aval, (*point_shape, params["output_size"]), role="output"
        )
        self.point_output_aval = output_aval

    def _grid_aval(self, aval, width):
        result = aval.update(shape=(*self.params["grid_shape"], width))
        _validate_aval(result, result.shape, role="derivative grid")
        return result

    def batch(self, axis_data, args, dims):
        if (axis_data.name is not no_axis_name or axis_data.spmd_name
                or axis_data.explicit_mesh_axis):
            raise NotImplementedError(
                "Eqiora JAX named axes, collectives and sharding are not supported"
            )
        size = axis_data.size
        if type(size) is not int or size < 0 or len(self.grid_shape) >= 32:
            raise ValueError(
                "Eqiora JAX batching requires at most 32 bounded static axes"
            )
        parameter_axis = dims[0]
        parameters = (args[0] if parameter_axis is None
                      else jnp.moveaxis(args[0], parameter_axis, 0))
        shifted = tuple(axis + 1 for axis in self.point_axes)
        params = dict(
            self.params, grid_shape=(size, *self.grid_shape),
            point_axes=shifted if parameter_axis is None else (0, *shifted),
        )
        input_aval = jax.typeof(parameters)
        output_aval = self.point_output_aval.update(
            shape=(*input_aval.shape[:-1], self.output_size)
        )
        operation = type(self)(input_aval, output_aval, params)
        arguments = [parameters]
        for value, axis in zip(args[1:], dims[1:], strict=True):
            arguments.append(jnp.broadcast_to(value, (size, *value.shape))
                             if axis is None else jnp.moveaxis(value, axis, 0))
        output_dims = ((None if parameter_axis is None else 0, 0)
                       if isinstance(self, _JvpCall) else 0)
        return operation(*arguments), output_dims

    def _expand(self, target, *inputs):
        return _ffi_call(
            target, self.out_aval, *inputs, program_key=self.program_key,
            grid_shape=self.grid_shape, point_axes=self.point_axes,
        )

    def jvp(self, primals, tangents):
        raise NotImplementedError(
            "Eqiora JAX supports first-order products only, not derivatives of products"
        )

    def vjp_fwd(self, nonzero_inputs, *args):
        raise NotImplementedError(
            "Eqiora JAX supports first-order products only, not derivatives of products"
        )

    def lin(self, nonzero_inputs, *args):
        raise NotImplementedError(
            "Eqiora JAX linearize and higher-order derivatives are not supported"
        )


class _PrimalCall(_Batching, VJPHiPrimitive):
    def __init__(self, input_aval, output_aval, params) -> None:
        self._initialize(input_aval, output_aval, params)
        self.in_avals = (input_aval,)
        self.out_aval = output_aval
        super().__init__()

    def expand(self, parameters):
        return self._expand(_PRIMAL_TARGET, parameters)


class _JvpCall(_Batching, VJPHiPrimitive):
    def __init__(self, input_aval, output_aval, params) -> None:
        self._initialize(input_aval, output_aval, params)
        self.in_avals = (input_aval, self._grid_aval(input_aval, params["input_size"]))
        self.out_aval = (output_aval, self._grid_aval(output_aval, params["output_size"]))
        super().__init__()

    def expand(self, parameters, tangent):
        return self._expand(_JVP_TARGET, parameters, tangent)


class _VjpCall(_Batching, VJPHiPrimitive):
    def __init__(self, input_aval, output_aval, params) -> None:
        self._initialize(input_aval, output_aval, params)
        self.in_avals = (input_aval, self._grid_aval(output_aval, params["output_size"]))
        self.out_aval = self._grid_aval(input_aval, params["input_size"])
        super().__init__()

    def expand(self, parameters, cotangent):
        return self._expand(_VJP_TARGET, parameters, cotangent)


class _Solve(_Batching, VJPHiPrimitive):
    def __init__(self, input_aval, output_aval, params) -> None:
        self._initialize(input_aval, output_aval, params)
        self.in_avals = (input_aval,)
        self.out_aval = output_aval
        super().__init__()

    def expand(self, parameters):
        return _PrimalCall(
            self.in_avals[0],
            self.out_aval,
            self.params,
        )(parameters)

    def vjp_fwd(self, nonzero_inputs, parameters):
        del nonzero_inputs
        output = _PrimalCall(
            self.in_avals[0],
            self.out_aval,
            self.params,
        )(parameters)
        return output, parameters

    def vjp_bwd_retval(self, parameters, cotangent):
        return (
            _VjpCall(
                self.in_avals[0],
                self.out_aval,
                self.params,
            )(parameters, cotangent),
        )

    def jvp(self, primals, tangents):
        (parameters,), (tangent,) = primals, tangents
        return _JvpCall(
            self.in_avals[0],
            self.out_aval,
            self.params,
        )(parameters, tangent)


class JaxProgram:
    """JAX view of a program in this process, with first-order ``vmap`` support.

    Maps ordered real/complex value leaves on one CPU device. Native signatures
    own their intrinsic shapes, units and associations. Private real-coordinate
    packing is differentiated by JAX around the native real-linear actions.
    Nested point and derivative-seed axes are static and limited to 32 entries. Each FFI buffer
    is limited to 64 MiB, and native maps and products have their own 64 MiB
    storage limit. Neither limit bounds peak memory use.
    """

    __slots__ = (
        "__weakref__",
        "_input_size",
        "_output_size",
        "_params",
        "_program",
        "_program_key",
    )

    def __init__(self, program: DifferentiableProgram) -> None:
        if not isinstance(program, DifferentiableProgram):
            raise TypeError("program must be an eqiora.DifferentiableProgram")
        if program.dtype != "float64" or program.device != "cpu:0":
            raise ValueError("this JAX adapter supports host-CPU float64 programs")
        self._program = program
        self._input_size, self._output_size = program._coordinate_sizes()
        self._program_key = program._register()
        self._params = MappingProxyType(
            _static_params(
                self._program_key,
                self._input_size,
                self._output_size,
            )
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if hasattr(self, name):
            raise AttributeError("Eqiora JAX program configuration is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        del name
        raise AttributeError("Eqiora JAX program configuration is immutable")

    @property
    def program(self) -> DifferentiableProgram:
        """The framework-neutral program wrapped by this adapter."""
        return self._program

    @property
    def inputs(self):
        return self._program.inputs

    @property
    def outputs(self):
        return self._program.outputs

    def __call__(self, parameters):
        coordinates = _pack(parameters, self.inputs)
        input_aval = jax.typeof(coordinates)
        output_aval = input_aval.update(shape=(self._output_size,), weak_type=False)
        output = _Solve(input_aval, output_aval, self._params)(coordinates)
        return _unpack(output, self.outputs)


def bind(program: DifferentiableProgram) -> JaxProgram:
    """Bind a framework-neutral program to the typed JAX/XLA FFI."""
    return JaxProgram(program)


__all__ = ["JaxProgram", "bind"]
