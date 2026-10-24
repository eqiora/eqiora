"""Use Eqiora differentiable programs with PyTorch transforms.

This optional module registers Eqiora programs as PyTorch operators. The
underlying Model, Plan, and derivative computations do not depend on PyTorch.
"""

from __future__ import annotations

from . import CompleteEvaluationMap, DifferentiableProgram, _eqiora

try:
    import torch
except ImportError as error:  # pragma: no cover - depends on optional install
    raise ImportError(
        "eqiora.torch requires the optional 'torch' dependency; "
        "install eqiora[torch]"
    ) from error


_SUPPORTED_TORCH_SERIES = (2, 14)


def _torch_series(version: str) -> tuple[int, int]:
    core = version.split("+", maxsplit=1)[0]
    parts = core.split(".")
    try:
        return (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError) as error:
        raise ImportError(f"cannot interpret PyTorch release {version!r}") from error


if _torch_series(torch.__version__) != _SUPPORTED_TORCH_SERIES:
    raise ImportError(
        "eqiora.torch supports PyTorch >=2.14,<2.15; "
        f"found {torch.__version__}"
    )


def _metadata(signatures):
    return tuple(
        (leaf.shape, torch.complex128 if leaf.value_type.scalar_domain == "complex"
         else torch.float64)
        for leaf in signatures
    )


def _validate(values, metadata, role):
    if not isinstance(values, (tuple, list)):
        raise TypeError(f'{role} must be an ordered tuple or list of tensors')
    if len(values) != len(metadata):
        raise ValueError(f'{role} must have {len(metadata)} leaves')
    for value, (shape, dtype) in zip(values, metadata, strict=True):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'{role} leaves must be tensors')
        if value.device.type != "cpu" or value.device.index not in (None, 0):
            raise ValueError(f'{role} must reside on CPU device 0')
        if value.layout != torch.strided:
            raise TypeError(f'{role} must use strided layout')
        if value.dtype != dtype:
            raise TypeError(f'{role} must have dtype {dtype}')
        if tuple(value.shape) != shape:
            raise ValueError(f'{role} must have exact shape {shape}')


def _detached(values):
    # Stage only at the native boundary; preserve framework views and version
    # counters in saved autograd inputs. No device transfer or numeric AD occurs.
    return tuple(v.detach().resolve_conj().resolve_neg().contiguous() for v in values)


def _shaped(signatures, prefix):
    return tuple(((*prefix, *shape), dtype) for shape, dtype in _metadata(signatures))


def _accepted(parameters, token, shape):
    program = _eqiora._differentiable_program(token)
    _validate(parameters, _shaped(program.inputs, shape), "parameters")
    result = program.map(_detached(parameters), point_shape=shape).execute()
    if not isinstance(result, CompleteEvaluationMap):
        raise RuntimeError(
            f"Eqiora evaluation stopped at member {result.stopped_index}: {result.diagnostics!r}"
        )
    return result


@torch.library.custom_op("eqiora::differentiable_solve", mutates_args=(), device_types="cpu")
def _solve(parameters: list[torch.Tensor], token: str, shape: list[int]) -> list[torch.Tensor]:
    return [torch.tensor(v, device="cpu") for v in _accepted(parameters, token, shape).primal()]


@_solve.register_fake
def _solve_fake(parameters, token, shape):
    # Immutable native signatures only: no data inspection or numerical work.
    program = _eqiora._differentiable_program(token)
    _validate(parameters, _shaped(program.inputs, shape), "parameters")
    return [torch.empty(s, dtype=d, device="cpu") for s, d in _shaped(program.outputs, shape)]


@torch.library.custom_op("eqiora::_differentiable_product", mutates_args=(), device_types="cpu")
def _product(parameters: list[torch.Tensor], directions: list[torch.Tensor], token: str,
             shape: list[int], seeds: list[int], transpose: bool) -> list[torch.Tensor]:
    complete = _accepted(parameters, token, shape)
    axes = list(range(len(seeds), len(seeds) + len(shape)))
    if transpose:
        result = complete.vjp(_detached(directions), seed_shape=seeds, point_axes=axes)
        values = result.mapped_cotangents
    else:
        result = complete.jvp(_detached(directions), seed_shape=seeds, point_axes=axes)
        values = result.tangent
    return [torch.tensor(v, device="cpu") for v in values]


@_product.register_fake
def _product_fake(parameters, directions, token, shape, seeds, transpose):
    program = _eqiora._differentiable_program(token)
    _validate(parameters, _shaped(program.inputs, shape), "parameters")
    inputs, outputs = ((program.outputs, program.inputs) if transpose
                       else (program.inputs, program.outputs))
    _validate(directions, _shaped(inputs, (*seeds, *shape)), "directions")
    return [torch.empty(s, dtype=d, device="cpu")
            for s, d in _shaped(outputs, (*seeds, *shape))]


def _batch_value(value, dim, size, axis=0):
    if dim is not None:
        return value.movedim(dim, axis)
    return value.unsqueeze(axis).expand(*value.shape[:axis], size, *value.shape[axis:])


def _batch_product(info, parameter_dims, direction_dims, parameters, directions,
                   token, shape, seeds, transpose):
    # Canonical layout is seed axes, then physical points, then native leaf axes.
    # A direction-only vmap adds seeds, never additional physical evaluations.
    physical = any(dim is not None for dim in parameter_dims)
    axis = len(seeds) if physical else 0
    if physical:
        parameters = [_batch_value(v, d, info.batch_size)
                      for v, d in zip(parameters, parameter_dims, strict=True)]
        shape = [info.batch_size, *shape]
    else:
        seeds = [info.batch_size, *seeds]
    directions = [_batch_value(v, d, info.batch_size, axis)
                  for v, d in zip(directions, direction_dims, strict=True)]
    result = _product(parameters, directions, token, shape, seeds, transpose)
    return result, [axis] * len(result)


@_product.register_vmap
def _product_vmap(info, in_dims, parameters, directions, token, shape, seeds, transpose):
    return _batch_product(info, in_dims[0], in_dims[1], parameters, directions,
                          token, shape, seeds, transpose)


class _Product(torch.autograd.Function):
    @staticmethod
    def forward(token, shape, seeds, transpose, count, *values):
        return tuple(_product(list(values[:count]), list(values[count:]),
                              token, shape, seeds, transpose))

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *cotangents):
        raise NotImplementedError("Eqiora native higher-order solution products are not connected")

    @staticmethod
    def jvp(ctx, *tangents):
        raise NotImplementedError("Eqiora native higher-order solution products are not connected")

    @staticmethod
    def vmap(info, in_dims, token, shape, seeds, transpose, count, *values):
        result, dims = _batch_product(
            info, in_dims[5:5 + count], in_dims[5 + count:],
            list(values[:count]), list(values[count:]), token, shape, seeds, transpose,
        )
        return tuple(result), tuple(dims)


@torch.compiler.allow_in_graph
class _Solve(torch.autograd.Function):
    # All tensors are direct arguments. Dynamo retains this transform boundary;
    # AOT traces its body, whose native operations have metadata-only fake kernels.
    @staticmethod
    def forward(token, shape, *parameters):
        return tuple(_solve(list(parameters), token, shape))

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.token, ctx.shape, *parameters = inputs
        ctx.save_for_backward(*parameters)
        ctx.save_for_forward(*parameters)

    @staticmethod
    def backward(ctx, *cotangents):
        parameters = ctx.saved_tensors
        # Keep the transform boundary only when constructing a derivative graph.
        if torch.is_grad_enabled():
            result = _Product.apply(ctx.token, ctx.shape, [], True, len(parameters),
                                    *parameters, *cotangents)
        else:
            result = _product(list(parameters), list(cotangents), ctx.token,
                              ctx.shape, [], True)
        return None, None, *result

    @staticmethod
    def jvp(ctx, _token, _shape, *tangents):
        parameters = ctx.saved_tensors
        return _Product.apply(ctx.token, ctx.shape, [], False, len(parameters),
                              *parameters, *tangents)

    @staticmethod
    def vmap(info, in_dims, token, shape, *parameters):
        values = [_batch_value(v, d, info.batch_size)
                  for v, d in zip(parameters, in_dims[2:], strict=True)]
        result = _Solve.apply(token, [info.batch_size, *shape], *values)
        return result, (0,) * len(result)


class TorchProgram:
    """Process-local PyTorch view of ordered real/complex native value leaves.

    Native signatures own each leaf's shape and scalar domain. CPU float64 and
    complex128 values support first-order autograd, native mapped products and
    fullgraph compilation. Higher-order solution products remain unavailable.
    """

    __slots__ = ("_program", "_token", "_input_metadata")

    def __init__(self, program: DifferentiableProgram) -> None:
        if not isinstance(program, DifferentiableProgram):
            raise TypeError("program must be an eqiora.DifferentiableProgram")
        if program.dtype != "float64" or program.device != "cpu:0":
            raise ValueError("this PyTorch adapter supports CPU:0 float64 programs")
        self._program = program
        self._token = program._register()
        self._input_metadata = _metadata(program.inputs)

    def __setattr__(self, name, value):
        if hasattr(self, name):
            raise AttributeError("Eqiora PyTorch program configuration is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise AttributeError("Eqiora PyTorch program configuration is immutable")

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
        # Reject unsupported dispatch keys before custom-op dispatch. Kernels
        # independently validate against the same immutable native signature.
        _validate(parameters, self._input_metadata, "parameters")
        return _Solve.apply(self._token, [], *parameters)


def bind(program: DifferentiableProgram) -> TorchProgram:
    """Register a framework-neutral program as a PyTorch operator in this process."""
    return TorchProgram(program)


__all__ = ["TorchProgram", "bind"]
