"""Use Eqiora differentiable programs with PyTorch autograd.

This optional module registers Eqiora programs as PyTorch operators. The
underlying Model, Plan, and derivative computations do not depend on PyTorch.
"""

from __future__ import annotations

from . import DifferentiableProgram, _eqiora

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
        if not value.is_contiguous():
            raise ValueError(f'{role} must be C-contiguous')


def _detached(values):
    # Native DLPack staging cannot consume lazy conjugate/negative tensor views.
    return tuple(v.detach().resolve_conj().resolve_neg() for v in values)


@torch.library.custom_op("eqiora::differentiable_solve", mutates_args=(), device_types="cpu")
def _solve(parameters: list[torch.Tensor], token: str) -> list[torch.Tensor]:
    program = _eqiora._differentiable_program(token)
    _validate(parameters, _metadata(program.inputs), "parameters")
    evaluation = program.evaluate(_detached(parameters))
    return [torch.tensor(v, device="cpu") for v in evaluation.primal().output]


@_solve.register_fake
def _solve_fake(parameters, token):
    # Registry entries are immutable and retained for the process lifetime. Read
    # only their signatures here: no tensor data or numerical evaluation occurs.
    program = _eqiora._differentiable_program(token)
    _validate(parameters, _metadata(program.inputs), "parameters")
    return [
        torch.empty(shape, dtype=dtype, device="cpu")
        for shape, dtype in _metadata(program.outputs)
    ]


@torch.library.custom_op("eqiora::_differentiable_solve_vjp", mutates_args=(), device_types="cpu")
def _vjp(parameters: list[torch.Tensor], cotangents: list[torch.Tensor], token: str) -> list[torch.Tensor]:
    program = _eqiora._differentiable_program(token)
    _validate(parameters, _metadata(program.inputs), "parameters")
    _validate(cotangents, _metadata(program.outputs), "cotangents")
    evaluation = program.evaluate(_detached(parameters))
    return [
        torch.tensor(value, device="cpu")
        for value in evaluation.vjp(_detached(cotangents)).input_cotangent
    ]


@_vjp.register_fake
def _vjp_fake(parameters, cotangents, token):
    program = _eqiora._differentiable_program(token)
    _validate(parameters, _metadata(program.inputs), "parameters")
    _validate(cotangents, _metadata(program.outputs), "cotangents")
    return [torch.empty_like(v) for v in parameters]


def _setup(ctx, inputs, output):
    del output
    parameters, token = inputs
    ctx.save_for_backward(*parameters)
    ctx.token = token


def _backward(ctx, cotangents):
    return _vjp(list(ctx.saved_tensors), [v.contiguous() for v in cotangents], ctx.token), None

_solve.register_autograd(_backward, setup_context=_setup)


class TorchProgram:
    """Process-local PyTorch view of ordered real/complex native value leaves.

    Native signatures own each leaf's shape and scalar domain. CPU float64 and
    complex128 values support first-order autograd and fullgraph compilation.
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
        return tuple(_solve(list(parameters), self._token))


def bind(program: DifferentiableProgram) -> TorchProgram:
    """Register a framework-neutral program as a PyTorch operator in this process."""
    return TorchProgram(program)


__all__ = ["TorchProgram", "bind"]
