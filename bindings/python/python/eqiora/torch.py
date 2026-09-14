"""PyTorch autograd projection of Eqiora differentiable programs.

This optional module owns framework registration and process-local program
tokens. Eqiora's native model, Plan, accepted-point, and derivative
contracts remain framework-neutral.
"""

from __future__ import annotations

import os
import threading
import uuid
from typing import NamedTuple

from . import DifferentiableProgram

try:
    import torch
except ImportError as error:  # pragma: no cover - depends on optional install
    raise ImportError(
        "eqiora.torch requires the optional 'torch' dependency; "
        "install eqiora[torch]"
    ) from error


_SUPPORTED_TORCH_SERIES = (2, 14)


class _ProgramIdentity(NamedTuple):
    model_digest: str
    plan_identity: str
    input_ids: tuple[str, ...]
    output_id: str
    input_shape: tuple[int]
    output_shape: tuple[int]
    dtype: str
    device: str
    derivative_contract: str


class _RegistryEntry(NamedTuple):
    pid: int
    program: DifferentiableProgram
    identity: _ProgramIdentity


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


_registry_lock = threading.Lock()
_programs: dict[str, _RegistryEntry] = {}
_tokens_by_identity: dict[_ProgramIdentity, str] = {}


def _program_identity(program: DifferentiableProgram) -> _ProgramIdentity:
    return _ProgramIdentity(
        model_digest=program.model_digest,
        plan_identity=program.plan_identity,
        input_ids=tuple(program.input_ids),
        output_id=program.output_id,
        input_shape=program.input_shape,
        output_shape=program.output_shape,
        dtype=program.dtype,
        device=program.device,
        derivative_contract=program.derivative_contract,
    )


def _register(program: DifferentiableProgram) -> str:
    identity = _program_identity(program)
    with _registry_lock:
        if token := _tokens_by_identity.get(identity):
            return token
        token = uuid.uuid4().hex
        _programs[token] = _RegistryEntry(os.getpid(), program, identity)
        _tokens_by_identity[identity] = token
        return token


def _resolve(
    token: str,
    input_size: int,
    output_size: int,
) -> DifferentiableProgram:
    with _registry_lock:
        entry = _programs.get(token)
    if entry is None:
        raise RuntimeError(
            "the process-local Eqiora PyTorch program token is not available"
        )
    if entry.pid != os.getpid():
        raise RuntimeError(
            "Eqiora PyTorch program tokens cannot cross a process boundary"
        )
    if (
        entry.identity.input_shape != (input_size,)
        or entry.identity.output_shape != (output_size,)
    ):
        raise RuntimeError("operator metadata does not match Eqiora program identity")
    return entry.program


def _validate_tensor(
    value: torch.Tensor,
    size: int,
    *,
    role: str,
) -> None:
    if value.device.type != "cpu" or value.device.index not in (None, 0):
        raise ValueError(f"{role} must reside on CPU device 0")
    if value.layout != torch.strided:
        raise TypeError(f"{role} must use strided layout")
    if value.dtype is not torch.float64:
        raise TypeError(f"{role} must have dtype torch.float64")
    if value.ndim != 1 or value.shape[0] != size:
        raise ValueError(f"{role} must have exact shape ({size},)")
    if not value.is_contiguous():
        raise ValueError(f"{role} must be C-contiguous")


def _to_tensor(array) -> torch.Tensor:
    # Array exports a fresh versioned CPU snapshot. The accepted native
    # evaluation and its immutable evidence are never shared with PyTorch.
    return torch.utils.dlpack.from_dlpack(array)


@torch.library.custom_op(
    "eqiora::differentiable_solve",
    mutates_args=(),
    device_types="cpu",
)
def _solve(
    parameters: torch.Tensor,
    token: str,
    input_size: int,
    output_size: int,
) -> torch.Tensor:
    _validate_tensor(parameters, input_size, role="parameters")
    program = _resolve(token, input_size, output_size)
    evaluation = program.evaluate(parameters.detach())
    output = _to_tensor(evaluation.primal().output)
    _validate_tensor(output, output_size, role="Eqiora output")
    return output


@_solve.register_fake
def _solve_fake(
    parameters: torch.Tensor,
    token: str,
    input_size: int,
    output_size: int,
) -> torch.Tensor:
    del token
    _validate_tensor(parameters, input_size, role="parameters")
    return parameters.new_empty((output_size,))


@torch.library.custom_op(
    "eqiora::_differentiable_solve_vjp",
    mutates_args=(),
    device_types="cpu",
)
def _vjp(
    parameters: torch.Tensor,
    cotangent: torch.Tensor,
    token: str,
    input_size: int,
    output_size: int,
) -> torch.Tensor:
    _validate_tensor(parameters, input_size, role="parameters")
    _validate_tensor(cotangent, output_size, role="cotangent")
    program = _resolve(token, input_size, output_size)
    evaluation = program.evaluate(parameters.detach())
    input_cotangent = _to_tensor(evaluation.vjp(cotangent.detach()).input_cotangent)
    _validate_tensor(
        input_cotangent,
        input_size,
        role="Eqiora input cotangent",
    )
    return input_cotangent


@_vjp.register_fake
def _vjp_fake(
    parameters: torch.Tensor,
    cotangent: torch.Tensor,
    token: str,
    input_size: int,
    output_size: int,
) -> torch.Tensor:
    del token
    _validate_tensor(parameters, input_size, role="parameters")
    _validate_tensor(cotangent, output_size, role="cotangent")
    return parameters.new_empty((input_size,))


def _setup_solve_context(ctx, inputs, output) -> None:
    del output
    parameters, token, input_size, output_size = inputs
    ctx.save_for_backward(parameters)
    ctx.token = token
    ctx.input_size = input_size
    ctx.output_size = output_size


def _solve_backward(ctx, cotangent: torch.Tensor):
    (parameters,) = ctx.saved_tensors
    input_cotangent = _vjp(
        parameters,
        cotangent.contiguous(),
        ctx.token,
        ctx.input_size,
        ctx.output_size,
    )
    return input_cotangent, None, None, None


_solve.register_autograd(_solve_backward, setup_context=_setup_solve_context)


class TorchProgram:
    """Process-local functional PyTorch view of one Eqiora program."""

    __slots__ = (
        "_input_size",
        "_output_size",
        "_program",
        "_token",
    )

    def __init__(self, program: DifferentiableProgram) -> None:
        if not isinstance(program, DifferentiableProgram):
            raise TypeError("program must be an eqiora.DifferentiableProgram")
        if program.dtype != "float64" or program.device != "cpu:0":
            raise ValueError("this PyTorch adapter supports CPU:0 float64 programs")
        self._program = program
        self._input_size = program.input_shape[0]
        self._output_size = program.output_shape[0]
        self._token = _register(program)

    @property
    def program(self) -> DifferentiableProgram:
        """The exact framework-neutral program retained by this adapter."""
        return self._program

    @property
    def input_shape(self) -> tuple[int]:
        return (self._input_size,)

    @property
    def output_shape(self) -> tuple[int]:
        return (self._output_size,)

    def __call__(self, parameters: torch.Tensor) -> torch.Tensor:
        # The public wrapper rejects unsupported dispatch keys before PyTorch
        # attempts to select a custom-op kernel. The kernel repeats the check
        # defensively for direct operator calls.
        _validate_tensor(parameters, self._input_size, role="parameters")
        return _solve(
            parameters,
            self._token,
            self._input_size,
            self._output_size,
        )


def bind(program: DifferentiableProgram) -> TorchProgram:
    """Bind a framework-neutral program to a process-local PyTorch operator."""
    return TorchProgram(program)


__all__ = ["TorchProgram", "bind"]
