"""Use Eqiora differentiable programs with PyTorch autograd.

Authority: ``bindings/python/python/eqiora/torch.py``.
"""

from . import DifferentiableProgram, DifferentiableValueSignature

import torch

class TorchProgram:
    """Functional PyTorch view of one Eqiora program, valid in this process.

    Authority: ``bindings/python/python/eqiora/torch.py::TorchProgram``.
    """

    def __init__(self, program: DifferentiableProgram) -> None: ...
    @property
    def program(self) -> DifferentiableProgram: ...
    @property
    def inputs(self) -> tuple[DifferentiableValueSignature, ...]: ...
    @property
    def outputs(self) -> tuple[DifferentiableValueSignature, ...]: ...
    def __call__(self, parameters: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> tuple[torch.Tensor, ...]: ...

def bind(program: DifferentiableProgram) -> TorchProgram:
    """Register a program as a PyTorch operator in this process.

    Authority: ``bindings/python/python/eqiora/torch.py::bind``.
    """

    ...

__all__ = ["TorchProgram", "bind"]
