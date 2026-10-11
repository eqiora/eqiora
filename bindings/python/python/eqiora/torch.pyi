"""Use Eqiora differentiable programs with PyTorch autograd.

Authority: ``bindings/python/python/eqiora/torch.py``.
"""

from . import DifferentiableProgram

import torch

class TorchProgram:
    """Process-local functional PyTorch view of one Eqiora program.

    Authority: ``bindings/python/python/eqiora/torch.py::TorchProgram``.
    """

    def __init__(self, program: DifferentiableProgram) -> None: ...
    @property
    def program(self) -> DifferentiableProgram: ...
    @property
    def input_shape(self) -> tuple[int]: ...
    @property
    def output_shape(self) -> tuple[int]: ...
    def __call__(self, parameters: torch.Tensor) -> torch.Tensor: ...

def bind(program: DifferentiableProgram) -> TorchProgram:
    """Bind a program to a process-local PyTorch operator.

    Authority: ``bindings/python/python/eqiora/torch.py::bind``.
    """

    ...

__all__ = ["TorchProgram", "bind"]
