from typing import assert_type

import torch

import eqiora
import eqiora.torch as eqtorch


def check_torch_adapter(
    program: eqiora.DifferentiableProgram,
    parameters: tuple[torch.Tensor, ...],
) -> None:
    direct = eqtorch.TorchProgram(program)
    bound = eqtorch.bind(program)
    assert_type(direct, eqtorch.TorchProgram)
    assert_type(bound, eqtorch.TorchProgram)
    assert_type(bound.program, eqiora.DifferentiableProgram)
    assert_type(bound.inputs, tuple[eqiora.DifferentiableValueSignature, ...])
    assert_type(bound.outputs, tuple[eqiora.DifferentiableValueSignature, ...])
    assert_type(bound(parameters), tuple[torch.Tensor, ...])
