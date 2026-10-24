from typing import assert_type

import torch

import eqiora
import eqiora.torch as eqtorch


def check_torch_adapter(
    program: eqiora.DifferentiableProgram,
    parameters: torch.Tensor,
) -> None:
    direct = eqtorch.TorchProgram(program)
    bound = eqtorch.bind(program)
    assert_type(direct, eqtorch.TorchProgram)
    assert_type(bound, eqtorch.TorchProgram)
    assert_type(bound.program, eqiora.DifferentiableProgram)
    assert_type(bound.input_shape, tuple[int])
    assert_type(bound.output_shape, tuple[int, ...])
    assert_type(bound(parameters), torch.Tensor)
