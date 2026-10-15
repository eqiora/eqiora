from typing import assert_type

import jax

import eqiora
import eqiora.jax as eqjax


def check_jax_adapter(
    program: eqiora.DifferentiableProgram,
    parameters: tuple[jax.Array, ...],
) -> None:
    direct = eqjax.JaxProgram(program)
    bound = eqjax.bind(program)
    assert_type(direct, eqjax.JaxProgram)
    assert_type(bound, eqjax.JaxProgram)
    assert_type(bound.program, eqiora.DifferentiableProgram)
    assert_type(bound.inputs, tuple[eqiora.DifferentiableValueSignature, ...])
    assert_type(bound.outputs, tuple[eqiora.DifferentiableValueSignature, ...])
    assert_type(bound(parameters), tuple[jax.Array, ...])
    assert_type(jax.vmap(bound)(parameters), tuple[jax.Array, ...])
