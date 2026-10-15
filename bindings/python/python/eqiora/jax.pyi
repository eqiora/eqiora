"""Use Eqiora differentiable programs with JAX.

Authority: ``bindings/python/python/eqiora/jax.py``.
"""

from __future__ import annotations

from typing import Sequence

import jax

from . import DifferentiableProgram, DifferentiableValueSignature

class JaxProgram:
    """JAX view of a program in this process, with first-order ``vmap`` support.

    Authority: ``bindings/python/python/eqiora/jax.py::JaxProgram``.
    """

    def __init__(self, program: DifferentiableProgram) -> None: ...
    @property
    def program(self) -> DifferentiableProgram: ...
    @property
    def inputs(self) -> tuple[DifferentiableValueSignature, ...]: ...
    @property
    def outputs(self) -> tuple[DifferentiableValueSignature, ...]: ...
    def __call__(self, parameters: Sequence[jax.Array]) -> tuple[jax.Array, ...]: ...

def bind(program: DifferentiableProgram) -> JaxProgram:
    """Wrap a DifferentiableProgram so that JAX can call and differentiate it.

    Authority: ``bindings/python/python/eqiora/jax.py::bind``.
    """

    ...

__all__ = ["JaxProgram", "bind"]
