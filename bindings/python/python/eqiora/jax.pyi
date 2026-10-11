"""Use Eqiora differentiable programs with JAX.

Authority: ``bindings/python/python/eqiora/jax.py``.
"""

from __future__ import annotations

import jax

from . import DifferentiableProgram

class JaxProgram:
    """JAX view of a program in this process, with first-order ``vmap`` support.

    Authority: ``bindings/python/python/eqiora/jax.py::JaxProgram``.
    """

    def __init__(self, program: DifferentiableProgram) -> None: ...
    @property
    def program(self) -> DifferentiableProgram: ...
    @property
    def input_shape(self) -> tuple[int]: ...
    @property
    def output_shape(self) -> tuple[int]: ...
    def __call__(self, parameters: jax.Array) -> jax.Array: ...

def bind(program: DifferentiableProgram) -> JaxProgram:
    """Wrap a DifferentiableProgram so that JAX can call and differentiate it.

    Authority: ``bindings/python/python/eqiora/jax.py::bind``.
    """

    ...

__all__ = ["JaxProgram", "bind"]
