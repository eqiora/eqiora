"""Use Eqiora differentiable programs with JAX.

Authority: ``bindings/python/python/eqiora/jax.py``.
"""

from __future__ import annotations

import jax

from . import DifferentiableProgram

class JaxProgram:
    """Process-local JAX view with native first-order ``vmap`` composition.

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
    """Bind a framework-neutral program to the typed JAX/XLA FFI.

    Authority: ``bindings/python/python/eqiora/jax.py::bind``.
    """

    ...

__all__ = ["JaxProgram", "bind"]
