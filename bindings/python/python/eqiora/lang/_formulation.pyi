"""Typed boundary for the private harmonic authoring adapter."""

from collections.abc import Sequence
from .. import ValueType
from . import Component, Expression, Relation

def harmonic(
    component: Component,
    name: str,
    relations: Sequence[Relation],
    angular_frequency: object,
    convention: str,
    normalization: str,
    excitations: Sequence[tuple[Expression, object]],
    amplitudes: Sequence[tuple[str, Expression, ValueType]],
    doc: str | None,
) -> None: ...
