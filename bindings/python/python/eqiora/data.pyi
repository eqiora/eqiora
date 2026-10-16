"""Explicit numeric adapters and canonical resolved-array artifacts.

Authority: ``bindings/python/python/eqiora/data.py``.
"""
from collections.abc import Sequence
from typing import Literal, final
from . import Dimension
from .units import Unit

__all__ = ["ResolvedArray", "csv_array"]

@final
class ResolvedArray:
    """Immutable row-major values with the native resolved-array identity and codec.

    Authority: ``crates/eqiora-artifact/src/resolved_array.rs::ResolvedArrayV1``.
    """
    @staticmethod
    def from_f64(shape: Sequence[int], values: Sequence[float]) -> ResolvedArray: ...
    @staticmethod
    def from_u64(shape: Sequence[int], values: Sequence[int]) -> ResolvedArray: ...
    @staticmethod
    def from_bytes(data: bytes) -> ResolvedArray: ...
    def to_bytes(self) -> bytes: ...
    @property
    def digest(self) -> str: ...
    @property
    def scalar(self) -> Literal["f64", "u64"]: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def values(self) -> tuple[float, ...] | tuple[int, ...]: ...

def csv_array(text: str, *, columns: Sequence[tuple[str, Unit]], delimiter: str = ",") -> tuple[ResolvedArray, tuple[Dimension, ...]]:
    """Map numeric CSV columns explicitly through exact Unit normalization.

    Authority: ``bindings/python/python/eqiora/data.py::csv_array``.
    """
    ...
