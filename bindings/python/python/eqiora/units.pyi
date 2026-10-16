"""Shared units for input quantities and explicit presentation.

Authority: ``bindings/python/python/eqiora/units.py``.
"""
from fractions import Fraction
from decimal import Decimal
from . import Dimension
from typing import Final, final

@final
class PresentedQuantity:
    """A display projection retaining its stored quantity and rounding policy.

    Authority: ``bindings/python/python/eqiora/units.py::PresentedQuantity``.
    """
    @property
    def stored_value(self) -> float: ...
    @property
    def dimension(self) -> Dimension: ...
    @property
    def value(self) -> float: ...
    @property
    def unit(self) -> str: ...
    @property
    def text(self) -> str: ...
    @property
    def decimal_places(self) -> int: ...
    @property
    def locale(self) -> str: ...
    @property
    def rounding(self) -> str: ...

@final
class Unit:
    """An immutable unit expression for input and presentation quantities.

    Authority: ``bindings/python/python/eqiora/units.py::Unit``.
    """
    def __mul__(self, other: Unit, /) -> Unit: ...
    def __truediv__(self, other: Unit, /) -> Unit: ...
    def __pow__(self, exponent: int | Fraction, /) -> Unit: ...
    def normalize(self, value: int | float | Decimal) -> tuple[float, Dimension]: ...
    def prefixed(self, prefix: str) -> Unit: ...
    def present(self, value: float, dimension: Dimension, *, decimal_places: int = 6, locale: str = "C") -> PresentedQuantity: ...

#: Input-unit symbol.
#:
#: Authority: ``bindings/python/python/eqiora/units.py``.
kg: Final[Unit]

#: Input-unit symbol.
#:
#: Authority: ``bindings/python/python/eqiora/units.py``.
m: Final[Unit]

#: Input-unit symbol.
#:
#: Authority: ``bindings/python/python/eqiora/units.py``.
s: Final[Unit]

#: Input-unit symbol.
#:
#: Authority: ``bindings/python/python/eqiora/units.py``.
A: Final[Unit]

#: Input-unit symbol.
#:
#: Authority: ``bindings/python/python/eqiora/units.py``.
K: Final[Unit]

#: Input-unit symbol.
#:
#: Authority: ``bindings/python/python/eqiora/units.py``.
mol: Final[Unit]

#: Input-unit symbol.
#:
#: Authority: ``bindings/python/python/eqiora/units.py``.
cd: Final[Unit]

#: Input-unit symbol.
#:
#: Authority: ``bindings/python/python/eqiora/units.py``.
Hz: Final[Unit]

#: Input-unit symbol.
#:
#: Authority: ``bindings/python/python/eqiora/units.py``.
N: Final[Unit]

#: Input-unit symbol.
#:
#: Authority: ``bindings/python/python/eqiora/units.py``.
Pa: Final[Unit]

#: Input-unit symbol.
#:
#: Authority: ``bindings/python/python/eqiora/units.py``.
J: Final[Unit]

#: Input-unit symbol.
#:
#: Authority: ``bindings/python/python/eqiora/units.py``.
W: Final[Unit]

#: Input-unit symbol.
#:
#: Authority: ``bindings/python/python/eqiora/units.py``.
C: Final[Unit]

#: Input-unit symbol.
#:
#: Authority: ``bindings/python/python/eqiora/units.py``.
V: Final[Unit]

#: Input-unit symbol.
#:
#: Authority: ``bindings/python/python/eqiora/units.py``.
Ohm: Final[Unit]

#: Input-unit symbol.
#:
#: Authority: ``bindings/python/python/eqiora/units.py``.
S: Final[Unit]

#: Input-unit symbol.
#:
#: Authority: ``bindings/python/python/eqiora/units.py``.
F: Final[Unit]

#: Input-unit symbol.
#:
#: Authority: ``bindings/python/python/eqiora/units.py``.
H: Final[Unit]

#: Input-unit symbol.
#:
#: Authority: ``bindings/python/python/eqiora/units.py``.
Wb: Final[Unit]

#: Input-unit symbol.
#:
#: Authority: ``bindings/python/python/eqiora/units.py``.
T: Final[Unit]

#: Input-unit symbol.
#:
#: Authority: ``bindings/python/python/eqiora/units.py``.
g: Final[Unit]

#: Input-unit symbol.
#:
#: Authority: ``bindings/python/python/eqiora/units.py``.
one: Final[Unit]

__all__ = ['Unit', 'kg', 'm', 's', 'A', 'K', 'mol', 'cd', 'Hz', 'N', 'Pa', 'J', 'W', 'C', 'V', 'Ohm', 'S', 'F', 'H', 'Wb', 'T', 'g', 'one']
