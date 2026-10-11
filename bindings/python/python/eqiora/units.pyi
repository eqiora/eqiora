"""Unit catalog for authoring Source quantities.

Authority: ``bindings/python/python/eqiora/units.py``.
"""
from fractions import Fraction
from typing import Final, final

@final
class Unit:
    """Compose an immutable input-unit expression.

    Authority: ``bindings/python/python/eqiora/units.py::Unit``.
    """
    def __mul__(self, other: Unit, /) -> Unit: ...
    def __truediv__(self, other: Unit, /) -> Unit: ...
    def __pow__(self, exponent: int | Fraction, /) -> Unit: ...
    def prefixed(self, prefix: str) -> Unit: ...

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
