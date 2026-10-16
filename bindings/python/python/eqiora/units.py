"""Shared units for input quantities and explicit presentation.

Combining units only builds unit syntax; the compiler converts values to
coherent SI.
"""

from __future__ import annotations
from fractions import Fraction
from decimal import Decimal
from dataclasses import dataclass

from ._eqiora import _unit_catalog, _normalize_input_quantity, _present_quantity, _AstExpression as _Ast, Dimension
from ._literals import quantity as _quantity_literal
from ._source_bounds import _MAX_EXPRESSION_DEPTH, _MAX_EXPRESSION_NODES

_CREATE_UNIT = object()
_MISSING_UNIT = object()


@dataclass(frozen=True, slots=True)
class PresentedQuantity:
    """A display projection retaining its stored quantity and rounding policy."""

    stored_value: float
    dimension: Dimension
    value: float
    unit: str
    text: str
    decimal_places: int
    locale: str
    rounding: str


class Unit:
    """An immutable unit expression for input and presentation quantities."""

    __slots__ = ("_ast", "_depth", "_nodes", "_prefixable", "_symbol")

    def __init__(self, _token: object = _MISSING_UNIT, _ast: _Ast | None = None,
                 _depth: int = 1, _nodes: int = 1, _prefixable: bool = False,
                 _symbol: str | None = None) -> None:
        if _token is not _CREATE_UNIT or not isinstance(_ast, _Ast):
            raise TypeError("units are composed from eqiora.units catalog values")
        if _depth > _MAX_EXPRESSION_DEPTH or _nodes > _MAX_EXPRESSION_NODES:
            raise ValueError("unit expression exceeds the 64-depth or 4096-node authoring limit")
        object.__setattr__(self, "_ast", _ast)
        object.__setattr__(self, "_depth", _depth)
        object.__setattr__(self, "_nodes", _nodes)
        object.__setattr__(self, "_prefixable", _prefixable)
        object.__setattr__(self, "_symbol", _symbol)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Unit values are immutable")

    def _binary(self, other: Unit, operator: str) -> Unit:
        if not isinstance(other, Unit):
            return NotImplemented
        depth = max(self._depth, other._depth) + 1
        nodes = self._nodes + other._nodes + 1
        # Check before concatenation, including repeated self-composition.
        if depth > _MAX_EXPRESSION_DEPTH or nodes > _MAX_EXPRESSION_NODES:
            raise ValueError("unit expression exceeds the 64-depth or 4096-node authoring limit")
        return Unit(_CREATE_UNIT, self._ast.binary(operator, other._ast), depth, nodes)

    def __mul__(self, other: Unit) -> Unit:
        return self._binary(other, "*")

    def __truediv__(self, other: Unit) -> Unit:
        return self._binary(other, "/")

    def __pow__(self, exponent: int | Fraction) -> Unit:
        if isinstance(exponent, bool) or not isinstance(exponent, (int, Fraction)):
            raise TypeError("unit exponents must be int or fractions.Fraction")
        numerator, denominator = exponent.numerator, exponent.denominator
        if abs(numerator) > 2147483647 or denominator > 2147483647:
            raise ValueError("unit exponent numerator and denominator must be bounded by 2147483647")
        depth = self._depth + 1
        nodes = self._nodes + (2 if denominator == 1 else 4)
        if depth > _MAX_EXPRESSION_DEPTH or nodes > _MAX_EXPRESSION_NODES:
            raise ValueError("unit expression exceeds the 64-depth or 4096-node authoring limit")
        power = _Ast.number(str(numerator))
        if denominator != 1:
            power = power.binary("/", _Ast.number(str(denominator)))
        return Unit(_CREATE_UNIT, self._ast.binary("^", power), depth, nodes)

    def normalize(self, value: int | float | Decimal) -> tuple[float, Dimension]:
        """Return the coherent-SI value and dimension through the compiler.

        Integers and Decimal inputs keep their exact decimal meaning; floats use
        their shortest round-trip decimal spelling, as in lang.quantity.
        Conversion rounds once, after composing the exact unit scale. Nonfinite
        values, nonzero underflow and unsupported affine units reject.
        """
        return _normalize_input_quantity(_quantity_literal(value), self._ast)

    def present(self, value: float, dimension: Dimension, *, decimal_places: int = 6,
                locale: str = "C") -> PresentedQuantity:
        """Display a stored coherent-SI value through the native unit owner.

        Conversion rounds once from the exact stored binary64. Copy/accessibility
        text uses explicit fixed decimal places and the C locale. Stored values
        and dimensions remain available; this introduces no affine, spectral or
        phase convention. Unsupported units, locales and numerical ranges reject.
        """
        if isinstance(decimal_places, bool) or not isinstance(decimal_places, int):
            raise TypeError("decimal_places must be an integer")
        stored, displayed, text, rounding = _present_quantity(value, dimension, self._ast,
                                                              decimal_places, locale)
        return PresentedQuantity(stored, dimension, displayed, self._ast.source,
                                 text, decimal_places, locale, rounding)

    def prefixed(self, prefix: str) -> Unit:
        """Apply an SI prefix to a unit symbol that accepts prefixes."""
        if prefix not in _PREFIXES:
            raise ValueError("input-unit prefix is not in the compiler catalog")
        if not self._prefixable:
            raise ValueError("prefixes require a bare prefixable catalog unit; use g for mass")
        return Unit(_CREATE_UNIT, _Ast.name(prefix + self._symbol))


_symbols, _prefixes = _unit_catalog()
_PREFIXES = frozenset(_prefixes)
__all__ = ["Unit", "PresentedQuantity"]
for _symbol, _prefixable in _symbols:
    _name = "one" if _symbol == "1" else _symbol
    globals()[_name] = Unit(_CREATE_UNIT, _Ast.number("1") if _symbol == "1" else _Ast.name(_symbol), _prefixable=_prefixable, _symbol=_symbol)
    __all__.append(_name)
del _symbols, _prefixes, _symbol, _prefixable, _name
