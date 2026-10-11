"""Compiler-owned input-unit catalog and bounded structural unit expressions.

Unit composition emits syntax; the compiler alone converts values to coherent SI.
"""

from __future__ import annotations
from fractions import Fraction

from ._eqiora import _input_unit_catalog, _AstExpression as _Ast
from ._source_bounds import _MAX_EXPRESSION_DEPTH, _MAX_EXPRESSION_NODES

_CREATE_UNIT = object()
_MISSING_UNIT = object()


class Unit:
    """One immutable, bounded Eqiora input-unit expression."""

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

    def prefixed(self, prefix: str) -> Unit:
        """Apply one compiler-admitted prefix to one prefixable catalog symbol."""
        if prefix not in _PREFIXES:
            raise ValueError("input-unit prefix is not in the compiler catalog")
        if not self._prefixable:
            raise ValueError("prefixes require a bare prefixable catalog unit; use g for mass")
        return Unit(_CREATE_UNIT, _Ast.name(prefix + self._symbol))


_symbols, _prefixes = _input_unit_catalog()
_PREFIXES = frozenset(_prefixes)
__all__ = ["Unit"]
for _symbol, _prefixable in _symbols:
    _name = "one" if _symbol == "1" else _symbol
    globals()[_name] = Unit(_CREATE_UNIT, _Ast.number("1") if _symbol == "1" else _Ast.name(_symbol), _prefixable=_prefixable, _symbol=_symbol)
    __all__.append(_name)
del _symbols, _prefixes, _symbol, _prefixable, _name
