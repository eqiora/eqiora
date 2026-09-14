"""Bounded Python authoring for deterministic Eqiora Language source.

These values only construct source syntax.  The existing native parser, type checker,
lowerer, and compiler remain the sole authority for mathematical meaning.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from decimal import Decimal, Context, DecimalException, Inexact, Rounded, Overflow, InvalidOperation, MAX_EMAX, MIN_EMIN
import math as _stdlib_math
import os
from pathlib import Path
import re
import tempfile
from builtins import property as _property
from typing import Final, Literal
from types import MappingProxyType

from .._eqiora import Dimension, FieldRole, ValueType, FiniteSpace, IndexSet, Enum as _NativeEnum, _nominal_type, Notation, ModuleError, _AstExpression as _Ast, _AstDeclaration, _AstDefinition, _AstModule, _module_from_declarations

from ..units import Unit
from .._source_bounds import _MAX_EXPRESSION_DEPTH, _MAX_EXPRESSION_NODES, _MAX_OUTPUT_BYTES

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_NAME_PATH = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z")
_MAX_DECLARATIONS = 256
_MAX_IDENTIFIER_BYTES = 1_024
_MAX_DOC_BYTES = 16_384
_CREATE = object()
_MISSING = object()


class PropertyContract:
    """An identity-bearing typed property contract declaration handle."""

    __slots__ = ("_doc", "_name", "_owner", "_value_type", "_profile")

    def __init__(
        self,
        _token: object = _MISSING,
        _owner: object = _MISSING,
        _name_value: str = "",
        _value_type: ValueType | None = None,
        _profile=None,
        _doc: tuple[str, ...] = (),
    ) -> None:
        if _token is not _CREATE or _value_type is None:
            raise TypeError("property contracts are created by Module")
        object.__setattr__(self, "_owner", _owner)
        object.__setattr__(self, "_name", _name_value)
        object.__setattr__(self, "_value_type", _value_type)
        object.__setattr__(self, "_profile", _profile)
        object.__setattr__(self, "_doc", _doc)

    def input(self, name: str) -> Expression:
        """Reference one exact ordered typed formal of this contract."""
        from ._property_profiles import formal
        return formal(self, name)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("PropertyContract handles are immutable")


class PropertyRelease:
    """An identity-bearing typed property release handle."""

    __slots__ = (
        "_citation",
        "_contract",
        "_doc",
        "_license",
        "_name",
        "_owner",
        "_source_scale",
        "_profile",
        "_source_unit",
        "_value",
        "_table",
    )

    def __init__(
        self,
        _token: object = _MISSING,
        *,
        _owner: object = _MISSING,
        _name_value: str = "",
        _contract: PropertyContract | None = None,
        _value: Expression | None = None,
        _source_unit: Unit | None = None,
        _source_scale: int | float = 1,
        _profile=None,
        _citation: str = "",
        _license: str = "",
        _doc: tuple[str, ...] = (),
    ) -> None:
        if _token is not _CREATE or _contract is None or _source_unit is None:
            raise TypeError("property releases are created by Module")
        object.__setattr__(self, "_owner", _owner)
        object.__setattr__(self, "_name", _name_value)
        object.__setattr__(self, "_contract", _contract)
        object.__setattr__(self, "_value", _value)
        object.__setattr__(self, "_table", None)
        object.__setattr__(self, "_source_unit", _source_unit)
        object.__setattr__(self, "_source_scale", _source_scale)
        object.__setattr__(self, "_profile", _profile)
        object.__setattr__(self, "_citation", _citation)
        object.__setattr__(self, "_license", _license)
        object.__setattr__(self, "_doc", _doc)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("PropertyRelease handles are immutable")


class MaterialComposition:
    """An immutable binding set from property requirements to exact releases."""

    __slots__ = ("_bindings", "_doc", "_name", "_owner")

    def __init__(
        self,
        _token: object = _MISSING,
        *,
        _owner: object = _MISSING,
        _name_value: str = "",
        _bindings: tuple[tuple[str, PropertyRelease], ...] = (),
        _doc: tuple[str, ...] = (),
    ) -> None:
        if _token is not _CREATE:
            raise TypeError("material compositions are created by Module")
        object.__setattr__(self, "_owner", _owner)
        object.__setattr__(self, "_name", _name_value)
        object.__setattr__(self, "_bindings", _bindings)
        object.__setattr__(self, "_doc", _doc)

    def __getitem__(self, name: str) -> PropertyRelease:
        """Select an exact release through this composition's declared member."""
        for member, release in self._bindings:
            if member == name:
                selected = object.__new__(PropertyRelease)
                for slot in PropertyRelease.__slots__:
                    object.__setattr__(selected, slot, getattr(release, slot))
                object.__setattr__(selected, "_name", f"{self._name}.{member}")
                return selected
        raise KeyError(name)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("MaterialComposition handles are immutable")


class Expression:
    """A closed Eqiora Language expression; equality is not an equation builder."""

    __slots__ = ("_owner", "_ast", "_binders", "_sources")

    def __init__(
        self,
        _token: object = _MISSING,
        _ast: _Ast | None = None,
        _owner: object | None = None,
        *, _binders: frozenset[object] = frozenset(),
        _sources: frozenset[object] = frozenset(),
    ) -> None:
        if _token is not _CREATE or not isinstance(_ast, _Ast):
            raise TypeError(
                "expressions are created by eqiora.lang declarations and operators"
            )
        object.__setattr__(self, "_sources", _sources)
        object.__setattr__(self, "_binders", _binders)
        object.__setattr__(self, "_ast", _ast)
        object.__setattr__(self, "_owner", _owner)

    @property
    def _depth(self) -> int:
        return self._ast.depth

    @property
    def _nodes(self) -> int:
        return self._ast.node_count

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Expression values are immutable")

    def __add__(self, other: object) -> Expression:
        return _binary(self, "+", other)

    def __radd__(self, other: object) -> Expression:
        return _binary(other, "+", self)

    def __sub__(self, other: object) -> Expression:
        return _binary(self, "-", other)

    def __rsub__(self, other: object) -> Expression:
        return _binary(other, "-", self)

    def __mul__(self, other: object) -> Expression:
        return _binary(self, "*", other)

    def __rmul__(self, other: object) -> Expression:
        return _binary(other, "*", self)

    def __truediv__(self, other: object) -> Expression:
        return _binary(self, "/", other)

    def __rtruediv__(self, other: object) -> Expression:
        return _binary(other, "/", self)

    def __pow__(self, exponent: int) -> Expression:
        if isinstance(exponent, bool) or not isinstance(exponent, int):
            raise TypeError("expression powers require an integer exponent")
        if not -32 <= exponent <= 32:
            raise ModuleError("expression exponent must be between -32 and 32")
        return Expression(
            _CREATE,
            self._ast.binary("^", _Ast.number(str(exponent))),
            self._owner,
            _binders=self._binders,
         _sources=self._sources)

    def __getitem__(self, index: int | Expression | slice) -> Expression:
        if isinstance(index, slice):
            if index.step is not None:
                raise ModuleError("expression slices do not accept a step")
            bounds = []
            for bound in (index.start, index.stop):
                if isinstance(bound, bool) or not isinstance(bound, (int, Expression)):
                    raise TypeError("expression slices require explicit exact integer bounds")
                if isinstance(bound, int) and bound < 0:
                    raise ModuleError("expression slice bounds must be nonnegative")
                bounds.append(_expression(bound))
            lower, upper = bounds
            # The shared compiler checks exact static values and array bounds;
            # no Python clipping, negative normalization, or runtime indexing.
            owner = _owner(self, lower)
            upper_owner = _owner(self, upper)
            if owner is not None and upper_owner is not None and owner is not upper_owner:
                raise ModuleError("cannot combine expressions from different Module or Component owners")
            return Expression(_CREATE, self._ast.slice(lower._ast, upper._ast), owner or upper_owner,
                              _binders=self._binders | lower._binders | upper._binders,
                              _sources=self._sources | lower._sources | upper._sources)
        if isinstance(index, Expression):
            return Expression(_CREATE, self._ast.index(index._ast), _owner(self, index),
                              _binders=self._binders | index._binders, _sources=self._sources | index._sources)
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("expression indices must be nonnegative integers")
        if index < 0:
            raise ModuleError("expression indices must be nonnegative")
        text = _number(index)
        return Expression(_CREATE, self._ast.index(_Ast.number(text)), self._owner, _binders=self._binders, _sources=self._sources)

    def __bool__(self) -> bool:
        raise TypeError("symbolic Eqiora expressions have no truth value; use explicit predicates")

    __hash__ = object.__hash__

    def __eq__(self, other: object) -> bool:
        raise TypeError("symbolic equality requires equal() or equation()")

    def __ne__(self, other: object) -> bool:
        raise TypeError("symbolic inequality requires not_equal()")

    def __lt__(self, other: object) -> bool:
        raise TypeError("symbolic ordering requires an explicit predicate")

    __le__ = __lt__
    __gt__ = __lt__
    __ge__ = __lt__

    def __neg__(self) -> Expression:
        return Expression(
            _CREATE,
            self._ast.unary("-"),
            self._owner,
            _binders=self._binders,
         _sources=self._sources)


class Equation:
    """Immutable ordered mathematical equality, never a Python truth value."""

    __slots__ = ("_lhs", "_rhs")

    def __init__(self, token: object, lhs: Expression, rhs: Expression) -> None:
        if token is not _CREATE:
            raise TypeError("use equation(lhs, rhs)")
        object.__setattr__(self, "_lhs", lhs)
        object.__setattr__(self, "_rhs", rhs)

    def __setattr__(self, name, value):
        raise AttributeError("Equation is immutable")

    @property
    def lhs(self) -> Expression:
        return self._lhs

    @property
    def rhs(self) -> Expression:
        return self._rhs

    def __bool__(self) -> bool:
        raise TypeError("mathematical equations have no Python truth value")


def equation(lhs: object, rhs: object) -> Equation:
    left, right = _expression(lhs), _expression(rhs)
    _owner(left, right)
    return Equation(_CREATE, left, right)


class Enum:
    """A closed enum declaration in one Module; members are exact symbolic paths."""

    __slots__ = ("_source", "_definition", "_syntax_name")

    def __init__(self, _token: object = _MISSING, *, _source: Module | None = None,
                 _definition: _NativeEnum | None = None, _syntax_name: str | None = None) -> None:
        if _token is not _CREATE:
            raise TypeError("Module enum handles are created by Module.enum()")
        object.__setattr__(self, "_source", _source)
        object.__setattr__(self, "_definition", _definition)
        object.__setattr__(self, "_syntax_name", _definition.name if _syntax_name is None else _syntax_name)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Enum handles are immutable")

    @property
    def name(self) -> str:
        return self._definition.name

    @property
    def value_type(self) -> ValueType:
        return self._definition.value_type

    @property
    def members(self) -> tuple[str, ...]:
        return self._definition.members

    def member(self, name: str) -> Expression:
        self._definition.member(name)
        return _EnumMember(self, name)


class _EnumMember(Expression):
    __slots__ = ("_enumeration", "_member_name")

    def __init__(self, enumeration: Enum, name: str) -> None:
        super().__init__(_CREATE, _Ast.name(f"{enumeration._syntax_name}.{name}"), None,
                         _sources=frozenset((enumeration._source._owner,)))
        object.__setattr__(self, "_enumeration", enumeration)
        object.__setattr__(self, "_member_name", name)


class Operator:
    """An immutable typed operator declared by one Module; call with named arguments."""

    __slots__ = ("_owner", "_name", "_inputs", "_result", "_body", "_doc")

    def __init__(self, _token: object = _MISSING, *, _source: Module | None = None,
                 _name: str = "", _inputs: tuple[tuple[str, str], ...] = (),
                 _result: str = "", _body: Expression | None = None,
                 _doc: tuple[str, ...] = ()) -> None:
        if _token is not _CREATE:
            raise TypeError("operators are created by Module.operator()")
        for key, value in (("_owner", None if _source is None else _source._owner),
                           ("_name", _name), ("_inputs", _inputs),
                           ("_result", _result), ("_body", _body), ("_doc", _doc)):
            object.__setattr__(self, key, value)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Operator values are immutable")

    def __call__(self, /, **arguments: object) -> Expression:
        names = tuple(name for name, _ in self._inputs)
        if set(arguments) != set(names):
            raise ModuleError("operator call must supply exactly its named inputs")
        values = tuple(_expression(arguments[name]) for name in names)
        owner = None
        for value in values:
            if value._sources - {self._owner}:
                raise ModuleError("operator arguments must belong to this Module")
            if owner is not None and value._owner is not None and owner is not value._owner:
                raise ModuleError("operator arguments must belong to the same Component")
            if value._owner is not None:
                owner = value._owner
        depth = max((value._depth for value in values), default=0) + 1
        nodes = sum(value._nodes for value in values) + 1
        if depth > _MAX_EXPRESSION_DEPTH or nodes > _MAX_EXPRESSION_NODES:
            raise ModuleError("operator call exceeds the expression depth or node limit")
        return Expression(_CREATE, _Ast.call(self._name, [value._ast for value in values], names), owner,
                          _binders=frozenset().union(*(value._binders for value in values)),
                          _sources=frozenset((self._owner,)))


class _Math:
    __slots__ = ()
    pi: Final = Expression(_CREATE, _Ast.name("math.pi"), None)
    i: Final = Expression(_CREATE, _Ast.name("math.i"), None)

    @staticmethod
    def complex(real: object, imaginary: object) -> Expression:
        left, right = _expression(real), _expression(imaginary)
        return Expression(_CREATE, _Ast.call("math.complex", [left._ast, right._ast]),
                          _owner(left, right), _binders=left._binders | right._binders, _sources=left._sources | right._sources)

    @staticmethod
    def sin(value: object) -> Expression:
        return _unary("math.sin", value)

    @staticmethod
    def sqrt(value: object) -> Expression:
        return _unary("math.sqrt", value)


    @staticmethod
    def abs(value: object) -> Expression:
        """Author absolute value, preserving its physical dimension."""
        return _unary("math.abs", value)

    @staticmethod
    def min(left: object, right: object) -> Expression:
        """Author a binary minimum; equality selects the first operand."""
        return _binary_function("math.min", left, right)

    @staticmethod
    def max(left: object, right: object) -> Expression:
        """Author a binary maximum; equality selects the first operand."""
        return _binary_function("math.max", left, right)

    @staticmethod
    def clamp(value: object, lower: object, upper: object) -> Expression:
        """Author a clamp; the compiler requires ordered compatible bounds."""
        return _ternary("math.clamp", value, lower, upper)

    @staticmethod
    def sign(value: object) -> Expression:
        """Author dimensionless sign: negative one, zero, or positive one."""
        return _unary("math.sign", value)

    @staticmethod
    def step(value: object) -> Expression:
        """Author dimensionless step: zero below zero, one at or above zero."""
        return _unary("math.step", value)


math: Final = _Math()


class _Parameter(Expression):
    __slots__ = ("_component", "_name")

    def __init__(self, component: object, name: str) -> None:
        super().__init__(_CREATE, _Ast.name(name), component)
        object.__setattr__(self, "_component", component)
        object.__setattr__(self, "_name", name)


class _Field(Expression):
    __slots__ = ("_component", "_name")

    def __init__(self, component: object, name: str) -> None:
        super().__init__(_CREATE, _Ast.name(name), component)
        object.__setattr__(self, "_component", component)
        object.__setattr__(self, "_name", name)


from ._records import _ImportedRecord, Record, RecordField, RecordParameter, declare_record as _declare_record


class ImportedRecord(_ImportedRecord):
    """A read-only public Record descriptor from an explicit import."""

    __slots__ = ()


class Relation:
    """An opaque Module-owned relation declaration handle."""

    __slots__ = ("_component", "_name", "_owner")

    def __init__(
        self,
        _token: object = _MISSING,
        _owner: object = _MISSING,
        _component: object = _MISSING,
        _name: str = "",
    ) -> None:
        if _token is not _CREATE:
            raise TypeError("relations are created by Component.relation()")
        object.__setattr__(self, "_owner", _owner)
        object.__setattr__(self, "_component", _component)
        object.__setattr__(self, "_name", _name)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Relation handles are immutable")


class PropertyRequirement(Expression):
    __slots__ = ("_component", "_contract", "_name")

    def __init__(
        self,
        _token: object,
        component: object,
        name: str,
        contract: PropertyContract,
    ) -> None:
        if _token is not _CREATE:
            raise TypeError("property requirements are created by Component")
        super().__init__(_CREATE, _Ast.name(name), component)
        object.__setattr__(self, "_component", component)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_contract", contract)

    def __call__(self, /, **arguments: object) -> Expression:
        from ._property_profiles import apply
        return apply(self, arguments)


class Support:
    """An identity-bearing support declaration handle."""

    __slots__ = ("_component", "_kind", "_name", "_owner")

    def __init__(
        self,
        _token: object = _MISSING,
        _owner: object = _MISSING,
        _component: object = _MISSING,
        _name: str = "",
        _kind: str = "",
    ) -> None:
        if _token is not _CREATE:
            raise TypeError(
                "supports are created by Component.volume() or Component.boundary()"
            )
        object.__setattr__(self, "_owner", _owner)
        object.__setattr__(self, "_component", _component)
        object.__setattr__(self, "_name", _name)
        object.__setattr__(self, "_kind", _kind)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Support handles are immutable")


class Clock:
    """An immutable nominal periodic clock belonging to one Component.

    Create clocks with Component.clock(); equal periods do not imply identity.
    """

    __slots__ = ("_component", "_name")

    def __init__(self, _token: object = _MISSING, _component: object = _MISSING,
                 _name: str = "") -> None:
        if _token is not _CREATE:
            raise TypeError("clocks are created by Component.clock()")
        object.__setattr__(self, "_component", _component)
        object.__setattr__(self, "_name", _name)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Clock handles are immutable")


class Event:
    """An immutable crossing event owned by one Component, distinct from a periodic Clock."""

    __slots__ = ("_component", "_name")

    def __init__(self, _token: object = _MISSING, _component: object = _MISSING,
                 _name: str = "") -> None:
        if _token is not _CREATE:
            raise TypeError("events are created by Component.event()")
        object.__setattr__(self, "_component", _component)
        object.__setattr__(self, "_name", _name)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Event handles are immutable")


def _clock_seconds(value: Fraction | int, *, positive: bool) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Fraction, int)):
        raise TypeError("clock seconds must be Fraction or int, not float or bool")
    exact = Fraction(value)
    if exact < 0 or (positive and exact == 0):
        raise ModuleError("clock period must be positive and phase must be nonnegative")
    if exact.numerator > (1 << 64) - 1 or exact.denominator > (1 << 64) - 1:
        raise ModuleError("clock rational numerator and denominator must fit u64")
    return exact


def _number(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("numeric literals must be finite int or float values, not bool")
    if isinstance(value, float) and not _stdlib_math.isfinite(value):
        raise ModuleError("numeric literals must be finite")
    if isinstance(value, int) and value.bit_length() > 1_024:
        raise ModuleError("integer literal exceeds the 1024-bit authoring limit")
    text = repr(value)
    if len(text) > 1_024:
        raise ModuleError("numeric literal exceeds the 1024-byte authoring limit")
    return text


def _expression(value: object) -> Expression:
    if isinstance(value, bool):
        return Expression(_CREATE, _Ast.boolean(value), None)
    if isinstance(value, Expression):
        return value
    if isinstance(value, complex):
        return math.complex(value.real, value.imag)
    text = _number(value)
    return Expression(_CREATE, _Ast.number(text), None)


def partial(value: object, *, wrt: Expression, holding: Sequence[Expression] = ()) -> Expression:
    """Differentiate an explicit scalar expression at a declared independent binding."""
    value = _expression(value)
    if not isinstance(wrt, Expression):
        raise TypeError("partial wrt requires a declared expression binding")
    if isinstance(holding, (str, bytes)) or not isinstance(holding, Sequence):
        raise TypeError("partial holding requires a sequence of declared bindings")
    if len(holding) > _MAX_EXPRESSION_NODES:
        raise ModuleError("partial holding exceeds the expression node limit")
    inputs = [value, wrt]
    for binding in holding:
        if not isinstance(binding, Expression):
            raise TypeError("partial holding requires declared expression bindings")
        inputs.append(binding)
    owner = None
    for item in inputs:
        if owner is not None and item._owner is not None and owner is not item._owner:
            raise ModuleError("partial bindings must belong to the same lexical owner")
        if item._owner is not None:
            owner = item._owner
    return Expression(_CREATE, _Ast.partial(value._ast, wrt._ast, [item._ast for item in holding]), owner,
                      _binders=frozenset().union(*(item._binders for item in inputs)),
                      _sources=frozenset().union(*(item._sources for item in inputs)))


def tensor_value(*, frame: Support, components: Sequence[object] | Expression) -> Expression:
    """Construct a uniform spatial value in an explicitly referenced Cartesian frame.

    Component axes are explicit; outer channel axes remain array constructions.
    The compiler checks frame eligibility, shape, units, and scalar domains.
    """
    if not isinstance(frame, Support):
        raise TypeError("tensor_value frame must be an eqiora.lang.Support")
    value = components if isinstance(components, Expression) else array(components)
    if value._owner is not None and value._owner is not frame._component:
        raise ModuleError("tensor_value components and frame must belong to the same Component")
    return Expression(_CREATE,
                      _Ast.call("tensor_value", [_Ast.name(frame._name), value._ast], ["frame", "components"]),
                      frame._component, _binders=value._binders, _sources=value._sources)


def array(values: Sequence[object]) -> Expression:
    """Construct ordered channel axes; arrays never infer spatial vector roles."""
    def build(items: Sequence[object], depth: int) -> Expression:
        if depth > _MAX_EXPRESSION_DEPTH:
            raise ModuleError("array expression depth exceeds the 64-node nesting limit")
        if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
            raise TypeError("array values must be a nonempty sequence")
        if not items:
            raise ModuleError("array values must be nonempty")
        if len(items) >= _MAX_EXPRESSION_NODES:
            raise ModuleError("array expression exceeds the 4096-node limit")
        expressions = []
        owner = None
        nodes = 1
        for item in items:
            value = build(item, depth + 1) if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) else _expression(item)
            if owner is not None and value._owner is not None and value._owner is not owner:
                raise ModuleError("cannot combine expressions from different Module or Component owners")
            owner = owner if value._owner is None else value._owner
            nodes += value._nodes
            if nodes > _MAX_EXPRESSION_NODES:
                raise ModuleError("array expression exceeds the 4096-node limit")
            expressions.append(value)
        return Expression(_CREATE, _Ast.array([value._ast for value in expressions]),
                          owner, _binders=frozenset().union(*(value._binders for value in expressions)), _sources=frozenset().union(*(value._sources for value in expressions)))
    return build(values, 1)


def _literal_expression(value: object) -> Expression:
    nodes = 0
    def literal(item: object, depth: int) -> Expression:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_EXPRESSION_NODES:
            raise ModuleError("literal exceeds the 4096-node limit")
        if depth > _MAX_EXPRESSION_DEPTH:
            raise ModuleError("literal depth exceeds the 64-node nesting limit")
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            if len(item) >= _MAX_EXPRESSION_NODES:
                raise ModuleError("literal exceeds the 4096-node limit")
            return array([literal(child, depth + 1) for child in item])
        if isinstance(item, Expression):
            raise TypeError("property release values must be numeric literals")
        return _expression(item)
    return literal(value, 1)


def quantity(value: int | float | Decimal, unit: Unit) -> Expression:
    """Author an exact decimal input quantity, normalized by the compiler.

    Decimal and integer inputs retain their authored decimal value. Floats use
    Python's shortest round-trip decimal spelling, not their exact binary ratio.
    Native numerical inputs remain binary64 coherent-SI values.
    """
    if not isinstance(unit, Unit):
        raise TypeError("unit must be an eqiora.units.Unit")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ModuleError("quantity literals must be finite")
        # Count significant digits before formatting; never expand an exponent
        # into a potentially enormous fixed-point decimal string.
        try:
            bounded = Context(prec=256, Emax=MAX_EMAX, Emin=MIN_EMIN,
                              traps=[Inexact, Rounded, Overflow, InvalidOperation]).create_decimal(value)
        except DecimalException as error:
            raise ModuleError("quantity literal exceeds the 256-byte limit") from error
        literal = str(bounded)
    else:
        literal = _number(value)
    if len(literal.encode("ascii")) > 256:
        raise ModuleError("quantity literal exceeds the 256-byte limit")
    return Expression(_CREATE, _Ast.quantity(literal, unit._ast), None)


def _owner(left: Expression, right: Expression) -> object | None:
    if (
        left._owner is not None
        and right._owner is not None
        and left._owner is not right._owner
    ):
        raise ModuleError("cannot combine expressions from different Module or Component owners")
    return left._owner if left._owner is not None else right._owner


def _ternary(operation: str, first: object, second: object, third: object) -> Expression:
    values = tuple(_expression(value) for value in (first, second, third))
    owner = None
    for value in values:
        if owner is not None and value._owner is not None and owner is not value._owner:
            raise ModuleError("cannot combine expressions from different Module or Component owners")
        if value._owner is not None:
            owner = value._owner
    depth = max(value._depth for value in values) + 1
    nodes = sum(value._nodes for value in values) + 1
    if depth > _MAX_EXPRESSION_DEPTH or nodes > _MAX_EXPRESSION_NODES:
        raise ModuleError("expression exceeds the depth or node limit")
    first, second, third = (value._ast for value in values)
    ast = (_Ast.select(first, second, third) if operation == "if"
           else _Ast.call(operation, [first, second, third]))
    return Expression(_CREATE, ast, owner,
                      _binders=frozenset().union(*(value._binders for value in values)),
                      _sources=frozenset().union(*(value._sources for value in values)))


def case(value: object, arms: Sequence[tuple[Expression, object]]) -> Expression:
    """Author ordered enum cases; the compiler checks exact coverage and branch types."""
    value = _expression(value)
    if isinstance(arms, (str, bytes)) or not isinstance(arms, Sequence) or not arms:
        raise TypeError("case arms must be a nonempty ordered sequence of member/value pairs")
    if len(arms) >= _MAX_EXPRESSION_NODES:
        raise ModuleError("case arms exceed the expression node limit")
    checked = []
    expressions = [value]
    nodes = value._nodes + 1
    owner = value._owner
    for arm in arms:
        if not isinstance(arm, (tuple, list)) or len(arm) != 2 or not isinstance(arm[0], _EnumMember):
            raise TypeError("case patterns require declared Module enum members")
        pattern, result = arm[0], _expression(arm[1])
        if owner is not None and result._owner is not None and owner is not result._owner:
            raise ModuleError("case expressions must belong to the same Component")
        if result._owner is not None:
            owner = result._owner
        nodes += pattern._nodes + result._nodes
        if nodes > _MAX_EXPRESSION_NODES:
            raise ModuleError("case exceeds the expression node limit")
        checked.append((pattern, result))
        expressions.extend((pattern, result))
    depth = max(item._depth for item in expressions) + 1
    if depth > _MAX_EXPRESSION_DEPTH:
        raise ModuleError("case exceeds the expression depth limit")
    ast = value._ast.case([f"{pattern._enumeration.name}.{pattern._member_name}" for pattern, _ in checked], [result._ast for _, result in checked])
    return Expression(_CREATE, ast, owner,
                      _binders=frozenset().union(*(item._binders for item in expressions)),
                      _sources=frozenset().union(*(item._sources for item in expressions)))


def if_else(condition: object, then_value: object, else_value: object) -> Expression:
    """Author a conditional expression without evaluating Python truthiness.

    All operands are authored and checked; execution selects one branch.
    """
    return _ternary("if", condition, then_value, else_value)


def _binary(left: object, operator: str, right: object) -> Expression:
    left_expr = _expression(left)
    right_expr = _expression(right)
    precedence = 10 if operator in ("+", "-") else 20
    return Expression(
        _CREATE,
        left_expr._ast.binary(operator, right_expr._ast),
        _owner(left_expr, right_expr),
        _binders=left_expr._binders | right_expr._binders, _sources=left_expr._sources | right_expr._sources)


def _predicate(left: object, operator: str, right: object) -> Expression:
    left, right = _expression(left), _expression(right)
    return Expression(_CREATE, left._ast.binary(operator, right._ast),
                      _owner(left, right), _binders=left._binders | right._binders, _sources=left._sources | right._sources)


def equal(left: object, right: object) -> Expression:
    """Author equal without evaluating Python equality or truthiness."""
    return _predicate(left, "==", right)


def not_equal(left: object, right: object) -> Expression:
    """Author not equal without evaluating Python equality or truthiness."""
    return _predicate(left, "!=", right)


def less(left: object, right: object) -> Expression:
    """Author less without evaluating Python equality or truthiness."""
    return _predicate(left, "<", right)


def less_equal(left: object, right: object) -> Expression:
    """Author less equal without evaluating Python equality or truthiness."""
    return _predicate(left, "<=", right)


def greater(left: object, right: object) -> Expression:
    """Author greater without evaluating Python equality or truthiness."""
    return _predicate(left, ">", right)


def greater_equal(left: object, right: object) -> Expression:
    """Author greater equal without evaluating Python equality or truthiness."""
    return _predicate(left, ">=", right)


def logical_and(left: object, right: object) -> Expression:
    """Author logical and without evaluating Python equality or truthiness."""
    return _predicate(left, "and", right)


def logical_or(left: object, right: object) -> Expression:
    """Author logical or without evaluating Python equality or truthiness."""
    return _predicate(left, "or", right)


def logical_not(value: object) -> Expression:
    """Author logical negation without evaluating Python truthiness."""
    value = _expression(value)
    return Expression(_CREATE, value._ast.unary("not"), value._owner, _binders=value._binders, _sources=value._sources)


def _unary(name: str, value: object) -> Expression:
    expression = _expression(value)
    return Expression(
        _CREATE,
        _Ast.call(name, [expression._ast]),
        expression._owner,
        _binders=expression._binders, _sources=expression._sources)


def coordinate(axis: int) -> Expression:
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError("coordinate axis must be an integer")
    if not 0 <= axis <= 15:
        raise ModuleError("coordinate axis must be between 0 and 15")
    return Expression(_CREATE, _Ast.call("coordinate", [_Ast.number(str(axis))]), None)


def grad(value: object) -> Expression:
    return _unary("grad", value)


def _binary_function(name: str, left: object, right: object) -> Expression:
    left_expression = _expression(left)
    right_expression = _expression(right)
    return Expression(
        _CREATE,
        _Ast.call(name, [left_expression._ast, right_expression._ast]),
        _owner(left_expression, right_expression),
        _binders=left_expression._binders | right_expression._binders, _sources=left_expression._sources | right_expression._sources)


def dot(left: object, right: object) -> Expression:
    return _binary_function("dot", left, right)


def quotient(left: object, right: object) -> Expression:
    """Exact integer quotient truncated toward zero; overflow and zero divisors reject."""
    return _binary_function("quotient", left, right)


def remainder(left: object, right: object) -> Expression:
    """Exact integer remainder with the dividend's sign when nonzero."""
    return _binary_function("remainder", left, right)


def ordinal(value: Expression) -> Expression:
    """Explicitly read an index's ordinary integer ordinal without changing its set."""
    return _unary("ordinal", value)


def to_real(value: object) -> Expression:
    """Explicitly convert an integer to real; binary64 rounding can lose integer precision."""
    return _unary("to_real", value)


def to_integer(value: object) -> Expression:
    """Convert only an integral, finite, dimensionless real in the signed 64-bit range."""
    return _unary("to_integer", value)


def integrate(domain: Support, integrand: object) -> Expression:
    if not isinstance(domain, Support) or domain._kind != "volume":
        raise ModuleError("integrate() requires a volume Support")
    expression = _expression(integrand)
    if expression._owner is not None and expression._owner is not domain._component:
        raise ModuleError("integrand and Support must belong to the same Component")
    return Expression(
        _CREATE,
        _Ast.call("integrate", [_Ast.name(domain._name), expression._ast]),
        domain._component,
        _binders=expression._binders, _sources=expression._sources)


def div(value: object) -> Expression:
    return _unary("div", value)


def derivative(value: Expression) -> Expression:
    """Author a continuous State derivative; the compiler checks role and activation."""
    return _unary("derivative", value)


def time() -> Expression:
    """Author the enclosing continuous timeline coordinate in seconds."""
    return Expression(_CREATE, _Ast.call("time", []))


def pre(value: Expression) -> Expression:
    """Read a State's pre-tick value; the compiler checks clock and context."""
    return _unary("pre", value)


def next(value: Expression) -> Expression:
    """Name a State's next-tick value; the compiler checks clock and context."""
    return _unary("next", value)


def trace(value: object) -> Expression:
    return _unary("trace", value)


def normal(value: object) -> Expression:
    return _unary("normal", value)


def frobenius(left: object, right: object) -> Expression:
    """Author the Frobenius contraction of two tensor expressions."""
    return _binary_function("frobenius", left, right)


def symmetric_part(value: object) -> Expression:
    return _unary("symmetric_part", value)


def isotropic_lift(value: object) -> Expression:
    return _unary("isotropic_lift", value)


def _name(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("declaration names must be strings")
    if not _NAME.fullmatch(value) or len(value.encode("utf-8")) > _MAX_IDENTIFIER_BYTES:
        raise ModuleError(f"invalid Eqiora Language declaration name {value!r}")
    return value


def _name_path(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if (
        not _NAME_PATH.fullmatch(value)
        or len(value.encode("utf-8")) > _MAX_IDENTIFIER_BYTES
    ):
        raise ModuleError(f"invalid Eqiora Language {label} {value!r}")
    return value


def _doc(value: object | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, str):
        raise TypeError("doc must be a string or None")
    if len(value.encode("utf-8")) > _MAX_DOC_BYTES:
        raise ModuleError(f"doc exceeds the {_MAX_DOC_BYTES}-byte limit")
    if "\r" in value or any(
        character < " " and character not in "\n\t" for character in value
    ):
        raise ModuleError("doc contains unsupported control characters")
    return tuple(value.split("\n"))


class Component:
    """The shared bounded draft for one public Model or Component definition."""

    __slots__ = (
        "_aliases",
        "_kind",
        "_requirements",
        "_defaults",
        "_causal",
        "_clocks",
        "_events",
        "_initials",
        "_index_sets",
        "_active_binders",
        "_reduction_names",
        "_component_token",
        "_declaration_count",
        "_doc",
        "_fields",
        "_ports",
        "_connections",
        "_formulations",
        "_test_restrictions",
        "_instances",
        "_name",
        "_names",
        "_notations",
        "_owner",
        "_parameters",
        "_properties",
        "_relations",
        "_observables",
        "_laws",
        "_source",
        "_supports",
    )

    def __init__(
        self,
        _token: object = _MISSING,
        _source: Module | None = None,
        _name_value: str = "",
        _doc_value: object | None = None,
    ) -> None:
        if _token is not _CREATE:
            raise TypeError("components are created by Module.component()")
        assert _source is not None
        self._source = _source
        self._kind = "component"
        self._requirements: set[object] = set()
        self._defaults: dict[Expression, Expression] = {}
        self._causal: dict[Expression, str] = {}
        self._owner = _source._owner
        self._component_token = object()
        self._name = _name(_name_value)
        self._doc = (
            _doc_value if isinstance(_doc_value, tuple) else _doc(_doc_value)
        )
        self._active_binders: dict[object, str] = {}
        self._reduction_names: set[str] = set()
        self._names: set[str] = set()
        self._notations: dict[str, Notation] = {}
        self._supports: list[tuple[Support, str, object, tuple[str, ...]]] = []
        self._clocks: list[tuple[Clock, Fraction | None, Fraction | None, tuple[str, ...]]] = []
        self._events: list[tuple[Event, Expression, str, tuple[str, ...]]] = []
        self._initials: list[tuple[tuple[tuple[Expression, Expression], ...], tuple[str, ...]]] = []
        self._index_sets: list[tuple[IndexSet, tuple[str, ...]]] = []
        self._parameters: list[tuple[_Parameter, str, tuple[str, ...]]] = []
        self._aliases: list[tuple[str, Expression, str | None, Support | None, Clock | Event | None, tuple[str, ...]]] = []
        self._properties: list[
            tuple[PropertyRequirement, PropertyContract, tuple[str, ...]]
        ] = []
        self._fields: list[
            tuple[
                Expression, Support, str, FieldRole, Clock | None, tuple[str, ...]
            ]
        ] = []
        self._observables: list[tuple[str, Expression, str, tuple[str, ...]]] = []
        self._relations: list[
            tuple[str, Support | None, tuple[tuple[str, Expression, Expression], ...], Clock | Event | None, tuple[str, ...]]
        ] = []
        self._laws: list[
            tuple[str, Support, Expression | None, Expression, Expression, tuple[str, ...]]
        ] = []
        self._test_restrictions: list[tuple[str, str, tuple[str, ...]]] = []
        self._formulations: list[
            tuple[str, tuple[Relation, ...], tuple[tuple[Expression, Expression], ...], tuple[str, ...]]
        ] = []
        self._instances: list[tuple[str, Component, tuple[tuple[str, str], ...], tuple[str, ...]]] = []
        self._ports = []
        self._connections = []
        self._declaration_count = 0

    def _type_syntax(self, value_type: ValueType) -> str:
        try:
            spaces = [(space, space.name) for space, _ in self._source._spaces]
            spaces += self._source._imported_spaces
            sets = [(item, item.name) for item, _ in self._index_sets]
            enums = [(item._definition, item.name) for item, _ in self._source._enums]
            enums += [(item._definition, name) for item, name in self._source._imported_enums]
            return _nominal_type(value_type, spaces, sets, enums)
        except ValueError as error:
            raise ModuleError(str(error)) from error

    def _space_syntax(self, space: FiniteSpace) -> str:
        for value, _ in self._source._spaces:
            if space == value:
                return value.name
        for value, syntax in self._source._imported_spaces:
            if space == value:
                return syntax
        raise ModuleError("finite space must belong to this Module")

    def _nominal_value(self, function: str, name: str, value: Expression) -> Expression:
        if value._owner is not None and value._owner is not self._component_token:
            raise ModuleError("nominal value components must belong to this Component")
        return Expression(_CREATE, _Ast.call(function, [_Ast.name(name), value._ast]), self._component_token, _binders=value._binders, _sources=value._sources)

    def counts(self, space: FiniteSpace, components: Sequence[object]) -> Expression:
        """Construct nonnegative counts in an exact basis registered by this Module."""
        if not isinstance(space, FiniteSpace):
            raise ModuleError("count space must belong to this Module")
        return self._nominal_value("counts", self._space_syntax(space), array(components))

    def coordinates(self, space: FiniteSpace, components: Sequence[object]) -> Expression:
        """Construct signed integer coordinates in this Module's exact finite basis."""
        if not isinstance(space, FiniteSpace):
            raise ModuleError("coordinate space must belong to this Module")
        return self._nominal_value("coordinates", self._space_syntax(space), array(components))

    def index(self, set: IndexSet, value: object) -> Expression:
        """Construct a checked ordinal in an index set registered by this Component."""
        if not isinstance(set, IndexSet) or not any(set == item for item, _ in self._index_sets):
            raise ModuleError("index set must belong to this Component")
        return self._nominal_value("index", set.name, _expression(value))

    def index_set(self, name: str, *, extent: int, doc: str | None = None) -> IndexSet:
        """Declare a constant nominal index set; expression extents require authored source."""
        self._source._ensure_open()
        doc_lines = _doc(doc)
        value = IndexSet(_name(name), extent=extent)
        self._add_name(name)
        self._index_sets.append((value, doc_lines))
        return value

    def sum(self, body: Callable[[Expression], object], *, over: IndexSet, name: str = "i") -> Expression:
        """Construct a finite sum; call body once with an exact scoped index."""
        return self._reduction("sum", body, over, name)

    def product(self, body: Callable[[Expression], object], *, over: IndexSet, name: str = "i") -> Expression:
        """Construct a finite product; the compiler checks element types and units."""
        return self._reduction("product", body, over, name)

    def min(self, body: Callable[[Expression], object], *, over: IndexSet, name: str = "i") -> Expression:
        """Construct a finite minimum of compatible real or integer scalars.

        The callback runs once. Evaluation is eager and retains the first tie.
        """
        return self._reduction("min", body, over, name)

    def max(self, body: Callable[[Expression], object], *, over: IndexSet, name: str = "i") -> Expression:
        """Construct a finite maximum of compatible real or integer scalars.

        The callback runs once. Evaluation is eager and retains the first tie.
        """
        return self._reduction("max", body, over, name)

    def _reduction(self, operation: str, body: Callable[[Expression], object],
                   over: IndexSet, name: str) -> Expression:
        self._source._ensure_open()
        if not isinstance(over, IndexSet) or not any(over == item for item, _ in self._index_sets):
            raise ModuleError("reduction index set must belong to this Component")
        name = _name(name)
        if name in self._names or name in self._source._top_names or name in self._active_binders.values():
            raise ModuleError("reduction binder must not capture an existing name")
        if not callable(body):
            raise TypeError("reduction body must be callable")
        if len(self._active_binders) >= _MAX_EXPRESSION_DEPTH:
            raise ModuleError("reduction nesting exceeds the expression depth limit")
        if name not in self._reduction_names and len(self._reduction_names) >= _MAX_EXPRESSION_NODES:
            raise ModuleError("reduction names exceed the expression node limit")
        token = object()
        self._active_binders[token] = name
        try:
            index = Expression(_CREATE, _Ast.name(name), self._component_token,
                               _binders=frozenset((token,)))
            value = _expression(body(index))
            if value._owner is not None and value._owner is not self._component_token:
                raise ModuleError("reduction body must belong to this Component")
            if not value._binders <= self._active_binders.keys():
                raise ModuleError("reduction body contains an escaped binder")
            result = Expression(_CREATE, value._ast.reduction(operation, name, over.name),
                                self._component_token,
                                _binders=value._binders - {token}, _sources=value._sources)
        finally:
            del self._active_binders[token]
        self._reduction_names.add(name)
        return result

    def _closed_expression(self, value: Expression) -> None:
        if value._sources - {self._source._owner}:
            raise ModuleError("operator expression must belong to this Module")
        if self._active_binders:
            raise ModuleError("reduction callbacks construct expressions, not declarations")
        if value._binders:
            raise ModuleError("declaration expression contains a free reduction binder")

    def _add_name(self, name: object) -> str:
        self._source._ensure_open()
        admitted = _name(name)
        if self._active_binders:
            raise ModuleError("reduction callbacks construct expressions, not declarations")
        if admitted in self._reduction_names:
            raise ModuleError("declaration name would capture a reduction binder")
        if admitted in self._names:
            raise ModuleError(f"duplicate declaration name {admitted!r}")
        if self._declaration_count >= _MAX_DECLARATIONS:
            raise ModuleError(
                f"Component exceeds the {_MAX_DECLARATIONS}-declaration limit"
            )
        self._names.add(admitted)
        self._declaration_count += 1
        return admitted

    def _support(self, support: object) -> Support:
        if (
            not isinstance(support, Support)
            or support._component is not self._component_token
        ):
            raise ModuleError("support must belong to this Component and Module")
        return support

    def _clock(self, clock: Clock | None) -> Clock | None:
        if clock is not None and (
            not isinstance(clock, Clock) or clock._component is not self._component_token
        ):
            raise ModuleError("clock must belong to this Component and Module")
        return clock

    def _activation(self, activation: Clock | Event | None) -> Clock | Event | None:
        if isinstance(activation, Event):
            if activation._component is not self._component_token:
                raise ModuleError("event must belong to this Component and Module")
            return activation
        return self._clock(activation)

    def event(self, name: str, guard: Expression | int | float, *,
              direction: Literal["any", "rising", "falling"], doc: str | None = None) -> Event:
        """Declare a crossing event with an explicit direction and exact lexical guard.

        The compiler checks guard types and activation semantics. Events may
        activate relations and aliases; they are not periodic Clock requirements.
        """
        self._source._ensure_open()
        if not isinstance(direction, str) or direction not in ("any", "rising", "falling"):
            raise ModuleError("event direction must be any, rising, or falling")
        expression = _expression(guard)
        self._closed_expression(expression)
        if expression._owner is not None and expression._owner is not self._component_token:
            raise ModuleError("event guard must belong to this Component and Module")
        if sum(value._nodes for _, value, _, _ in self._events) + expression._nodes > _MAX_EXPRESSION_NODES:
            raise ModuleError("event guards exceed the expression node limit")
        documentation = _doc(doc)
        admitted = self._add_name(name)
        event = Event(_CREATE, self._component_token, admitted)
        self._events.append((event, expression, direction, documentation))
        return event

    def clock(
        self, name: str, *, period_s: Fraction | int,
        phase_s: Fraction | int = 0, doc: str | None = None,
    ) -> Clock:
        """Declare a nominal clock with exact rational period and phase in seconds."""
        period = _clock_seconds(period_s, positive=True)
        phase = _clock_seconds(phase_s, positive=False)
        doc_lines = _doc(doc)
        admitted = self._add_name(name)
        clock = Clock(_CREATE, self._component_token, admitted)
        self._clocks.append((clock, period, phase, doc_lines))
        return clock

    def initial(self, *equations: tuple[object, object],
                left: object = None, right: object = None, doc: str | None = None) -> None:
        """Add simultaneous equations with explicit sides, or one left/right pair.

        Equations are not guesses or ordered writes. Both sides retain their
        exact type, Component ownership, and authored expression budget.
        """
        self._source._ensure_open()
        if (left is None) != (right is None):
            raise TypeError("initial requires both left and right")
        if left is not None and equations:
            raise TypeError("initial left/right cannot be combined with equation pairs")
        pairs = ((left, right),) if left is not None else equations
        if any(not isinstance(pair, (tuple, list)) or len(pair) != 2 for pair in pairs):
            raise TypeError("each initial equation requires two explicit sides")
        pairs = tuple((_expression(left), _expression(right)) for left, right in pairs)
        expressions = tuple(value for pair in pairs for value in pair)
        for value in expressions:
            self._closed_expression(value)
        if any(value._owner is not None and value._owner is not self._component_token for value in expressions):
            raise ModuleError("initial expressions must belong to this Component")
        total_nodes = sum(value._nodes for equations, _ in self._initials for pair in equations for value in pair)
        total_nodes += sum(value._nodes for value in expressions)
        if total_nodes > _MAX_EXPRESSION_NODES:
            raise ModuleError(f"Component initial expressions exceed the {_MAX_EXPRESSION_NODES}-node limit")
        doc_lines = _doc(doc)
        if self._declaration_count >= _MAX_DECLARATIONS:
            raise ModuleError(f"Component exceeds the {_MAX_DECLARATIONS}-declaration limit")
        self._declaration_count += 1
        self._initials.append((pairs, doc_lines))

    def volume(
        self,
        name: str,
        *,
        dimensions: int,
        doc: str | None = None,
    ) -> Support:
        if isinstance(dimensions, bool) or not isinstance(dimensions, int):
            raise TypeError("volume dimensions must be an integer")
        if not 1 <= dimensions <= 15:
            raise ModuleError("volume dimensions must be between 1 and 15")
        doc_lines = _doc(doc)
        admitted = self._add_name(name)
        support = Support(
            _CREATE, self._owner, self._component_token, admitted, "volume"
        )
        self._supports.append((support, "volume", dimensions, doc_lines))
        return support

    def complete_exterior(self, name: str, *, parent: Support, doc: str | None = None) -> BoundarySet:
        """Require the complete exterior of one exact caller-bound volume."""
        return _boundaries.complete_exterior(self, name, parent=parent, doc=doc)

    def boundaries(self, *members: Support) -> BoundarySelectionSet:
        """Select an explicit finite set of exact boundaries for an occurrence binding."""
        return _boundaries.boundaries(self, *members)

    def boundary(
        self,
        name: str,
        *,
        parent: Support,
        doc: str | None = None,
    ) -> Support:
        parent = self._support(parent)
        if parent._kind != "volume":
            raise ModuleError("a boundary parent must be a volume from this Component")
        doc_lines = _doc(doc)
        admitted = self._add_name(name)
        support = Support(
            _CREATE, self._owner, self._component_token, admitted, "boundary"
        )
        self._supports.append((support, "boundary", parent, doc_lines))
        return support

    def parameter(
        self,
        name: str,
        *,
        value_type: ValueType | Record | ImportedRecord,
        doc: str | None = None,
    ) -> Expression:
        if not isinstance(value_type, (ValueType, Record, ImportedRecord)):
            raise TypeError("value_type must be an eqiora.ValueType or Module Record")
        is_record = isinstance(value_type, (Record, ImportedRecord))
        syntax = value_type._type_syntax(self._source) if is_record else self._type_syntax(value_type)
        doc_lines = _doc(doc)
        admitted = self._add_name(name)
        parameter = RecordParameter(self._component_token, admitted, value_type) if is_record else _Parameter(self._component_token, admitted)
        self._parameters.append((parameter, syntax, doc_lines))
        self._requirements.add(parameter)
        return parameter

    def set_default(self, parameter: Expression, value: Expression | int | float | complex) -> None:
        """Set a signature Parameter default after declaring its lexical dependencies."""
        self._source._ensure_open()
        if not isinstance(parameter, _Parameter) or parameter not in self._requirements:
            raise ModuleError("default target must be this Component's Parameter requirement")
        expression = _expression(value)
        self._closed_expression(expression)
        if expression._owner is not None and expression._owner is not self._component_token:
            raise ModuleError("Parameter defaults must belong to this Component")
        total = expression._nodes + sum(value._nodes for target, value in self._defaults.items() if target is not parameter)
        if total > _MAX_EXPRESSION_NODES:
            raise ModuleError("Parameter defaults exceed the 4096-node expression limit")
        self._defaults[parameter] = expression

    def clock_requirement(self, name: str, *, doc: str | None = None) -> Clock:
        """Declare a borrowed nominal clock in the external signature."""
        doc_lines = _doc(doc)
        admitted = self._add_name(name)
        clock = Clock(_CREATE, self._component_token, admitted)
        self._clocks.append((clock, None, None, doc_lines))
        self._requirements.add(clock)
        return clock

    def field_requirement(
        self, name: str, *, on: Support | None = None, value_type: ValueType, role: FieldRole,
        at: Clock | None = None, doc: str | None = None,
    ) -> Expression:
        """Declare a borrowed Field; an occurrence never allocates its storage."""
        field = self.field(name, on=on, value_type=value_type, role=role, at=at, doc=doc)
        self._requirements.add(field)
        return field

    def let_alias(
        self,
        name: str,
        expression: Expression | int | float | complex,
        *,
        value_type: ValueType | None = None,
        on: Support | None = None,
        at: Clock | Event | None = None,
        doc: str | None = None,
    ) -> Expression:
        """Name a private immutable expression in this Component's lexical scope.

        The compiler infers type and intrinsic support; aliases add no storage.
        ``on`` asserts the exact inferred support; it cannot move or broadcast
        an expression. ``at`` asserts one exact activation in the inferred dependency
        profile; it grants no pre/next permissions. Context-dependent coordinate,
        trace, or normal aliases are not admitted.
        """
        value = _expression(expression)
        self._closed_expression(value)
        if value._owner is not None and value._owner is not self._component_token:
            raise ModuleError("alias expressions must belong to this Component")
        if value_type is not None and not isinstance(value_type, ValueType):
            raise TypeError("value_type must be an eqiora.ValueType")
        if on is not None:
            self._support(on)
        at = self._activation(at)
        syntax = None if value_type is None else self._type_syntax(value_type)
        doc_lines = _doc(doc)
        if sum(item[1]._nodes for item in self._aliases) + value._nodes > _MAX_EXPRESSION_NODES:
            raise ModuleError(
                f"Component alias expressions exceed the {_MAX_EXPRESSION_NODES}-node limit"
            )
        admitted = self._add_name(name)
        self._aliases.append((admitted, value, syntax, on, at, doc_lines))
        return Expression(_CREATE, _Ast.name(admitted), self._component_token)

    def property(
        self,
        name: str,
        *,
        contract: PropertyContract,
        doc: str | None = None,
    ) -> PropertyRequirement:
        if not isinstance(contract, PropertyContract) or contract._owner is not self._owner:
            raise ModuleError("property contract must belong to this Module")
        admitted = self._add_name(name)
        requirement = PropertyRequirement(
            _CREATE, self._component_token, admitted, contract
        )
        self._properties.append((requirement, contract, _doc(doc)))
        return requirement

    def field(
        self,
        name: str,
        *,
        on: Support | None = None,
        value_type: ValueType | Record | ImportedRecord,
        role: FieldRole,
        at: Clock | None = None,
        doc: str | None = None,
    ) -> Expression:
        """Declare a spatial Field, optionally activated by an exact local clock."""
        at = self._clock(at)
        on = None if on is None else self._support(on)
        if on is not None and on._kind != "volume":
            raise ModuleError(
                "the initial Module vocabulary admits fields on volumes only"
            )
        if not isinstance(value_type, (ValueType, Record, ImportedRecord)):
            raise TypeError("value_type must be an eqiora.ValueType or Module Record")
        is_record = isinstance(value_type, (Record, ImportedRecord))
        syntax = value_type._type_syntax(self._source) if is_record else self._type_syntax(value_type)
        if not isinstance(role, FieldRole):
            raise TypeError("role must be an eqiora.FieldRole")
        doc_lines = _doc(doc)
        admitted = self._add_name(name)
        expression = RecordField(self._component_token, admitted, value_type) if is_record else _Field(self._component_token, admitted)
        self._fields.append((expression, on, syntax, role, at, doc_lines))
        return expression

    def input(
        self, name: str, *, value_type: ValueType, on: Support | None = None,
        at: Clock | None = None, doc: str | None = None,
    ) -> Expression:
        """Declare a causal input; runtime samples are supplied after static compilation."""
        field = self.field(name, on=on, value_type=value_type, role=FieldRole.Variable, at=at, doc=doc)
        self._causal[field] = "input"
        self._requirements.add(field)
        return field

    def output(
        self, name: str, *, value_type: ValueType, on: Support | None = None,
        at: Clock | None = None, doc: str | None = None,
    ) -> Expression:
        """Declare a causal output defined by an ordinary body relation."""
        field = self.field(name, on=on, value_type=value_type, role=FieldRole.Variable, at=at, doc=doc)
        self._causal[field] = "output"
        return field

    def observable(
        self, name: str, expression: Expression | int | float | complex, *,
        value_type: ValueType, doc: str | None = None,
    ) -> None:
        """Declare a typed derived output without creating an expression symbol."""
        value = _expression(expression)
        self._closed_expression(value)
        if value._owner is not None and value._owner is not self._component_token:
            raise ModuleError("observable expression must belong to this Component")
        if not isinstance(value_type, ValueType):
            raise TypeError("value_type must be an eqiora.ValueType")
        syntax, doc_lines = self._type_syntax(value_type), _doc(doc)
        if sum(item[1]._nodes for item in self._observables) + value._nodes > _MAX_EXPRESSION_NODES:
            raise ModuleError(f"Component observable expressions exceed the {_MAX_EXPRESSION_NODES}-node limit")
        admitted = self._add_name(name)
        self._observables.append((admitted, value, syntax, doc_lines))

    def relation(
        self,
        name: str,
        condition: Equation | Inequality | Complementarity,
        *additional_conditions: Equation | Inequality | Complementarity,
        on: Support | None = None,
        at: Clock | Event | None = None,
        doc: str | None = None,
    ) -> Relation:
        """Declare ordered mathematical conditions owned by one Relation and activation."""
        equalities = (condition, *additional_conditions)
        if len(equalities) > _MAX_DECLARATIONS:
            raise ModuleError("relation exceeds the 256-condition limit")
        if any(not isinstance(item, (Equation, Inequality, Complementarity)) for item in equalities):
            raise TypeError("relation requires equation(), inequality(), or complementarity()")
        at = self._activation(at)
        on = None if on is None else self._support(on)
        if isinstance(on, BoundarySet):
            raise ModuleError("a relation needs an individual boundary member, not a boundary set")
        if isinstance(on, BoundaryMember) and at is not None:
            raise ModuleError("boundary relation families require continuous activation")
        def admit(value: Expression | int | float | complex) -> Expression:
            expression = _boundaries.close_expression(_expression(value), on)
            self._closed_expression(expression)
            if expression._owner is None:
                return Expression(
                    _CREATE,
                    expression._ast,
                    self._component_token,
                    _binders=expression._binders, _sources=expression._sources)
            if expression._owner is not self._component_token:
                raise ModuleError("relation expressions must belong to this Component")
            return expression

        pairs = tuple(("equality" if isinstance(item, Equation) else item._kind,
                       admit(item.lhs), admit(item.rhs)) for item in equalities)
        doc_lines = _doc(doc)
        total_nodes = (
            sum(
                left._nodes + right._nodes
                for item in self._relations for _, left, right in item[2]
            )
            + sum(sum(term._nodes for term in item[2:5] if term is not None) for item in self._laws)
            + sum(left._nodes + right._nodes for _, left, right in pairs)
        )
        if total_nodes > _MAX_EXPRESSION_NODES:
            raise ModuleError(
                f"Component relation expressions exceed the {_MAX_EXPRESSION_NODES}-node limit"
            )
        admitted = self._add_name(name)
        self._relations.append(
            (admitted, on, pairs, at, doc_lines)
        )
        return Relation(_CREATE, self._owner, self._component_token, admitted)

    def law(
        self,
        name: str,
        *,
        on: Support,
        flux: Expression,
        source: Expression,
        storage: Expression | None = None,
        doc: str | None = None,
    ) -> Relation:
        """Declare d(storage)/dt + div(outward flux) = source on a fixed volume.

        Omitted storage denotes a steady Law. Source is always explicit.
        """
        from ._law import declare
        return declare(self, name, on, flux, source, storage, doc)

    def test(self, name: str, *, for_: Expression,
             zero_on: Support | BoundarySelectionSet | None = None) -> Expression:
        """Declare a test for an exact trial and an optional homogeneous boundary restriction."""
        self._source._ensure_open()
        if not isinstance(for_, _Field) or for_._owner is not self._component_token:
            raise ModuleError("test trial must be a Field from this Component")
        if any(trial == for_._name for _, trial, _ in self._test_restrictions):
            raise ModuleError("a trial Field may have only one test declaration")
        if len(self._test_restrictions) >= 8:
            raise ModuleError("weak form exceeds the 8-test limit")
        if zero_on is None:
            boundaries = ()
        elif isinstance(zero_on, BoundarySelectionSet):
            if zero_on._component is not self._component_token:
                raise ModuleError("test boundary selection must belong to this Component")
            boundaries = tuple(member._name for member in zero_on._members)
        else:
            self._support(zero_on)
            if isinstance(zero_on, BoundaryMember):
                raise ModuleError("test restriction must be closed outside a boundary binder")
            boundaries = (zero_on._name,)
        admitted = self._add_name(name)
        self._test_restrictions.append((admitted, for_._name, boundaries))
        return Expression(_CREATE, _Ast.name(admitted), self._component_token)

    def weak_form(
        self,
        name: str,
        relations: Sequence[Relation],
        *,
        equations: Sequence[tuple[object, object]],
        doc: str | None = None,
    ) -> None:
        """Attach one named weak form with ordered relations, tests, and equalities."""
        self._source._ensure_open()
        if not isinstance(relations, Sequence) or not 1 <= len(relations) <= 8:
            raise ModuleError("weak form requires between 1 and 8 relations")
        if any(not isinstance(relation, Relation)
               or relation._component is not self._component_token for relation in relations):
            raise ModuleError("form relations must belong to this Component")
        if len({relation._name for relation in relations}) != len(relations):
            raise ModuleError("weak form relations must be distinct")
        if not isinstance(equations, Sequence) or not 1 <= len(equations) <= 8:
            raise ModuleError("weak form requires between 1 and 8 equations")
        pairs = []
        for pair in equations:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError("weak form equations must be pairs of expressions")
            left, right = (_expression(value) for value in pair)
            for expression in (left, right):
                self._closed_expression(expression)
                if expression._owner is not None and expression._owner is not self._component_token:
                    raise ModuleError("form expressions must belong to this Component")
            pairs.append((left, right))
        if self._formulations:
            raise ModuleError("a Component admits one named weak form")
        total_nodes = (
            sum(left._nodes + right._nodes
                for item in self._relations for _, left, right in item[2])
            + sum(sum(term._nodes for term in item[2:5] if term is not None) for item in self._laws)
            + sum(left._nodes + right._nodes for left, right in pairs)
        )
        if total_nodes > _MAX_EXPRESSION_NODES:
            raise ModuleError(
                f"Component relation and form expressions exceed the {_MAX_EXPRESSION_NODES}-node limit"
            )
        if not self._test_restrictions:
            raise ModuleError("weak form requires explicit test declarations")
        if self._declaration_count >= _MAX_DECLARATIONS:
            raise ModuleError(f"Component exceeds the {_MAX_DECLARATIONS}-declaration limit")
        admitted, doc_lines = _name(name), _doc(doc)
        self._declaration_count += 1
        self._formulations.append((admitted, tuple(relations), tuple(pairs), doc_lines))

    @_property
    def _qualified_name(self) -> str:
        return self._name

    def port(self, name: str, *, connector: Connector, on: Support | None = None,
             doc: str | None = None) -> Port:
        """Declare a nominal physical endpoint, optionally over an exact boundary."""
        return _connections.port(self, name, connector=connector, on=on, doc=doc)

    def connect(self, *ports: Port, over: BoundaryMember | None = None,
                doc: str | None = None) -> None:
        """Declare a conserving physical net or explicit finite boundary family."""
        _connections.connect(self, *ports, over=over, doc=doc)

    def connect_periodic(self, first: FieldPort, second: FieldPort,
                         *, doc: str | None = None) -> None:
        """Identify exactly two field boundaries in a Model's spatial topology."""
        _connections.connect(self, first, second, periodic=True, doc=doc)

    def instance(
        self, name: str, *, component: Component | ComponentRef,
        bindings: Mapping[str, object], doc: str | None = None,
    ) -> Mapping[str, Expression | Port]:
        """Bind explicit signature names to typed enclosing values."""
        if not isinstance(component, (Component, ComponentRef)) or component._owner is not self._owner:
            raise ModuleError("instance component must belong to this Module's explicit imports")
        if component is self or isinstance(component, Component) and component._kind != "component":
            raise ModuleError("an instance requires another Component definition")
        if not isinstance(bindings, Mapping):
            raise TypeError("bindings must map target signature names to enclosing values")
        if len(bindings) > _MAX_DECLARATIONS:
            raise ModuleError("instance bindings exceed the declaration limit")
        handles = {}
        if isinstance(component, ComponentRef):
            signature = component._signature
            handles.update(component._support_contracts)
        else:
            targets = (set(component._requirements)
                       | {item[0] for item in component._supports}
                       | {item[0] for item in component._properties})
            signature = []
            for target in targets:
                handles[target._name] = target
                role = ("support" if isinstance(target, Support)
                        else "clock" if isinstance(target, Clock)
                        else "property" if isinstance(target, PropertyRequirement)
                        else "input" if component._causal.get(target) == "input"
                        else "field" if isinstance(target, _Field)
                        else "parameter")
                signature.append((target._name, role, target not in component._defaults))
        targets = {name: role for name, role, _ in signature if role not in ("output", "port")}
        required = {name for name, role, required in signature if required and role not in ("output", "port")}
        if not required <= set(bindings) or not set(bindings) <= targets.keys():
            raise ModuleError("instance bindings must satisfy the exact required signature")
        admitted_bindings = []
        for target, value in bindings.items():
            target = _name(target)
            role = targets[target]
            if role in ("support", "clock"):
                if role == "support":
                    _boundaries.support_binding(self, handles.get(target), value, bindings)
                else:
                    if not isinstance(value, Clock):
                        raise ModuleError("clock requirement needs an enclosing Clock")
                    self._clock(value)
                expression = _boundaries.support_expression(value) if role == "support" else _Ast.name(value._name)
            elif role == "property":
                required_property = handles.get(target)
                from ._imported_properties import same_contract
                if (not isinstance(value, PropertyRelease) or value._owner is not self._owner
                        or required_property is not None and not same_contract(value._contract, required_property._contract)):
                    raise ModuleError("property binding requires the exact Module contract release")
                expression = _Ast.name(value._name)
            else:
                if role == "field" and (not isinstance(value, _Field) or value._component is not self._component_token):
                    raise ModuleError("Field requirement needs an enclosing Field")
                value = _expression(value)
                self._closed_expression(value)
                if value._owner is not None and value._owner is not self._component_token:
                    raise ModuleError("instance bindings must belong to this Component")
                expression = value._ast
            admitted_bindings.append((target, expression))
        outputs = (component._outputs if isinstance(component, ComponentRef)
                   else tuple(field._name for field, kind in component._causal.items() if kind == "output"))
        physical = (_connections.instance_ports(self, component, _name(name), bindings)
                    if isinstance(component, Component)
                    else _imports.instance_ports(self, component, _name(name), bindings))
        documentation = _doc(doc)
        admitted = self._add_name(name)
        self._instances.append((admitted, component, tuple(admitted_bindings), documentation))
        exposed = {
            field: Expression(_CREATE, _Ast.name(f"{admitted}.{field}"),
                              self._component_token)
            for field in outputs
        }
        exposed.update(physical)
        return MappingProxyType(exposed)

    def set_notation(self, name: str, notation: Notation) -> None:
        """Attach validated notation to an existing declaration in this lexical scope."""
        self._source._ensure_open()
        if name not in self._names:
            raise ModuleError("notation target must be an existing declaration in this Component")
        if not isinstance(notation, Notation):
            raise TypeError("notation must be an eqiora.lang.Notation")
        self._notations[name] = notation

    def _declaration(self, allocate) -> _AstDefinition:
        if self._test_restrictions and not self._formulations:
            raise ModuleError("test declaration requires its owning weak form")
        declarations = []
        def add(name, doc, factory):
            ordinal = allocate(doc, self._notations.get(name))
            declarations.append(factory(ordinal))
        for support, kind, detail, doc in self._supports:
            if kind == "complete_exterior":
                add(support._name, doc, lambda n: _AstDeclaration.complete_exterior(
                    support._name, detail._name, n))
            else:
                add(support._name, doc, lambda n: _AstDeclaration.support(
                    support._name, detail if kind == "volume" else None,
                    detail._name if kind == "boundary" else None, n))
        for port, doc in self._ports:
            add(port._name, doc, lambda n: _boundaries.port_declaration(port, n))
        for ports, over, periodic, doc in self._connections:
            add("", doc, lambda n: _boundaries.connection_declaration(ports, over, periodic, n))
        for parameter, kind, doc in self._parameters:
            default = self._defaults.get(parameter)
            add(parameter._name, doc, lambda n: _AstDeclaration.parameter(
                parameter._name, kind, None if default is None else default._ast, n))
        for requirement, contract, doc in self._properties:
            add(requirement._name, doc, lambda n: _AstDeclaration.property(
                requirement._name, contract._name, n))
        for clock, period, phase, doc in self._clocks:
            def declaration(n):
                if clock in self._requirements:
                    return _AstDeclaration.clock(clock._name, None, None, n)
                second = _Ast.name("s")
                def seconds(value):
                    return _Ast.quantity(str(value.numerator), second).binary(
                        "/", _Ast.quantity(str(value.denominator), _Ast.number("1")))
                return _AstDeclaration.clock(clock._name, seconds(period), seconds(phase), n)
            add(clock._name, doc, declaration)
        for index_set, doc in self._index_sets:
            add(index_set.name, doc, lambda n: _AstDeclaration.index_set(
                index_set.name, _Ast.number(str(index_set.extent)), n))
        for field, support, kind, role, clock, doc in self._fields:
            position = self._causal.get(field, "field" if field in self._requirements else "")
            add(field._name, doc, lambda n: _AstDeclaration.field(
                field._name, kind, "state" if role == FieldRole.State else "variable",
                None if support is None else support._name,
                None if clock is None else clock._name, position, n))
        for name, value, kind, support, clock, doc in self._aliases:
            add(name, doc, lambda n: _AstDeclaration.alias(
                name, kind, None if support is None else support._name,
                None if clock is None else clock._name, value._ast, n))
        for name, value, kind, doc in self._observables:
            add(name, doc, lambda n: _AstDeclaration.observable(name, kind, value._ast, n))
        for event, guard, direction, doc in self._events:
            add(event._name, doc, lambda n: _AstDeclaration.event(
                event._name, guard._ast, direction, n))
        for equations, doc in self._initials:
            add("", doc, lambda n: _AstDeclaration.initial(
                [(left._ast, right._ast) for left, right in equations], n))
        for name, support, pairs, clock, doc in self._relations:
            add(name, doc, lambda n: _boundaries.relation_declaration(
                name, support, pairs, clock, n))
        for name, support, storage, flux, source, doc in self._laws:
            add(name, doc, lambda n: _AstDeclaration.law(
                name, support._name, None if storage is None else storage._ast,
                flux._ast, source._ast, n))
        for name, component, bindings, doc in self._instances:
            add(name, doc, lambda n: _AstDeclaration.instance(
                name, component._qualified_name, bindings, n))
        form = None
        if self._formulations:
            name, relations, equations, doc = self._formulations[0]
            form = (name, [relation._name for relation in relations], self._test_restrictions,
                    [(left._ast, right._ast) for left, right in equations], allocate(doc))
        ordinal = allocate(self._doc, self._source._notations.get(self._name))
        return _AstDefinition(self._name, self._kind == "model", declarations, form, ordinal)

class ComponentRef:
    """Immutable reference to one public Component in an explicit import."""

    __slots__ = ("_owner", "_name", "_qualified_name", "_outputs", "_signature",
                 "_module", "_ports", "_support_contracts")

    def __init__(self, token, imported, name, signature, ports, supports):
        if token is not _CREATE:
            raise TypeError("Component references come from an explicit import")
        object.__setattr__(self, "_owner", imported._owner)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_qualified_name", f"{imported._alias}.{name}")
        object.__setattr__(self, "_module", imported)
        object.__setattr__(self, "_ports", tuple(ports))
        object.__setattr__(self, "_support_contracts", MappingProxyType({
            name: (kind, parent, dimensions) for name, kind, parent, dimensions in supports}))
        object.__setattr__(self, "_signature", tuple(signature))
        object.__setattr__(self, "_outputs", tuple(name for name, role, _ in signature if role == "output"))

    @property
    def name(self) -> str:
        return self._name

    def __setattr__(self, name, value):
        raise AttributeError("ComponentRef is immutable")


class ModuleRef:
    """An explicit module import, with no ambient lookup registry."""

    __slots__ = ("_source", "_owner", "_alias", "_target")

    def __init__(self, token, owner, alias, target):
        if token is not _CREATE:
            raise TypeError("Module references come from import_module()")
        object.__setattr__(self, "_source", owner)
        object.__setattr__(self, "_owner", owner._owner)
        object.__setattr__(self, "_alias", alias)
        object.__setattr__(self, "_target", target)

    def __setattr__(self, name, value):
        raise AttributeError("ModuleRef is immutable")

    def property_contract(self, name: str) -> PropertyContract:
        """Reference one public property contract in this explicit import."""
        from ._imported_properties import contract
        return contract(self, name)

    def property_release(self, name: str) -> PropertyRelease:
        """Reference one public release without copying its scientific definition."""
        from ._imported_properties import release
        return release(self, name)

    def component(self, name: str) -> ComponentRef:
        name = _name(name)
        graph = self._target._freeze()
        return ComponentRef(_CREATE, self, name, graph.component(name),
                            graph.component_ports(name), graph.component_supports(name))

    def dimension(self, name: str) -> Dimension:
        """Resolve one public imported structural dimension through the compiler."""
        from . import _dimensions
        return _dimensions.imported(self, name)

    def record(self, name: str) -> ImportedRecord:
        """Refer to one public closed Record in this exact imported Module."""
        name = _name(name)
        members = self._target._freeze().record_descriptor(name)
        return ImportedRecord(_CREATE, source=self._source, name=name,
                              syntax_name=f"{self._alias}.{name}", members=tuple(members))

    def enum(self, name: str) -> Enum:
        """Refer to one public Enum in this exact imported Module."""
        return self._source._imported_enum(self, _name(name))

    def space(self, name: str) -> FiniteSpace:
        """Refer to one public finite Space in this exact imported Module."""
        return self._source._imported_space(self, _name(name))

    def operator(self, name: str) -> Operator:
        """Call one public pure operator from this exact imported Module."""
        from ._import_operators import operator
        return operator(self, name)

    def connector(self, name: str) -> Connector:
        """Refer to one public nominal connector in this exact imported Module."""
        return _imports.connector(self, name)


class Module:
    """Author bounded Model and Component definitions; freeze on first emission."""

    __slots__ = (
        "_dimensions",
        "_components",
        "_operators",
        "_connectors",
        "_operator_building",
        "_contracts",
        "_frozen_text",
        "_owner",
        "_releases",
        "_materials",
        "_top_names",
        "_notations",
        "_spaces",
        "_imported_spaces",
        "_enums",
        "_imported_enums",
        "_records",
        "_name",
        "_package",
        "_imports",
        "_graph",
    )

    def __init__(self, name: str, *declarations: object, package: str = "eqiora.local_project") -> None:
        self._name = _name_path(name, "module name")
        self._package = _AstModule.namespace(package, self._name)
        self._imports: dict[str, Module] = {}
        self._graph = None if not declarations else _module_from_declarations(name, *declarations)
        self._owner = object()
        self._dimensions = []
        self._components: list[Component] = []
        self._operators: list[Operator] = []
        self._connectors: list[Connector] = []
        self._operator_building = False
        self._contracts: list[PropertyContract] = []
        self._releases: list[PropertyRelease] = []
        self._materials: list[MaterialComposition] = []
        self._top_names: set[str] = set()
        self._notations: dict[str, Notation] = {}
        self._spaces: list[tuple[FiniteSpace, tuple[str, ...]]] = []
        self._imported_spaces: list[tuple[FiniteSpace, str]] = []
        self._enums: list[tuple[Enum, tuple[str, ...]]] = []
        self._imported_enums: list[tuple[Enum, str]] = []
        self._records: list[Record] = []
        self._frozen_text: str | None = None

    def __repr__(self) -> str:
        state = "frozen" if self._graph is not None else "open"
        return f"Module({self._name!r}, state={state!r})"

    def dimension(self, name: str, value: Dimension, *, doc: str | None = None) -> Dimension:
        """Export a structural Dimension; input-unit spelling remains separate."""
        from . import _dimensions
        return _dimensions.declare(self, name, value, doc)

    def _type_syntax(self, value_type: ValueType) -> str:
        try:
            spaces = [(space, space.name) for space, _ in self._spaces] + self._imported_spaces
            enums = [(item._definition, item.name) for item, _ in self._enums]
            enums += [(item._definition, name) for item, name in self._imported_enums]
            return _nominal_type(value_type, spaces, [], enums)
        except ValueError as error:
            raise ModuleError(str(error)) from error

    def _imported_enum(self, imported: ModuleRef, name: str) -> Enum:
        self._ensure_open()
        syntax_name = f"{imported._alias}.{name}"
        for value, syntax in self._imported_enums:
            if syntax == syntax_name:
                return value
        members = imported._target._freeze().enum_descriptor(name)
        value = Enum(_CREATE, _source=self,
                     _definition=_NativeEnum(name, members=members),
                     _syntax_name=syntax_name)
        self._imported_enums.append((value, syntax_name))
        return value

    def _imported_space(self, imported: ModuleRef, name: str) -> FiniteSpace:
        self._ensure_open()
        syntax_name = f"{imported._alias}.{name}"
        for value, syntax in self._imported_spaces:
            if syntax == syntax_name:
                return value
        value = FiniteSpace(name, labels=imported._target._freeze().finite_space_descriptor(name))
        self._imported_spaces.append((value, syntax_name))
        return value

    def field_connector(self, name: str, *, trace: tuple[str, ValueType],
                        flux: tuple[str, ValueType], spatial_vector: bool = False,
                        doc: str | None = None) -> FieldConnector:
        """Declare exact trace/flux types with parent-outward boundary duality."""
        return _boundaries.field_connector(self, name, trace=trace, flux=flux,
                                           spatial_vector=spatial_vector, doc=doc)

    def connector(self, name: str, *, across: tuple[str, ValueType],
                  through: tuple[str, ValueType], doc: str | None = None) -> Connector:
        """Declare a nominal scalar physical connector with named across/through quantities."""
        return _connections.connector(self, name, across=across, through=through, doc=doc)

    def operator(self, name: str, *, inputs: Mapping[str, ValueType], result_type: ValueType,
                 body: Callable[..., object], doc: str | None = None) -> Operator:
        """Declare a closed real-scalar operator from one symbolic callback invocation.

        Formal types include physical dimensions. Calls use named arguments;
        hidden Model captures and values from another Module are rejected.
        """
        self._ensure_open()
        admitted = _name(name)
        if admitted in self._top_names:
            raise ModuleError("duplicate top-level declaration name")
        if not isinstance(inputs, Mapping):
            raise TypeError("operator inputs must map names to ValueType")
        if len(inputs) > _MAX_DECLARATIONS:
            raise ModuleError("operator input count exceeds the declaration limit")
        def checked(value: ValueType) -> str:
            if not isinstance(value, ValueType):
                raise TypeError("operator contracts require ValueType")
            if value.scalar_domain != "real" or value.shape:
                raise ModuleError("Python operators require ordinary real scalar types")
            return self._type_syntax(value)
        signature = tuple((_name(key), checked(value)) for key, value in inputs.items())
        result = checked(result_type)
        documentation = _doc(doc)
        if not callable(body):
            raise TypeError("operator body must be callable")
        token = object()
        formals = {key: Expression(_CREATE, _Ast.name(key), token) for key, _ in signature}
        self._operator_building = True
        try:
            expression = _expression(body(**formals))
        finally:
            self._operator_building = False
        if expression._owner is not None and expression._owner is not token:
            raise ModuleError("operator body must use only its own formal inputs")
        if expression._binders or expression._sources - {self._owner}:
            raise ModuleError("operator body contains a foreign Module or free binder")
        if sum(item._body._nodes for item in self._operators) + expression._nodes > _MAX_EXPRESSION_NODES:
            raise ModuleError("operator bodies exceed the expression node limit")
        self._add_top_name(admitted)
        operator = Operator(_CREATE, _source=self, _name=admitted, _inputs=signature,
                            _result=result, _body=expression, _doc=documentation)
        self._operators.append(operator)
        return operator

    def record(self, name: str, *, members: Mapping[str, ValueType], doc: str | None = None) -> Record:
        """Declare one exact closed heterogeneous record in this Module."""
        return _declare_record(self, name, members=members, doc=doc)

    def enum(self, name: str, *, members: Sequence[str], doc: str | None = None) -> Enum:
        """Declare a closed enum shared by all occurrences in this Module."""
        self._ensure_open()
        if isinstance(members, (str, bytes)) or not isinstance(members, Sequence):
            raise TypeError("enum members must be an ordered sequence of names")
        definition = _NativeEnum(_name(name), members=members)
        documentation = _doc(doc)
        self._add_top_name(name)
        result = Enum(_CREATE, _source=self, _definition=definition)
        self._enums.append((result, documentation))
        return result

    def space(self, name: str, *, labels: Sequence[str], doc: str | None = None) -> FiniteSpace:
        """Declare one exact ordered finite basis shared by this Module's components."""
        self._ensure_open()
        doc_lines = _doc(doc)
        if isinstance(labels, str) or not isinstance(labels, Sequence):
            raise TypeError("space labels must be an ordered sequence")
        value = FiniteSpace(_name(name), labels=labels)
        self._add_top_name(name)
        self._spaces.append((value, doc_lines))
        return value

    def _ensure_open(self) -> None:
        if self._operator_building:
            raise ModuleError("operator callbacks construct expressions, not declarations")
        if self._graph is not None:
            raise ModuleError("Module is frozen after emission or compilation")

    @classmethod
    def parse(cls, name: str, source: str, *, package: str = "eqiora.local_project") -> Module:
        module = cls(name, package=package)
        if not isinstance(source, str):
            raise TypeError("source must be Unicode text")
        module._graph = _AstModule.parse(f"src/{name.replace('.', '/')}.eqi", source)
        return module

    def import_module(self, alias: str, module: Module | None = None, *, path=None) -> ModuleRef:
        alias = _name(alias)
        if (module is None) == (path is None):
            raise TypeError("import_module requires exactly one Module or path")
        if path is not None:
            path = Path(os.fspath(path))
            if path.suffix != ".eqi" or not path.is_file():
                raise ModuleError("an imported path must name an explicit .eqi file")
            if path.stat().st_size > _MAX_OUTPUT_BYTES:
                raise ModuleError("imported source exceeds the source byte limit")
            module = Module.parse(_name(path.stem), path.read_text(encoding="utf-8"), package=self._package)
        if not isinstance(module, Module):
            raise TypeError("import target must be a Module")
        if module is self:
            raise ModuleError("a Module cannot import itself")
        previous = self._imports.get(alias)
        if previous is not None and previous is not module:
            raise ModuleError("import alias already has an attached module")
        if self._graph is not None:
            imports = dict(self._graph.imports())
            if imports.get(alias) != f"{module._package}.{module._name}":
                raise ModuleError("a frozen Module only permits attaching an exact existing import")
        self._imports[alias] = module
        return ModuleRef(_CREATE, self, alias, module)

    def _closure(self):
        units, active, visited = {}, set(), set()
        def visit(module):
            if module in visited:
                return
            identity = (module._package, module._name)
            if identity in active:
                raise ModuleError("recursive module imports are not supported")
            graph = module._freeze()
            previous = units.get(identity)
            if previous is not None:
                if not graph.same_graph(previous._freeze()):
                    raise ModuleError("one logical module name has conflicting contents")
                attachments = lambda item: {alias: (target._package, target._name)
                                            for alias, target in item._imports.items()}
                if attachments(module) != attachments(previous):
                    raise ModuleError("one logical module name has conflicting explicit imports")
            if len(visited) + len(active) >= _MAX_DECLARATIONS:
                raise ModuleError("module closure exceeds the bounded module count")
            active.add(identity)
            for alias, target in graph.imports():
                imported = module._imports.get(alias)
                if imported is None or f"{imported._package}.{imported._name}" != target:
                    raise ModuleError(f"explicit module import {alias!r} has no matching attached source")
                visit(imported)
            active.remove(identity)
            visited.add(module)
            units[identity] = module
        visit(self)
        return [module for _, module in sorted(units.items())]

    def _units(self):
        return [(module._package, module._name, module._freeze()) for module in self._closure()]

    def _dependencies(self):
        return sorted({(module._package, target._package) for module in self._closure()
                       for target in module._imports.values() if module._package != target._package})

    def _add_top_name(self, name: object) -> str:
        self._ensure_open()
        admitted = _name(name)
        if any(admitted in component._reduction_names or admitted in component._active_binders.values()
               for component in self._components):
            raise ModuleError("top-level declaration would capture a reduction binder")
        if admitted in self._top_names:
            raise ModuleError(f"duplicate top-level declaration name {admitted!r}")
        if len(self._top_names) >= _MAX_DECLARATIONS:
            raise ModuleError(
                f"Module exceeds the {_MAX_DECLARATIONS}-declaration limit"
            )
        self._top_names.add(admitted)
        return admitted

    def component(
        self,
        name: str,
        *,
        doc: str | None = None,
    ) -> Component:
        self._ensure_open()
        doc_lines = _doc(doc)
        admitted = self._add_top_name(name)
        component = Component(_CREATE, self, admitted, doc_lines)
        self._components.append(component)
        return component

    def model(self, name: str, *, doc: str | None = None) -> Component:
        """Author a selected Model using the same signature and body vocabulary."""
        model = self.component(name, doc=doc)
        model._kind = "model"
        return model

    def property_contract(
        self,
        name: str,
        *,
        value_type: ValueType,
        inputs: Mapping[str, ValueType] | None = None,
        derivatives: Literal["value_only", "first_partials", "first_open_intervals"] = "value_only",
        branch: str | None = None,
        doc: str | None = None,
    ) -> PropertyContract:
        """Declare ordered typed inputs, result, branch and derivative requirements."""
        self._ensure_open()
        if self._components:
            raise ModuleError("property declarations must precede Components")
        if not isinstance(value_type, ValueType):
            raise TypeError("value_type must be an eqiora.ValueType")
        self._type_syntax(value_type)
        from ._property_profiles import contract_profile
        profile = contract_profile(self, inputs, derivatives, branch)
        doc_lines = _doc(doc)
        admitted = self._add_top_name(name)
        contract = PropertyContract(
            _CREATE,
            self._owner,
            admitted,
            value_type,
            profile,
            doc_lines,
        )
        self._contracts.append(contract)
        return contract

    def property_table_release(self, name: str, *, implements: PropertyContract,
                               data: str, axis_unit: Unit, source_unit: Unit,
                               validity: tuple[object, object], citation: str, license: str,
                               branch: str = "single", doc: str | None = None) -> PropertyRelease:
        """Reference a package-owned table; the compiler verifies its closed assets."""
        from ._property_tables import release
        return release(self, name, implements=implements, data=data, axis_unit=axis_unit,
                       source_unit=source_unit, validity=validity, citation=citation,
                       license=license, branch=branch, doc=doc)

    def property_release(
        self,
        name: str,
        *,
        implements: PropertyContract,
        value: Expression | int | float | complex | Sequence[object],
        source_unit: Unit,
        source_scale: int | float,
        citation: str,
        license: str,
        validity: Expression | bool | None = None,
        branch: str | None = None,
        outside: Literal["reject"] = "reject",
        doc: str | None = None,
    ) -> PropertyRelease:
        """Declare ordered numeric components; the compiler scales every component to SI."""
        self._ensure_open()
        if self._components:
            raise ModuleError("property declarations must precede Components")
        from ._imported_properties import admitted_contract
        if not admitted_contract(self, implements):
            raise ModuleError("release contract must be the exact contract from this Module")
        if not isinstance(source_unit, Unit):
            raise TypeError("source_unit must be an eqiora.units.Unit")
        from ._property_profiles import release_profile
        literal, profile = release_profile(implements, value, validity, branch, outside)
        _number(source_scale)
        if source_scale <= 0:
            raise ModuleError("source_scale must be finite and strictly positive")
        citation_identity = _name_path(citation, "citation identity")
        license_identity = _name_path(license, "license identity")
        doc_lines = _doc(doc)
        admitted = self._add_top_name(name)
        release = PropertyRelease(
            _CREATE,
            _owner=self._owner,
            _name_value=admitted,
            _contract=implements,
            _value=literal,
            _source_unit=source_unit,
            _source_scale=source_scale,
            _profile=profile,
            _citation=citation_identity,
            _license=license_identity,
            _doc=doc_lines,
        )
        self._releases.append(release)
        return release

    def material_composition(
        self,
        name: str,
        *,
        properties: Mapping[str, PropertyRelease],
        doc: str | None = None,
    ) -> MaterialComposition:
        self._ensure_open()
        if not isinstance(properties, Mapping):
            raise TypeError("properties must be a mapping of property requirements to releases")
        if not properties:
            raise ModuleError("material composition requires at least one property")
        bindings: list[tuple[str, PropertyRelease]] = []
        for requirement, release in properties.items():
            admitted_requirement = _name(requirement)
            if not isinstance(release, PropertyRelease) or release._owner is not self._owner:
                raise ModuleError("material property releases must belong to this Module")
            bindings.append((admitted_requirement, release))
        bindings.sort(key=lambda binding: binding[0])
        doc_lines = _doc(doc)
        admitted = self._add_top_name(name)
        material = MaterialComposition(
            _CREATE,
            _owner=self._owner,
            _name_value=admitted,
            _bindings=tuple(bindings),
            _doc=doc_lines,
        )
        self._materials.append(material)
        return material

    def set_notation(self, name: str, notation: Notation) -> None:
        """Attach validated notation to an existing top-level declaration."""
        self._ensure_open()
        if name not in self._top_names:
            raise ModuleError("notation target must be an existing top-level declaration")
        if not isinstance(notation, Notation):
            raise TypeError("notation must be an eqiora.lang.Notation")
        self._notations[name] = notation

    def _freeze(self, *, commit: bool = True) -> _AstModule:
        if self._graph is not None:
            return self._graph
        self._ensure_open()
        metadata = []
        metadata_bytes = 0
        ordinal = 0
        def allocate(doc=(), notation=None):
            nonlocal ordinal, metadata_bytes
            ordinal += 1
            if ordinal > _MAX_DECLARATIONS * _MAX_DECLARATIONS:
                raise ModuleError("Module exceeds the total declaration limit")
            if doc or notation is not None:
                metadata_bytes += sum(len(line.encode("utf-8")) + 1 for line in doc)
                metadata_bytes += len(notation.canonical.encode("utf-8")) if notation is not None else 0
                if metadata_bytes > _MAX_OUTPUT_BYTES:
                    raise ModuleError(f"Module exceeds the {_MAX_OUTPUT_BYTES}-byte limit")
                metadata.append((ordinal, "\n".join(doc) if doc else None, notation))
            return ordinal
        definitions = [component._declaration(allocate) for component in self._components]
        operators = [
            (operator._name, [(name, kind, allocate()) for name, kind in operator._inputs], operator._result, operator._body._ast,
             allocate(operator._doc, self._notations.get(operator._name)))
            for operator in self._operators
        ]
        connectors = [
            (connector._name, *connector._across, *connector._through,
             allocate(connector._doc, self._notations.get(connector._name)))
            for connector in self._connectors if not isinstance(connector, FieldConnector)
        ]
        field_connectors = [
            (connector._name, *connector._across, *connector._through,
             connector._spatial_vector,
             allocate(connector._doc, self._notations.get(connector._name)))
            for connector in self._connectors if isinstance(connector, FieldConnector)
        ]
        dimensions = [(name, self._type_syntax(ValueType.real(value)),
                       allocate(doc, self._notations.get(name)))
                      for name, value, doc in self._dimensions]
        graph = None if not (definitions or operators or connectors or field_connectors or dimensions) and (self._contracts or self._releases) else _AstModule(definitions, operators, connectors, field_connectors, dimensions)
        for contract in self._contracts:
            graph = _AstModule.with_contract(graph, contract._name, self._type_syntax(contract._value_type),
                                        ([(name, self._type_syntax(kind)) for name, kind in contract._profile.inputs],
                                         contract._profile.derivatives, contract._profile.branch),
                                        allocate(contract._doc, self._notations.get(contract._name)))
        for record in self._records:
            graph = graph.with_record(record.name, [(name, kind, allocate()) for name, kind in record._syntax],
                                      allocate(record._doc, self._notations.get(record.name)))
        for enumeration, doc in self._enums:
            graph = graph.with_enum(enumeration.name, enumeration.members,
                                    allocate(doc, self._notations.get(enumeration.name)))
        for space, doc in self._spaces:
            graph = graph.with_space(space.name, space.labels,
                                     allocate(doc, self._notations.get(space.name)))
        for release in self._releases:
            if release._table is not None:
                from ._property_tables import emit
                graph = emit(graph, release, allocate(release._doc, self._notations.get(release._name)))
                continue
            graph = _AstModule.with_release(
                graph,
                release._name, release._contract._name,
                (release._value._ast, release._source_unit._ast,
                 _Ast.number(_number(release._source_scale))),
                (release._citation, release._license),
                (None if release._profile.validity is None else release._profile.validity._ast,
                 release._profile.branch, release._profile.outside),
                allocate(release._doc, self._notations.get(release._name)))
        for material in self._materials:
            graph = graph.with_material(
                material._name, [(name, release._name, allocate()) for name, release in material._bindings],
                allocate(material._doc, self._notations.get(material._name)))
        for alias, module in sorted(self._imports.items()):
            graph = graph.with_import(f"{module._package}.{module._name}", alias, allocate())
        for declaration, doc, notation in metadata:
            graph = graph.metadata(declaration, doc, notation)
        graph.check()
        if commit:
            self._graph = graph
        return graph

    def to_eqi(self) -> str:
        """Canonically format the compiler-owned module and freeze its declarations."""
        if self._frozen_text is None:
            graph = self._freeze(commit=False)
            text = graph.source()
            self._graph = graph
            self._frozen_text = text
        return self._frozen_text

    def write_eqi(self, path: str | os.PathLike[str]) -> None:
        """Atomically replace one regular path with this Module's UTF-8 text."""

        target = Path(os.fspath(path))
        if "\x00" in str(target):
            raise ValueError("output path contains a null byte")
        if target.exists() and (target.is_dir() or target.is_symlink()):
            raise ValueError(
                "output path must be a regular file, not a directory or symlink"
            )
        text = self.to_eqi()
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as stream:
                temporary = stream.name
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass


from . import _connections
from ._connections import Connector, Port
from . import _boundaries
from ._boundaries import BoundarySet, BoundaryMember, BoundarySelectionSet, FieldConnector, FieldPort
from . import _imports
from ._constraints import Inequality, Complementarity, inequality, complementarity


__all__ = [
    "BoundarySet",
    "BoundaryMember",
    "BoundarySelectionSet",
    "FieldConnector",
    "FieldPort",
    "Inequality", "Complementarity", "inequality", "complementarity",
    "Connector",
    "Port",
    "Equation",
    "equation",
    "ComponentRef",
    "ModuleRef",
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "logical_and",
    "logical_or",
    "logical_not",

    "Clock",
    "Component",
    "Expression",
    "Enum",
    "Record",
    "ImportedRecord",
    "RecordField",
    "RecordParameter",
    "Event",
    "MaterialComposition",
    "Notation",
    "Operator",
    "PropertyContract",
    "PropertyRequirement",
    "PropertyRelease",
    "Relation",
    "Module",
    "ModuleError",
    "Support",
    "array",
    "case",
    "coordinate",
    "dot",
    "div",
    "grad",
    "integrate",
    "if_else",
    "isotropic_lift",
    "math",
    "normal",
    "ordinal",
    "partial",
    "derivative",
    "time",
    "pre",
    "next",
    "quantity",
    "quotient",
    "remainder",
    "to_real",
    "to_integer",
    "frobenius",
    "symmetric_part",
    "tensor_value",
    "trace",
]
