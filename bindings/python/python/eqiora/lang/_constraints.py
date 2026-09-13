"""Typed condition syntax; the shared compiler owns units, signs and supports."""

from . import Expression, _CREATE, _expression, _owner


class _Constraint:
    __slots__ = ("_lhs", "_rhs")

    def __init__(self, token, lhs, rhs):
        if token is not _CREATE:
            raise TypeError("use inequality() or complementarity()")
        object.__setattr__(self, "_lhs", lhs)
        object.__setattr__(self, "_rhs", rhs)

    def __setattr__(self, name, value):
        raise AttributeError("mathematical conditions are immutable")

    @property
    def lhs(self) -> Expression:
        return self._lhs

    @property
    def rhs(self) -> Expression:
        return self._rhs

    def __bool__(self):
        raise TypeError("mathematical conditions have no Python truth value")


class Inequality(_Constraint):
    """A non-strict mathematical condition with lhs greater than or equal to rhs."""

    __slots__ = ()
    _kind = "inequality"


class Complementarity(_Constraint):
    """Two explicit nonnegativity predicates whose physical operands are complementary."""

    __slots__ = ()
    _kind = "complementarity"


def inequality(lhs: object, rhs: object) -> Inequality:
    """Require lhs >= rhs; realization requires a separate enforcement policy."""
    left, right = _expression(lhs), _expression(rhs)
    _owner(left, right)
    return Inequality(_CREATE, left, right)


def complementarity(left: Expression, right: Expression) -> Complementarity:
    """Require two nonnegative operands with at least one zero.

    Pass explicit greater_equal(operand, typed_zero) or less_equal(typed_zero,
    operand) predicates. The shared compiler rejects absent or reversed signs,
    incompatible units and foreign supports before constructing a Model.
    """
    if not isinstance(left, Expression) or not isinstance(right, Expression):
        raise TypeError("complementarity requires two symbolic nonnegativity predicates")
    _owner(left, right)
    return Complementarity(_CREATE, left, right)
