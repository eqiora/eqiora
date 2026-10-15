"""Bounded decimal spelling shared by syntax and numerical input adapters."""
from decimal import Decimal, Context, DecimalException, Inexact, Rounded, Overflow, InvalidOperation, MAX_EMAX, MIN_EMIN
import math as _stdlib_math

from ._eqiora import ModuleError


def number(value: object) -> str:
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

def quantity(value):
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
        literal = number(value)
    if len(literal.encode("ascii")) > 256:
        raise ModuleError("quantity literal exceeds the 256-byte limit")
    return literal
