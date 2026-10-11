"""Structural dimensions use the native exact-module resolver and ordinary Dimension."""
from .._eqiora import Dimension
from . import _doc, _name


def declare(module, name, value, doc):
    if not isinstance(value, Dimension):
        raise TypeError("dimension aliases require a Dimension")
    documentation = _doc(doc)
    name = module._add_top_name(name)
    module._dimensions.append((name, value, documentation))
    return value


def imported(reference, name):
    name = _name(name)
    target = reference._target
    return target._freeze().dimension_descriptor(
        (target._package, target._name), target._units(), target._dependencies(), name)
