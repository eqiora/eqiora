"""Exact declaration references shared by imported authoring adapters."""

from . import _CREATE, _name


class DeclarationRef:
    """Private adapter metadata, never a new local nominal declaration."""

    __slots__ = ("_module", "_kind", "_local_name", "_name", "_owner", "_descriptor", "_identity")

    def __init__(self, token, imported, kind, name, descriptor, *, origin=None):
        if token is not _CREATE:
            raise TypeError("declaration references require an explicit Module import")
        for key, value in (("_module", imported), ("_kind", kind), ("_local_name", name),
                           ("_name", f"{imported._alias}.{name}"), ("_owner", imported._owner),
                           ("_descriptor", descriptor),
                           ("_identity", ((imported._target if origin is None else origin)._package,
                                          (imported._target if origin is None else origin)._name,
                                          kind, name.split(".")[-1]))):
            object.__setattr__(self, key, value)

    def __setattr__(self, name, value):
        raise AttributeError("imported declaration references are immutable")


def reference(imported, name, kind, describe):
    """Use one native public-declaration lookup, retaining its exact Module owner.

    Consumer leaves supply ``describe(graph, local_name)``. That lookup must
    enforce declaration kind and visibility through the existing source AST.
    The returned metadata never authorizes a local declaration or value lookup.
    """
    name = _name(name)
    descriptor = describe(imported._target._freeze(), name)
    return DeclarationRef(_CREATE, imported, kind, name, descriptor)


def same_declaration(left, right):
    """Compare an origin descriptor rather than equal member types or spelling."""
    left_ref = getattr(left, "_reference", None)
    right_ref = getattr(right, "_reference", None)
    if left_ref is not None and right_ref is not None:
        return left_ref._identity == right_ref._identity
    return left is right
