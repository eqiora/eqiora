"""Physical Law authoring using the Component's ordinary identity and bounds."""

from __future__ import annotations

from . import (
    Component, Expression, ModuleError, Relation, Support, _CREATE,
    _MAX_EXPRESSION_NODES, _doc, _expression,
)


def declare(
    component: Component, name: str, on: Support, flux: Expression,
    source: Expression, storage: Expression | None, doc: str | None,
) -> Relation:
    on = component._support(on)
    if on._kind != "volume":
        raise ModuleError("a fixed-domain Law requires a volume support")
    terms = tuple(None if value is None else _expression(value)
                  for value in (storage, flux, source))
    for value in terms:
        if value is not None:
            component._closed_expression(value)
            if value._owner is not None and value._owner is not component._component_token:
                raise ModuleError("Law expressions must belong to this Component")
    total = sum(left._nodes + right._nodes
                for item in component._relations for _, left, right in item[2])
    total += sum(sum(term._nodes for term in item[2:5] if term is not None)
                 for item in component._laws)
    total += sum(term._nodes for term in terms if term is not None)
    total += sum(item[1]._nodes + item[2]._nodes for item in component._formulations)
    if total > _MAX_EXPRESSION_NODES:
        raise ModuleError(f"Component Law expressions exceed the {_MAX_EXPRESSION_NODES}-node limit")
    docs = _doc(doc)
    admitted = component._add_name(name)
    stored, outward, production = terms
    component._laws.append((admitted, on, stored, outward, production, docs))
    return Relation(_CREATE, component._owner, component._component_token, admitted)
