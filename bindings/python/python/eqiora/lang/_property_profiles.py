"""Bounded immutable authoring profiles for exact nominal property declarations."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import Expression, PropertyContract, ValueType, PropertyRequirement


@dataclass(frozen=True, slots=True)
class ContractProfile:
    inputs: tuple[tuple[str, ValueType], ...]
    derivatives: str
    branch: str | None


@dataclass(frozen=True, slots=True)
class ReleaseProfile:
    validity: Expression | None
    branch: str | None
    outside: str


def contract_profile(module, inputs, derivatives, branch) -> ContractProfile:
    from . import ModuleError, ValueType, _MAX_DECLARATIONS, _name, _name_path

    if inputs is None:
        inputs = {}
    if not isinstance(inputs, Mapping):
        raise TypeError("property inputs must be an ordered mapping of names to ValueType")
    if len(inputs) > _MAX_DECLARATIONS:
        raise ModuleError("property inputs exceed the declaration limit")
    admitted = []
    seen = set()
    for name, kind in inputs.items():
        name = _name(name)
        if name in seen:
            raise ModuleError("duplicate property input name")
        seen.add(name)
        if not isinstance(kind, ValueType):
            raise TypeError("property input types must be eqiora.ValueType values")
        module._type_syntax(kind)
        admitted.append((name, kind))
    if derivatives not in ("value_only", "first_partials", "first_open_intervals"):
        raise ModuleError("property derivatives must be value_only, first_partials or first_open_intervals")
    if branch is not None:
        branch = _name_path(branch, "property branch")
    return ContractProfile(tuple(admitted), derivatives, branch)


def formal(contract: PropertyContract, name: str) -> Expression:
    from . import Expression, ModuleError, _Ast, _CREATE

    if name not in tuple(name for name, _ in contract._profile.inputs):
        raise ModuleError("property input must name a declared contract formal")
    return Expression(_CREATE, _Ast.name(name), contract,
                      _sources=frozenset((contract._owner,)))


def release_profile(contract: PropertyContract, value, validity, branch, outside):
    from . import Expression, ModuleError, _expression, _literal_expression, _name_path

    if branch is not None:
        branch = _name_path(branch, "property branch")
    if branch != contract._profile.branch:
        raise ModuleError("property release branch must match its exact contract")
    if outside != "reject":
        raise ModuleError("property outside-domain policy must be reject")
    body = value if isinstance(value, Expression) else _literal_expression(value)
    predicate = None if validity is None else _expression(validity)
    for expression in (body, predicate):
        if expression is None:
            continue
        if expression._owner is not None and expression._owner is not contract:
            raise ModuleError("property expressions must use this exact contract's inputs")
        if expression._sources - {contract._owner}:
            raise ModuleError("property expressions must belong to this Module")
        if expression._binders:
            raise ModuleError("property expressions cannot capture an index binder")
    return body, ReleaseProfile(predicate, branch, outside)


def apply(requirement: PropertyRequirement, arguments: dict[str, object]) -> Expression:
    from . import (Expression, ModuleError, _Ast, _CREATE, _expression,
                   _MAX_EXPRESSION_DEPTH, _MAX_EXPRESSION_NODES)

    names = tuple(name for name, _ in requirement._contract._profile.inputs)
    if set(arguments) != set(names):
        raise ModuleError("property call must supply exactly its named inputs")
    values = tuple(_expression(arguments[name]) for name in names)
    for value in values:
        if value._owner is not None and value._owner is not requirement._component:
            raise ModuleError("property arguments must belong to the same Component")
        if value._sources - {requirement._contract._owner}:
            raise ModuleError("property arguments must belong to this Module")
    if (max((value._depth for value in values), default=0) + 1 > _MAX_EXPRESSION_DEPTH
            or sum(value._nodes for value in values) + 1 > _MAX_EXPRESSION_NODES):
        raise ModuleError("property call exceeds the expression depth or node limit")
    return Expression(_CREATE, _Ast.call(requirement._name, [v._ast for v in values], names),
                      requirement._component,
                      _binders=frozenset().union(*(v._binders for v in values)),
                      _sources=frozenset((requirement._contract._owner,)))
