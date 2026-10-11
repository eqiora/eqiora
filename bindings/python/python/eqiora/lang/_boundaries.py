"""Exact boundary support and field interface authoring over the shared AST."""

from . import (
    Expression, ModuleError, Support, ValueType, _Ast, _AstDeclaration, _CREATE,
    _MAX_DECLARATIONS, _doc, _name,
)
from ._connections import Connector, Port


class BoundarySet(Support):
    """One complete exterior requirement with an exact parent support."""

    __slots__ = ("_parent", "_scope", "_members")

    def __init__(self, token, component, name, parent):
        super().__init__(token, component._owner, component._component_token,
                         name, "complete_exterior")
        object.__setattr__(self, "_parent", parent)
        object.__setattr__(self, "_scope", component)
        object.__setattr__(self, "_members", {})

    def member(self, name: str):
        """Bind an exact member for a port, relation, or connection family."""
        self._scope._source._ensure_open()
        if self._scope._active_binders:
            raise ModuleError("reduction callbacks construct expressions, not boundary binders")
        name = _name(name)
        if name in self._members:
            return self._members[name]
        if name in self._scope._names or name in self._scope._source._top_names:
            raise ModuleError("boundary member would capture an existing declaration")
        if len(self._members) >= _MAX_DECLARATIONS:
            raise ModuleError("complete exterior exceeds 256 authored member binders")
        member = BoundaryMember(_CREATE, self, name)
        self._members[name] = member
        self._scope._reduction_names.add(name)
        return member


class BoundaryMember(Support):
    """A lexical boundary binder retaining its exact complete exterior and parent."""

    __slots__ = ("_set",)

    def __init__(self, token, boundary_set, name):
        super().__init__(token, boundary_set._owner, boundary_set._component,
                         name, "boundary_member")
        object.__setattr__(self, "_set", boundary_set)


class BoundarySelectionSet:
    """An explicit finite collection of exact boundary handles sharing one parent."""

    __slots__ = ("_component", "_parent", "_members", "_kind", "_ast")

    def __init__(self, token, component, parent, members):
        if token is not _CREATE:
            raise TypeError("boundary selections are created by Component.boundaries()")
        for name, value in (("_component", component._component_token),
                            ("_parent", parent), ("_members", members),
                            ("_kind", "complete_exterior"),
                            ("_ast", _Ast.call("boundaries", [_Ast.name(x._name) for x in members]))):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError("BoundarySelectionSet is immutable")


class FieldConnector(Connector):
    """A nominal trace/flux pair with one shape, frame, and outward boundary duality."""

    __slots__ = ("_spatial_vector",)

    def __init__(self, token, owner, name, trace, flux, spatial_vector, doc):
        super().__init__(token, owner, name, trace, flux, doc)
        object.__setattr__(self, "_spatial_vector", spatial_vector)


class FieldPort(Port):
    """An exact field endpoint or a finite family awaiting a boundary selection."""

    __slots__ = ("_support", "_family", "_selector", "_scope")

    def __init__(self, token, component, name, connector, support, family=None,
                 selector=None):
        super().__init__(token, component._component_token, name, connector)
        object.__setattr__(self, "_scope", component)
        object.__setattr__(self, "_support", support)
        object.__setattr__(self, "_family", family)
        object.__setattr__(self, "_selector", selector)

    def __getitem__(self, boundary):
        if self._family is None or self._selector is not None:
            raise ModuleError("only an unselected boundary port family accepts a selector")
        self._scope._support(boundary)
        if boundary._kind not in ("boundary", "boundary_member"):
            raise ModuleError("port family selection needs an individual boundary")
        _, boundary_set = self._family
        if isinstance(boundary, BoundaryMember):
            if boundary._set is not boundary_set:
                raise ModuleError("boundary member belongs to a different complete exterior")
        elif boundary_set is not None:
            parent = next(detail for support, _, detail, _ in self._scope._supports
                          if support is boundary)
            if parent is not boundary_set._parent:
                raise ModuleError("boundary selector has a different exact parent")
            if isinstance(boundary_set, BoundarySelectionSet) and boundary not in boundary_set._members:
                raise ModuleError("boundary selector is absent from the exact selected exterior")
        return FieldPort(_CREATE, self._scope, self._name, self._connector,
                         self._support, self._family, boundary)

    def member(self, name: str) -> Expression:
        if name not in (self._connector._across[0], self._connector._through[0]):
            raise AttributeError(f"Connector {self._connector._name!r} has no quantity {name!r}")
        if self._family is not None:
            if self._selector is None:
                raise ModuleError("select an exact boundary before reading a port family")
            expression = _Ast.name(self._name).boundary_port(
                self._family[0], self._selector._name).member(name)
        else:
            expression = _Ast.name(f"{self._name}.{name}")
        binders = (frozenset((self._selector,))
                   if isinstance(self._selector, BoundaryMember) else frozenset())
        return Expression(_CREATE, expression, self._owner, _binders=binders)


def complete_exterior(component, name, *, parent, doc=None):
    parent = component._support(parent)
    if parent._kind != "volume":
        raise ModuleError("a complete exterior parent must be a volume from this Component")
    documentation = _doc(doc)
    admitted = component._add_name(name)
    value = BoundarySet(_CREATE, component, admitted, parent)
    component._supports.append((value, "complete_exterior", parent, documentation))
    return value


def boundaries(component, *members):
    component._source._ensure_open()
    if not 1 <= len(members) <= _MAX_DECLARATIONS:
        raise ModuleError("boundary selection needs between 1 and 256 exact members")
    parent = None
    for member in members:
        component._support(member)
        if member._kind != "boundary":
            raise ModuleError("boundary selections require individual boundary handles")
        member_parent = next(detail for support, _, detail, _ in component._supports
                             if support is member)
        if parent is None:
            parent = member_parent
        elif parent is not member_parent:
            raise ModuleError("boundary selections must share one exact parent")
    if len(set(members)) != len(members):
        raise ModuleError("boundary selection contains a duplicate member")
    return BoundarySelectionSet(_CREATE, component, parent, members)


def field_connector(module, name, *, trace, flux, spatial_vector=False, doc=None):
    def quantity(value):
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError("connector quantity must be a (name, ValueType) pair")
        member, kind = value
        member = _name(member)
        if not isinstance(kind, ValueType):
            raise TypeError("connector quantity needs an eqiora.ValueType")
        if kind.scalar_domain != "real" or kind.array_rank:
            raise ModuleError("field connector quantities require real scalar or spatial tensor types")
        return member, kind
    trace, flux = quantity(trace), quantity(flux)
    if trace[0] == flux[0]:
        raise ModuleError("connector quantity names must be distinct")
    if (trace[1].shape, trace[1].frame) != (flux[1].shape, flux[1].frame):
        raise ModuleError("field connector quantities require identical shape and frame")
    if not isinstance(spatial_vector, bool):
        raise TypeError("spatial_vector must be a Boolean")
    if spatial_vector and trace[1].shape:
        raise ModuleError("generic spatial_vector requires scalar quantity types")
    documentation = _doc(doc)
    admitted = module._add_top_name(name)
    value = FieldConnector(_CREATE, module._owner, admitted, trace, flux,
                           spatial_vector, documentation)
    module._connectors.append(value)
    return value


def port(component, name, *, connector, on, doc=None):
    if connector._owner is not component._owner:
        raise ModuleError("port connector must belong to this Module")
    on = component._support(on)
    if on._kind not in ("boundary", "boundary_member"):
        raise ModuleError("a field port requires an individual boundary or boundary member")
    documentation = _doc(doc)
    admitted = component._add_name(name)
    family = (on._name, on._set) if isinstance(on, BoundaryMember) else None
    value = FieldPort(_CREATE, component, admitted, connector, on, family)
    component._ports.append((value, documentation))
    return value


def close_expression(value, on):
    """Discharge only the exact relation-family binder before ordinary admission."""
    boundary_binders = {binder for binder in value._binders
                        if isinstance(binder, BoundaryMember)}
    if boundary_binders - ({on} if isinstance(on, BoundaryMember) else set()):
        raise ModuleError("relation contains a foreign or unbound boundary family member")
    if not boundary_binders:
        return value
    return Expression(_CREATE, value._ast, value._owner,
                      _binders=value._binders - boundary_binders, _sources=value._sources)


def support_binding(component, required, value, bindings):
    if isinstance(value, BoundarySelectionSet):
        if value._component is not component._component_token:
            raise ModuleError("boundary selection must belong to this Component")
    else:
        component._support(value)
    if isinstance(value, BoundaryMember):
        raise ModuleError("an instance binding cannot escape a boundary family binder")
    if required is None:
        return
    if isinstance(required, tuple):
        kind, parent_name, dimensions = required
        if kind != value._kind:
            raise ModuleError("support binding must preserve volume, boundary, or boundary-set kind")
        detail = (value._parent if isinstance(value, (BoundarySet, BoundarySelectionSet))
                  else next(detail for support, _, detail, _ in component._supports if support is value))
        if parent_name is not None and bindings.get(parent_name) is not detail:
            raise ModuleError("support binding must preserve the imported exact parent contract")
        if dimensions is not None and dimensions != detail:
            raise ModuleError("volume binding must preserve the imported ambient dimension")
        return
    if required._kind != value._kind:
        raise ModuleError("support binding must preserve volume, boundary, or boundary-set kind")
    if isinstance(required, BoundarySet):
        if bindings.get(required._parent._name) is not value._parent:
            raise ModuleError("complete exterior binding must preserve its exact bound parent")
    else:
        required_detail = next(detail for definition in component._source._components
                               if definition._component_token is required._component
                               for support, _, detail, _ in definition._supports
                               if support is required)
        actual_detail = next(detail for support, _, detail, _ in component._supports
                             if support is value)
        if required._kind == "volume" and required_detail != actual_detail:
            raise ModuleError("volume binding must preserve the ambient dimension")
        if required._kind == "boundary" and bindings.get(required_detail._name) is not actual_detail:
            raise ModuleError("boundary binding must preserve its exact bound parent")


def support_expression(value):
    return value._ast if isinstance(value, BoundarySelectionSet) else _Ast.name(value._name)


def port_declaration(port, ordinal):
    if isinstance(port, FieldPort):
        return _AstDeclaration.field_port(
            port._name, port._connector._name, port._support._name,
            port._family[1]._name if port._family is not None else None, ordinal)
    return _AstDeclaration.scalar_port(port._name, port._connector._name, ordinal)


def endpoint(port):
    selector = None
    if isinstance(port, FieldPort) and port._selector is not None:
        selector = port._family[0], port._selector._name
    return port._name, selector


def connection_declaration(ports, over, periodic, ordinal):
    endpoints = [endpoint(port) for port in ports]
    if periodic or over is not None or any(selector is not None for _, selector in endpoints):
        binder = (over._name, over._set._name) if over is not None else None
        return _AstDeclaration.boundary_connection(endpoints, binder, periodic, ordinal)
    return _AstDeclaration.conserving_connection(
        [_Ast.name(port._name) for port in ports], ordinal)


def relation_declaration(name, support, pairs, clock, ordinal):
    conditions = [(kind, left._ast, right._ast) for kind, left, right in pairs]
    if isinstance(support, BoundaryMember):
        return _AstDeclaration.boundary_relation(
            name, support._name, support._set._name, conditions, ordinal)
    return _AstDeclaration.relation(name, None if support is None else support._name,
                                   None if clock is None else clock._name, conditions, ordinal)
