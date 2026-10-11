"""Immutable named physical ports; the compiler owns connection semantics."""

from . import (
    Expression, ModuleError, ValueType, _Ast, _CREATE, _MAX_DECLARATIONS,
    _doc, _name,
)


class Connector:
    """A nominal scalar across/through declaration in one Module.

    Authority: ``bindings/python/python/eqiora/lang/_connections.py::Connector``.
    """

    __slots__ = ("_owner", "_name", "_across", "_through", "_doc")

    def __init__(self, token, owner, name, across, through, doc):
        if token is not _CREATE:
            raise TypeError("connectors are created by Module.connector()")
        for key, value in (("_owner", owner), ("_name", name), ("_across", across),
                           ("_through", through), ("_doc", doc)):
            object.__setattr__(self, key, value)

    def __setattr__(self, name, value):
        raise AttributeError("Connector is immutable")

    def __repr__(self):
        return f"Connector({self._name!r})"


class Port:
    """An exact physical endpoint; declared quantities are symbolic expressions.

    Use member(name) when a declared name overlaps a Python attribute.

    Authority: ``bindings/python/python/eqiora/lang/_connections.py::Port``.
    """

    __slots__ = ("_owner", "_name", "_connector")

    def __init__(self, token, owner, name, connector):
        if token is not _CREATE:
            raise TypeError("ports are created by Component.port() or instance()")
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_connector", connector)

    def __getattr__(self, name):
        return self.member(name)

    def member(self, name: str) -> Expression:
        """Select an exact declared quantity, including Python attribute names.

        Authority: ``bindings/python/python/eqiora/lang/_connections.py::Port.member``.
        """
        if name not in (self._connector._across[0], self._connector._through[0]):
            raise AttributeError(f"Connector {self._connector._name!r} has no quantity {name!r}")
        return Expression(_CREATE, _Ast.name(f"{self._name}.{name}"), self._owner)

    def __setattr__(self, name, value):
        raise AttributeError("Port is immutable")

    def __repr__(self):
        return f"Port({self._name!r}, connector={self._connector._name!r})"


def connector(module, name, *, across, through, doc=None):
    def quantity(value):
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError("connector quantity must be a (name, ValueType) pair")
        member, kind = value
        member = _name(member)
        if not isinstance(kind, ValueType):
            raise TypeError("connector quantity needs an eqiora.ValueType")
        return member, module._type_syntax(kind)

    across, through = quantity(across), quantity(through)
    if across[0] == through[0]:
        raise ModuleError("connector quantity names must be distinct")
    documentation = _doc(doc)
    admitted = module._add_top_name(name)
    value = Connector(_CREATE, module._owner, admitted, across, through, documentation)
    module._connectors.append(value)
    return value


def port(component, name, *, connector, on=None, doc=None):
    from . import _boundaries
    if isinstance(connector, _boundaries.FieldConnector):
        return _boundaries.port(component, name, connector=connector, on=on, doc=doc)
    if not isinstance(connector, Connector) or connector._owner is not component._owner:
        raise ModuleError("port connector must belong to this Module")
    if on is not None:
        raise ModuleError("a scalar connector port cannot have a boundary support")
    documentation = _doc(doc)
    admitted = component._add_name(name)
    value = Port(_CREATE, component._component_token, admitted, connector)
    component._ports.append((value, documentation))
    return value


def connect(component, *ports, over=None, periodic=False, doc=None):
    from ._boundaries import BoundaryMember, FieldPort
    component._source._ensure_open()
    if not 2 <= len(ports) <= _MAX_DECLARATIONS:
        raise ModuleError("a connection needs between 2 and 256 physical ports")
    for port in ports:
        if not isinstance(port, Port) or port._owner is not component._component_token:
            raise ModuleError("connection endpoints must be physical ports of this Component")
        if isinstance(port, FieldPort) and port._family is not None and port._selector is None:
            raise ModuleError("connection needs an exact selection of every port family")
        if isinstance(port, FieldPort) and isinstance(port._selector, BoundaryMember):
            if port._selector is not over:
                raise ModuleError("connection has a foreign or unbound boundary family member")
    if over is not None:
        if not isinstance(over, BoundaryMember):
            raise ModuleError("connection family requires a boundary member binder")
        component._support(over)
        if any(not isinstance(port, FieldPort) for port in ports):
            raise ModuleError("boundary connection families require field ports")
    if periodic:
        if component._kind != "model":
            raise ModuleError("a spatial-periodic connection belongs only to a Model")
        if len(ports) != 2 or over is not None:
            raise ModuleError("a spatial-periodic connection requires exactly two fixed endpoints")
        if any(not isinstance(port, FieldPort) for port in ports):
            raise ModuleError("a spatial-periodic connection requires field boundary ports")
    documentation = _doc(doc)
    if component._declaration_count >= _MAX_DECLARATIONS:
        raise ModuleError("Component exceeds the 256-declaration limit")
    component._declaration_count += 1
    component._connections.append((ports, over, periodic, documentation))


def instance_ports(component, target, name, bindings):
    """Expose authored local physical endpoints with their occurrence identity."""
    from ._boundaries import FieldPort
    result = {}
    for port, _ in target._ports:
        path = f"{name}.{port._name}"
        if isinstance(port, FieldPort):
            family = None
            support = bindings.get(port._support._name)
            if port._family is not None:
                family = port._family[0], bindings[port._family[1]._name]
            result[port._name] = FieldPort(_CREATE, component, path,
                                            port._connector, support, family)
        else:
            result[port._name] = Port(_CREATE, component._component_token,
                                      path, port._connector)
    return result
