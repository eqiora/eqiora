"""Descriptors retain their explicit imported Module and compiler-qualified name."""

from . import ModuleError, _CREATE, _name
from ._connections import Connector, Port
from ._boundaries import FieldConnector, FieldPort
from ._import_references import DeclarationRef


class ImportedConnector(Connector):
    __slots__ = ("_reference",)

    def __init__(self, reference, first, second):
        super().__init__(_CREATE, reference._owner, reference._name,
                         (first, None), (second, None), ())
        object.__setattr__(self, "_reference", reference)


class ImportedFieldConnector(FieldConnector):
    __slots__ = ("_reference",)

    def __init__(self, reference, first, second):
        super().__init__(_CREATE, reference._owner, reference._name,
                         (first, None), (second, None), False, ())
        object.__setattr__(self, "_reference", reference)


def connector(imported, name, *, internal=False):
    if not internal:
        name = _name(name)
    # Component interface metadata can refer through its own explicit imports.
    # These are already-loaded module edges, never ambient lookup or new loading.
    target = imported._target
    segments = name.split(".")
    for alias in segments[:-1]:
        try:
            target = target._imports[alias]
        except KeyError as error:
            raise ModuleError("imported port descriptor refers to an absent explicit import") from error
    descriptor = target._freeze().connector_descriptor(segments[-1], not internal)
    ref = DeclarationRef(_CREATE, imported, "connector", name, descriptor, origin=target)
    kind, first, second = ref._descriptor
    cls = ImportedFieldConnector if kind == "field" else ImportedConnector
    return cls(ref, first, second)


def instance_ports(component, target, name, bindings):
    result = {}
    for port_name, connector_name, support, family in target._ports:
        contract = connector(target._module, connector_name, internal=True)
        path = f"{name}.{port_name}"
        if support is None:
            result[port_name] = Port(_CREATE, component._component_token, path, contract)
        else:
            selected_support = bindings.get(support)
            bound_family = ((family[0], bindings[family[1]]) if family is not None else None)
            result[port_name] = FieldPort(_CREATE, component, path, contract,
                                          selected_support, bound_family)
    return result
