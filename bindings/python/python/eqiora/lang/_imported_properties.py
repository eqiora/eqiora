"""Property references retain exact explicit-import identity and qualified names."""
from . import ModuleError, PropertyContract, PropertyRelease, ValueType, _CREATE, _name
from ._import_references import DeclarationRef, reference, same_declaration
from ._property_profiles import ContractProfile


class ImportedPropertyContract(PropertyContract):
    __slots__ = ("_reference",)

    def __init__(self, ref):
        result, inputs, derivatives, branch = ref._descriptor
        if not isinstance(result, ValueType) or any(not isinstance(kind, ValueType) for _, kind in inputs):
            raise ModuleError("imported property descriptor requires fully resolved ValueType metadata")
        # Read-only metadata never creates a local declaration.
        super().__init__(_CREATE, ref._owner, ref._name, result,
                         ContractProfile(tuple(inputs),
                                         derivatives,
                                         branch), ())
        object.__setattr__(self, "_reference", ref)


class ImportedPropertyRelease(PropertyRelease):
    __slots__ = ("_reference",)

    def __init__(self, ref, required_contract):
        # Imported releases are immutable references, never copied release bodies.
        for name in PropertyRelease.__slots__:
            object.__setattr__(self, name, None)
        object.__setattr__(self, "_owner", ref._owner)
        object.__setattr__(self, "_name", ref._name)
        object.__setattr__(self, "_contract", required_contract)
        object.__setattr__(self, "_reference", ref)


def contract(imported, name, *, internal=False):
    if not internal:
        name = _name(name)
    target = imported._target
    segments = name.split(".")
    for alias in segments[:-1]:
        try:
            target = target._imports[alias]
        except KeyError as error:
            raise ModuleError("property descriptor requires an existing explicit import") from error
    descriptor = target._freeze().property_contract_descriptor((target._package, target._name), target._units(), target._dependencies(), segments[-1])
    ref = DeclarationRef(_CREATE, imported, "property_contract", name, descriptor, origin=target)
    return ImportedPropertyContract(ref)


def release(imported, name):
    ref = reference(imported, name, "property_release",
                    lambda graph, local: graph.property_release_descriptor(local))
    return ImportedPropertyRelease(ref, contract(imported, ref._descriptor, internal=True))


def same_contract(left, right):
    return same_declaration(left, right)


def admitted_contract(module, value):
    return (isinstance(value, PropertyContract) and value._owner is module._owner
            and (value in module._contracts or isinstance(value, ImportedPropertyContract)))
