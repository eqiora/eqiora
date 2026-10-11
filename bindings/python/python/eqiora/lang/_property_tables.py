"""Typed table references emit the ordinary exact package declaration."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TableSource:
    data: str
    axis: str
    axis_unit: object
    validity: tuple


def release(module, name, *, implements, data, axis_unit, source_unit,
            validity, citation, license, branch="single", doc=None):
    from . import (ModuleError, PropertyRelease, Unit, _CREATE, _doc, _expression,
                   _name_path)
    from ._imported_properties import admitted_contract
    from ._property_profiles import ReleaseProfile
    module._ensure_open()
    if module._components:
        raise ModuleError("property declarations must precede Components")
    if not admitted_contract(module, implements):
        raise ModuleError("table release requires the exact Module contract")
    inputs = implements._profile.inputs
    if (len(inputs) != 1 or any(kind.scalar_domain != "real" or kind.shape
                               for kind in (inputs[0][1], implements._value_type))):
        raise ModuleError("table requires one concrete real scalar axis and result")
    if implements._profile.derivatives not in ("value_only", "first_open_intervals"):
        raise ModuleError("table derivatives require first_open_intervals")
    if not isinstance(axis_unit, Unit) or not isinstance(source_unit, Unit):
        raise TypeError("table axis and result require coherent-SI Unit values")
    if branch != (implements._profile.branch or "single"):
        raise ModuleError("table branch must match its exact contract")
    if not isinstance(validity, (tuple, list)) or len(validity) != 2:
        raise TypeError("table validity requires its two closed interval endpoints")
    endpoints = tuple(_expression(value) for value in validity)
    if any(value._owner is not None or value._sources or value._binders for value in endpoints):
        raise ModuleError("table validity endpoints must be closed quantities")
    table = TableSource(_name_path(data, "exact table asset"), inputs[0][0], axis_unit, endpoints)
    citation = _name_path(citation, "exact citation asset")
    license = _name_path(license, "exact license asset")
    doc = _doc(doc)
    admitted = module._add_top_name(name)
    value = PropertyRelease(_CREATE, _owner=module._owner, _name_value=admitted,
                            _contract=implements, _source_unit=source_unit,
                            _source_scale=1, _profile=ReleaseProfile(None, branch, "reject"),
                            _citation=citation, _license=license, _doc=doc)
    object.__setattr__(value, "_table", table)
    module._releases.append(value)
    return value


def emit(graph, release, ordinal):
    from . import _AstModule
    table = release._table
    return _AstModule.with_table_release(
        graph,
        release._name, release._contract._name,
        (table.data, table.axis, table.axis_unit._ast, release._source_unit._ast,
         table.validity[0]._ast, table.validity[1]._ast),
        (release._citation, release._license, release._profile.branch), ordinal)
