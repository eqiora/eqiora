"""Bounded harmonic authoring; mathematical admission belongs to the compiler."""
from collections.abc import Sequence

from . import (ModuleError, Relation, ValueType, _AstFormulation, _Field,
              _MAX_DECLARATIONS, _MAX_EXPRESSION_NODES, _doc, _expression, _name)


def harmonic(component, name, relations, angular_frequency, convention, normalization,
             excitations, amplitudes, doc):
    component._source._ensure_open()
    if component._formulation is not None:
        raise ModuleError("a Component admits one named form")
    if component._test_restrictions:
        raise ModuleError("harmonic forms do not own weak test declarations")
    if convention != "negative_exponential" or normalization != "peak":
        raise ModuleError("harmonic forms require negative_exponential convention and peak normalization")
    for label, items in (("relations", relations), ("excitations", excitations), ("amplitudes", amplitudes)):
        if not isinstance(items, Sequence) or len(items) > _MAX_DECLARATIONS:
            raise ModuleError(f"harmonic {label} require a bounded sequence")
    if not relations or not amplitudes:
        raise ModuleError("harmonic forms require relations and amplitude mappings")
    if any(not isinstance(item, Relation) or item._component is not component._component_token
           for item in relations):
        raise ModuleError("form relations must belong to this Component")
    if len({item._name for item in relations}) != len(relations):
        raise ModuleError("form relations must be distinct")

    def expression(value):
        value = _expression(value)
        component._closed_expression(value)
        if value._owner is not None and value._owner is not component._component_token:
            raise ModuleError("form expressions must belong to this Component")
        return value

    frequency = expression(angular_frequency)
    nodes = frequency._nodes
    inputs = []
    for item in excitations:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("excitations must be (original input, amplitude expression) pairs")
        original, value = item
        if (not isinstance(original, _Field) or original._owner is not component._component_token
                or component._causal.get(original) != "input"):
            raise ModuleError("excitation must name an input from this Component")
        value = expression(value)
        inputs.append((original._name, value._ast))
        nodes += value._nodes
    if len({name for name, _ in inputs}) != len(inputs):
        raise ModuleError("harmonic excitations must be distinct")
    mappings = []
    for item in amplitudes:
        if not isinstance(item, tuple) or len(item) != 3:
            raise TypeError("amplitudes must be (name, original Field, ValueType) triples")
        amplitude_name, original, kind = item
        if (not isinstance(original, _Field) or original._owner is not component._component_token
                or original in component._requirements or original in component._causal):
            raise ModuleError("amplitude must map a body Field from this Component")
        if not isinstance(kind, ValueType):
            raise TypeError("amplitude type must be an eqiora.ValueType")
        support = next((on for field, on, *_ in component._fields if field is original), None)
        mappings.append((_name(amplitude_name), original._name, component._type_syntax(kind),
                         None if support is None else support._name))
    if len({item[0] for item in mappings}) != len(mappings) or len({item[1] for item in mappings}) != len(mappings):
        raise ModuleError("harmonic amplitude names and original Fields must be distinct")
    total = nodes + sum(left._nodes + right._nodes
                        for item in component._relations for _, left, right in item[2])
    total += sum(sum(term._nodes for term in item[2:5] if term is not None) for item in component._laws)
    if total > _MAX_EXPRESSION_NODES:
        raise ModuleError("Component relation and form expressions exceed the node limit")
    if component._declaration_count >= _MAX_DECLARATIONS:
        raise ModuleError("Component exceeds the declaration limit")
    documentation = _doc(doc)
    native = _AstFormulation.harmonic(_name(name), [item._name for item in relations],
                                     frequency._ast, inputs, mappings)
    component._formulation = (native, documentation, nodes)
    component._declaration_count += 1
