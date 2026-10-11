"""Imported operator calls reuse the consumer-owned qualified call adapter."""

from . import Operator, _CREATE
from ._import_references import reference


class ImportedOperator(Operator):
    __slots__ = ("_reference",)

    def __init__(self, ref):
        inputs, result = ref._descriptor
        super().__init__(_CREATE, _name=ref._name,
                         _inputs=tuple(inputs), _result=result)
        object.__setattr__(self, "_owner", ref._owner)
        object.__setattr__(self, "_reference", ref)


def operator(imported, name):
    return ImportedOperator(reference(imported, name, "operator",
                                      lambda graph, local: graph.operator_descriptor(local)))
