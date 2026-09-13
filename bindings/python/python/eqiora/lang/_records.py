"""Closed Module-owned record syntax and ordinary member expression handles."""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from . import Expression, ModuleError, ValueType, _Ast, _AstModule, _CREATE, _MISSING, _Field, _Parameter


class Record:
    """One ordered, closed record declaration belonging to an exact Module."""

    __slots__ = ("_source", "_name", "_syntax_name", "_members", "_syntax", "_doc")

    def __init__(self, token=_MISSING, *, source=None, name="", syntax_name=None,
                 members=(), syntax=(), doc=()):
        if token is not _CREATE:
            raise TypeError("records are created by Module.record()")
        for key, value in (("_source", source), ("_name", name),
                           ("_syntax_name", name if syntax_name is None else syntax_name),
                           ("_members", members),
                           ("_syntax", syntax), ("_doc", doc)):
            object.__setattr__(self, key, value)

    def __setattr__(self, name, value):
        raise AttributeError("Record handles are immutable")

    @property
    def name(self) -> str:
        return self._name

    @property
    def members(self) -> Mapping[str, ValueType]:
        return MappingProxyType(dict(self._members))

    def _type_syntax(self, source):
        if self._source is not source:
            raise ModuleError("record declaration must belong to this Module")
        return _AstModule.record_type(self._syntax_name)

    def __call__(self, /, **members: object) -> Expression:
        from . import _expression, _MAX_EXPRESSION_DEPTH, _MAX_EXPRESSION_NODES
        names = tuple(name for name, _ in self._members)
        if set(members) != set(names):
            raise ModuleError("record constructor requires exactly its declared named members")
        values = tuple(_expression(members[name]) for name in names)
        owner = None
        for value in values:
            if value._sources - {self._source._owner}:
                raise ModuleError("record members must belong to this Module")
            if owner is not None and value._owner is not None and owner is not value._owner:
                raise ModuleError("record members must belong to the same Component")
            if value._owner is not None:
                owner = value._owner
        if max((value._depth for value in values), default=0) + 1 > _MAX_EXPRESSION_DEPTH:
            raise ModuleError("record constructor exceeds the expression depth limit")
        if sum(value._nodes for value in values) + 1 > _MAX_EXPRESSION_NODES:
            raise ModuleError("record constructor exceeds the expression node limit")
        return Expression(_CREATE, _Ast.call(self._syntax_name, [value._ast for value in values], names), owner,
                          _binders=frozenset().union(*(value._binders for value in values)),
                          _sources=frozenset((self._source._owner,)))


class ImportedRecord(Record):
    """Read-only descriptor for one public Record in an explicit import."""

    __slots__ = ()

    @property
    def members(self) -> Mapping[str, str]:
        """Declaration-ordered member names mapped to exact qualified source types."""
        return MappingProxyType(dict(self._members))


def declare_record(source, name, *, members, doc=None) -> Record:
    from . import _name, _doc, _MAX_DECLARATIONS
    source._ensure_open()
    if not isinstance(members, Mapping):
        raise TypeError("record members must map names to ValueType")
    if not 1 <= len(members) <= _MAX_DECLARATIONS:
        raise ModuleError("record member count exceeds the declaration limit or is empty")
    admitted = _name(name)
    checked = []
    syntax = []
    for member, value_type in members.items():
        member = _name(member)
        if member == "_":
            raise ModuleError("record member cannot be a wildcard")
        if not isinstance(value_type, ValueType):
            raise TypeError("record members require ValueType; nested records are not supported")
        checked.append((member, value_type))
        syntax.append((member, source._type_syntax(value_type)))
    documentation = _doc(doc)
    source._add_top_name(admitted)
    record = Record(_CREATE, source=source, name=admitted, members=tuple(checked),
                    syntax=tuple(syntax), doc=documentation)
    source._records.append(record)
    return record


class RecordField(_Field):
    """A record Field whose members are ordinary Field expressions."""

    __slots__ = ("_record",)

    def __init__(self, component, name, record):
        super().__init__(component, name)
        object.__setattr__(self, "_record", record)

    def member(self, name: str) -> _Field:
        if name not in self._record.members:
            raise ModuleError("record has no such declared member")
        return _Field(self._component, f"{self._name}.{name}")


class RecordParameter(_Parameter):
    """A record Parameter whose members are ordinary Parameter expressions."""

    __slots__ = ("_record",)

    def __init__(self, component, name, record):
        super().__init__(component, name)
        object.__setattr__(self, "_record", record)

    def member(self, name: str) -> _Parameter:
        if name not in self._record.members:
            raise ModuleError("record has no such declared member")
        return _Parameter(self._component, f"{self._name}.{name}")
