"""Model authoring, numerical plans, execution, and results.

Authority: ``bindings/python/python/eqiora/__init__.py``.
"""

from collections.abc import Generator, Iterator, Sequence
from fractions import Fraction
import os
from os import PathLike
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    NamedTuple,
    Never,
    Protocol,
    Self,
    TypeVar,
    final,
    overload,
)

import numpy as np
import numpy.typing as npt

from . import fluid as fluid
from . import formulation as formulation
from . import fem as fem
from . import fsi as fsi
from . import fvm as fvm
from . import geometry as geometry
from . import lang as lang
from .lang import Module as Module
from . import units as units
from . import meshing as meshing
from . import solid as solid
from . import solve as solve
from . import solve as solve_module
from . import time as time
from . import trajectory as trajectory
from .viewer import View as View

_Float64Array = npt.NDArray[np.float64]

__version__: str

class _DLPackProducer(Protocol):
    def __dlpack_device__(self) -> tuple[int, int]: ...
    def __dlpack__(
        self,
        *,
        stream: object | None = ...,
        max_version: tuple[int, int] | None = ...,
        dl_device: tuple[int, int] | None = ...,
        copy: bool | None = ...,
    ) -> object: ...

@final
class Diagnostic:
    """Immutable lossless projection of a current Rust diagnostic.

    Authority: ``crates/eqiora-python/src/error.rs::PyDiagnostic``.
    """

    @property
    def source(self) -> str: ...
    @property
    def code(self) -> str: ...
    @property
    def severity(self) -> str: ...
    @property
    def message(self) -> str: ...
    @property
    def graph_path(self) -> list[str] | None: ...
    @property
    def source_span(self) -> tuple[str, int, int] | None: ...
    @property
    def suggestion(self) -> str | None: ...

class EqioraError(Exception):
    """Base failure for an Eqiora operation rejected with diagnostics.

    Authority: ``crates/eqiora-python/src/error.rs::EqioraError``.
    """

    category: str
    diagnostics: tuple[Diagnostic, ...]

class ValidationError(EqioraError):
    """Failure caused by a model or request violating a typed contract.

    Authority: ``crates/eqiora-python/src/error.rs::ValidationError``.
    """

    ...

class CompatibilityError(EqioraError):
    """Failure caused by an incompatible versioned or persisted value.

    Authority: ``crates/eqiora-python/src/error.rs::CompatibilityError``.
    """

    ...

class CapabilityError(EqioraError):
    """Failure caused by an adapter lacking a required capability.

    Authority: ``crates/eqiora-python/src/error.rs::CapabilityError``.
    """

    ...

class ExecutionError(EqioraError):
    """Failure during execution.

    Authority: ``crates/eqiora-python/src/error.rs::ExecutionError``.
    """

    ...

class CancellationError(EqioraError):
    """Failure reporting cancellation of an Eqiora operation.

    Authority: ``crates/eqiora-python/src/error.rs::CancellationError``.
    """

    ...

class InternalError(EqioraError):
    """Internal failure that does not expose implementation details.

    Authority: ``crates/eqiora-python/src/error.rs::InternalError``.
    """

    ...

class PackageConformancePackage(NamedTuple):
    """Identity fields for one package in a conformance report.

    Authority: ``bindings/python/python/eqiora/__init__.py::check_package_conformance``.
    """

    name: str
    version: str
    semantic_digest: str
    source_digest: str

class PackageConformanceReport(NamedTuple):
    """Structural-conformance report for one locked package closure.

    Authority: ``bindings/python/python/eqiora/__init__.py::check_package_conformance``.
    """

    profile: str
    eqiora_version: str
    compiler: str
    compiler_version: str
    semantic_canonicalization_version: int
    source_bundle_version: int
    resolution_version: int
    root_package: PackageConformancePackage
    packages: tuple[PackageConformancePackage, ...]
    entry_model: str
    resolution_digest: str
    package_compilation_digest: str
    model_id: str
    model_revision: int
    model_digest: str
    deterministic_replay_agreement: bool

@final
class PropertyBinding:
    """Inspect an exact constant, analytic or table release bound to a Model occurrence.

    Authority: ``crates/eqiora-python/src/model/property.rs::PyPropertyBinding``.
    """

    @property
    def composition(self) -> str | None: ...
    @property
    def contract(self) -> str: ...
    @property
    def release(self) -> str: ...
    @property
    def component(self) -> str: ...
    @property
    def requirement(self) -> str: ...
    @property
    def normalized_value(self) -> _TypedValue | None: ...
    @property
    def inputs(self) -> list[str]: ...
    @property
    def branch(self) -> str | None: ...
    @property
    def first_partials(self) -> bool: ...
    @property
    def derivatives(self) -> str: ...
    @property
    def value_type(self) -> ValueType: ...
    @property
    def validity(self) -> str: ...
    @property
    def citation(self) -> str: ...
    @property
    def license(self) -> str: ...

@final
class AuthoredFormulation:
    """Fresh-compile authored Formulation inspection.

    Authority: ``crates/eqiora-python/src/model/authored_formulation.rs``.
    """

    @property
    def implication(self) -> Literal["strong-implies-weak", "strong-implies-interval-conservation"]: ...
    @property
    def assumptions(self) -> list[str]: ...
    @property
    def name(self) -> str: ...
    @property
    def test_restrictions(self) -> list[tuple[str, str, list[str]]]: ...
    @property
    def interval(self) -> tuple[str, str, str] | None: ...
    @property
    def kind(self) -> str: ...
    @property
    def source_identity(self) -> str: ...
    @property
    def relation_ids(self) -> list[str]: ...
    @property
    def domain_id(self) -> str: ...
    @property
    def trial_field_ids(self) -> list[str]: ...
    @property
    def filename(self) -> str: ...
    @property
    def source_range(self) -> tuple[int, int]: ...

@final
class Dimension:
    """SI base-dimension exponents in M, L, T, I, Θ, N, J order.

    Authority: ``crates/eqiora-python/src/modeling.rs::PyDimension``.
    """

    def __new__(
        cls,
        *,
        mass: int | Fraction = 0,
        length: int | Fraction = 0,
        time: int | Fraction = 0,
        current: int | Fraction = 0,
        temperature: int | Fraction = 0,
        amount: int | Fraction = 0,
        luminous_intensity: int | Fraction = 0,
    ) -> Self: ...
    @property
    def exponents(self) -> tuple[Fraction, Fraction, Fraction, Fraction, Fraction, Fraction, Fraction]: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __ne__(self, other: object, /) -> bool: ...

@final
class Enum:
    """An exact closed enum declaration; replay may have no retained lexical name.

    Authority: ``crates/eqiora-python/src/modeling/enumeration.rs::PyEnum``.
    """
    def __new__(cls, name: str, *, members: Sequence[str]) -> Self: ...
    @property
    def name(self) -> str | None: ...
    @property
    def id(self) -> str: ...
    @property
    def members(self) -> tuple[str, ...]: ...
    @property
    def value_type(self) -> ValueType: ...
    def member(self, name: str) -> EnumValue: ...

@final
class EnumValue:
    """An immutable nominal enum member without numeric or Boolean coercion.

    Authority: ``crates/eqiora-python/src/modeling/enumeration.rs::PyEnumValue``.
    """
    @property
    def value_type(self) -> ValueType: ...
    @property
    def enum_id(self) -> str: ...
    def __bool__(self) -> bool: ...

@final
class FiniteSpace:
    """Exact nominal ordered basis, distinct from a numerical discretization space.

    Authority: ``crates/eqiora-python/src/modeling/nominal.rs::PyFiniteSpace``.
    """
    def __new__(cls, name: str, *, labels: list[str] | tuple[str, ...]) -> Self: ...
    @property
    def name(self) -> str: ...
    @property
    def id(self) -> str: ...
    @property
    def labels(self) -> tuple[str, ...]: ...

@final
class IndexSet:
    """An exact nominal zero-based set with a positive constant integer extent.

    Authority: ``crates/eqiora-python/src/modeling/nominal.rs::PyIndexSet``.
    """
    def __new__(cls, name: str, *, extent: int) -> Self: ...
    @property
    def name(self) -> str: ...
    @property
    def id(self) -> str: ...
    @property
    def extent(self) -> int: ...

@final
class ValueType:
    """Exact scalar domain, dimension, channel axes and spatial frame.

    Authority: ``crates/eqiora-python/src/modeling/value_type.rs::PyValueType``.
    """

    def to_eqi(self) -> str: ...

    def render(
        self, profile: Literal["latex", "mathml", "unicode", "plain", "speech"] = "latex",
    ) -> MathRendering: ...

    @staticmethod
    def integer() -> ValueType: ...
    @staticmethod
    def boolean() -> ValueType: ...
    @staticmethod
    def real(dimension: Dimension | None = None) -> ValueType: ...
    @staticmethod
    def complex(dimension: Dimension | None = None) -> ValueType: ...
    @staticmethod
    def coordinates(space: FiniteSpace) -> ValueType: ...
    @staticmethod
    def counts(space: FiniteSpace) -> ValueType: ...
    @staticmethod
    def index(set: IndexSet) -> ValueType: ...
    @staticmethod
    def vector(scalar: ValueType, extent: int) -> ValueType: ...
    @staticmethod
    def tensor(scalar: ValueType, *extents: int) -> ValueType: ...
    @staticmethod
    def array(element: ValueType, extent: int) -> ValueType: ...
    @property
    def scalar_domain(self) -> Literal["real", "complex", "integer", "boolean", "enum"]: ...
    @property
    def dimension(self) -> Dimension: ...
    @property
    def shape(self) -> list[int]: ...
    @property
    def array_rank(self) -> int: ...
    @property
    def frame(self) -> Literal["invariant", "spatial_cartesian"]: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class BoundarySide:
    """Closed orientation of one Cartesian boundary domain.

    Authority: ``crates/eqiora-python/src/modeling.rs::PyBoundarySide``.
    """

    Lower: ClassVar[BoundarySide]
    Upper: ClassVar[BoundarySide]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class Domain:
    """Immutable draft-local Cartesian volume or oriented boundary.

    Authority: ``crates/eqiora-python/src/modeling.rs::PyDomain``.
    """

    @staticmethod
    def box(name: str, *bounds: tuple[float, float]) -> Domain: ...
    def boundary(
        self,
        name: str,
        *,
        axis: int,
        side: BoundarySide,
    ) -> Domain: ...
    @property
    def name(self) -> str: ...
    @property
    def bounds(self) -> list[tuple[float, float]] | None: ...
    @property
    def parent(self) -> Domain | None: ...
    @property
    def axis(self) -> int | None: ...
    @property
    def side(self) -> BoundarySide | None: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class FieldRole:
    """Author-declared evolution role independent of spatial support.

    Authority: ``crates/eqiora-python/src/modeling.rs::PyFieldRole``.
    """

    Variable: ClassVar[FieldRole]
    State: ClassVar[FieldRole]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class Initial:
    """Simultaneous fresh-initialization equations with explicit sides.

    Authority: ``crates/eqiora-python/src/modeling.rs::PyInitial``.
    """
    def __new__(cls, *equations: tuple[_ExpressionLike, _ExpressionLike]) -> Self: ...
    @property
    def equations(self) -> list[tuple[Expression, Expression]]: ...

@final
class Expression:
    """Immutable symbolic expression whose shape and support Rust infers.

    Authority: ``crates/eqiora-python/src/modeling.rs::PyExpression``.
    """

    def __getitem__(self, index: int | slice, /) -> Expression: ...
    def __neg__(self) -> Expression: ...
    def __add__(self, right: _ExpressionLike, /) -> Expression: ...
    def __radd__(self, left: _ExpressionLike, /) -> Expression: ...
    def __sub__(self, right: _ExpressionLike, /) -> Expression: ...
    def __rsub__(self, left: _ExpressionLike, /) -> Expression: ...
    def __mul__(self, right: _ExpressionLike, /) -> Expression: ...
    def __rmul__(self, left: _ExpressionLike, /) -> Expression: ...
    def __truediv__(self, right: _ExpressionLike, /) -> Expression: ...
    def __rtruediv__(self, left: _ExpressionLike, /) -> Expression: ...
    def __bool__(self) -> bool: ...

@final
class Field:
    """Immutable typed field declaration.

    Authority: ``crates/eqiora-python/src/modeling.rs::PyField``.
    """

    def __new__(
        cls,
        name: str,
        *,
        domain: Domain | None = None,
        role: FieldRole,
        value_type: ValueType | None = None,
    ) -> Self: ...
    @property
    def name(self) -> str: ...
    @property
    def dimension(self) -> Dimension: ...
    @property
    def role(self) -> FieldRole: ...
    @property
    def value_type(self) -> ValueType: ...
    @property
    def domain(self) -> Domain | None: ...
    def __getitem__(self, index: int | slice, /) -> Expression: ...
    def __neg__(self) -> Expression: ...
    def __add__(self, right: _ExpressionLike, /) -> Expression: ...
    def __radd__(self, left: _ExpressionLike, /) -> Expression: ...
    def __sub__(self, right: _ExpressionLike, /) -> Expression: ...
    def __rsub__(self, left: _ExpressionLike, /) -> Expression: ...
    def __mul__(self, right: _ExpressionLike, /) -> Expression: ...
    def __rmul__(self, left: _ExpressionLike, /) -> Expression: ...
    def __truediv__(self, right: _ExpressionLike, /) -> Expression: ...
    def __rtruediv__(self, left: _ExpressionLike, /) -> Expression: ...
    def __bool__(self) -> bool: ...

@final
class Parameter:
    """Immutable complete typed parameter declaration.

    Nonzero spatial values require an explicit frame Domain registered in the
    native Model. This is a uniform coefficient, not a distributed Field.

    Authority: ``crates/eqiora-python/src/modeling.rs::PyParameter``.
    """

    def __new__(
        cls,
        name: str,
        *,
        value: _TypedValue,
        value_type: ValueType | None = None,
        frame: Domain | None = None,
    ) -> Self: ...
    @property
    def name(self) -> str: ...
    @property
    def dimension(self) -> Dimension: ...
    @property
    def value(self) -> _TypedValue: ...
    @property
    def value_type(self) -> ValueType: ...
    def __getitem__(self, index: int | slice, /) -> Expression: ...
    def __neg__(self) -> Expression: ...
    def __add__(self, right: _ExpressionLike, /) -> Expression: ...
    def __radd__(self, left: _ExpressionLike, /) -> Expression: ...
    def __sub__(self, right: _ExpressionLike, /) -> Expression: ...
    def __rsub__(self, left: _ExpressionLike, /) -> Expression: ...
    def __mul__(self, right: _ExpressionLike, /) -> Expression: ...
    def __rmul__(self, left: _ExpressionLike, /) -> Expression: ...
    def __truediv__(self, right: _ExpressionLike, /) -> Expression: ...
    def __rtruediv__(self, left: _ExpressionLike, /) -> Expression: ...
    def __bool__(self) -> bool: ...

@final
class Observable:
    """Typed derived output, separate from solve unknowns.

    Authority: ``crates/eqiora-python/src/modeling/observable.rs::PyObservable``.
    """
    def __new__(cls, name: str, *, value_type: ValueType, expression: _ExpressionLike) -> Self: ...
    @property
    def name(self) -> str: ...
    @property
    def value_type(self) -> ValueType: ...
    @property
    def expression(self) -> Expression: ...

def integral(value: _ExpressionLike, measure: _ExpressionLike) -> Expression:
    """Construct an Observable integral with an explicit Domain measure.

    Authority: ``crates/eqiora-python/src/modeling/observable.rs::integral``.
    """
    ...
def measure(domain: Domain) -> Expression:
    """Select the exact Domain measure for an Observable integral.

    Authority: ``crates/eqiora-python/src/modeling/observable.rs::measure``.
    """
    ...

@final
class PhysicalDomain:
    """Immutable nominal scalar physical domain with explicitly named quantities.

    Authority: ``crates/eqiora-python/src/modeling.rs::PyPhysicalDomain``.
    """

    def __new__(
        cls,
        name: str,
        *,
        across_name: str,
        across_type: ValueType,
        through_name: str,
        through_type: ValueType,
    ) -> Self: ...
    @property
    def name(self) -> str: ...
    @property
    def across_name(self) -> str: ...
    @property
    def through_name(self) -> str: ...
    @property
    def across_type(self) -> ValueType: ...
    @property
    def through_type(self) -> ValueType: ...

@final
class ConservingPort:
    """Immutable scalar conserving-port declaration.

    Authority: ``crates/eqiora-python/src/modeling.rs::PyConservingPort``.
    """

    def __new__(cls, name: str, *, domain: PhysicalDomain) -> Self: ...
    @property
    def name(self) -> str: ...
    @property
    def domain(self) -> PhysicalDomain: ...

@final
class Connection:
    """Immutable anonymous conserving connection declaration.

    Authority: ``crates/eqiora-python/src/modeling.rs::PyConnection``.
    """

    ...

_ExpressionLike = Expression | Field | Parameter | bool | int | float | complex

@final
class Relation:
    """Immutable continuous equations with explicit typed sides.

    Authority: ``crates/eqiora-python/src/modeling.rs::PyRelation``.
    """
    def __new__(cls, name: str, *, equations: Sequence[tuple[_ExpressionLike, _ExpressionLike]],
                domain: Domain | None = None) -> Self: ...
    @property
    def name(self) -> str: ...
    @property
    def equations(self) -> list[tuple[Expression, Expression]]: ...
    @property
    def domain(self) -> Domain | None: ...

@final
class Array:
    """Immutable dense one-dimensional CPU ``float64`` result buffer.

    Authority: ``crates/eqiora-python/src/array.rs::PyArrayBuffer``.
    """

    def __getitem__(self, index: int, /) -> float: ...
    def numpy(self, *, copy: bool | None = None) -> _Float64Array: ...
    def __array__(
        self, dtype: object | None = None, copy: bool | None = None
    ) -> _Float64Array: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...
    def __dlpack__(
        self,
        *,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> object: ...
    @property
    def device(self) -> str: ...
    @property
    def device_id(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    @property
    def byte_order(self) -> str: ...
    @property
    def shape(self) -> tuple[int]: ...
    @property
    def strides(self) -> tuple[int]: ...
    @property
    def c_contiguous(self) -> bool: ...
    @property
    def aligned(self) -> bool: ...
    @property
    def readonly(self) -> bool: ...
    @property
    def ownership(self) -> str: ...
    @property
    def origin_copy_occurred(self) -> bool: ...
    def __len__(self) -> int: ...

@final
class Revision:
    """Identity of an immutable model artifact.

    Authority: ``crates/eqiora-python/src/model.rs::PyRevision``.
    """

    @property
    def model_id(self) -> str: ...
    @property
    def digest(self) -> str: ...
    @property
    def number(self) -> int: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class StructuralSemanticFingerprint:
    """Alpha-normalized comparison evidence, not exact model identity.

    Authority: ``crates/eqiora-python/src/model.rs::PyStructuralSemanticFingerprint``.
    """

    @property
    def generation(self) -> str: ...
    @property
    def digest(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class ValueEdit:
    """Immutable exact-base value edit prepared by the Rust facade.

    Authority: ``crates/eqiora-python/src/model.rs::PyValueEdit``.
    """

    @property
    def key(self) -> str: ...
    @property
    def base_digest(self) -> str: ...
    @property
    def base_revision(self) -> int: ...
    @property
    def target_id(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class ParameterRef:
    """Parameter selected from an immutable model.

    Authority: ``crates/eqiora-python/src/model.rs::PyModelParameterRef``.
    """

    @property
    def value(self) -> _TypedValue: ...
    @property
    def value_type(self) -> ValueType: ...
    @property
    def model_digest(self) -> str: ...
    @property
    def id(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class FieldRef:
    """Field selected from an immutable model.

    Authority: ``crates/eqiora-python/src/model.rs::PyModelFieldRef``.
    """

    @property
    def model_digest(self) -> str: ...
    @property
    def id(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class DomainRef:
    """Domain selected from an immutable Model.

    Authority: ``crates/eqiora-python/src/model.rs::PyModelDomainRef``.
    """
    @property
    def model_digest(self) -> str: ...
    @property
    def id(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class InitialField:
    """Immutable exact-Field-bound coherent-SI initial coefficients.

    Authority: ``crates/eqiora-python/src/trajectory.rs::PyInitialField``.
    """
    def __new__(
        cls,
        field: FieldRef,
        /,
        *,
        vertex_values: object | None = None,
        cell_values: object | None = None,
    ) -> InitialField: ...
    @property
    def field(self) -> FieldRef: ...

@final
class FieldOutput:
    """Immutable coefficients for one exact Model Field on one exact Mesh.

    Authority: ``crates/eqiora-python/src/result/field_output.rs::PyFieldOutput``.
    """

    @property
    def field(self) -> FieldRef: ...
    @property
    def mesh(self) -> meshing.Mesh: ...
    @property
    def dimension(self) -> tuple[Fraction, Fraction, Fraction, Fraction, Fraction, Fraction, Fraction]: ...
    @property
    def value_shape(self) -> tuple[int, ...]: ...
    @property
    def space(self) -> str: ...
    @property
    def associations(self) -> tuple[str, ...]: ...
    def values(self, association: str, /) -> Array: ...
    def coefficient_count(self, association: str, /) -> int: ...
    def logical_shape(self, association: str, /) -> tuple[int, ...]: ...

@final
class ClockDomain:
    """An immutable standalone nominal clock with exact rational seconds.

    Authority: ``crates/eqiora-python/src/clock.rs::PyClockDomain``.
    """
    def __new__(cls, *, period_s: Fraction | int, phase_s: Fraction | int = 0) -> Self: ...
    @property
    def id(self) -> str: ...
    @property
    def period_s(self) -> Fraction: ...
    @property
    def phase_s(self) -> Fraction: ...

@final
class ExecutionCheckpoint:
    """An immutable in-memory reference checkpoint bound to its complete Model.

    Authority: ``crates/eqiora-python/src/execution_session.rs::PyExecutionCheckpoint``.
    """
    ...

@final
class ExecutionSession:
    """Reference execution through fully stabilized boundaries with explicit input tables.

    Input Port, Clock and output selectors accept source aliases or exact Model-owned ULIDs.

    Authority: ``crates/eqiora-python/src/execution_session.rs::PyExecutionSession``.
    """
    def advance(self) -> bool:
        """Advance to the next stabilized boundary; false means completion.

        Authority: ``crates/eqiora-python/src/execution_session.rs::PyExecutionSession``.
        """
        ...
    @property
    def progress(self) -> dict[str, float | int]:
        """Fresh model_time, end_time, accepted_steps, maximum_steps observation.

        Authority: ``crates/eqiora-python/src/execution_session.rs::PyExecutionSession``.
        """
        ...
    @property
    def activation_sequence(self) -> tuple[tuple[str, ...], ...]:
        """Exact activation ULIDs per microstep at the last stabilized boundary.

        Authority: ``crates/eqiora-python/src/execution_session.rs::PyExecutionSession``.
        """
        ...
    def advance_ticks(self, count: int) -> int: ...
    @property
    def next_tick(self) -> Fraction | None: ...
    def checkpoint(self) -> ExecutionCheckpoint: ...
    def across(self, name: str) -> float | None:
        """Read an accepted coherent-SI across value by Port alias or exact ULID.

        Authority: ``crates/eqiora-python/src/execution_session/physical.rs::read``.
        """
        ...
    def through(self, name: str) -> float | None:
        """Read an accepted coherent-SI through value by Port alias or exact ULID.

        Authority: ``crates/eqiora-python/src/execution_session/physical.rs::read``.
        """
        ...
    def field(self, name: str) -> _TypedValue | None:
        """Read a Field by source alias or exact Model-owned ULID.

        Reopened Model artifacts retain ULIDs, not source lookup aliases.

        Authority: ``crates/eqiora-python/src/execution_session.rs::PyExecutionSession``.
        """
        ...
    def output(self, name: str, tick_index: int) -> tuple[Fraction, _TypedValue] | None: ...

@final
class QuantityLabel:
    """Immutable label for one exact occurrence in a full Model rendering scope.

    Authority: ``crates/eqiora-python/src/model/notation.rs::PyQuantityLabel``.
    """
    def __repr__(self) -> str: ...
    @property
    def identity(self) -> str: ...
    @property
    def scope(self) -> str: ...
    @property
    def occurrence(self) -> str: ...
    @property
    def role(self) -> str: ...
    @property
    def family_member(self) -> str | None: ...
    @property
    def selector(self) -> str: ...
    @property
    def graph_id(self) -> str | None: ...
    @property
    def definition_span(self) -> tuple[str, int, int] | None: ...
    @property
    def instance_span(self) -> tuple[str, int, int] | None: ...
    @property
    def label(self) -> str: ...

@final
class MathReference:
    """Exact semantic targets shared by every mathematical presentation profile.

    Authority: ``crates/eqiora-python/src/model/rendering.rs::PyMathReference``.
    """

    def __repr__(self) -> str: ...
    @property
    def graph_id(self) -> str | None: ...
    @property
    def role(self) -> str | None: ...
    @property
    def declarations(self) -> tuple[str, ...]: ...
    @property
    def operator(self) -> str | None: ...

@final
class MathRendering:
    """Immutable equation or type presentation with accessible text and exact references.

    If ``used_fallback`` is true, ``text`` is plain text rather than rich markup.

    Authority: ``crates/eqiora-python/src/model/rendering.rs::PyMathRendering``.
    """

    def __repr__(self) -> str: ...
    @property
    def profile(self) -> str: ...
    @property
    def text(self) -> str: ...
    @property
    def plain(self) -> str: ...
    @property
    def speech(self) -> str: ...
    @property
    def used_fallback(self) -> bool: ...
    @property
    def references(self) -> tuple[MathReference, ...]: ...

@final
class Model:
    """Immutable model artifact with all semantic references resolved.

    Authority: ``crates/eqiora-python/src/model.rs::PyModel``.
    """

    @staticmethod
    def from_bytes(data: bytes) -> Model: ...
    @staticmethod
    def read(path: str | PathLike[str]) -> Model: ...
    def to_bytes(self) -> bytes: ...
    def write(self, path: str | PathLike[str]) -> None: ...
    def preview_value_edit(self, target: str, value: _TypedValue) -> ValueEdit: ...
    def commit(self, edit: ValueEdit) -> Model: ...
    def execution_session(
        self, *, end_time_s: float, max_step_s: float,
        inputs: dict[str, tuple[str, list[_TypedValue] | tuple[_TypedValue, ...]]],
    ) -> ExecutionSession: ...
    def resume_execution(self, checkpoint: ExecutionCheckpoint) -> ExecutionSession: ...
    def enum(self, selection: str) -> Enum:
        """Inspect an exact enum by source alias or retained canonical ULID.

        Authority: ``crates/eqiora-python/src/model.rs::PyModel``.
        """
        ...
    def parameter(self, selection: str) -> ParameterRef: ...
    def field(self, selection: str) -> FieldRef: ...
    def activation(self, selection: str) -> ActivationRef: ...
    def observable(self, selection: str) -> ObservableRef: ...
    def constraint(self, selection: str, ordinal: int) -> ConstraintRef: ...
    def domain(self, selection: str) -> DomainRef: ...
    def notation_labels(
        self, profile: Literal["latex", "mathml", "unicode", "plain", "speech"] = "latex", *,
        identities: Sequence[str] | None = None,
    ) -> tuple[QuantityLabel, ...]: ...
    def render_equations(
        self, relation: str,
        profile: Literal["latex", "mathml", "unicode", "plain", "speech"] = "latex",
    ) -> tuple[MathRendering, ...]: ...
    def render_formulations(
        self, profile: Literal["latex", "mathml", "unicode", "plain", "speech"] = "latex",
    ) -> tuple[MathRendering, ...]: ...
    def structurally_equivalent(self, other: Model) -> bool: ...
    @property
    def digest(self) -> str: ...
    @property
    def package_compilation_digest(self) -> str | None: ...
    @property
    def property_bindings(self) -> tuple[PropertyBinding, ...]: ...
    @property
    def authored_formulations(self) -> tuple[AuthoredFormulation, ...]: ...
    @property
    def structural_fingerprint(self) -> StructuralSemanticFingerprint: ...
    @property
    def revision(self) -> Revision: ...
    @property
    def model_id(self) -> str: ...
    @property
    def field_ids(self) -> list[str]: ...
    @property
    def parameter_ids(self) -> list[str]: ...
    @property
    def domain_ids(self) -> list[str]: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class ScalarPlanView:
    """Scalar-valued Fields resolved from one Model.

    Authority: ``crates/eqiora-python/src/common_plan/capability_view.rs::PyScalarPlanView``.
    """
    @property
    def kind(self) -> str: ...
    @property
    def fields(self) -> tuple[FieldRef, ...]: ...
    @property
    def coefficient_sampling(self) -> str: ...
    @property
    def face_coefficient_policy(self) -> str: ...

@final
class ResolvedExecution:
    """Exact scalar, layout, schedule, provider, and placement selected for execution.

    Authority: ``crates/eqiora-python/src/common_plan/resolved_execution.rs::PyResolvedExecution``.
    """
    @property
    def scalar_type(self) -> str: ...
    @property
    def vector_layout(self) -> str: ...
    @property
    def schedule(self) -> str: ...
    @property
    def provider(self) -> str: ...
    @property
    def provider_version(self) -> str: ...
    @property
    def placement(self) -> str: ...
    @property
    def workers(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class FormulationKind:
    """Closed mathematical Formulation families accepted by exact override.

    Authority: ``crates/eqiora-python/src/common_plan/capability_view.rs::PyFormulationKind``.
    """

    PrimalGalerkin: ClassVar[FormulationKind]
    MixedGalerkin: ClassVar[FormulationKind]
    IntegralConservative: ClassVar[FormulationKind]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class FormulationSelectionMode:
    """Whether a Formulation was selected automatically or supplied explicitly.

    Authority: ``crates/eqiora-python/src/common_plan/capability_view.rs::PyFormulationSelectionMode``.
    """

    Automatic: ClassVar[FormulationSelectionMode]
    Exact: ClassVar[FormulationSelectionMode]
    Authored: ClassVar[FormulationSelectionMode]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class FormulationView:
    """Effective mathematical form selected between Model and Realization.

    Authority: ``crates/eqiora-python/src/common_plan/capability_view.rs::PyFormulationView``.
    """
    @property
    def requested(self) -> FormulationSelectionMode: ...
    @property
    def effective(self) -> FormulationKind: ...
    @property
    def boundary_treatment(self) -> str: ...
    @property
    def rule_ids(self) -> list[str]: ...
    @property
    def selection_reason_codes(self) -> list[str]: ...
    @property
    def requested_source_identity(self) -> str | None: ...
    def __repr__(self) -> str: ...

@final
class Plan:
    """Immutable numerical Plan containing a Model and its resources.

    The Model determines the physics. ``capability`` exposes
    capability-specific field roles and policies through one closed typed view;
    ``fields`` remains the capability-neutral exact FieldRef inventory.

    Authority: ``crates/eqiora-python/src/common_plan.rs::PyPlan``.
    """
    @property
    def enforcement(self) -> solve.ActiveSet | None: ...
    @staticmethod
    def from_bytes(data: bytes) -> Plan: ...
    @staticmethod
    def read(path: str | PathLike[str]) -> Plan: ...
    def to_bytes(self) -> bytes: ...
    def write(self, path: str | PathLike[str]) -> None: ...
    @property
    def identity(self) -> str: ...
    @property
    def model_id(self) -> str: ...
    @property
    def model_digest(self) -> str: ...
    @property
    def model_revision(self) -> int: ...
    @property
    def package_compilation_digest(self) -> str | None: ...
    @property
    def geometry_digest(self) -> str | None: ...
    @property
    def mesh_digest(self) -> str | None: ...
    @property
    def correspondence_digest(self) -> str | None: ...
    @property
    def production_digest(self) -> str | None: ...
    @property
    def realization_digest(self) -> str | None: ...
    @property
    def model(self) -> Model: ...
    @property
    def mesh(self) -> meshing.Mesh | None: ...
    @property
    def formulation(self) -> FormulationView | None: ...
    @property
    def capability(self) -> ScalarPlanView | solve.AlgebraicPlanView | time.OdePlanView | solid.ElasticityPlanView | fluid.IncompressibleFlowPlanView | fsi.FixedReferenceFsiPlanView: ...
    @property
    def fields(self) -> tuple[FieldRef, ...]: ...
    @property
    def spatial(self) -> fem.Q1 | fem.MiniP1 | fvm.CellCenteredTpfa | fvm.CellCentered | tuple[fem.ScopedSpatialPolicy, ...] | None: ...
    @property
    def solve(self) -> solve_module.ResolvedLinear | solve_module.ResolvedNewton | None: ...
    @property
    def requested_solve(self) -> solve_module.Linear | solve_module.Newton | None: ...
    @property
    def temporal(self) -> time.BackwardEuler | time.Tsitouras45 | None: ...
    @property
    def execution(self) -> ResolvedExecution: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class State:
    """Physical state associated with one Plan.

    Authority: ``crates/eqiora-python/src/trajectory.rs::PyState``.
    """

    def to_bytes(self) -> bytes: ...
    @staticmethod
    def from_bytes(plan: Plan, data: bytes) -> State: ...
    @staticmethod
    def initial(
        plan: Plan,
        /,
        *,
        fields: tuple[InitialField, ...] | None = None,
        time_s: float | None = None,
    ) -> State: ...
    @staticmethod
    def zero(plan: Plan, /, *, time_s: float = 0.0) -> State: ...
    @staticmethod
    def from_result(plan: Plan, result: Result, /, *, time_s: float) -> State: ...
    @property
    def digest(self) -> str: ...
    @property
    def step(self) -> int: ...
    @property
    def time_s(self) -> float: ...
    @property
    def fields(self) -> tuple[trajectory.FieldSnapshot, ...]: ...
    @property
    def field_refs(self) -> tuple[FieldRef, ...]: ...
    @property
    def state_space_identity(self) -> str: ...
    @property
    def mesh(self) -> meshing.Mesh | None: ...
    @property
    def model(self) -> Model | None: ...
    @property
    def source_plan_identity(self) -> str | None: ...
    @property
    def source_request_identity(self) -> str | None: ...
    @property
    def source_trajectory_identity(self) -> str | None: ...
    @property
    def source_kind(self) -> str | None: ...
    def field(self, field: FieldRef, /) -> trajectory.FieldSnapshot: ...
    def curl(self, field: FieldRef, /) -> trajectory.DerivedFieldSnapshot: ...
    def sample(
        self,
        field: FieldRef,
        /,
        *,
        at: tuple[float, float],
    ) -> trajectory.FieldSample: ...
    def boundary_force(
        self,
        selection: geometry.GeometrySelection,
        /,
    ) -> trajectory.BoundaryForce: ...
    def value(self, field: FieldRef, /) -> float: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class ConvergenceReason:
    """Accepted reason for linear-solve convergence.

    Authority: ``crates/eqiora-python/src/realization.rs::PyConvergenceReason``.
    """

    InitialResidualSatisfied: ClassVar[ConvergenceReason]
    ResidualToleranceSatisfied: ClassVar[ConvergenceReason]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class LinearSolveSummary:
    """Linear-solve convergence report.

    Authority: ``crates/eqiora-python/src/realization.rs::PyLinearSolveSummary``.
    """

    @property
    def backend(self) -> str: ...
    @property
    def adapter(self) -> str: ...
    @property
    def verification_adapter(self) -> str: ...
    @property
    def orientation(self) -> str: ...
    @property
    def algorithm(self) -> str: ...
    @property
    def preconditioner(self) -> str: ...
    @property
    def reduction(self) -> str: ...
    @property
    def relative_tolerance(self) -> float: ...
    @property
    def absolute_tolerance(self) -> float: ...
    @property
    def maximum_iterations(self) -> int: ...
    @property
    def reason(self) -> ConvergenceReason: ...
    @property
    def completed_iterations(self) -> int: ...
    @property
    def initial_residual_norm(self) -> float: ...
    @property
    def reported_residual_norm(self) -> float: ...
    @property
    def true_residual_norm(self) -> float: ...
    @property
    def residual_target(self) -> float: ...

@final
class DifferentiationMode:
    """Primal, JVP, or VJP occurrence kind.

    Authority: ``crates/eqiora-python/src/differentiation.rs::PyDifferentiationMode``.
    """

    Primal: ClassVar[DifferentiationMode]
    Jvp: ClassVar[DifferentiationMode]
    Vjp: ClassVar[DifferentiationMode]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class DerivativeImplementation:
    """Source of the derivative action used by an occurrence.

    Authority: ``crates/eqiora-python/src/differentiation.rs::PyDerivativeImplementation``.
    """

    AnalyticAssembled: ClassVar[DerivativeImplementation]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class LinearizationState:
    """Whether an accepted linearization was established or reused.

    Authority: ``crates/eqiora-python/src/differentiation.rs::PyLinearizationState``.
    """

    Established: ClassVar[LinearizationState]
    Reused: ClassVar[LinearizationState]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class DifferentiationEvidence:
    """Typed in-memory provenance for one differentiation occurrence.

    Authority: ``crates/eqiora-python/src/differentiation.rs::PyDifferentiationEvidence``.
    """

    @property
    def model_digest(self) -> str: ...
    @property
    def plan_identity(self) -> str: ...
    @property
    def input_ids(self) -> list[str]: ...
    @property
    def output_id(self) -> str: ...
    @property
    def mode(self) -> DifferentiationMode: ...
    @property
    def implementation(self) -> DerivativeImplementation: ...
    @property
    def linearization_state(self) -> LinearizationState: ...
    @property
    def state_system_fingerprint(self) -> str: ...
    @property
    def primal_residual_norm(self) -> float: ...
    @property
    def residual_tolerance(self) -> float: ...
    @property
    def primal_solve(self) -> LinearSolveSummary: ...
    @property
    def derivative_solve(self) -> LinearSolveSummary | None: ...

@final
class DifferentiablePrimal:
    """Accepted complete primary field from a primal evaluation.

    Authority: ``crates/eqiora-python/src/differentiation.rs::PyDifferentiablePrimal``.
    """

    @property
    def output(self) -> Array: ...
    @property
    def evidence(self) -> DifferentiationEvidence: ...

@final
class DifferentiableJvp:
    """Accepted primary field and its forward tangent.

    Authority: ``crates/eqiora-python/src/differentiation.rs::PyDifferentiableJvp``.
    """

    @property
    def output(self) -> Array: ...
    @property
    def tangent(self) -> Array: ...
    @property
    def evidence(self) -> DifferentiationEvidence: ...

@final
class DifferentiableVjp:
    """Accepted primary field and its reverse input cotangent.

    Authority: ``crates/eqiora-python/src/differentiation.rs::PyDifferentiableVjp``.
    """

    @property
    def output(self) -> Array: ...
    @property
    def input_cotangent(self) -> Array: ...
    @property
    def evidence(self) -> DifferentiationEvidence: ...

@final
class DifferentiableEvaluation:
    """Immutable accepted evaluation at one numerical parameter point.

    Authority: ``crates/eqiora-python/src/differentiation.rs::PyDifferentiableEvaluation``.
    """

    @property
    def point(self) -> Array: ...
    def primal(self) -> DifferentiablePrimal: ...
    def jvp(
        self, tangent: Array | _Float64Array | _DLPackProducer
    ) -> DifferentiableJvp: ...
    def vjp(
        self, cotangent: Array | _Float64Array | _DLPackProducer
    ) -> DifferentiableVjp: ...

@final
class DifferentiableProgram:
    """Immutable program over one fixed input-coordinate set.

    Authority: ``crates/eqiora-python/src/differentiation.rs::PyDifferentiableProgram``.
    """

    @property
    def model_digest(self) -> str: ...
    @property
    def plan_identity(self) -> str: ...
    @property
    def input_ids(self) -> list[str]: ...
    @property
    def output_id(self) -> str: ...
    @property
    def input_shape(self) -> tuple[int]: ...
    @property
    def output_shape(self) -> tuple[int]: ...
    @property
    def dtype(self) -> str: ...
    @property
    def device(self) -> str: ...
    @property
    def derivative_contract(self) -> str: ...
    def evaluate(
        self, parameters: Array | _Float64Array | _DLPackProducer
    ) -> DifferentiableEvaluation: ...
    def map(
        self,
        mapped: Array | _Float64Array | _DLPackProducer,
        *,
        shared_inputs: Sequence[ParameterRef] | None = None,
        shared: Array | _Float64Array | _DLPackProducer | None = None,
        retained_bytes_limit: int = 67_108_864,
    ) -> EvaluationMapPlan: ...
    def primal(self) -> DifferentiablePrimal: ...
    def jvp(
        self, tangent: Array | _Float64Array | _DLPackProducer
    ) -> DifferentiableJvp: ...
    def vjp(
        self, cotangent: Array | _Float64Array | _DLPackProducer
    ) -> DifferentiableVjp: ...

@final
class EvaluationMapCancellation:
    """Cooperative cancellation between ordered occurrences.

    Authority: ``crates/eqiora-python/src/differentiation/batch.rs::PyEvaluationMapCancellation``.
    """

    def __init__(self) -> None: ...
    def cancel(self) -> None: ...
    @property
    def requested(self) -> bool: ...

@final
class EvaluationMapPlan:
    """Frozen complete points; planning validates metadata without solving.

    Authority: ``crates/eqiora-python/src/differentiation/batch.rs::PyEvaluationMapPlan``.
    """

    def __len__(self) -> int: ...
    def __getitem__(self, index: int, /) -> _Float64Array: ...
    @property
    def program(self) -> DifferentiableProgram: ...
    @property
    def point_shape(self) -> tuple[int, ...]: ...
    @property
    def input_shape(self) -> tuple[int, ...]: ...
    @property
    def output_shape(self) -> tuple[int, ...]: ...
    @property
    def shared_input_ids(self) -> list[str]: ...
    @property
    def mapped_input_ids(self) -> list[str]: ...
    @property
    def estimated_storage_bytes(self) -> int: ...
    @property
    def points(self) -> _Float64Array: ...
    def occurrence_coordinates(self, index: int) -> tuple[int, ...]: ...
    def execute(
        self, *, cancellation: EvaluationMapCancellation | None = None
    ) -> CompleteEvaluationMap | EvaluationMapTerminalReport: ...

@final
class CompleteEvaluationMap:
    """Batch retaining every accepted occurrence and linearization.

    Authority: ``crates/eqiora-python/src/differentiation/batch/results.rs::PyCompleteEvaluationMap``.
    """

    def __len__(self) -> int: ...
    def __getitem__(self, index: int, /) -> DifferentiableEvaluation: ...
    @property
    def plan(self) -> EvaluationMapPlan: ...
    @property
    def statuses(self) -> list[str]: ...
    def member(self, index: int) -> DifferentiableEvaluation: ...
    def primal(self) -> _Float64Array: ...
    def jvp(
        self, mapped: Array | _Float64Array | _DLPackProducer, *,
        shared: Array | _Float64Array | _DLPackProducer | None = None,
        seed_shape: Sequence[int] | None = None,
        point_axes: Sequence[int] | None = None,
        numerical_bytes_limit: int = 67_108_864,
    ) -> EvaluationMapJvp: ...
    def vjp(
        self, cotangents: Array | _Float64Array | _DLPackProducer, *,
        seed_shape: Sequence[int] | None = None,
        point_axes: Sequence[int] | None = None,
        numerical_bytes_limit: int = 67_108_864,
    ) -> EvaluationMapVjp: ...

@final
class EvaluationMapTerminalReport:
    """Inspectable failed or cancelled prefix, never a complete primal or product.

    Authority: ``crates/eqiora-python/src/differentiation/batch/results.rs::PyEvaluationMapTerminalReport``.
    """

    def __len__(self) -> int: ...
    def __getitem__(self, index: int, /) -> DifferentiableEvaluation | None: ...
    @property
    def plan(self) -> EvaluationMapPlan: ...
    @property
    def stopped_index(self) -> int: ...
    @property
    def cancelled(self) -> bool: ...
    @property
    def diagnostics(self) -> list[Diagnostic]: ...
    @property
    def statuses(self) -> list[str]: ...
    def member(self, index: int) -> DifferentiableEvaluation | None: ...

@final
class EvaluationMapJvp:
    """Mapped JVPs in explicit point/seed order, with per-member reports.

    Authority: ``crates/eqiora-python/src/differentiation/batch/products.rs::PyEvaluationMapJvp``.
    """

    @property
    def plan(self) -> EvaluationMapPlan: ...
    @property
    def seed_shape(self) -> tuple[int, ...]: ...
    @property
    def point_axes(self) -> tuple[int, ...]: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def output(self) -> _Float64Array: ...
    @property
    def tangent(self) -> _Float64Array: ...
    def member(self, index: int) -> DifferentiableJvp: ...

@final
class EvaluationMapVjp:
    """Mapped VJPs; globally shared cotangents sum over point axes.

    Authority: ``crates/eqiora-python/src/differentiation/batch/products.rs::PyEvaluationMapVjp``.
    """

    @property
    def plan(self) -> EvaluationMapPlan: ...
    @property
    def seed_shape(self) -> tuple[int, ...]: ...
    @property
    def point_axes(self) -> tuple[int, ...]: ...
    @property
    def shared_shape(self) -> tuple[int, ...]: ...
    @property
    def mapped_shape(self) -> tuple[int, ...]: ...
    @property
    def shared_input_ids(self) -> list[str]: ...
    @property
    def mapped_input_ids(self) -> list[str]: ...
    @property
    def shared_cotangents(self) -> _Float64Array: ...
    @property
    def mapped_cotangents(self) -> _Float64Array: ...
    def member(self, index: int) -> DifferentiableVjp: ...

@final
class Series:
    """Read-only field-local sampled series in SI units.

    Authority: ``crates/eqiora-python/src/result.rs::PySeries``.
    """

    @property
    def field(self) -> FieldRef | None: ...
    @property
    def id(self) -> str: ...
    @property
    def name(self) -> str | None: ...
    @property
    def dimension(self) -> tuple[Fraction, Fraction, Fraction, Fraction, Fraction, Fraction, Fraction]: ...
    @property
    def time(self) -> Array: ...
    @property
    def values(self) -> Array: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[tuple[float, float]]: ...

@final
class ConstraintRef:
    """One exact Model-owned mathematical inequality or complementarity condition.

    Authority: ``crates/eqiora-python/src/model/constraint.rs::PyConstraintRef``.
    """
    @property
    def model_digest(self) -> str: ...
    @property
    def relation_id(self) -> str: ...
    @property
    def ordinal(self) -> int: ...
    @property
    def kind(self) -> Literal["inequality", "complementarity"]: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class ConstraintMeasurement:
    """Independently evaluated original operands, their units, and accepted activity.

    Authority: ``crates/eqiora-python/src/result/constraints.rs::PyConstraintMeasurement``.
    """
    @property
    def reference(self) -> ConstraintRef: ...
    @property
    def activity(self) -> Literal["active", "inactive", "biactive", "inequality"]: ...
    @property
    def left_value(self) -> float: ...
    @property
    def right_value(self) -> float: ...
    @property
    def left_dimension(self) -> Dimension: ...
    @property
    def right_dimension(self) -> Dimension: ...
    @property
    def left_tolerance(self) -> float: ...
    @property
    def right_tolerance(self) -> float | None: ...

@final
class ActivationRef:
    """Exact event guard Activation selected from one immutable Model.

    Authority: ``crates/eqiora-python/src/model/activation_ref.rs::PyActivationRef``.
    """
    @property
    def model_digest(self) -> str: ...
    @property
    def id(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class ObservableRef:
    """Exact derived output selected from one immutable Model.

    Authority: ``crates/eqiora-python/src/model/observable_ref.rs::PyObservableRef``.
    """
    @property
    def model_digest(self) -> str: ...
    @property
    def id(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class Observation:
    """Typed Result-owned evaluation with explicit effective quadrature.

    ``quadrature_points`` counts Gauss–Legendre points per axis; a point
    boundary uses exactly one point. Values are expressed in SI units.

    Authority: ``crates/eqiora-python/src/result/observe.rs::PyObservation``.
    """
    @property
    def value(self) -> _TypedValue: ...
    @property
    def value_type(self) -> ValueType: ...
    @property
    def result_identity(self) -> str: ...
    @property
    def observable_id(self) -> str: ...
    @property
    def evaluation_kind(self) -> Literal["value", "state-jvp"]: ...
    @property
    def quadrature(self) -> Literal["Point", "GaussLegendre"] | None: ...
    @property
    def quadrature_points(self) -> int | None: ...
    @property
    def quadrature_dimension(self) -> int | None: ...

@final
class TrajectoryObservation:
    """Typed terminal or time-integrated value from exact accepted history.

    Values use coherent SI units. Time integration multiplies the declared
    Observable dimension by seconds; it does not integrate output samples.

    Authority: ``crates/eqiora-python/src/result/time_observe.rs::PyTrajectoryObservation``.
    """
    @property
    def interval_s(self) -> tuple[float, float]: ...
    @property
    def endpoint_convention(self) -> str: ...
    @property
    def value(self) -> _TypedValue: ...
    @property
    def value_type(self) -> ValueType: ...
    @property
    def result_identity(self) -> str: ...
    @property
    def trajectory_identity(self) -> str: ...
    @property
    def observable_id(self) -> str: ...
    @property
    def evaluation_kind(self) -> Literal["terminal", "time-integral", "terminal-parameter-jvp", "time-integral-parameter-jvp"]: ...
    @property
    def quadrature(self) -> time.TimeFunctionalQuadrature | None: ...
    @property
    def parameter_jvp(self) -> bool: ...

@final
class ObservableStateTangent:
    """Dimensioned coefficient variation bound to one exact accepted Result.

    Omitted fields have zero variation. Coefficients follow the accepted field's
    canonical vertex order, with units supplied explicitly by ``Dimension``.

    Authority: ``crates/eqiora-python/src/result/observe.rs::PyObservableStateTangent``.
    """
    @property
    def result_identity(self) -> str: ...

@final
class ProfilePhase:
    """Aggregate timing for one hierarchical execution phase.

    Authority: ``crates/eqiora-python/src/profile.rs::PyProfilePhase``.
    """
    @property
    def path(self) -> list[str]: ...
    @property
    def name(self) -> str: ...
    @property
    def count(self) -> int: ...
    @property
    def total_seconds(self) -> float: ...

@final
class ProfileEvent:
    """Structured metadata for one phase or solver observation.

    Authority: ``crates/eqiora-python/src/profile.rs::PyProfileEvent``.
    """
    @property
    def path(self) -> list[str]: ...
    @property
    def fields(self) -> dict[str, str]: ...

@final
class Profile:
    """Process-local timings and events for one profiled Run.

    Authority: ``crates/eqiora-python/src/profile.rs::PyProfile``.
    """
    @property
    def phases(self) -> list[ProfilePhase]: ...
    @property
    def events(self) -> list[ProfileEvent]: ...
    @property
    def total_seconds(self) -> float: ...
    def summary(self) -> str: ...

@final
class Result:
    """Accepted execution occurrence with typed output relationships.

    Authority: ``crates/eqiora-python/src/result.rs::PyRunResult``.
    """

    @property
    def constraints(self) -> tuple[ConstraintMeasurement, ...]: ...
    @property
    def model_id(self) -> str: ...
    @property
    def model_digest(self) -> str: ...
    @property
    def model_revision(self) -> int: ...
    @property
    def plan_key(self) -> str: ...
    @property
    def adapter(self) -> str: ...
    @property
    def adapter_version(self) -> str: ...
    @property
    def elapsed_seconds(self) -> float: ...
    @property
    def profile(self) -> Profile | None: ...
    def to_bytes(self) -> bytes: ...
    @staticmethod
    def from_bytes(plan: Plan, data: bytes) -> Result: ...
    @staticmethod
    def read(plan: Plan, path: str | PathLike[str]) -> Result: ...
    def write(self, path: str | PathLike[str]) -> None: ...
    @property
    def fields(self) -> list[Series]: ...
    @property
    def solve(self) -> LinearSolveSummary: ...
    def observe_terminal(self, observable: ObservableRef) -> TrajectoryObservation: ...
    def observe_time_integral(self, observable: ObservableRef, *, quadrature: time.TimeFunctionalQuadrature) -> TrajectoryObservation: ...
    def observe_terminal_parameter_jvp(self, observable: ObservableRef, direction: dict[ParameterRef, tuple[Dimension, float]]) -> TrajectoryObservation: ...
    def observe_time_integral_parameter_jvp(self, observable: ObservableRef, direction: dict[ParameterRef, tuple[Dimension, float]], *, quadrature: time.TimeFunctionalQuadrature) -> TrajectoryObservation: ...
    def observe(self, observable: ObservableRef, *, quadrature_points: int | None = None) -> Observation: ...
    def observable_state_tangent(self, directions: dict[FieldRef, tuple[Dimension, Sequence[float]]]) -> ObservableStateTangent: ...
    def observe_state_jvp(self, observable: ObservableRef, tangent: ObservableStateTangent, *, quadrature_points: int) -> Observation: ...
    def output(self, field: FieldRef, /) -> FieldOutput: ...
    def boundary_force(
        self, selection: geometry.GeometrySelection, /
    ) -> trajectory.BoundaryForce: ...
    def boundary_flux(
        self, selection: geometry.GeometrySelection, /
    ) -> trajectory.BoundaryFlux: ...
    def series(self, field: FieldRef, /) -> Series: ...
    def mesh(self, field: FieldRef, /) -> meshing.Mesh: ...
    @property
    def trajectory(self) -> trajectory.Trajectory: ...

@final
class RunStatus:
    """Monotone state of one execution occurrence.

    Authority: ``crates/eqiora-python/src/execution/evidence.rs::PyRunStatus``.
    """

    Created: ClassVar[RunStatus]
    Validating: ClassVar[RunStatus]
    Queued: ClassVar[RunStatus]
    Running: ClassVar[RunStatus]
    Cancelling: ClassVar[RunStatus]
    Cancelled: ClassVar[RunStatus]
    Completed: ClassVar[RunStatus]
    Failed: ClassVar[RunStatus]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class TransientRunProgress:
    """Last fully accepted common transient step boundary.

    Authority: ``crates/eqiora-python/src/execution/evidence.rs::PyCommonTransientRunProgress``.
    """

    @property
    def accepted_steps(self) -> int: ...
    @property
    def maximum_steps(self) -> int: ...
    @property
    def model_time_s(self) -> float: ...

@final
class TransientRunCancellation:
    """Exact accepted common transient boundary where cancellation terminated.

    Authority: ``crates/eqiora-python/src/execution/evidence.rs::PyCommonTransientRunCancellation``.
    """

    @property
    def progress(self) -> TransientRunProgress: ...
    @property
    def request_identity(self) -> str: ...
    @property
    def state(self) -> State: ...

_RunResultT = TypeVar("_RunResultT", bound=Result)

class Run(Generic[_RunResultT]):
    """Awaitable handle for one execution occurrence.

    Authority: ``bindings/python/python/eqiora/__init__.py::Run``.
    """

    def __init__(self, native: Never) -> None: ...
    @property
    def status(self) -> RunStatus: ...
    @property
    def history(self) -> tuple[RunStatus, ...]: ...
    @property
    def progress(self) -> TransientRunProgress | None: ...
    @property
    def cancellation(
        self,
    ) -> TransientRunCancellation | None: ...
    @property
    def done(self) -> bool: ...
    @property
    def model_id(self) -> str: ...
    @property
    def model_digest(self) -> str: ...
    @property
    def model_revision(self) -> int: ...
    @property
    def package_compilation_digest(self) -> str | None: ...
    @property
    def plan_key(self) -> str: ...
    @property
    def adapter(self) -> str: ...
    @property
    def adapter_version(self) -> str: ...
    def cancel(self) -> bool: ...
    def result(self) -> _RunResultT: ...
    def __await__(self) -> Generator[Any, None, _RunResultT]: ...

_TypedValue = EnumValue | int | float | complex | list["_TypedValue"] | tuple["_TypedValue", ...]

_ModelDeclaration = (
    Enum
    | FiniteSpace
    | IndexSet
    | Domain
    | Initial
    | Field
    | Parameter
    | Observable
    | PhysicalDomain
    | ConservingPort
    | Relation
    | Connection
)

def across(port: ConservingPort) -> Expression:
    """Return the across variable of a scalar conserving port.

    Authority: ``crates/eqiora-python/src/modeling.rs::across``.
    """

    ...

def compile(
    *,
    path: str | PathLike[str] | None = None,
    source: str | Module | None = None,
    filename: str | None = None,
    geometry: geometry.Geometry | None = None,
    bindings: dict[str, _TypedValue | ClockDomain | geometry.GeometrySelection | tuple[geometry.GeometrySelection, geometry.GeometrySelection] | tuple[tuple[geometry.GeometrySelection, ...], geometry.GeometrySelection]] | None = None,
    entry: str | None = None,
) -> Model:
    """Compile one source and its optional exact Geometry closure.

    Authority: ``bindings/python/python/eqiora/__init__.py::compile``.
    """

    ...

def compile_package(
    store_root: str | PathLike[str],
    resolution: bytes,
    *,
    entry: str,
    geometry: geometry.Geometry | None = None,
    bindings: dict[str, _TypedValue | ClockDomain | geometry.GeometrySelection | tuple[geometry.GeometrySelection, geometry.GeometrySelection] | tuple[tuple[geometry.GeometrySelection, ...], geometry.GeometrySelection]] | None = None,
) -> Model:
    """Compile one locked Model or one Component using the supplied Geometry.

    Authority: ``crates/eqiora-python/src/package.rs::compile_package``.
    """

    ...

@final
class ProjectUpdate:
    """A validated selection that can be inspected and committed once.

    Authority: ``crates/eqiora-python/src/package/update.rs::PyProjectUpdate``.
    """

    @property
    def resolution(self) -> bytes: ...
    @property
    def lock(self) -> bytes: ...
    @property
    def explanation(self) -> str: ...
    def __repr__(self) -> str: ...
    def commit(self, store_root: str | PathLike[str]) -> bytes:
        """Publish the frozen selection; a consumed proposal needs a fresh preview."""
        ...

def preview_local_project(project_root: str | PathLike[str]) -> ProjectUpdate:
    """Preview the validated selection without installing or publishing it.

    Authority: ``crates/eqiora-python/src/package/update.rs::preview_local_project``.
    """
    ...

def resolve_local_project(
    project_root: str | PathLike[str],
    store_root: str | PathLike[str],
) -> bytes:
    """Resolve a local package project, write ``eqiora.lock``, and populate a store.

    Authority: ``crates/eqiora-python/src/package.rs::resolve_local_project``.
    """

    ...

def add_local_dependency(
    project_root: str | PathLike[str],
    store_root: str | PathLike[str],
    name: str,
    *,
    version: str,
    path: str,
) -> bytes:
    """Add or replace a dependency request and publish its exact selection.

    Authority: ``crates/eqiora-python/src/package.rs::add_local_dependency``.
    """
    ...

def remove_local_dependency(
    project_root: str | PathLike[str],
    store_root: str | PathLike[str],
    name: str,
) -> bytes:
    """Remove a direct dependency and publish the validated manifest and lock.

    Authority: ``crates/eqiora-python/src/package.rs::remove_local_dependency``.
    """
    ...

def add_bundled_dependency(
    project_root: str | PathLike[str],
    store_root: str | PathLike[str],
    name: str,
    *,
    version: str,
) -> bytes:
    """Add one exact bundled package through the shared manifest/lock transaction.

    Authority: ``crates/eqiora-python/src/package.rs::add_bundled_dependency``.
    """
    ...

def add_git_dependency(project_root: str | PathLike[str], store_root: str | PathLike[str], name: str, *, version: str, repository: str, revision: str) -> bytes:
    """Add an immutable Git package to the project.

    Authority: ``crates/eqiora-python/src/package.rs::add_git_dependency``.
    """
    ...

def fetch_project(project_root: str | PathLike[str], store_root: str | PathLike[str]) -> bytes:
    """Materialize the accepted lock from explicit sources without updating it.

    Authority: ``crates/eqiora-python/src/package.rs::fetch_project``.
    """
    ...

def open_project(project_root: str | PathLike[str], store_root: str | PathLike[str]) -> bytes:
    """Validate the current root and exact closure using only the supplied offline store.

    Authority: ``crates/eqiora-python/src/package.rs::open_project``.
    """
    ...

def update_project(project_root: str | PathLike[str], store_root: str | PathLike[str]) -> bytes:
    """Re-derive the exact lock from current explicit sources and requests.

    Authority: ``crates/eqiora-python/src/package.rs::update_project``.
    """
    ...

def vendor_project(project_root: str | PathLike[str], store_root: str | PathLike[str], destination: str | PathLike[str]) -> bytes:
    """Copy the validated accepted closure to an explicit offline store.

    Authority: ``crates/eqiora-python/src/package.rs::vendor_project``.
    """
    ...

def check_package_conformance(
    store_root: str | os.PathLike[str],
    resolution_bytes: bytes,
    *,
    entry_model: str,
    profile: str,
) -> PackageConformanceReport:
    """Check one exact locked package closure by deterministic replay.

    Authority: ``crates/eqiora-python/src/package.rs::_check_package_conformance``.
    """

    ...

def connect(*ports: ConservingPort) -> Connection:
    """Build an anonymous conserving connection declaration.

    Authority: ``crates/eqiora-python/src/modeling.rs::connect``.
    """

    ...

def derivative(field: Field) -> Expression:
    """Return the time derivative of a field.

    Authority: ``crates/eqiora-python/src/modeling.rs::derivative``.
    """

    ...

def div(value: _ExpressionLike) -> Expression:
    """Return the spatial divergence of a symbolic expression.

    Authority: ``crates/eqiora-python/src/modeling.rs::div``.
    """

    ...

def grad(value: _ExpressionLike) -> Expression:
    """Return the spatial gradient of a symbolic expression.

    Authority: ``crates/eqiora-python/src/modeling.rs::grad``.
    """

    ...

def resolve(
    model: Model,
    *,
    mesh: meshing.Mesh | None = None,
    spatial: fem.Q1 | fem.MiniP1 | fvm.CellCenteredTpfa | fvm.CellCentered | tuple[fem.ScopedSpatialPolicy, ...] | None = None,
    formulation: FormulationKind | None = None,
    solve: solve.Linear | solve.Newton | None = None,
    scaling: fluid.IncompressibleScaling | None = None,
    temporal: time.BackwardEuler | time.Tsitouras45 | None = None,
    enforcement: solve.ActiveSet | None = None,
) -> Plan:
    """Resolve an exact Model and typed numerical policies into a common Plan.

    Typed spatial and solve policies select numerics, never physics. The
    resolved Plan retains the exact caller Model and every applicable caller
    resource. Spatial paths retain their exact Mesh without regeneration;
    structural no-Mesh ODE paths reject spatial resources.

    Authority: ``bindings/python/python/eqiora/__init__.py::resolve``.
    """

    ...

def run(
    plan: Plan,
    *,
    state: State | None = None,
    until_s: float | None = None,
    output_times_s: tuple[float, ...] | None = None,
    steps: int | None = None,
    output_steps: tuple[int, ...] | None = None,
    profile: bool = False,
) -> Result:
    """Execute a steady Plan or a transient Plan with a specified time interval synchronously.

    Authority: ``bindings/python/python/eqiora/__init__.py::run``.
    """

    ...

def submit(
    plan: Plan,
    *,
    state: State | None = None,
    until_s: float | None = None,
    output_times_s: tuple[float, ...] | None = None,
    steps: int | None = None,
    output_steps: tuple[int, ...] | None = None,
    profile: bool = False,
) -> Run[Result]:
    """Submit a steady Plan or a transient Plan with a specified time interval.

    Authority: ``bindings/python/python/eqiora/__init__.py::submit``.
    """

    ...

def through(port: ConservingPort) -> Expression:
    """Return the through variable of a scalar conserving port.

    Authority: ``crates/eqiora-python/src/modeling.rs::through``.
    """

    ...

def trace(value: _ExpressionLike) -> Expression:
    """Return the boundary trace of a symbolic expression.

    Authority: ``crates/eqiora-python/src/modeling.rs::trace``.
    """

    ...

from . import diff as diff

__all__ = [
    "Module",
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "logical_not",
    "logical_and",
    "logical_or",

    "__version__",
    "Array",
    "AuthoredFormulation",
    "BoundarySide",
    "CancellationError",
    "CapabilityError",
    "CompatibilityError",
    "Connection",
    "ClockDomain",
    "ExecutionSession",
    "ExecutionCheckpoint",
    "ConservingPort",
    "ConvergenceReason",
    "DerivativeImplementation",
    "Diagnostic",
    "DifferentiableEvaluation",
    "DifferentiableJvp",
    "DifferentiablePrimal",
    "DifferentiableProgram",
    "DifferentiableVjp",
    "CompleteEvaluationMap",
    "EvaluationMapPlan",
    "EvaluationMapTerminalReport",
    "EvaluationMapCancellation",
    "EvaluationMapJvp",
    "EvaluationMapVjp",
    "DifferentiationEvidence",
    "DifferentiationMode",
    "Dimension",
    "ValueType",
    "Enum",
    "EnumValue",
    "FiniteSpace",
    "IndexSet",
    "DomainRef",
    "Domain",
    "EqioraError",
    "ExecutionError",
    "Expression",
    "Field",
    "FieldOutput",
    "FieldRef",
    "FormulationView",
    "FormulationKind",
    "FormulationSelectionMode",
    "InitialField",
    "InternalError",
    "LinearSolveSummary",
    "LinearizationState",
    "MathReference",
    "MathRendering",
    "Model",
    "PackageConformancePackage",
    "PackageConformanceReport",
    "Parameter",
    "Observable",
    "ObservableRef",
    "ConstraintRef",
    "ConstraintMeasurement",
    "ActivationRef",
    "Observation",
    "TrajectoryObservation",
    "ObservableStateTangent",
    "integral",
    "measure",
    "ParameterRef",
    "PhysicalDomain",
    "PropertyBinding",
    "QuantityLabel",
    "Plan",
    "FieldRole",
    "Initial",
    "Relation",
    "Result",
    "Profile",
    "ProfileEvent",
    "ProfilePhase",
    "Revision",
    "ResolvedExecution",
    "ScalarPlanView",
    "Run",
    "RunStatus",
    "Series",
    "State",
    "TransientRunCancellation",
    "TransientRunProgress",
    "StructuralSemanticFingerprint",
    "ValidationError",
    "ValueEdit",
    "View",
    "across",
    "check_package_conformance",
    "compile",
    "compile_package",
    "connect",
    "derivative",
    "div",
    "grad",
    "lang",
    "units",
    "resolve",
    "ProjectUpdate",
    "preview_local_project",
    "resolve_local_project",
    "add_local_dependency",
    "remove_local_dependency",
    "run",
    "submit",
    "through",
    "trace",
    "diff",
    "fem",
    "fluid",
    "formulation",
    "fsi",
    "fvm",
    "geometry",
    "meshing",
    "solid",
    "solve",
    "time",
    "trajectory",
    "add_bundled_dependency",
    "add_git_dependency",
    "fetch_project",
    "open_project",
    "update_project",
    "vendor_project",
]


def equal(left: _ExpressionLike, right: _ExpressionLike) -> Expression:
    """Author a typed equal predicate.

    Authority: ``crates/eqiora-python/src/modeling/predicates.rs::equal``.
    """
    ...


def not_equal(left: _ExpressionLike, right: _ExpressionLike) -> Expression:
    """Author a typed not equal predicate.

    Authority: ``crates/eqiora-python/src/modeling/predicates.rs::not_equal``.
    """
    ...


def less(left: _ExpressionLike, right: _ExpressionLike) -> Expression:
    """Author a typed less predicate.

    Authority: ``crates/eqiora-python/src/modeling/predicates.rs::less``.
    """
    ...


def less_equal(left: _ExpressionLike, right: _ExpressionLike) -> Expression:
    """Author a typed less equal predicate.

    Authority: ``crates/eqiora-python/src/modeling/predicates.rs::less_equal``.
    """
    ...


def greater(left: _ExpressionLike, right: _ExpressionLike) -> Expression:
    """Author a typed greater predicate.

    Authority: ``crates/eqiora-python/src/modeling/predicates.rs::greater``.
    """
    ...


def greater_equal(left: _ExpressionLike, right: _ExpressionLike) -> Expression:
    """Author a typed greater equal predicate.

    Authority: ``crates/eqiora-python/src/modeling/predicates.rs::greater_equal``.
    """
    ...


def logical_not(value: _ExpressionLike) -> Expression:
    """Author a typed logical not predicate.

    Authority: ``crates/eqiora-python/src/modeling/predicates.rs::logical_not``.
    """
    ...


def logical_and(left: _ExpressionLike, right: _ExpressionLike) -> Expression:
    """Author a typed logical and predicate.

    Authority: ``crates/eqiora-python/src/modeling/predicates.rs::logical_and``.
    """
    ...


def logical_or(left: _ExpressionLike, right: _ExpressionLike) -> Expression:
    """Author a typed logical or predicate.

    Authority: ``crates/eqiora-python/src/modeling/predicates.rs::logical_or``.
    """
    ...
