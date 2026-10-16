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
from . import data as data
from . import formulation as formulation
from . import fem as fem
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
    """Immutable compiler or runtime diagnostic, carrying every field of the underlying Rust diagnostic.

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
    """Base class for Eqiora errors; `diagnostics` explains what was rejected.

    Authority: ``crates/eqiora-python/src/error.rs::EqioraError``.
    """

    category: str
    diagnostics: tuple[Diagnostic, ...]

class ValidationError(EqioraError):
    """Raised when a model or request is invalid.

    Authority: ``crates/eqiora-python/src/error.rs::ValidationError``.
    """

    ...

class CompatibilityError(EqioraError):
    """Raised when a versioned or saved value is incompatible with this release.

    Authority: ``crates/eqiora-python/src/error.rs::CompatibilityError``.
    """

    ...

class CapabilityError(EqioraError):
    """Raised when an adapter lacks a required capability.

    Authority: ``crates/eqiora-python/src/error.rs::CapabilityError``.
    """

    ...

class ExecutionError(EqioraError):
    """Failure during execution.

    Authority: ``crates/eqiora-python/src/error.rs::ExecutionError``.
    """

    ...

class CancellationError(EqioraError):
    """Raised when an Eqiora operation is cancelled.

    Authority: ``crates/eqiora-python/src/error.rs::CancellationError``.
    """

    ...

class InternalError(EqioraError):
    """Unexpected internal failure; implementation details are not exposed.

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
    """Conformance report for a locked package and its dependencies.

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
    """Property data (constant, analytic, or tabulated) bound into a Model, with its source and validity range.

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
    """Read-only view of a Formulation written in source, taken from a fresh compilation.

    Authority: ``crates/eqiora-python/src/model/authored_formulation.rs``.
    """

    @property
    def gauge_field_ids(self) -> list[str] | None: ...
    @property
    def implication(self) -> Literal["strong-implies-weak", "strong-implies-interval-conservation", "strong-equivalent-finite"]: ...
    @property
    def assumptions(self) -> list[str]: ...
    @property
    def name(self) -> str: ...
    @property
    def test_restrictions(self) -> list[tuple[str, str, list[str], list[tuple[int, int]], str | None]]: ...
    @property
    def interval(self) -> tuple[str, str, str] | None: ...
    @property
    def kind(self) -> str: ...
    @property
    def source_identity(self) -> str: ...
    @property
    def relation_ids(self) -> list[str]: ...
    @property
    def domain_id(self) -> str | None: ...
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
    """Enum declaration with a fixed set of members. A replayed enum may have no name.

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
    """Enum member. It does not convert to a number or a Boolean.

    Authority: ``crates/eqiora-python/src/modeling/enumeration.rs::PyEnumValue``.
    """
    @property
    def value_type(self) -> ValueType: ...
    @property
    def enum_id(self) -> str: ...
    def __bool__(self) -> bool: ...

@final
class FiniteSpace:
    """Named finite basis with ordered labels. It is unrelated to finite-element spaces.

    Authority: ``crates/eqiora-python/src/modeling/nominal.rs::PyFiniteSpace``.
    """
    def __new__(cls, name: str, *, labels: list[str] | tuple[str, ...]) -> Self: ...
    @staticmethod
    def product(name: str, left: FiniteSpace, right: FiniteSpace) -> FiniteSpace: ...
    @property
    def factors(self) -> tuple[FiniteSpace, FiniteSpace] | None: ...
    @property
    def name(self) -> str: ...
    @property
    def id(self) -> str: ...
    @property
    def labels(self) -> tuple[str, ...] | None: ...

@final
class IndexSet:
    """Named, zero-based index set with a fixed positive size.

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
    """Type of a value: scalar domain, dimension, shape, and spatial frame.

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
    def coordinates(scalar: ValueType, space: FiniteSpace, *, dual: bool = False) -> ValueType: ...
    @staticmethod
    def linear_map(scalar: ValueType, source: FiniteSpace, target: FiniteSpace, *, source_dual: bool = False, target_dual: bool = False) -> ValueType: ...
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
    """Which side, lower or upper, a Cartesian boundary domain lies on.

    Authority: ``crates/eqiora-python/src/modeling.rs::PyBoundarySide``.
    """

    Lower: ClassVar[BoundarySide]
    Upper: ClassVar[BoundarySide]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class CoordinateInterval:
    """Finite lower and upper bounds, in coherent SI units, for one dimensioned coordinate.

    The bounds must be strictly increasing. The interval holds only the bounds and
    their dimension; the support it is attached to determines which coordinate it describes.

    Authority: ``crates/eqiora-python/src/coordinate_interval.rs::PyCoordinateInterval``.
    """
    def __new__(cls, lower: float, upper: float, *, dimension: Dimension) -> Self: ...
    @property
    def lower(self) -> float: ...
    @property
    def upper(self) -> float: ...
    @property
    def dimension(self) -> Dimension: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __repr__(self) -> str: ...
    def __hash__(self) -> int: ...

@final
class Domain:
    """Cartesian box, or one of its oriented boundaries, used while building a Model draft.

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
class SpatialRegularity:
    """Function-space regularity declared for a field, independent of the discretization.

    Authority: ``crates/eqiora-python/src/modeling/regularity.rs::PySpatialRegularity``.
    """

    Unspecified: ClassVar[SpatialRegularity]
    L2: ClassVar[SpatialRegularity]
    H1: ClassVar[SpatialRegularity]
    HCurl: ClassVar[SpatialRegularity]
    HDiv: ClassVar[SpatialRegularity]
    Smooth: ClassVar[SpatialRegularity]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class FieldRole:
    """Whether a field is a time-evolving state or an algebraic variable.

    Authority: ``crates/eqiora-python/src/modeling.rs::PyFieldRole``.
    """

    Variable: ClassVar[FieldRole]
    State: ClassVar[FieldRole]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class Initial:
    """Equations solved together to set initial values.

    Authority: ``crates/eqiora-python/src/modeling.rs::PyInitial``.
    """
    def __new__(cls, *equations: tuple[_ExpressionLike, _ExpressionLike]) -> Self: ...
    @property
    def equations(self) -> list[tuple[Expression, Expression]]: ...

@final
class Expression:
    """Symbolic expression. Its shape and support are inferred when it is built.

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
        spatial_regularity: SpatialRegularity = SpatialRegularity.Unspecified,
    ) -> Self: ...
    @property
    def spatial_regularity(self) -> SpatialRegularity: ...
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
    """Typed parameter declaration with its value.

    A Parameter is uniform in space; use a Field for values that vary. A nonzero
    spatial vector or tensor value needs a `frame` Domain registered in the Model.

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
    """Output derived from the solution; it is not an unknown of the solve.

    Authority: ``crates/eqiora-python/src/modeling/observable.rs::PyObservable``.
    """
    def __new__(cls, name: str, *, value_type: ValueType, expression: _ExpressionLike) -> Self: ...
    @property
    def name(self) -> str: ...
    @property
    def value_type(self) -> ValueType: ...
    @property
    def expression(self) -> Expression: ...
    def __neg__(self) -> Expression: ...
    def __bool__(self) -> bool: ...
    def __add__(self, other: _ExpressionLike, /) -> Expression: ...
    def __radd__(self, other: _ExpressionLike, /) -> Expression: ...
    def __sub__(self, other: _ExpressionLike, /) -> Expression: ...
    def __rsub__(self, other: _ExpressionLike, /) -> Expression: ...
    def __mul__(self, other: _ExpressionLike, /) -> Expression: ...
    def __rmul__(self, other: _ExpressionLike, /) -> Expression: ...
    def __truediv__(self, other: _ExpressionLike, /) -> Expression: ...
    def __rtruediv__(self, other: _ExpressionLike, /) -> Expression: ...

def integral(value: _ExpressionLike, measure: _ExpressionLike) -> Expression:
    """Construct an Observable integral with an explicit Domain measure.

    Authority: ``crates/eqiora-python/src/modeling/observable.rs::integral``.
    """
    ...
def measure(domain: Domain) -> Expression:
    """Return the measure of a Domain, for use with `integral`.

    Authority: ``crates/eqiora-python/src/modeling/observable.rs::measure``.
    """
    ...

@final
class PhysicalDomain:
    """Named physical domain, such as electrical, with its across and through quantities.

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
    """Immutable declaration of a scalar conserving port.

    Authority: ``crates/eqiora-python/src/modeling.rs::PyConservingPort``.
    """

    def __new__(cls, name: str, *, domain: PhysicalDomain) -> Self: ...
    @property
    def name(self) -> str: ...
    @property
    def domain(self) -> PhysicalDomain: ...

@final
class Connection:
    """Immutable declaration of an unnamed conserving connection.

    Authority: ``crates/eqiora-python/src/modeling.rs::PyConnection``.
    """

    ...

_ExpressionLike = Expression | Field | Parameter | Observable | bool | int | float | complex

@final
class Relation:
    """Named set of equations, each given as a (left, right) pair.

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
    """Immutable dense one-dimensional CPU ``float64`` or ``complex128`` result buffer.

    Authority: ``crates/eqiora-python/src/array.rs::PyArrayBuffer``.
    """

    def __getitem__(self, index: int, /) -> float | complex: ...
    def numpy(
        self, *, copy: bool | None = None
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]: ...
    def __array__(
        self, dtype: object | None = None, copy: bool | None = None
    ) -> npt.NDArray[Any]: ...
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
    """Identity of a Model: its ID, digest, and revision number.

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
    """Fingerprint for comparing Model structure while ignoring names. Equal fingerprints do not mean identical Models.

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
    """Prepared change to one value in a specific Model revision; apply it with `Model.commit`.

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
    """Initial coefficients for one Field, in coherent SI units.

    Authority: ``crates/eqiora-python/src/trajectory/field.rs::PyInitialField``.
    """
    def __new__(
        cls,
        field: FieldRef,
        /,
        *,
        vertex_values: object | None = None,
        cell_values: object | None = None,
        value: object | None = None,
    ) -> InitialField: ...
    @property
    def field(self) -> FieldRef: ...

@final
class FieldOutput:
    """Computed coefficients of one Model Field on one Mesh.

    Authority: ``crates/eqiora-python/src/result/field_output.rs::PyFieldOutput``.
    """

    @property
    def field(self) -> FieldRef: ...
    @property
    def mesh(self) -> meshing.Mesh: ...
    @property
    def dimension(self) -> tuple[Fraction, Fraction, Fraction, Fraction, Fraction, Fraction, Fraction]: ...
    @property
    def coefficient_dimension(self) -> tuple[Fraction, Fraction, Fraction, Fraction, Fraction, Fraction, Fraction]: ...
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
    """Immutable standalone periodic clock whose period and phase are exact rational seconds.

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
    """Exact continuation of reference or supported adaptive execution.

    Adaptive checkpoints retain their original Plan, Run and numerical history;
    reference execution checkpoints currently remain in memory.

    Authority: ``crates/eqiora-python/src/execution_checkpoint.rs::PyExecutionCheckpoint``.
    """
    def to_bytes(self) -> bytes: ...
    @staticmethod
    def from_bytes(plan: Plan, data: bytes, /) -> ExecutionCheckpoint: ...
    def write(self, path: str | os.PathLike[str]) -> None: ...
    @staticmethod
    def read(plan: Plan, path: str | os.PathLike[str], /) -> ExecutionCheckpoint: ...

@final
class ExecutionSession:
    """Step-by-step reference execution of a Model, driven by explicit input tables.

    Each advance finishes stabilizing the current execution boundary before returning.

    Input Ports, Clocks, and outputs can be selected by their source name or by their ULID in the Model.

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
    """Display label for one quantity in a rendered Model.

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
    """Model declarations and operators that part of a rendering refers to; the same for every profile.

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
    """Rendered equation or type, with plain-text and spoken alternatives and references back to the Model.

    If `used_fallback` is true, `text` is plain text rather than rich markup.

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
    """Compiled model with all references resolved. A Model is immutable; `commit` returns a new one.

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
class SpatialPlanView:
    """Plan view for a spatial problem: its scalar or vector Fields and coefficient settings.

    Authority: ``crates/eqiora-python/src/common_plan/capability_view.rs::PySpatialPlanView``.
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
    """Scalar type, data layout, schedule, provider, and placement chosen for execution.

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
    """Formulation families that a resolved Plan can use.

    `FirstOrderEvolution` is the first-order form produced by automatic time normalization.

    Authority: ``crates/eqiora-python/src/common_plan/capability_view.rs::PyFormulationKind``.
    """

    FiniteHermitianPencil: ClassVar[FormulationKind]
    FirstOrderEvolution: ClassVar[FormulationKind]
    HarmonicResponse: ClassVar[FormulationKind]
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
    """Mathematical formulation chosen when the Model was resolved into a Plan.

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
    @property
    def source_relation_id(self) -> str | None: ...
    @property
    def state_coordinates(self) -> tuple[tuple[FieldRef, int, int, bool], ...]: ...
    def __repr__(self) -> str: ...

@final
class Plan:
    """Numerical plan: a Model together with its mesh, discretization, solver, and time settings.

    The Model defines the physics. `capability` is a typed view of the field roles
    and settings for this kind of problem; `fields` lists every FieldRef regardless
    of problem kind.

    Authority: ``crates/eqiora-python/src/common_plan.rs::PyPlan``.
    """
    @property
    def enforcement(self) -> solve.ActiveSet | solve.StrictInterior | None: ...
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
    def harmonic_original_model(self) -> Model | None: ...
    @property
    def harmonic_amplitudes(self) -> tuple[tuple[str, FieldRef, FieldRef], ...]: ...
    @property
    def harmonic_angular_frequency(self) -> float | None: ...
    @property
    def mesh(self) -> meshing.Mesh | None: ...
    @property
    def formulation(self) -> FormulationView | None: ...
    @property
    def capability(self) -> SpatialPlanView | solve.AlgebraicPlanView | solve.EigenPlanView | time.OdePlanView | fluid.IncompressibleFlowPlanView: ...
    @property
    def fields(self) -> tuple[FieldRef, ...]: ...
    def field_coefficient_entities(self, field: FieldRef, /) -> tuple[tuple[int, int], ...]:
        """Mesh (dimension, index) entities in exact coefficient order."""
        ...
    def field_gradient_modes(self, field: FieldRef, /) -> dict[tuple[int, int], tuple[tuple[tuple[int, int], int], ...]]:
        """Independent vertex-gradient columns with signed edge entries.

        Removes one constant potential per connected component. These span
        discrete gradients, not every possible null mode of the Model operator.
        Entity pairs are (dimension, index) in this Plan's exact Mesh.
        """
        ...
    def field_exterior_derivative(self, field: FieldRef, /) -> dict[tuple[int, int], tuple[tuple[tuple[int, int], int], ...]]:
        """Signed face-integrated curl or cell-integrated divergence rows.

        Entries reference coefficient entities; zero row action is the
        corresponding differential constraint. No measure division is implied.
        Entity pairs are (dimension, index) in this Plan's exact Mesh.
        """
        ...
    @property
    def spatial(self) -> fem.Q1 | fem.TetrahedralEdge | fem.TetrahedralFace | fem.MiniP1 | fvm.CellCenteredTpfa | fvm.CellCentered | tuple[fem.ScopedSpatialPolicy, ...] | None: ...
    @property
    def solve(self) -> solve_module.ResolvedLinear | solve_module.ResolvedNewton | solve_module.HermitianEigen | None: ...
    @property
    def requested_solve(self) -> solve_module.Linear | solve_module.Newton | solve_module.HermitianEigen | None: ...
    @property
    def temporal(self) -> time.BackwardEuler | time.Tsitouras45 | time.ImplicitMidpoint | None: ...
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
    @staticmethod
    def interpolate(plan: Plan, result: Result, /) -> State:
        """Interpolate real stationary P1 fields onto a retained refinement; impose target traces."""
        ...
    @property
    def digest(self) -> str: ...
    @property
    def step(self) -> int: ...
    @property
    def time_s(self) -> float | None: ...
    @property
    def fields(self) -> tuple[trajectory.FieldSnapshot, ...]: ...
    @property
    def state_coordinates(self) -> tuple[tuple[FieldRef, int, int, bool], ...]: ...
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
    def value(self, field: FieldRef, /, *, derivative_order: int = 0) -> _TypedValue: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class ConvergenceReason:
    """Why a linear solve was considered converged.

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
    """Kind of differentiation: primal, JVP, or VJP.

    Authority: ``crates/eqiora-python/src/differentiation.rs::PyDifferentiationMode``.
    """

    Primal: ClassVar[DifferentiationMode]
    Jvp: ClassVar[DifferentiationMode]
    Vjp: ClassVar[DifferentiationMode]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class DerivativeImplementation:
    """Which implementation supplied the derivative for an evaluation.

    Authority: ``crates/eqiora-python/src/differentiation.rs::PyDerivativeImplementation``.
    """

    AnalyticAssembled: ClassVar[DerivativeImplementation]
    OperatorIr: ClassVar[DerivativeImplementation]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class LinearizationState:
    """Whether the linearization was computed afresh or reused.

    Authority: ``crates/eqiora-python/src/differentiation.rs::PyLinearizationState``.
    """

    Established: ClassVar[LinearizationState]
    Reused: ClassVar[LinearizationState]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class DifferentiableValueSignature:
    """Exact native Model/type/support identity of one selected numerical leaf.

    Authority: ``crates/eqiora-python/src/differentiation/signature.rs::PyDifferentiableValueSignature``.
    """

    @property
    def id(self) -> str: ...
    @property
    def model_digest(self) -> str: ...
    @property
    def value_type(self) -> ValueType: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def mesh_digest(self) -> str | None: ...
    @property
    def space(self) -> str | None: ...
    @property
    def coefficient_entities(self) -> tuple[tuple[int, int], ...]: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class DifferentiationEvidence:
    """Provenance of one differentiation: model, plan, solve summaries, and residual.

    Authority: ``crates/eqiora-python/src/differentiation.rs::PyDifferentiationEvidence``.
    """

    @property
    def model_digest(self) -> str: ...
    @property
    def plan_identity(self) -> str: ...
    @property
    def input_ids(self) -> list[str]: ...
    @property
    def output_ids(self) -> list[str]: ...
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
    def primal_solve(self) -> LinearSolveSummary | NonlinearSolveSummary: ...
    @property
    def initial_state_identity(self) -> str | None: ...
    @property
    def derivative_solve(self) -> LinearSolveSummary | None: ...

@final
class DifferentiablePrimal:
    """Ordered typed output leaves from a primal evaluation.

    Authority: ``crates/eqiora-python/src/differentiation.rs::PyDifferentiablePrimal``.
    """

    @property
    def output(self) -> tuple[npt.NDArray[np.float64 | np.complex128], ...]: ...
    @property
    def evidence(self) -> DifferentiationEvidence: ...

@final
class DifferentiableJvp:
    """Ordered output leaves and their shape-preserving forward tangent leaves.

    Authority: ``crates/eqiora-python/src/differentiation.rs::PyDifferentiableJvp``.
    """

    @property
    def output(self) -> tuple[npt.NDArray[np.float64 | np.complex128], ...]: ...
    @property
    def tangent(self) -> tuple[npt.NDArray[np.float64 | np.complex128], ...]: ...
    @property
    def evidence(self) -> DifferentiationEvidence: ...

@final
class DifferentiableVjp:
    """Ordered output leaves and the shape-preserving input cotangent leaves.

    Authority: ``crates/eqiora-python/src/differentiation.rs::PyDifferentiableVjp``.
    """

    @property
    def output(self) -> tuple[npt.NDArray[np.float64 | np.complex128], ...]: ...
    @property
    def input_cotangent(self) -> tuple[npt.NDArray[np.float64 | np.complex128], ...]: ...
    @property
    def evidence(self) -> DifferentiationEvidence: ...

@final
class DifferentiableEvaluation:
    """Converged evaluation at one parameter point.

    Authority: ``crates/eqiora-python/src/differentiation.rs::PyDifferentiableEvaluation``.
    """

    @property
    def point(self) -> tuple[npt.NDArray[np.float64 | np.complex128], ...]: ...
    def primal(self) -> DifferentiablePrimal: ...
    def jvp(
        self, tangent: Sequence[float | complex | Array | npt.NDArray[np.float64 | np.complex128] | _DLPackProducer]
    ) -> DifferentiableJvp: ...
    def vjp(
        self, cotangent: Sequence[float | complex | Array | npt.NDArray[np.float64 | np.complex128] | _DLPackProducer]
    ) -> DifferentiableVjp: ...
    @property
    def accepted_unknowns(self) -> Array: ...
    def residual_jvp(self, unknown: Array | _Float64Array | _DLPackProducer, parameter: Array | _Float64Array | _DLPackProducer) -> Array: ...
    def residual_vjp(self, cotangent: Array | _Float64Array | _DLPackProducer) -> tuple[Array, Array]: ...
    def output_partial_jvp(self, unknown: Array | _Float64Array | _DLPackProducer, parameter: Array | _Float64Array | _DLPackProducer) -> Array: ...
    def output_partial_vjp(self, cotangent: Array | _Float64Array | _DLPackProducer) -> tuple[Array, Array]: ...

@final
class DifferentiableProgram:
    """Differentiable map between ordered typed input and output products.

    Authority: ``crates/eqiora-python/src/differentiation.rs::PyDifferentiableProgram``.
    """

    @property
    def model_digest(self) -> str: ...
    @property
    def plan_identity(self) -> str: ...
    @property
    def inputs(self) -> tuple[DifferentiableValueSignature, ...]: ...
    @property
    def outputs(self) -> tuple[DifferentiableValueSignature, ...]: ...
    @property
    def dtype(self) -> str: ...
    @property
    def device(self) -> str: ...
    def evaluate(
        self,
        parameters: Sequence[float | complex | Array | npt.NDArray[np.float64 | np.complex128] | _DLPackProducer],
    ) -> DifferentiableEvaluation: ...
    def map(
        self,
        mapped: Sequence[float | complex | Array | npt.NDArray[np.float64 | np.complex128] | _DLPackProducer],
        *,
        point_shape: Sequence[int],
        shared_inputs: Sequence[ParameterRef] | None = None,
        shared: Sequence[float | complex | Array | npt.NDArray[np.float64 | np.complex128] | _DLPackProducer] | None = None,
        retained_bytes_limit: int = 67_108_864,
    ) -> EvaluationMapPlan: ...
    def primal(self) -> DifferentiablePrimal: ...
    def jvp(
        self, tangent: Sequence[float | complex | Array | npt.NDArray[np.float64 | np.complex128] | _DLPackProducer]
    ) -> DifferentiableJvp: ...
    def vjp(
        self, cotangent: Sequence[float | complex | Array | npt.NDArray[np.float64 | np.complex128] | _DLPackProducer]
    ) -> DifferentiableVjp: ...

@final
class EvaluationMapCancellation:
    """Cancellation token, checked between batch members.

    Authority: ``crates/eqiora-python/src/differentiation/batch.rs::PyEvaluationMapCancellation``.
    """

    def __init__(self) -> None: ...
    def cancel(self) -> None: ...
    @property
    def requested(self) -> bool: ...

@final
class EvaluationMapPlan:
    """Batch of parameter points that has been validated but not yet solved.

    Authority: ``crates/eqiora-python/src/differentiation/batch.rs::PyEvaluationMapPlan``.
    """

    def __len__(self) -> int: ...
    def __getitem__(self, index: int, /) -> tuple[npt.NDArray[np.float64 | np.complex128], ...]: ...
    @property
    def program(self) -> DifferentiableProgram: ...
    @property
    def point_shape(self) -> tuple[int, ...]: ...
    @property
    def shared_input_ids(self) -> list[str]: ...
    @property
    def mapped_input_ids(self) -> list[str]: ...
    @property
    def estimated_storage_bytes(self) -> int: ...
    @property
    def points(self) -> tuple[npt.NDArray[np.float64 | np.complex128], ...]: ...
    def occurrence_coordinates(self, index: int) -> tuple[int, ...]: ...
    def execute(
        self, *, cancellation: EvaluationMapCancellation | None = None
    ) -> CompleteEvaluationMap | EvaluationMapTerminalReport: ...

@final
class CompleteEvaluationMap:
    """Completed batch that keeps every member's evaluation and linearization.

    Authority: ``crates/eqiora-python/src/differentiation/batch/results.rs::PyCompleteEvaluationMap``.
    """

    def __len__(self) -> int: ...
    def __getitem__(self, index: int, /) -> DifferentiableEvaluation: ...
    @property
    def plan(self) -> EvaluationMapPlan: ...
    @property
    def statuses(self) -> list[str]: ...
    def member(self, index: int) -> DifferentiableEvaluation: ...
    def primal(self) -> tuple[npt.NDArray[np.float64 | np.complex128], ...]: ...
    def jvp(
        self, mapped: Sequence[float | complex | Array | npt.NDArray[np.float64 | np.complex128] | _DLPackProducer], *,
        shared: Sequence[float | complex | Array | npt.NDArray[np.float64 | np.complex128] | _DLPackProducer] | None = None,
        seed_shape: Sequence[int] | None = None,
        point_axes: Sequence[int] | None = None,
        numerical_bytes_limit: int = 67_108_864,
    ) -> EvaluationMapJvp: ...
    def vjp(
        self, cotangents: Sequence[float | complex | Array | npt.NDArray[np.float64 | np.complex128] | _DLPackProducer], *,
        seed_shape: Sequence[int] | None = None,
        point_axes: Sequence[int] | None = None,
        numerical_bytes_limit: int = 67_108_864,
    ) -> EvaluationMapVjp: ...

@final
class EvaluationMapTerminalReport:
    """Members evaluated before a batch failed or was cancelled; it never holds a complete primal or derivative product.

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
    """JVPs with ordered immutable output leaves; shape describes point and seed axes only.

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
    def output(self) -> tuple[npt.NDArray[np.float64 | np.complex128], ...]: ...
    @property
    def tangent(self) -> tuple[npt.NDArray[np.float64 | np.complex128], ...]: ...
    def member(self, index: int) -> DifferentiableJvp: ...

@final
class EvaluationMapVjp:
    """VJPs with ordered immutable input leaves; shared covectors sum over point axes.

    Shapes describe collection axes only; each leaf retains its intrinsic shape.

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
    def shared_cotangents(self) -> tuple[npt.NDArray[np.float64 | np.complex128], ...]: ...
    @property
    def mapped_cotangents(self) -> tuple[npt.NDArray[np.float64 | np.complex128], ...]: ...
    def member(self, index: int) -> DifferentiableVjp: ...

@final
class Series:
    """Read-only time series of one field, in SI units.

    Authority: ``crates/eqiora-python/src/result.rs::PySeries``.
    """

    @property
    def field(self) -> FieldRef | None: ...
    @property
    def derivative_order(self) -> int: ...
    @property
    def component(self) -> int: ...
    @property
    def imaginary(self) -> bool: ...
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
    """One inequality or complementarity condition in a Model.

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
    """Both sides of a constraint evaluated on the result, with their units and whether the constraint is active.

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
    """Event-guard activation selected from a Model.

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
    """Observable selected from a Model.

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
    """Value of an Observable evaluated on a Result, with the quadrature that was used.

    `quadratures` maps Domain IDs to (kind, dimension, points per axis).
    Point boundaries use one point. Values are expressed in SI units.
    `output` is (retained output position, time in seconds, State identity),
    or None for an instantaneous Result.

    Authority: ``crates/eqiora-python/src/result/observe.rs::PyObservation``.
    """
    def project_component(
        self,
        projection: Literal["real", "imaginary", "magnitude", "squared_magnitude", "phase"],
        index: int = 0,
    ) -> tuple[float, Dimension] | None:
        """Native mathematical projection in SI; phase is radians and None at exact zero.

        The source Observation retains exact shape, basis and Result lineage.
        No threshold, phase unwrapping, peak/RMS, power or probability is inferred.
        """
        ...
    @property
    def value(self) -> _TypedValue: ...
    @property
    def value_type(self) -> ValueType: ...
    @property
    def point(self) -> tuple[str, list[tuple[float, Dimension]]] | None: ...
    @property
    def output(self) -> tuple[int, float, str] | None: ...
    @property
    def result_identity(self) -> str: ...
    @property
    def observable_id(self) -> str: ...
    @property
    def evaluation_kind(self) -> Literal["value", "parameter-jvp", "state-jvp", "state-second-variation"]: ...
    @property
    def quadratures(self) -> dict[str, tuple[Literal["Point", "GaussLegendre", "SimplexDuffyGaussLegendre"], int, int]]: ...

@final
class FiniteSpectrum:
    """Discrete Fourier transform of uniformly spaced samples of an Observable.

    The forward transform uses a `+i` exponent with a `1/N` factor and the inverse a
    `-i` exponent; phases are measured from the first sample. Coefficients and
    amplitudes keep the input units, power squares them, and power density and the
    rectangle-transform estimate multiply them by seconds. `reconstruct_sample`
    returns the windowed samples. Aliasing is not excluded, and this is not a
    harmonic-response solve.

    Authority: ``crates/eqiora-python/src/result/spectrum.rs::PyFiniteSpectrum``.
    """
    @property
    def result_identity(self) -> str: ...
    @property
    def trajectory_identity(self) -> str: ...
    @property
    def observable_id(self) -> str: ...
    @property
    def interval_s(self) -> tuple[float, float]: ...
    @property
    def phase_reference_s(self) -> float: ...
    @property
    def spacing_s(self) -> float: ...
    @property
    def sample_count(self) -> int: ...
    @property
    def window(self) -> str: ...
    @property
    def endpoint_convention(self) -> str: ...
    @property
    def convention(self) -> str: ...
    @property
    def input_type(self) -> ValueType: ...
    @property
    def coefficient_type(self) -> ValueType: ...
    @property
    def coefficients(self) -> list[Any]: ...
    def frequency_hz(self, bin: int) -> float: ...
    def angular_frequency_rad_s(self, bin: int) -> float: ...
    def phase_rad(self, bin: int, *, component: int | None = None) -> float: ...
    def amplitude(self, bin: int, *, component: int | None = None) -> float: ...
    def one_sided_amplitude(self, bin: int, *, component: int | None = None) -> float: ...
    def power(self, bin: int, *, component: int | None = None) -> float: ...
    def power_density_per_hz(self, bin: int, *, component: int | None = None) -> float: ...
    def one_sided_power_density_per_hz(self, bin: int, *, component: int | None = None) -> float: ...
    def rectangle_transform_estimate(self, bin: int) -> Any: ...
    def reconstruct_sample(self, sample: int) -> Any: ...
    def projection_type(self, projection: str) -> ValueType: ...

@final
class TrajectoryObservation:
    """Terminal value or temporal reduction of an Observable over a computed trajectory.

    Values use coherent SI units. A time integral has the Observable's dimension
    multiplied by seconds and is computed from the solver's accepted steps, not
    from the saved output samples. Extrema retain the original units and return
    extremum_time_s. Closed windows include both event sides; exact computed ties
    choose earliest time then accepted occurrence. No extrema derivatives or
    continuous-solution error bounds are implied.

    Authority: ``crates/eqiora-python/src/result/time_observe.rs::PyTrajectoryObservation``.
    """
    @property
    def spatial_quadratures(self) -> dict[str, tuple[str, int, int]]: ...
    @property
    def extremum_time_s(self) -> float | None: ...
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
    def evaluation_kind(self) -> Literal["terminal", "time-integral", "terminal-parameter-jvp", "time-integral-parameter-jvp", "time-minimum", "time-maximum"]: ...
    @property
    def interpolation(self) -> time.TimeObservationInterpolation | None: ...
    @property
    def parameter_jvp(self) -> bool: ...

@final
class ObservableStateTangent:
    """Variation of a Result's field coefficients, with units.

    Fields that are left out have zero variation. Coefficients follow the field's
    vertex order, and their unit is given by a `Dimension`.

    Authority: ``crates/eqiora-python/src/result/observe.rs::PyObservableStateTangent``.
    """
    @property
    def result_identity(self) -> str: ...

@final
class ProfilePhase:
    """Total timing for one phase in the profile hierarchy.

    Inclusive time includes nested phases; self time excludes them. `fields`
    distinguishes phases that share the same path.

    Authority: ``crates/eqiora-python/src/profile.rs::PyProfilePhase``.
    """
    @property
    def path(self) -> list[str]: ...
    @property
    def name(self) -> str: ...
    @property
    def fields(self) -> dict[str, str]: ...
    @property
    def calls(self) -> int: ...
    @property
    def inclusive_seconds(self) -> float: ...
    @property
    def self_seconds(self) -> float: ...
    @property
    def mean_seconds(self) -> float: ...

@final
class ProfileEvent:
    """Metadata recorded for a profiled phase or solver event.

    Some phases keep every event. Phases that are only aggregated keep one
    representative event, and `ProfilePhase.calls` gives the full count.

    Authority: ``crates/eqiora-python/src/profile.rs::PyProfileEvent``.
    """
    @property
    def path(self) -> list[str]: ...
    @property
    def fields(self) -> dict[str, str]: ...

@final
class Profile:
    """Timings and events recorded in this process for one profiled run.

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
class NonlinearSolveSummary:
    """Convergence report for a Newton solve, measured on the original residual.

    Authority: ``crates/eqiora-python/src/result/nonlinear.rs::PyNonlinearSolveSummary``.
    """
    @property
    def completed_iterations(self) -> int: ...
    @property
    def initial_residual_norm(self) -> float: ...
    @property
    def true_residual_norm(self) -> float: ...
    @property
    def residual_target(self) -> float: ...
    @property
    def linear_solves(self) -> tuple[LinearSolveSummary, ...] | None:
        """Actual Newton correction records; empty for zero updates, unavailable for differentiation summaries."""
        ...


@final
class Eigenpair:
    """Converged eigenvalue and mode, linked to their Fields and Result.

    Authority: ``crates/eqiora-python/src/result/eigen.rs::PyEigenpair``.
    """
    def field(self, field: FieldRef) -> tuple[Any, ValueType]:
        """Return a source Field's value and exact type for this selected mode."""
        ...
    @property
    def eigenvalue(self) -> float: ...
    @property
    def mode(self) -> tuple[float | complex, ...]: ...
    @property
    def eigenvalue_type(self) -> ValueType: ...
    @property
    def mode_type(self) -> ValueType: ...
    @property
    def eigenvalue_field(self) -> FieldRef: ...
    @property
    def mode_field(self) -> FieldRef: ...
    @property
    def result_identity(self) -> str: ...
    @property
    def relative_residual(self) -> float: ...
    @property
    def normalization_defect(self) -> float: ...

@final
class SpatialBalanceTerm:
    """Directed contribution from one equation in one mesh cell.

    Authority: ``crates/eqiora-python/src/result/spatial.rs::PySpatialBalanceTerm``.
    """
    @property
    def domain_id(self) -> str: ...
    @property
    def cell_index(self) -> int: ...
    @property
    def relation_id(self) -> str: ...
    @property
    def tested_field_id(self) -> str: ...
    @property
    def trial_field_id(self) -> str | None: ...
    @property
    def term_index(self) -> int: ...
    @property
    def role(self) -> Literal["storage", "state-potential", "dissipation", "constraint-work", "applied-volume-work"]: ...
    @property
    def dimension(self) -> Dimension: ...
    @property
    def previous(self) -> float: ...
    @property
    def current(self) -> float: ...
    @property
    def increment(self) -> float: ...
    @property
    def work(self) -> float: ...
    def __repr__(self) -> str: ...

@final
class SpatialBoundaryWork:
    """Boundary work and its weak action, with units and coefficient identity.

    Authority: ``crates/eqiora-python/src/result/spatial.rs::PySpatialBoundaryWork``.
    """
    @property
    def domain_id(self) -> str: ...
    @property
    def kind(self) -> Literal["natural", "prescribed", "interface"]: ...
    @property
    def field_id(self) -> str: ...
    @property
    def connection_id(self) -> str | None: ...
    @property
    def cell_index(self) -> int | None: ...
    @property
    def entity(self) -> tuple[int, int] | None: ...
    @property
    def slot(self) -> int | None: ...
    @property
    def component(self) -> int | None: ...
    @property
    def work(self) -> float: ...
    @property
    def work_dimension(self) -> Dimension: ...
    @property
    def action(self) -> float | None: ...
    @property
    def action_dimension(self) -> Dimension | None: ...
    def __repr__(self) -> str: ...

@final
class SpatialWorkDefect:
    """Signed work residual for one domain and physical dimension.

    Authority: ``crates/eqiora-python/src/result/spatial.rs::PySpatialWorkDefect``.
    """
    @property
    def domain_id(self) -> str: ...
    @property
    def dimension(self) -> Dimension: ...
    @property
    def value(self) -> float: ...
    def __repr__(self) -> str: ...

@final
class SpatialStepObservation:
    """Solver and work observations for one retained spatial output.

    Authority: ``crates/eqiora-python/src/result/spatial.rs::PySpatialStepObservation``.
    """
    @property
    def result_identity(self) -> str: ...
    @property
    def trajectory_identity(self) -> str: ...
    @property
    def output_index(self) -> int: ...
    @property
    def step_index(self) -> int: ...
    @property
    def previous_state_identity(self) -> str: ...
    @property
    def state_identity(self) -> str: ...
    @property
    def assembly_counts(self) -> tuple[int, int] | None: ...
    @property
    def balance_error(self) -> str | None: ...
    @property
    def solve(self) -> LinearSolveSummary | NonlinearSolveSummary | None: ...
    @property
    def volume_terms(self) -> tuple[SpatialBalanceTerm, ...]: ...
    @property
    def boundary_work(self) -> tuple[SpatialBoundaryWork, ...]: ...
    @property
    def defects(self) -> tuple[SpatialWorkDefect, ...]: ...
    def __repr__(self) -> str: ...

@final
class Result:
    """Outcome of a completed run, with access to its fields and observations.

    Authority: ``crates/eqiora-python/src/result.rs::PyRunResult``.
    """

    @property
    def initial_state(self) -> State | None:
        """The supplied initial State retained by this Run, when present."""
        ...
    def reconstruct_harmonic_fields(
        self, *, time_seconds: float
    ) -> tuple[tuple[FieldRef, object, ValueType], ...]:
        """Original finite real Fields, SI values and types; no initial-value equivalence."""
        ...
    def reconstruct_harmonic_field_block(
        self, field: FieldRef, /, *, time_seconds: float, block: int = 0
    ) -> Array:
        """Real SI coefficients with the mapped amplitude block's association and layout."""
        ...
    @property
    def eigenpair_count(self) -> int: ...
    @property
    def eigen_convergence(self) -> str | None: ...
    @property
    def eigen_candidate_counts(self) -> tuple[int, int] | None: ...
    def spatial_step(self, output_index: int, /) -> SpatialStepObservation: ...
    def eigenpair(self, index: int) -> Eigenpair: ...
    def eigenprojector(self, indices: list[int]) -> tuple[object, ValueType]: ...
    @property
    def original_residual_norm(self) -> float | None: ...
    @property
    def compatibility_residual(self) -> float | None: ...
    @property
    def gauge_residual(self) -> float | None: ...
    @property
    def gauge_multiplier(self) -> float | None: ...
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
    def export_xdmf(self, path: str | PathLike[str]) -> None: ...
    @property
    def fields(self) -> list[Series]: ...
    @property
    def solve(self) -> LinearSolveSummary | NonlinearSolveSummary: ...
    def observe_spectrum(self, observable: ObservableRef, *, start_s: float, spacing_s: float, count: int, window: str, max_products: int) -> FiniteSpectrum: ...
    def observe_terminal(self, observable: ObservableRef) -> TrajectoryObservation: ...
    def observe_time_integral(self, observable: ObservableRef, *, interpolation: time.TimeObservationInterpolation, interval_s: tuple[float, float] | None = None, quadrature_points: int | None = None) -> TrajectoryObservation: ...
    def observe_time_minimum(self, observable: ObservableRef, *, interpolation: time.TimeObservationInterpolation, interval_s: tuple[float, float] | None = None, quadrature_points: int | None = None) -> TrajectoryObservation: ...
    def observe_time_maximum(self, observable: ObservableRef, *, interpolation: time.TimeObservationInterpolation, interval_s: tuple[float, float] | None = None, quadrature_points: int | None = None) -> TrajectoryObservation: ...
    def observe_parameter_jvp(self, observable: ObservableRef, direction: dict[ParameterRef, tuple[Dimension, float]], *, output_index: int) -> Observation: ...
    def observe_terminal_parameter_jvp(self, observable: ObservableRef, direction: dict[ParameterRef, tuple[Dimension, float]]) -> TrajectoryObservation: ...
    def observe_time_integral_parameter_jvp(self, observable: ObservableRef, direction: dict[ParameterRef, tuple[Dimension, float]], *, interpolation: time.TimeObservationInterpolation, interval_s: tuple[float, float] | None = None) -> TrajectoryObservation: ...
    def observe(self, observable: ObservableRef, *, quadrature_points: int | None = None, output_index: int | None = None) -> Observation: ...
    def discretization_error_indicators(self, field: FieldRef, /, *, quadrature_points: int) -> tuple[Array, Dimension]:
        """Return canonical-cell H1 seminorm residual estimates and their Dimension.

        Supports one steady real scalar P1 triangular diffusion Region, constant
        positive diffusion, and homogeneous essential boundaries. Square and sum
        the entries for the global squared estimate. The source-squared integral
        uses the requested Duffy quadrature. This is not a certified error bound.
        """
        ...
    def observe_at(self, observable: ObservableRef, coordinates: Sequence[tuple[float, Dimension]], *, quadrature_points: int | None = None, output_index: int | None = None) -> Observation: ...
    def observable_state_tangent(self, directions: dict[FieldRef, tuple[Dimension, Sequence[float]]]) -> ObservableStateTangent: ...
    def observe_state_jvp(self, observable: ObservableRef, tangent: ObservableStateTangent, *, quadrature_points: int) -> Observation: ...
    def observe_state_second_variation(self, observable: ObservableRef, first: ObservableStateTangent, second: ObservableStateTangent, *, wrt: FieldRef, quadrature_points: int) -> Observation: ...
    def output(self, field: FieldRef, /) -> FieldOutput: ...
    def boundary_force(
        self, selection: geometry.GeometrySelection, /, *, output_index: int | None = None
    ) -> trajectory.BoundaryForce: ...
    def boundary_flux(
        self, selection: geometry.GeometrySelection, /
    ) -> trajectory.BoundaryFlux: ...
    def series(self, field: FieldRef, /, *, derivative_order: int = 0, component: int = 0, imaginary: bool = False) -> Series: ...
    def mesh(self, field: FieldRef, /) -> meshing.Mesh: ...
    @property
    def trajectory(self) -> trajectory.Trajectory: ...

@final
class RunStatus:
    """Status of a run. A run only moves forward through these states.

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
class RunProgress:
    """Last observed accepted numerical boundary; adaptive Runs have no fixed maximum step count.

    Authority: ``crates/eqiora-python/src/execution/evidence.rs::PyRunProgress``.
    """

    @property
    def accepted_steps(self) -> int: ...
    @property
    def maximum_steps(self) -> int | None: ...
    @property
    def model_time_s(self) -> float: ...

@final
class RunCancellation:
    """Last accepted State and optional exact checkpoint when cancellation terminates.

    Authority: ``crates/eqiora-python/src/execution/evidence.rs::PyRunCancellation``.
    """

    @property
    def progress(self) -> RunProgress: ...
    @property
    def request_identity(self) -> str: ...
    @property
    def state(self) -> State: ...
    @property
    def checkpoint(self) -> ExecutionCheckpoint | None: ...

_RunResultT = TypeVar("_RunResultT", bound=Result)

class Run(Generic[_RunResultT]):
    """Handle to a submitted run. Await it, or call `result()`, to get the Result.

    Authority: ``bindings/python/python/eqiora/__init__.py::Run``.
    """

    def __init__(self, native: Never) -> None: ...
    @property
    def status(self) -> RunStatus: ...
    @property
    def history(self) -> tuple[RunStatus, ...]: ...
    @property
    def progress(self) -> RunProgress | None: ...
    @property
    def cancellation(
        self,
    ) -> RunCancellation | None: ...
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
    bindings: dict[str, _TypedValue | ClockDomain | CoordinateInterval | geometry.GeometrySelection | tuple[geometry.GeometrySelection, geometry.GeometrySelection] | tuple[tuple[geometry.GeometrySelection, ...], geometry.GeometrySelection]] | None = None,
    entry: str | None = None,
) -> Model:
    """Compile Eqiora source, from a file or a string, into a Model, optionally with a Geometry.

    Authority: ``bindings/python/python/eqiora/__init__.py::compile``.
    """

    ...

def compile_package(
    store_root: str | PathLike[str],
    resolution: bytes,
    *,
    entry: str,
    geometry: geometry.Geometry | None = None,
    bindings: dict[str, _TypedValue | ClockDomain | CoordinateInterval | geometry.GeometrySelection | tuple[geometry.GeometrySelection, geometry.GeometrySelection] | tuple[tuple[geometry.GeometrySelection, ...], geometry.GeometrySelection]] | None = None,
) -> Model:
    """Compile a Model or Component from a locked package project, optionally with a Geometry.

    Authority: ``crates/eqiora-python/src/package.rs::compile_package``.
    """

    ...

@final
class ProjectUpdate:
    """Validated dependency resolution that you can inspect and then commit once.

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
    """Resolve a project's dependencies without installing them or writing the lock.

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
    """Add or replace a local path dependency, then update the manifest and lock.

    Authority: ``crates/eqiora-python/src/package.rs::add_local_dependency``.
    """
    ...

def remove_local_dependency(
    project_root: str | PathLike[str],
    store_root: str | PathLike[str],
    name: str,
) -> bytes:
    """Remove a direct dependency and update the manifest and lock.

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
    """Add a package bundled with Eqiora as a dependency, updating the manifest and lock.

    Authority: ``crates/eqiora-python/src/package.rs::add_bundled_dependency``.
    """
    ...

def add_git_dependency(project_root: str | PathLike[str], store_root: str | PathLike[str], name: str, *, version: str, repository: str, revision: str) -> bytes:
    """Add a package from a Git repository at a fixed revision, updating the manifest and lock.

    Authority: ``crates/eqiora-python/src/package.rs::add_git_dependency``.
    """
    ...

def fetch_project(project_root: str | PathLike[str], store_root: str | PathLike[str]) -> bytes:
    """Download the packages listed in the existing lock into the store, without changing the lock.

    Authority: ``crates/eqiora-python/src/package.rs::fetch_project``.
    """
    ...

def open_project(project_root: str | PathLike[str], store_root: str | PathLike[str]) -> bytes:
    """Open an installed project and check it against its lock, using only the given store (no network).

    Authority: ``crates/eqiora-python/src/package.rs::open_project``.
    """
    ...

def update_project(project_root: str | PathLike[str], store_root: str | PathLike[str]) -> bytes:
    """Resolve the project's dependency requests again and write a new lock.

    Authority: ``crates/eqiora-python/src/package.rs::update_project``.
    """
    ...

def vendor_project(project_root: str | PathLike[str], store_root: str | PathLike[str], destination: str | PathLike[str]) -> bytes:
    """Copy the project's locked dependencies into a directory for offline use.

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
    """Check a locked package and its dependencies by recompiling them and comparing the results.

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
    spatial: fem.Q1 | fem.TetrahedralEdge | fem.TetrahedralFace | fem.MiniP1 | fvm.CellCenteredTpfa | fvm.CellCentered | tuple[fem.ScopedSpatialPolicy, ...] | None = None,
    formulation: FormulationKind | None = None,
    solve: solve.Linear | solve.Newton | solve.HermitianEigen | None = None,
    scaling: fluid.IncompressibleScaling | dict[ConstraintRef, tuple[float, Dimension]] | None = None,
    temporal: time.BackwardEuler | time.Tsitouras45 | time.ImplicitMidpoint | None = None,
    enforcement: solve.ActiveSet | solve.StrictInterior | None = None,
) -> Plan:
    """Combine a Model with numerical settings to produce a Plan.

    The spatial, solve, and time settings choose the numerical method; they never
    change the physics. The Plan keeps the Model and Mesh you pass in unchanged
    (the Mesh is not regenerated). A Model without spatial fields rejects `mesh`
    and `spatial`.

    Authority: ``bindings/python/python/eqiora/__init__.py::resolve``.
    """

    ...

def run(
    plan: Plan,
    *,
    state: State | None = None,
    checkpoint: ExecutionCheckpoint | None = None,
    until_s: float | None = None,
    output_times_s: tuple[float, ...] | None = None,
    steps: int | None = None,
    output_steps: tuple[int, ...] | None = None,
    profile: bool = False,
) -> Result:
    """Run a Plan and wait for the Result. Transient Plans need an end time or a step count.

    Authority: ``bindings/python/python/eqiora/__init__.py::run``.
    """

    ...

def submit(
    plan: Plan,
    *,
    state: State | None = None,
    checkpoint: ExecutionCheckpoint | None = None,
    until_s: float | None = None,
    output_times_s: tuple[float, ...] | None = None,
    steps: int | None = None,
    output_steps: tuple[int, ...] | None = None,
    profile: bool = False,
) -> Run[Result]:
    """Start a Plan running and return a `Run` handle immediately. Transient Plans need an end time or a step count.

    Authority: ``bindings/python/python/eqiora/__init__.py::submit``.
    """

    ...

def through(port: ConservingPort) -> Expression:
    """Return the through variable of a scalar conserving port.

    Authority: ``crates/eqiora-python/src/modeling.rs::through``.
    """

    ...

def trace(value: _ExpressionLike, *, on: Domain | None = None, from_: Domain | None = None) -> Expression:
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
    "CoordinateInterval",
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
    "DifferentiableValueSignature",
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
    "NonlinearSolveSummary",
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
    "Eigenpair",
    "TrajectoryObservation",
    "FiniteSpectrum",
    "ObservableStateTangent",
    "integral",
    "measure",
    "ParameterRef",
    "PhysicalDomain",
    "PropertyBinding",
    "QuantityLabel",
    "Plan",
    "FieldRole",
    "SpatialRegularity",
    "Initial",
    "Relation",
    "Result",
    "SpatialBalanceTerm",
    "SpatialBoundaryWork",
    "SpatialWorkDefect",
    "SpatialStepObservation",
    "Profile",
    "ProfileEvent",
    "ProfilePhase",
    "Revision",
    "ResolvedExecution",
    "SpatialPlanView",
    "Run",
    "RunStatus",
    "Series",
    "State",
    "RunCancellation",
    "RunProgress",
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
    "data",
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
    """Build the typed predicate `left == right`.

    Authority: ``crates/eqiora-python/src/modeling/predicates.rs::equal``.
    """
    ...


def not_equal(left: _ExpressionLike, right: _ExpressionLike) -> Expression:
    """Build the typed predicate `left != right`.

    Authority: ``crates/eqiora-python/src/modeling/predicates.rs::not_equal``.
    """
    ...


def less(left: _ExpressionLike, right: _ExpressionLike) -> Expression:
    """Build the typed predicate `left < right`.

    Authority: ``crates/eqiora-python/src/modeling/predicates.rs::less``.
    """
    ...


def less_equal(left: _ExpressionLike, right: _ExpressionLike) -> Expression:
    """Build the typed predicate `left <= right`.

    Authority: ``crates/eqiora-python/src/modeling/predicates.rs::less_equal``.
    """
    ...


def greater(left: _ExpressionLike, right: _ExpressionLike) -> Expression:
    """Build the typed predicate `left > right`.

    Authority: ``crates/eqiora-python/src/modeling/predicates.rs::greater``.
    """
    ...


def greater_equal(left: _ExpressionLike, right: _ExpressionLike) -> Expression:
    """Build the typed predicate `left >= right`.

    Authority: ``crates/eqiora-python/src/modeling/predicates.rs::greater_equal``.
    """
    ...


def logical_not(value: _ExpressionLike) -> Expression:
    """Build the typed predicate `not value`.

    Authority: ``crates/eqiora-python/src/modeling/predicates.rs::logical_not``.
    """
    ...


def logical_and(left: _ExpressionLike, right: _ExpressionLike) -> Expression:
    """Build the typed predicate `left and right`.

    Authority: ``crates/eqiora-python/src/modeling/predicates.rs::logical_and``.
    """
    ...


def logical_or(left: _ExpressionLike, right: _ExpressionLike) -> Expression:
    """Build the typed predicate `left or right`.

    Authority: ``crates/eqiora-python/src/modeling/predicates.rs::logical_or``.
    """
    ...
