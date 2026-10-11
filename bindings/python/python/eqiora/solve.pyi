"""Linear, nonlinear, and eigenvalue solver settings.

Authority: ``crates/eqiora-python/src/common_plan/policy.rs::PyLinear``.
"""
from . import ConstraintRef, Dimension, FieldRef, ValueType
from typing import Any, ClassVar, Final, Self, final

@final
class HermitianEigen:
    """Dense Hermitian eigenvalue solve with an explicit provider and settings.

    Authority: ``crates/eqiora-python/src/common_plan/eigen.rs::PyHermitianEigen``.
    """
    def __new__(cls, *, count: int, provider: SolverProvider, residual_tolerance: float,
                normalization_tolerance: float, target: tuple[float, Dimension] | None = None,
                interval: tuple[tuple[float, Dimension], tuple[float, Dimension]] | None = None) -> HermitianEigen: ...
    @property
    def count(self) -> int: ...
    @property
    def provider(self) -> SolverProvider: ...
    @property
    def residual_tolerance(self) -> float: ...
    @property
    def normalization_tolerance(self) -> float: ...
    @property
    def target(self) -> tuple[float, Dimension] | None: ...
    @property
    def interval(self) -> tuple[tuple[float, Dimension], tuple[float, Dimension]] | None: ...

@final
class SolverPlanningObjective:
    """Objective used by the automatic (serial, CPU) solver planner.

    Authority: ``crates/eqiora-python/src/common_plan/policy.rs::PySolverPlanningObjective``.
    """
    Robust: ClassVar[SolverPlanningObjective]
    Fast: ClassVar[SolverPlanningObjective]
    LowMemory: ClassVar[SolverPlanningObjective]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

#: Prefer a solver with reproducible reductions.
#:
#: Authority: ``crates/eqiora-python/src/common_plan/policy.rs::PySolverPlanningObjective``.
Robust: Final[SolverPlanningObjective]
#: Prefer fast reductions, then a direct solver.
#:
#: Authority: ``crates/eqiora-python/src/common_plan/policy.rs::PySolverPlanningObjective``.
Fast: Final[SolverPlanningObjective]
#: Prefer iterative solvers over direct factorization. Memory use is not guaranteed to be lower.
#:
#: Authority: ``crates/eqiora-python/src/common_plan/policy.rs::PySolverPlanningObjective``.
LowMemory: Final[SolverPlanningObjective]

@final
class LinearSolver:
    """Linear solver algorithm.

    Authority: ``crates/eqiora-python/src/common_plan/solver_request.rs::PyLinearSolver``.
    """
    ConjugateGradient: ClassVar[LinearSolver]
    MinimumResidual: ClassVar[LinearSolver]
    BiConjugateGradientStabilized: ClassVar[LinearSolver]
    SparseLu: ClassVar[LinearSolver]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class Preconditioner:
    """Preconditioner.

    Authority: ``crates/eqiora-python/src/common_plan/solver_request.rs::PyPreconditioner``.
    """
    Identity: ClassVar[Preconditioner]
    Jacobi: ClassVar[Preconditioner]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class Reduction:
    """Floating-point reduction order for the linear solve.

    ``Reproducible`` fixes the order; ``Fast`` uses the backend-native order.

    Authority: ``crates/eqiora-python/src/common_plan/solver_request.rs::PyReduction``.
    """
    Reproducible: ClassVar[Reduction]
    Fast: ClassVar[Reduction]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class SolverProvider:
    """A compiled solver backend, identified by name and release.

    Authority: ``crates/eqiora-python/src/common_plan/solver_request.rs::PySolverProvider``.
    """
    @staticmethod
    def reference() -> SolverProvider: ...
    @staticmethod
    def faer() -> SolverProvider: ...
    @property
    def id(self) -> str: ...
    @property
    def implementation_version(self) -> str: ...
    @property
    def libraries(self) -> list[tuple[str, str]]: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class Linear:
    """Linear solve settings for the Model operator.

    Either give an ``objective`` and let the planner choose, or give all of
    ``algorithm``, ``preconditioner``, ``reduction``, and ``provider``. Partial
    or mixed settings are rejected before resolution.

    Authority: ``crates/eqiora-python/src/common_plan/policy.rs::PyLinear``.
    """
    def __new__(
        cls,
        *,
        relative_tolerance: float,
        absolute_tolerance: float,
        maximum_iterations: int,
        objective: SolverPlanningObjective | None = None,
        algorithm: LinearSolver | None = None,
        preconditioner: Preconditioner | None = None,
        reduction: Reduction | None = None,
        provider: SolverProvider | None = None,
    ) -> Self: ...
    @property
    def relative_tolerance(self) -> float: ...
    @property
    def absolute_tolerance(self) -> float: ...
    @property
    def maximum_iterations(self) -> int: ...
    @property
    def objective(self) -> SolverPlanningObjective | None: ...
    @property
    def algorithm(self) -> LinearSolver | None: ...
    @property
    def preconditioner(self) -> Preconditioner | None: ...
    @property
    def reduction(self) -> Reduction | None: ...
    @property
    def provider(self) -> SolverProvider | None: ...

    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class Newton:
    """Newton solver settings, including the nested linear solve.

    Resolution may require the nested linear tolerances to be at least as strict
    as the Newton absolute tolerance, so that an unconverged iteration cannot
    stop on a zero correction.

    Authority: ``crates/eqiora-python/src/common_plan/policy.rs::PyNewton``.
    """
    def __new__(
        cls,
        *,
        linear: Linear,
        relative_tolerance: float = 1e-9,
        absolute_tolerance: float = 1e-11,
        maximum_iterations: int = 16,
        maximum_line_search_steps: int = 12,
    ) -> Self: ...
    @property
    def linear(self) -> Linear: ...
    @property
    def relative_tolerance(self) -> float: ...
    @property
    def absolute_tolerance(self) -> float: ...
    @property
    def maximum_iterations(self) -> int: ...
    @property
    def maximum_line_search_steps(self) -> int: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class ResolvedLinear:
    """Linear algorithm, operator class, and provider chosen during resolution.

    Authority: ``crates/eqiora-python/src/common_plan/resolved_solve.rs::PyResolvedLinear``.
    """
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
    def operator(self) -> str: ...
    @property
    def backend(self) -> str: ...
    @property
    def backend_version(self) -> str: ...
    @property
    def provider(self) -> SolverProvider: ...
    @property
    def objective(self) -> SolverPlanningObjective | None: ...
    @property
    def planning_policy_id(self) -> str | None: ...
    @property
    def selected_candidate_id(self) -> str | None: ...
    @property
    def selected_evidence_case(self) -> str | None: ...
    @property
    def planning_reasons(self) -> list[tuple[str, str]]: ...
    def __repr__(self) -> str: ...

@final
class ResolvedNewton:
    """Newton settings and the nested linear solver chosen during resolution.

    Authority: ``crates/eqiora-python/src/common_plan/resolved_solve.rs::PyResolvedNewton``.
    """
    @property
    def linear(self) -> ResolvedLinear: ...
    @property
    def relative_tolerance(self) -> float: ...
    @property
    def absolute_tolerance(self) -> float: ...
    @property
    def maximum_iterations(self) -> int: ...
    @property
    def maximum_line_search_steps(self) -> int: ...
    def __repr__(self) -> str: ...

__all__ = [
    "ConstraintTolerance",
    "ActiveSet",
    "StrictInterior",
    "SolverPlanningObjective",
    "Robust",
    "Fast",
    "LowMemory",
    "Linear",
    "HermitianEigen",
    "EigenPlanView",
    "EigenCoordinateMap",
    "EigenExclusion",
    "AlgebraicPlanView",
    "LinearSolver",
    "Preconditioner",
    "Reduction",
    "SolverProvider",
    "Newton",
    "ResolvedLinear",
    "ResolvedNewton",
]

@final
class AlgebraicPlanView:
    """Resolved finite algebraic solve, including the list of unknowns.

    Authority: ``crates/eqiora-python/src/common_plan/algebraic.rs::PyAlgebraicPlanView``.
    """
    @property
    def kind(self) -> str: ...
    @property
    def unknown_count(self) -> int:
        """Number of real numerical coordinates, including both parts of complex values."""
        ...


@final
class ConstraintTolerance:
    """Absolute tolerances for constraint operands, in physical units.

    Authority: ``crates/eqiora-python/src/common_plan/enforcement.rs::PyConstraintTolerance``.
    """
    @staticmethod
    def inequality(reference: ConstraintRef, value: float, dimension: Dimension) -> ConstraintTolerance: ...
    @staticmethod
    def complementarity(reference: ConstraintRef, left_value: float, left_dimension: Dimension, right_value: float, right_dimension: Dimension) -> ConstraintTolerance: ...
    @property
    def reference(self) -> ConstraintRef: ...
    @property
    def left_value(self) -> float: ...
    @property
    def left_dimension(self) -> Dimension: ...
    @property
    def right_value(self) -> float | None: ...
    @property
    def right_dimension(self) -> Dimension | None: ...

@final
class ActiveSet:
    """Enforce affine inequality constraints with an active-set method.

    Authority: ``crates/eqiora-python/src/common_plan/enforcement.rs::PyActiveSet``.
    """
    def __new__(cls, *, tolerances: tuple[ConstraintTolerance, ...], max_active_sets: int) -> Self: ...
    @property
    def tolerances(self) -> tuple[ConstraintTolerance, ...]: ...
    @property
    def model_digest(self) -> str: ...
    @property
    def max_active_sets(self) -> int: ...

@final
class StrictInterior:
    """Keep a nonlinear solve strictly inside its inequality constraints, by explicit positive margins.

    Authority: ``crates/eqiora-python/src/common_plan/enforcement/strict_interior.rs::PyStrictInterior``.
    """
    def __new__(cls, *, margins: tuple[ConstraintTolerance, ...]) -> Self: ...
    @property
    def margins(self) -> tuple[ConstraintTolerance, ...]: ...
    @property
    def model_digest(self) -> str: ...

@final
class EigenPlanView:
    """Resolved Hermitian eigenproblem: where each matrix comes from and how coordinates map to the model.

    Authority: ``crates/eqiora-python/src/common_plan/eigen.rs::PyEigenPlanView``.
    """
    @property
    def mode_field(self) -> FieldRef: ...
    @property
    def eigenvalue_field(self) -> FieldRef: ...
    @property
    def coordinate_embeddings(self) -> tuple[EigenCoordinateMap, ...]: ...
    @property
    def excluded_space(self) -> EigenExclusion | None: ...

@final
class EigenCoordinateMap:
    """One coordinate relation ``target = mapping * coordinate`` from the model.

    Authority: ``crates/eqiora-python/src/common_plan/eigen.rs::PyEigenCoordinateMap``.
    """
    @property
    def relation_id(self) -> str: ...
    @property
    def target_field(self) -> FieldRef: ...
    @property
    def coordinate_field(self) -> FieldRef: ...
    @property
    def mapping(self) -> Any: ...
    @property
    def mapping_type(self) -> ValueType: ...

@final
class EigenExclusion:
    """Directions removed from the eigenproblem, with separate nullspace tests for the operator and the metric.

    Defects are normalized Frobenius norms of the operator or metric acting on
    the excluded directions. The null flags compare them with the Plan's
    residual tolerance; they do not identify a physical gauge. The
    dimensionless projector can be oblique in coefficient space.

    Authority: ``crates/eqiora-python/src/common_plan/eigen.rs::PyEigenExclusion``.
    """
    @property
    def dimension(self) -> int: ...
    @property
    def projector(self) -> Any: ...
    @property
    def projector_type(self) -> ValueType: ...
    @property
    def operator_defect(self) -> float: ...
    @property
    def metric_defect(self) -> float: ...
    @property
    def tolerance(self) -> float: ...
    @property
    def is_operator_null(self) -> bool: ...
    @property
    def is_metric_null(self) -> bool: ...
