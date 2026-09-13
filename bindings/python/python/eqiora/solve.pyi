"""Linear and nonlinear solver policies.

Authority: ``crates/eqiora-python/src/common_plan/policy.rs::PyLinear``.
"""
from . import ConstraintRef, Dimension
from typing import ClassVar, Final, Self, final

@final
class SolverPlanningObjective:
    """Preference consumed by the versioned host-serial solver planner.

    Authority: ``crates/eqiora-python/src/common_plan/policy.rs::PySolverPlanningObjective``.
    """
    Robust: ClassVar[SolverPlanningObjective]
    Fast: ClassVar[SolverPlanningObjective]
    LowMemory: ClassVar[SolverPlanningObjective]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

#: Prefer the reproducible-reduction catalog member.
#:
#: Authority: ``crates/eqiora-python/src/common_plan/policy.rs::PySolverPlanningObjective``.
Robust: Final[SolverPlanningObjective]
#: Prefer Fast reduction and then the direct catalog member.
#:
#: Authority: ``crates/eqiora-python/src/common_plan/policy.rs::PySolverPlanningObjective``.
Fast: Final[SolverPlanningObjective]
#: Prefer admitted iterative candidates before direct factorization; no memory guarantee.
#:
#: Authority: ``crates/eqiora-python/src/common_plan/policy.rs::PySolverPlanningObjective``.
LowMemory: Final[SolverPlanningObjective]

@final
class LinearSolver:
    """Exact existing algorithm identity.

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
    """Exact existing preconditioner policy.

    Authority: ``crates/eqiora-python/src/common_plan/solver_request.rs::PyPreconditioner``.
    """
    Identity: ClassVar[Preconditioner]
    Jacobi: ClassVar[Preconditioner]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class Reduction:
    """Exact existing reduction policy.

    Authority: ``crates/eqiora-python/src/common_plan/solver_request.rs::PyReduction``.
    """
    Reproducible: ClassVar[Reduction]
    Fast: ClassVar[Reduction]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class SolverProvider:
    """Complete identity of one compiled backend release.

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
    """Explicit exact or program-controlled solve intent for the Model operator.

    Supply either an objective or all algorithm/preconditioner/reduction/provider
    fields. Incomplete or mixed intent is rejected before resolution.

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
    """Newton solver policy with nested linear-solve controls.

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
    """Exact linear algorithm, operator class, and provider selected by resolution.

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
    """Exact Newton policy and nested resolved linear solver.

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
    "SolverPlanningObjective",
    "Robust",
    "Fast",
    "LowMemory",
    "Linear",
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
    """Resolved finite affine solve with its exact unknown inventory.

    Authority: ``crates/eqiora-python/src/common_plan/algebraic.rs::PyAlgebraicPlanView``.
    """
    @property
    def kind(self) -> str: ...
    @property
    def unknown_count(self) -> int: ...


@final
class ConstraintTolerance:
    """Explicit absolute operand tolerances carrying physical dimensions.

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
    """Bounded finite affine active-set realization; Model meaning is unchanged.

    Authority: ``crates/eqiora-python/src/common_plan/enforcement.rs::PyActiveSet``.
    """
    def __new__(cls, *, tolerances: tuple[ConstraintTolerance, ...], max_active_sets: int) -> Self: ...
    @property
    def tolerances(self) -> tuple[ConstraintTolerance, ...]: ...
    @property
    def model_digest(self) -> str: ...
    @property
    def max_active_sets(self) -> int: ...
