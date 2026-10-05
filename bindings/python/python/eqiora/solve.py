"""Closed algebraic solve policies with executable Eqiora consumers."""

from ._eqiora import (
    Linear,
    HermitianEigen,
    EigenPlanView,
    EigenCoordinateMap,
    EigenExclusion,
    ConstraintTolerance,
    ActiveSet,
    StrictInterior,
    AlgebraicPlanView,
    LinearSolver,
    Preconditioner,
    Reduction,
    SolverProvider,
    Newton,
    ResolvedLinear,
    ResolvedNewton,
    SolverPlanningObjective,
)

Robust = SolverPlanningObjective.Robust
Fast = SolverPlanningObjective.Fast
LowMemory = SolverPlanningObjective.LowMemory

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
