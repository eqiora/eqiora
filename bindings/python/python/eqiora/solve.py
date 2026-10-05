"""Closed algebraic solve policies with executable Eqiora consumers."""

from ._eqiora import (
    Linear,
    HermitianEigen,
    EigenPlanView,
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
    "AlgebraicPlanView",
    "LinearSolver",
    "Preconditioner",
    "Reduction",
    "SolverProvider",
    "Newton",
    "ResolvedLinear",
    "ResolvedNewton",
]
