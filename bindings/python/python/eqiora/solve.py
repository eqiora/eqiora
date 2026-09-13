"""Closed algebraic solve policies with executable Eqiora consumers."""

from ._eqiora import (
    Linear,
    ConstraintTolerance,
    ActiveSet,
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
