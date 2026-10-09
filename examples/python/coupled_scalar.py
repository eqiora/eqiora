"""Solve two coupled diffusion/reaction equations on the unit interval.

The exact solution is u=x(1-x), v=2x(1-x), with zero endpoint values.
Substitution gives forcing 2+x(1-x) and 8+6x(1-x), respectively.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import eqiora

SOURCE = """
public component CoupledScalar(
  support body: volume(ambient_dimension = 1),
  support left: boundary(parent = body),
  support right: boundary(parent = body)
) {

  variable u: 1 on body in h1;
  variable v: 1 on body in h1;
  parameter length: m = 1;
  parameter reaction_scale: 1 / m ^ 2 = 1;

  relation first on body {
    -div(grad(u)) + reaction_scale * (3 * u - v)
      - reaction_scale * (2 + coordinate(0) / length
        * (1 - coordinate(0) / length)) = 0;
  }
  relation second on body {
    -div(2 * grad(v)) + reaction_scale * (-2 * u + 4 * v)
      - reaction_scale * (8 + 6 * coordinate(0) / length
        * (1 - coordinate(0) / length)) = 0;
  }
  relation u_left on left { trace(u) = 0; }
  relation v_left on left { trace(v) = 0; }
  relation u_right on right { trace(u) = 0; }
  relation v_right on right { trace(v) = 0; }
}
"""


class Solution(NamedTuple):
    plan: eqiora.Plan
    result: eqiora.Result
    x: np.ndarray
    u: np.ndarray
    v: np.ndarray


def solve(cells: int = 16) -> Solution:
    graph = eqiora.geometry.GeometryGraph()
    interval = graph.interval(bounds=(0.0, 1.0))
    geometry = graph.build(interval, named_topology={
        "body": interval.region,
        "left": interval.boundaries[0],
        "right": interval.boundaries[1],
    })
    mesh = eqiora.meshing.generate(eqiora.meshing.resolve(
        geometry, eqiora.meshing.CartesianMesher(cells=(cells,))
    ))
    model = eqiora.compile(
        source=SOURCE, geometry=geometry, entry="CoupledScalar",
        bindings={
            "body": geometry.selection("body"),
            "left": (geometry.selection("left"), geometry.selection("body")),
            "right": (geometry.selection("right"), geometry.selection("body")),
        },
    )
    plan = eqiora.resolve(
        model,
        mesh=mesh,
        spatial=eqiora.fem.Q1(),
        solve=eqiora.solve.Linear(
            algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
            preconditioner=eqiora.solve.Preconditioner.Identity,
            reduction=eqiora.solve.Reduction.Reproducible,
            provider=eqiora.solve.SolverProvider.reference(),
            relative_tolerance=1.0e-12,
            absolute_tolerance=1.0e-14,
            maximum_iterations=1_000,
        ),
    )
    result = eqiora.run(plan)
    u = model.field("definition.u")
    v = model.field("definition.v")
    return Solution(
        plan, result, mesh.coordinates[:, 0],
        result.output(u).values("vertex").numpy().reshape(-1),
        result.output(v).values("vertex").numpy().reshape(-1),
    )


if __name__ == "__main__":
    solution = solve()
    exact = solution.x * (1.0 - solution.x)
    print(f"u maximum nodal error: {np.max(np.abs(solution.u - exact)):.3e}")
    print(f"v maximum nodal error: {np.max(np.abs(solution.v - 2 * exact)):.3e}")
