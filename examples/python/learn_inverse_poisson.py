"""Recover a Poisson source amplitude and inspect solved-field sensitivities."""

from __future__ import annotations

import argparse
from pathlib import Path

import eqiora
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--cells", type=int, default=12)
    args = parser.parse_args()
    if args.cells < 2:
        parser.error("--cells must be at least 2")

    graph = eqiora.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    names = ("x_lower", "x_upper", "y_lower", "y_upper")
    geometry = graph.build(
        rectangle,
        named_topology={
            "square": rectangle.region,
            **dict(zip(names, rectangle.boundaries, strict=True)),
        },
    )
    mesh = eqiora.meshing.generate(
        eqiora.meshing.resolve(
            geometry, eqiora.meshing.CartesianMesher(cells=(args.cells, args.cells))
        )
    )
    model = eqiora.compile(
        path=args.model,
        geometry=geometry,
        entry="InversePoisson",
        bindings={
            "square": geometry.selection("square"),
            **{
                name: (geometry.selection(name), geometry.selection("square"))
                for name in names
            },
            "diffusion": 1.0,
            "wave_number": np.pi,
            "source_scale": 2.0 * np.pi**2,
            "boundary_offset": 0.0,
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
            maximum_iterations=10_000,
        ),
    )
    program = eqiora.diff.compile(
        plan,
        inputs=[
            model.parameter("source_scale"),
            model.parameter("diffusion"),
            model.parameter("boundary_offset"),
        ],
        output=plan.capability.fields[0],
    )
    nominal = np.array([2.0 * np.pi**2, 1.0, 0.0], dtype=np.float64)
    response = program.primal().output.numpy()
    # Linear source response: g is the field produced per unit source scale.
    g = response / nominal[0]
    # A controlled synthetic recovery: use the same discretization for both solves.
    true_source = 1.4 * nominal[0]
    data = program.evaluate(
        np.array([true_source, 1.0, 0.0], dtype=np.float64)
    ).primal().output.numpy()
    recovered = float(np.dot(g, data) / np.dot(g, g))
    print(f"synthetic source: {true_source:.10f}")
    print(f"recovered source: {recovered:.10f}")
    print(f"relative recovery error: {abs(recovered / true_source - 1.0):.3e}")

    # Analytic continuum area mean: integral sin(pi*x) sin(pi*y) = 4/pi**2.
    # Trapezoidal weights integrate the uniform Q1 field, including its boundary.
    weights = np.ones((args.cells + 1, args.cells + 1), dtype=np.float64)
    weights[[0, -1], :] *= 0.5
    weights[:, [0, -1]] *= 0.5
    weights = weights.ravel() / args.cells**2
    print(f"Q1 area mean: {float(np.dot(weights, response)):.10f}")
    print(f"continuum area mean: {4.0 / np.pi**2:.10f}")

    # Differentiate the area mean with respect to [source, diffusion, boundary].
    gradient = program.vjp(weights).input_cotangent.numpy()
    print("area-mean gradient:", gradient)
    print("continuum gradient:", np.array([2.0 / np.pi**4, -4.0 / np.pi**2, 1.0]))
    direction = np.array([0.7, -0.2, 0.3], dtype=np.float64)
    tangent = program.jvp(direction).tangent.numpy()
    pairing_difference = abs(float(np.dot(weights, tangent) - np.dot(direction, gradient)))
    print(f"forward/reverse pairing difference: {pairing_difference:.3e}")
    step = 1.0e-4
    for _ in range(4):
        plus = program.evaluate(nominal + step * direction).primal().output.numpy()
        minus = program.evaluate(nominal - step * direction).primal().output.numpy()
        finite_difference = (plus - minus) / (2.0 * step)
        difference = np.max(np.abs(finite_difference - tangent))
        print(f"step={step:.0e}, maximum derivative difference={difference:.3e}")
        step *= 0.1


if __name__ == "__main__":
    main()
