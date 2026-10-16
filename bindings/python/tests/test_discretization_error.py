"""Uniform P1 refinement against independently integrated rational Galerkin values."""
import json
from pathlib import Path

import numpy as np
import pytest

import eqiora as q


def solve(cells, boundary="0[K]", *, simplex=True):
    graph = q.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0., 1.), y_bounds=(0., 1.))
    sides = dict(zip(("left", "right", "bottom", "top"), rectangle.boundaries, strict=True))
    geometry = graph.build(rectangle, named_topology={"body": rectangle.region, **sides})
    provider = q.meshing.AffineTriangleMesher if simplex else q.meshing.CartesianMesher
    mesh = q.meshing.generate(q.meshing.resolve(geometry, provider(cells=(cells, cells))))
    source = """
public component Diffusion(
    support body:volume(ambient_dimension=2),
    support left:boundary(parent=body), support right:boundary(parent=body),
    support bottom:boundary(parent=body), support top:boundary(parent=body)
) {
    variable u:K on body in h1;
    variable x:1 on body;
    variable y:1 on body;
    relation x_value on body { x=coordinate(0)/1[m]; }
    relation y_value on body { y=coordinate(1)/1[m]; }
    relation balance on body { -div(grad(u))=2[K/m^2]*(x*(1-x)+y*(1-y)); }
    relation fixed_left on left { trace(u)=BOUNDARY; }
    relation fixed_right on right { trace(u)=BOUNDARY; }
    relation fixed_bottom on bottom { trace(u)=BOUNDARY; }
    relation fixed_top on top { trace(u)=BOUNDARY; }
}
""".replace("BOUNDARY", boundary)
    model = q.compile(source=source, entry="Diffusion", geometry=geometry, bindings={
        "body": geometry.selection("body"),
        **{name: (geometry.selection(name), geometry.selection("body")) for name in sides},
    })
    plan = q.resolve(model, mesh=mesh, spatial=(q.fem.P1().at(model.domain("body")),) if simplex else q.fem.Q1(),
                     solve=q.solve.Linear(
                         algorithm=q.solve.LinearSolver.BiConjugateGradientStabilized,
                         preconditioner=q.solve.Preconditioner.Identity,
                         reduction=q.solve.Reduction.Reproducible,
                         provider=q.solve.SolverProvider.reference(),
                         relative_tolerance=1e-12, absolute_tolerance=1e-14,
                         maximum_iterations=1000,
                     ))
    return model, mesh, plan, q.run(plan)


def test_residual_estimation_on_installed_results_converges_under_uniform_refinement():
    # On this uniform diagonal grid the integrated P1 matrix is the five-point
    # stencil (4,-1,-1,-1,-1). The load at (x,y) is
    # h^2 * [2*x*(1-x)+2*y*(1-y) - 2*h^2/3], from integrating the
    # six incident hats (their coordinate second moments are h^2/6).
    # Exact rational elimination of that independent stencil gives these values.
    references = [(2, 131/11520, 5/96), (4, 20377/5898240, 61/1024),
                  (8, 15567421/17112760320, 103181/1671168)]
    estimates = []
    for n, exact_error_squared, center in references:
        model, mesh, plan, result = solve(n)
        field = model.field("definition.u")
        values = np.asarray(result.output(field).values("vertex"))
        coordinates, triangles = np.asarray(mesh.coordinates), np.asarray(mesh.cells)
        center_index = np.flatnonzero(np.all(coordinates == (0.5, 0.5), axis=1)).item()
        assert values[center_index] == pytest.approx(center, abs=1e-12, rel=0)
        energy = 0.
        for vertices in triangles:
            p = coordinates[vertices]
            edge_matrix = p[1:] - p[0]
            gradient = np.linalg.solve(edge_matrix, values[vertices[1:]] - values[vertices[0]])
            energy += abs(np.linalg.det(edge_matrix))/2 * np.dot(gradient, gradient)
        # For u=x(1-x)y(1-y), the exact continuous energy is 1/45.
        # Galerkin orthogonality then gives the exact squared H1 error.
        assert 1/45-energy == pytest.approx(exact_error_squared, abs=1e-12, rel=0)
        data = result.to_bytes()
        before = None
        for candidate in (result, q.Result.from_bytes(q.Plan.from_bytes(plan.to_bytes()), data)):
            indicators, dimension = candidate.discretization_error_indicators(field, quadrature_points=3)
            indicators = np.asarray(indicators)
            assert dimension == q.Dimension(temperature=1)
            assert indicators.shape == (len(triangles),)
            assert not indicators.flags.writeable
            assert np.all(np.isfinite(indicators)) and np.all(indicators >= 0)
            if before is not None:
                np.testing.assert_array_equal(indicators, before)
            before = indicators
            assert candidate.to_bytes() == data
        estimates.append(float(np.dot(before, before)))
    assert estimates[2] < estimates[1] < estimates[0]


def test_residual_estimation_rejects_foreign_fields_boundaries_and_spaces():
    model, _, _, result = solve(2)
    foreign, _, _, nonhomogeneous = solve(2, "1[K]")
    with pytest.raises(ValueError, match="different exact Model"):
        result.discretization_error_indicators(foreign.field("definition.u"), quadrature_points=3)
    with pytest.raises(q.ValidationError, match="homogeneous essential"):
        nonhomogeneous.discretization_error_indicators(foreign.field("definition.u"), quadrature_points=3)
    with pytest.raises(q.ValidationError):
        result.discretization_error_indicators(model.field("definition.u"), quadrature_points=0)
    cartesian, _, _, unsupported = solve(2, simplex=False)
    with pytest.raises(q.ValidationError, match="triangular Mesh"):
        unsupported.discretization_error_indicators(cartesian.field("definition.u"), quadrature_points=3)


def test_full_scratch_mesh_error_notebook(tmp_path, monkeypatch):
    notebook = Path(__file__).resolve().parents[3] / "examples/python/mesh_error_estimation.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    namespace = {"__name__": "__main__"}
    monkeypatch.chdir(tmp_path)
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            assert "eqiora.examples" not in source
            exec(compile(source, f"{notebook}:cell{index}", "exec"), namespace)
    assert [row["n"] for row in namespace["history"]] == [2, 4, 8]
    assert len(list((tmp_path / "mesh-error-study").glob("*.eqresult"))) == 3
