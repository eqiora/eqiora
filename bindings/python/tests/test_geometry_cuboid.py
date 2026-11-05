"""Source-owned cuboid authoring through the installed ordinary Python API."""
import inspect

import numpy as np
import pytest

import eqiora as q


SIDES = ("x_lower", "x_upper", "y_lower", "y_upper", "z_lower", "z_upper")


def cuboid(*, bounds=((0.0, 1.0),) * 3):
    graph = q.geometry.GeometryGraph()
    operation = graph.cuboid(x_bounds=bounds[0], y_bounds=bounds[1], z_bounds=bounds[2])
    names = {"body": operation.region, **dict(zip(SIDES, operation.boundaries, strict=True))}
    return graph, operation, graph.build(operation, named_topology=names)


def test_cuboid_names_one_volume_and_six_exact_owned_faces():
    graph, operation, geometry = cuboid(bounds=((-1.0, 2.0), (3.0, 5.0), (7.0, 11.0)))
    assert str(inspect.signature(graph.cuboid)) == "(*, x_bounds, y_bounds, z_bounds)"
    assert operation.region.dimension == geometry.dimension == 3
    assert len(operation.boundaries) == 6
    assert all(face.dimension == 2 for face in operation.boundaries)
    assert geometry.bounds == ((-1.0, 2.0), (3.0, 5.0), (7.0, 11.0))
    assert geometry.classification_tolerance is None
    mesh = q.meshing.generate(geometry, q.meshing.CartesianMesher(cells=(2, 3, 4)))
    assert mesh.dimension == 3
    assert (mesh.vertex_count, mesh.cell_count) == (60, 24)
    assert mesh.coordinates.shape == (60, 3)
    assert mesh.cells.shape == (24, 8)
    assert [mesh.selection_entity_count(geometry.selection(name)) for name in SIDES] == [12, 12, 8, 8, 6, 6]
    assert mesh.selection_entity_count(geometry.selection("body")) == 24
    replayed = q.meshing.Mesh.from_bytes(mesh.to_bytes())
    assert replayed.source_digest == geometry.digest
    np.testing.assert_array_equal(replayed.coordinates, mesh.coordinates)
    with pytest.raises(q.ValidationError):
        q.meshing.generate(geometry, q.meshing.CartesianMesher(cells=(2, 3)))


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("bad", [(1.0, 1.0), (2.0, 1.0), (float("nan"), 1.0), (0.0, float("inf"))])
def test_cuboid_bounds_fail_in_the_shared_geometry_owner(axis, bad):
    bounds = [(0.0, 1.0)] * 3
    bounds[axis] = bad
    with pytest.raises(q.ValidationError):
        cuboid(bounds=bounds)


def test_cuboid_rejects_foreign_stale_mixed_and_incomplete_handles():
    graph, source, _ = cuboid()
    _, foreign, _ = cuboid()
    stale = graph.cuboid(x_bounds=(0.0, 2.0), y_bounds=(0.0, 1.0), z_bounds=(0.0, 1.0))
    rectangle = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    for wrong in (foreign.boundaries[0], stale.boundaries[0], rectangle.region):
        names = {"body": source.region, **dict(zip(SIDES, source.boundaries, strict=True))}
        names["x_lower"] = wrong
        with pytest.raises(q.ValidationError):
            graph.build(source, named_topology=names)
    for names in (
        {"body": source.region, "faces": source.boundaries[:5]},
        {"mixed": (source.region, source.boundaries[0]), "faces": source.boundaries[1:]},
    ):
        with pytest.raises(q.ValidationError):
            graph.build(source, named_topology=names)


SOURCE = """
public component Harmonic3d(
    support body: volume(ambient_dimension=3),
    support x_lower: boundary(parent=body), support x_upper: boundary(parent=body),
    support y_lower: boundary(parent=body), support y_upper: boundary(parent=body),
    support z_lower: boundary(parent=body), support z_upper: boundary(parent=body)
) {
    variable u: K on body in h1;
    relation balance on body { -div(grad(u)) = 0[K/m^2]; }
    BOUNDARY_LAWS
}
""".replace("BOUNDARY_LAWS", "\n".join(
    f"relation fixed_{name} on {name} {{ trace(u) = 1[K]*(1+coordinate(0)/1[m]+2*coordinate(1)/1[m]+3*coordinate(2)/1[m]); }}"
    for name in SIDES
))


def test_cuboid_compiles_and_runs_positive_analytic_q1_through_the_ordinary_api():
    _, _, geometry = cuboid()
    mesh = q.meshing.generate(geometry, q.meshing.CartesianMesher(cells=(2, 2, 2)))
    model = q.compile(source=SOURCE, entry="Harmonic3d", geometry=geometry, bindings={
        "body": geometry.selection("body"),
        **{name: (geometry.selection(name), geometry.selection("body")) for name in SIDES},
    })
    plan = q.resolve(model, mesh=mesh, spatial=q.fem.Q1(), solve=q.solve.Linear(
        algorithm=q.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=q.solve.Preconditioner.Identity,
        reduction=q.solve.Reduction.Reproducible,
        provider=q.solve.SolverProvider.reference(),
        relative_tolerance=1e-12, absolute_tolerance=1e-14, maximum_iterations=1000,
    ))
    result = q.run(plan)
    coordinates = np.asarray(mesh.coordinates)
    expected = 1.0 + coordinates @ np.array([1.0, 2.0, 3.0])
    # This affine harmonic polynomial is in Q1. There is one free coefficient;
    # the requested relative residual tolerance bounds its error below 1e-11.
    values = np.asarray(result.output(model.field("definition.u")).values("vertex"))
    np.testing.assert_allclose(values, expected, rtol=0.0, atol=1e-11)
    assert np.all(values > 0.0)


def test_generalized_handles_preserve_existing_rectangle_canonical_bytes():
    graph = q.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    geometry = graph.build(rectangle, named_topology={
        "body": rectangle.region,
        **dict(zip(("left", "right", "bottom", "top"), rectangle.boundaries, strict=True)),
    })
    # Independent literal from the existing rectangle-v2 wire contract: field
    # order is fixed; selections sort by dimension then name, not by handles.
    expected = (
        b'{"schema":"eqiora.planar-rectangle-envelope/v2",'
        b'"encoding":"eqiora.canonical-json/v1","length_unit":"metre",'
        b'"bounds":[[0.0,1.0],[0.0,1.0]],"entity_sets":['
        b'{"name":"bottom","dimension":1,"members":[2]},'
        b'{"name":"left","dimension":1,"members":[0]},'
        b'{"name":"right","dimension":1,"members":[1]},'
        b'{"name":"top","dimension":1,"members":[3]},'
        b'{"name":"body","dimension":2,"members":[0]}]}'
    )
    assert geometry.canonical_bytes == expected
