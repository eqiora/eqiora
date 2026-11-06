"""Ordinary authoring; analytic geometry only, no numerical solves."""
import math

import pytest
import eqiora


def build(graph, op):
    return graph.build(op, named_topology={
        "domain": op.region,
        **{f"boundary_{i}": h for i, h in enumerate(op.boundaries)},
    })


def test_skew_polygon_named_handles_bounds_and_repr():
    graph = eqiora.geometry.GeometryGraph()
    vertices = [(0.25, -0.5), (1.375, 1.0), (0.0625, 5.5), (-1.0625, 4.0)]
    op = graph.polygon(vertices=vertices, tolerance=1e-12)
    geometry = build(graph, op)
    assert geometry.bounds == ((-17 / 16, 11 / 8), (-0.5, 5.5))
    assert geometry.dimension == 2
    assert geometry.selection_dimension("domain") == 2
    assert geometry.selection_dimension("boundary_0") == 1
    assert geometry.classification_tolerance == 1e-12
    assert "bounds=" in repr(geometry)
    import json
    from eqiora._eqiora import _compose_view
    metadata = json.loads(_compose_view((geometry,)).metadata_json)
    assert any(layer["kind"] == "geometry" for layer in metadata["layers"])
    reversed_op = graph.polygon(vertices=[vertices[i] for i in (0, 3, 2, 1)], tolerance=1e-12)
    names = {"domain": reversed_op.region}
    names.update({f"boundary_{i}": reversed_op.boundaries[3-i] for i in range(4)})
    assert graph.build(reversed_op, named_topology=names) == geometry


def test_convex_polyhedron_bounds_names_and_foreign_handles():
    graph = eqiora.geometry.GeometryGraph()
    vertices = [(x, y, z) for z in (0., 4.) for y in (0., 3.) for x in (0., 2.)]
    faces = [[0, 4, 6, 2], [1, 3, 7, 5], [0, 1, 5, 4],
             [2, 6, 7, 3], [0, 2, 3, 1], [4, 5, 7, 6]]
    op = graph.convex_polyhedron(vertices=vertices, faces=faces, tolerance=1e-12)
    geometry = build(graph, op)
    assert geometry.bounds == ((0., 2.), (0., 3.), (0., 4.))
    assert geometry.dimension == 3
    assert geometry.selection_dimension("domain") == 3
    assert geometry.selection_dimension("boundary_0") == 2
    assert "bounds=" in repr(geometry)
    from eqiora._eqiora import _compose_view
    with pytest.raises(eqiora.ValidationError, match="render projection"):
        _compose_view((geometry,))
    foreign = eqiora.geometry.GeometryGraph()
    with pytest.raises(eqiora.ValidationError):
        build(foreign, op)
    with pytest.raises(eqiora.ValidationError):
        graph.convex_polyhedron(vertices=vertices, faces=faces[:-1], tolerance=1e-12)
    with pytest.raises(eqiora.ValidationError):
        graph.convex_polyhedron(vertices=vertices, faces=[f[::-1] for f in faces], tolerance=1e-12)


@pytest.mark.parametrize("value", [0., -1., math.inf, math.nan])
def test_invalid_tolerance_rejects(value):
    graph = eqiora.geometry.GeometryGraph()
    with pytest.raises(eqiora.ValidationError):
        graph.polygon(vertices=[(0., 0.), (1., 0.), (0., 1.)], tolerance=value)
