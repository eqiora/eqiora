import json

import pytest

import eqiora


def rectangle(xmax: float = 2.0) -> eqiora.geometry.Geometry:
    graph = eqiora.geometry.GeometryGraph()
    source = graph.rectangle(x_bounds=(0.0, xmax), y_bounds=(-1.0, 2.0))
    return graph.build(
        source,
        named_topology={
            "region": source.region,
            "left": source.boundaries[0],
            "right": source.boundaries[1],
            "bottom": source.boundaries[2],
            "top": source.boundaries[3],
        },
    )


def test_affine_triangle_mesher_publishes_exact_source_owned_common_mesh() -> None:
    provider = eqiora.meshing.AffineTriangleMesher(cells=(2, 3))
    request = provider
    source = rectangle()
    plan = eqiora.meshing.resolve(source, request)
    mesh = eqiora.meshing.generate(plan)

    assert provider.cells == (2, 3)
    assert provider.diagonal == "lower-left-to-upper-right"
    assert plan.provider == provider
    assert mesh.source_digest == mesh.realized_geometry_digest == source.digest
    assert (mesh.vertex_count, mesh.cell_count) == (12, 12)
    assert mesh.coordinates.shape == (12, 2)
    assert mesh.cells.shape == (12, 3)
    assert mesh.coordinates[[0, -1]].tolist() == [[0.0, -1.0], [2.0, 2.0]]
    assert mesh.cells[:2].tolist() == [[0, 4, 5], [0, 5, 1]]
    assert {
        name: mesh.selection_entity_count(name)
        for name in ("left", "right", "bottom", "top", "region")
    } == {"left": 3, "right": 3, "bottom": 2, "top": 2, "region": 12}
    assert json.loads(mesh.production_lineage_bytes)["effective_policy"] == {
        "kind": "affine-triangle",
        "cells": [2, 3],
        "diagonal": "lower-left-to-upper-right",
        "refinements": [],
    }
    with pytest.raises(TypeError):
        eqiora.meshing.generate(rectangle(3.0), plan=plan)


@pytest.mark.parametrize("cells", [(0, 3), (2, 0), (2,), (True, 3)])
def test_affine_triangle_mesher_rejects_invalid_cells(cells: tuple[object, ...]) -> None:
    with pytest.raises((eqiora.ValidationError, TypeError)):
        eqiora.meshing.AffineTriangleMesher(cells=cells)


def test_local_refinement_preserves_geometry_and_replayable_mesh_history() -> None:
    source = rectangle(3.0)
    mesh = eqiora.meshing.generate(
        eqiora.meshing.resolve(source, eqiora.meshing.AffineTriangleMesher(cells=(2, 2)))
    )
    original = mesh.to_bytes()
    first = mesh.refine([0, 0], maximum_cells=10)
    refined = first.refine([1], maximum_cells=11)
    assert (mesh.cell_count, first.cell_count, refined.cell_count) == (8, 10, 11)
    assert refined.coordinates[9:].tolist() == [[0.75, -0.25], [0.75, -1.0]]
    assert mesh.to_bytes() == original
    assert refined.source_digest == mesh.source_digest == source.digest
    assert len({mesh.digest, first.digest, refined.digest}) == 3
    assert len({mesh.correspondence_digest, first.correspondence_digest, refined.correspondence_digest}) == 3
    assert refined.selection_entity_count(source.selection("bottom")) == 3
    assert refined.selection_entity_count(source.selection("region")) == 11
    assert json.loads(refined.production_lineage_bytes)["effective_policy"]["refinements"] == [
        {"marked_cells": [0], "maximum_cells": 10},
        {"marked_cells": [1], "maximum_cells": 11},
    ]
    replay = eqiora.meshing.Mesh.from_bytes(refined.to_bytes())
    assert replay.to_bytes() == refined.to_bytes()
    assert replay.refine([0], maximum_cells=100).to_bytes() == refined.refine([0], maximum_cells=100).to_bytes()
    assert first.to_bytes() == mesh.refine([0], maximum_cells=10).to_bytes()


@pytest.mark.parametrize("marked,budget", [([], 20), ([True], 20), ([-1], 20), ([1.0], 20), ([8], 20), ([0], 9), ([0], True)])
def test_local_refinement_rejects_invalid_marks_and_completion_budgets(marked, budget) -> None:
    mesh = eqiora.meshing.generate(eqiora.meshing.resolve(
        rectangle(3.0), eqiora.meshing.AffineTriangleMesher(cells=(2, 2))
    ))
    before = mesh.to_bytes()
    with pytest.raises((eqiora.ValidationError, TypeError, OverflowError)):
        mesh.refine(marked, maximum_cells=budget)
    assert mesh.to_bytes() == before


def test_local_refinement_rejects_a_different_mesh_family() -> None:
    mesh = eqiora.meshing.generate(eqiora.meshing.resolve(
        rectangle(), eqiora.meshing.CartesianMesher(cells=(2, 2))
    ))
    with pytest.raises(eqiora.ValidationError, match="source-owned affine-triangle"):
        mesh.refine([0], maximum_cells=20)
