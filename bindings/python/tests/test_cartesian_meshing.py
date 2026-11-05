import json

import numpy as np
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


def interval(upper: float = 2.0) -> eqiora.geometry.Geometry:
    graph = eqiora.geometry.GeometryGraph()
    source = graph.interval(bounds=(-1.0, upper))
    return graph.build(
        source,
        named_topology={
            "body": source.region,
            "left": source.boundaries[0],
            "right": source.boundaries[1],
        },
    )


def test_cartesian_mesher_publishes_exact_source_owned_common_mesh() -> None:
    provider = eqiora.meshing.CartesianMesher(cells=(2, 3))
    request = provider
    source = rectangle()
    plan = eqiora.meshing.resolve(source, request)
    mesh = eqiora.meshing.generate(plan)

    assert provider.cells == (2, 3)
    assert plan.provider == provider
    assert mesh.source_digest == mesh.realized_geometry_digest == source.digest
    assert (mesh.vertex_count, mesh.cell_count) == (12, 6)
    assert mesh.coordinates.shape == (12, 2)
    assert mesh.cells.shape == (6, 4)
    assert not mesh.coordinates.flags.writeable
    assert not mesh.cells.flags.writeable
    assert np.allclose(mesh.coordinates[[0, -1]], [[0.0, -1.0], [2.0, 2.0]])
    assert {
        name: mesh.selection_entity_count(name)
        for name in ("left", "right", "bottom", "top", "region")
    } == {"left": 3, "right": 3, "bottom": 2, "top": 2, "region": 6}
    assert json.loads(mesh.production_lineage_bytes)["effective_policy"] == {
        "kind": "cartesian-cells",
        "cells": [2, 3],
    }
    with pytest.raises(eqiora.CapabilityError):
        _ = mesh.minimum_mean_ratio
    with pytest.raises(TypeError):
        eqiora.meshing.generate(rectangle(3.0), plan=plan)
    with pytest.raises(TypeError):
        eqiora.meshing.generate(plan=plan)
    with pytest.raises(TypeError):
        eqiora.meshing.generate(object())


def test_interval_cartesian_mesh_round_trips_exact_common_resources() -> None:
    source = interval()
    provider = eqiora.meshing.CartesianMesher(cells=(3,))
    mesh = eqiora.meshing.generate(eqiora.meshing.resolve(source, provider))

    assert source.dimension == mesh.dimension == 1
    assert source.bounds == ((-1.0, 2.0),)
    assert source.selection("body").dimension == 1
    assert source.selection("left").dimension == 0
    assert provider.cells == (3,)
    assert repr(provider) == "CartesianMesher(cells=(3,))"
    assert mesh.coordinates.tolist() == [[-1.0], [0.0], [1.0], [2.0]]
    assert mesh.cells.tolist() == [[0, 1], [1, 2], [2, 3]]
    assert {
        name: mesh.selection_entity_count(source.selection(name))
        for name in ("body", "left", "right")
    } == {"body": 3, "left": 1, "right": 1}
    replayed = eqiora.meshing.Mesh.from_bytes(mesh.to_bytes())
    assert replayed.digest == mesh.digest
    assert replayed.source_digest == source.digest
    assert replayed.coordinates.tolist() == mesh.coordinates.tolist()
    assert json.loads(mesh.production_lineage_bytes)["provider"]["version"] == "2"
    with pytest.raises(eqiora.ValidationError):
        mesh.selection_entity_count(interval(3.0).selection("left"))
    with pytest.raises(eqiora.ValidationError):
        eqiora.meshing.resolve(source, eqiora.meshing.CartesianMesher(cells=(2, 3)))
    with pytest.raises(eqiora.ValidationError):
        eqiora.meshing.resolve(rectangle(), provider)


@pytest.mark.parametrize(
    "cells", [(0, 3), (2, 0), (), (1, 1, 1, 1), (4_000_001,), (True, 3)]
)
def test_cartesian_mesher_rejects_invalid_cells(cells: tuple[object, ...]) -> None:
    with pytest.raises((eqiora.ValidationError, TypeError)):
        eqiora.meshing.CartesianMesher(cells=cells)


def test_mesh_plan_none_provider_matches_omission_without_defaulting_geometry() -> None:
    source = rectangle()
    provider = eqiora.meshing.CartesianMesher(cells=(2, 3))
    plan = eqiora.meshing.resolve(source, provider)
    omitted = eqiora.meshing.generate(plan)
    explicit_none = eqiora.meshing.generate(plan, None)
    assert explicit_none.source_digest == omitted.source_digest == source.digest
    assert explicit_none.production_lineage_bytes == omitted.production_lineage_bytes
    np.testing.assert_array_equal(explicit_none.coordinates, omitted.coordinates)
    np.testing.assert_array_equal(explicit_none.cells, omitted.cells)
    for arguments in ((source,), (source, None)):
        with pytest.raises(TypeError, match="requires an explicit provider"):
            eqiora.meshing.generate(*arguments)
    with pytest.raises(TypeError, match="does not accept a provider"):
        eqiora.meshing.generate(plan, provider)
