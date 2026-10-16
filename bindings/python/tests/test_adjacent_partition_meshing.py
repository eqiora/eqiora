import json

import pytest

import eqiora


def partition_geometry(fluid_name: str = "fluid") -> eqiora.geometry.Geometry:
    graph = eqiora.geometry.GeometryGraph()
    left = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    right = graph.rectangle(x_bounds=(1.0, 2.0), y_bounds=(0.0, 1.0))
    partition = graph.partition(
        left,
        right,
        interface=(left.boundaries[1], right.boundaries[0]),
    )
    return graph.build(
        partition,
        named_topology={
            fluid_name: left.region,
            "solid": right.region,
            "interface": (left.boundaries[1], right.boundaries[0]),
            "inlet": left.boundaries[0],
            "outlet": right.boundaries[1],
            "walls": (
                left.boundaries[2],
                left.boundaries[3],
                right.boundaries[2],
                right.boundaries[3],
            ),
        },
    )


def test_exact_adjacent_partition_publishes_one_source_owned_common_mesh() -> None:
    source = partition_geometry()
    provider = eqiora.meshing.AffineTriangleMesher(cells=(2, 2))
    plan = eqiora.meshing.resolve(source, provider)
    mesh = eqiora.meshing.generate(plan)

    assert source.bounds == ((0.0, 2.0), (0.0, 1.0))
    assert source.selection_names == (
        "inlet",
        "interface",
        "outlet",
        "walls",
        "fluid",
        "solid",
    )
    assert (mesh.vertex_count, mesh.cell_count) == (9, 8)
    assert mesh.coordinates.tolist() == [
        [0.0, 0.0], [0.0, 0.5], [0.0, 1.0],
        [1.0, 0.0], [1.0, 0.5], [1.0, 1.0],
        [2.0, 0.0], [2.0, 0.5], [2.0, 1.0],
    ]
    assert mesh.cells.tolist() == [
        [0, 3, 4], [0, 4, 1], [1, 4, 5], [1, 5, 2],
        [3, 6, 7], [3, 7, 4], [4, 7, 8], [4, 8, 5],
    ]
    assert {
        name: mesh.selection_entity_count(source.selection(name))
        for name in source.selection_names
    } == {
        "inlet": 2,
        "interface": 2,
        "outlet": 2,
        "walls": 4,
        "fluid": 4,
        "solid": 4,
    }
    assert mesh.source_digest == source.digest
    assert mesh.realized_geometry_digest == source.digest
    assert json.loads(mesh.production_lineage_bytes)["effective_policy"] == {
        "kind": "affine-triangle",
        "cells": [2, 2],
        "diagonal": "lower-left-to-upper-right",
        "refinements": [],
    }


def test_partition_rejects_wrong_interface_incomplete_ownership_and_crosswire() -> None:
    graph = eqiora.geometry.GeometryGraph()
    left = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    right = graph.rectangle(x_bounds=(1.0, 2.0), y_bounds=(0.0, 1.0))
    with pytest.raises(eqiora.ValidationError):
        graph.partition(
            left,
            right,
            interface=(right.boundaries[0], left.boundaries[1]),
        )
    partition = graph.partition(
        left,
        right,
        interface=(left.boundaries[1], right.boundaries[0]),
    )
    with pytest.raises(eqiora.ValidationError):
        graph.build(
            partition,
            named_topology={"fluid": left.region, "solid": right.region},
        )

    source = partition_geometry()
    plan = eqiora.meshing.resolve(
        source,
        eqiora.meshing.AffineTriangleMesher(cells=(2, 2)),
    )
    with pytest.raises(TypeError):
        eqiora.meshing.generate(partition_geometry("liquid"), plan=plan)
    with pytest.raises(eqiora.ValidationError):
        eqiora.meshing.resolve(
            source,
            eqiora.meshing.AffineTriangleMesher(cells=(1, 2)),
        )
