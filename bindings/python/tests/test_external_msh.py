"""Ordinary installed Python import with explicit units and exact Geometry."""
import json

import pytest

import eqiora


SOURCE = b"""$MeshFormat
4.1 0 8
$EndMeshFormat
$Nodes
1 4 1 4
2 1 0 4
1
2
3
4
0 0 0
1 0 0
1 1 0
0 1 0
$EndNodes
$Elements
1 2 1 2
2 1 2 2
1 1 2 3
2 1 3 4
$EndElements
"""


def geometry() -> eqiora.geometry.Geometry:
    graph = eqiora.geometry.GeometryGraph()
    face = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    return graph.build(
        face, named_topology={"body": face.region, "left": face.boundaries[0], "right": face.boundaries[1], "bottom": face.boundaries[2], "top": face.boundaries[3]}
    )


def test_external_msh_retains_original_cells_units_binding_and_replay() -> None:
    source = geometry()
    mesh = eqiora.meshing.Mesh.import_msh(source, SOURCE, coordinate_unit="m")
    assert mesh.source_digest == source.digest
    assert (mesh.dimension, mesh.vertex_count, mesh.cell_count) == (2, 4, 2)
    assert mesh.cells.tolist() == [[0, 1, 2], [0, 2, 3]]
    assert mesh.coordinates.tolist() == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    assert mesh.selection_entity_count("body") == 2
    for side in ("left", "right", "bottom", "top"):
        assert mesh.selection_entity_count(side) == 1
        assert mesh.selection_entity_count(source.selection(side)) == 1
    lineage = json.loads(mesh.production_lineage_bytes)
    assert lineage["provider"] == {"identity": "eqiora.external-msh", "version": "4.1"}
    assert lineage["effective_policy"]["coordinate_unit"] == "m"
    assert "maximum_target_size_m" not in lineage["effective_policy"]
    assert eqiora.meshing.Mesh.from_bytes(mesh.to_bytes()).to_bytes() == mesh.to_bytes()


def test_external_msh_requires_supported_units_and_rejects_policy_overrides() -> None:
    source = geometry()
    with pytest.raises(TypeError):
        eqiora.meshing.Mesh.import_msh(source, SOURCE)  # type: ignore[call-arg]
    for unit in ("mm", "s", "", "metres"):
        with pytest.raises(eqiora.ValidationError, match="unit"):
            eqiora.meshing.Mesh.import_msh(source, SOURCE, coordinate_unit=unit)
    for options in (
        {"maximum_boundary_facets": 3},
        {"maximum_source_bytes": len(SOURCE) - 1},
        {"minimum_mean_ratio": 0.0},
    ):
        with pytest.raises(eqiora.ValidationError):
            eqiora.meshing.Mesh.import_msh(source, SOURCE, coordinate_unit="m", **options)
