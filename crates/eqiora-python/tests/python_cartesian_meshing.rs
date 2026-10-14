mod support;
use support::public_module;

use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[test]
fn python_rectangle_cartesian_meshing_owns_exact_common_resources() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let module = public_module(py)?;
        let locals = PyDict::new(py);
        locals.set_item("eqiora", module)?;
        py.run(
            c_str!(
                r#"
import json
import eqiora

def rectangle(xmax=2.0):
    import eqiora
    graph = eqiora.geometry.GeometryGraph()
    source = graph.rectangle(x_bounds=(0.0, xmax), y_bounds=(-1.0, 2.0))
    return graph.build(source, named_topology={
        "region": source.region,
        "left": source.boundaries[0],
        "right": source.boundaries[1],
        "bottom": source.boundaries[2],
        "top": source.boundaries[3],
    })

provider = eqiora.meshing.CartesianMesher(cells=(2, 3))
assert provider.cells == (2, 3)
assert repr(provider) == "CartesianMesher(cells=(2, 3))"
request = provider
source = rectangle()
plan = eqiora.meshing.resolve(source, request)
assert plan.provider == provider
assert not hasattr(plan, "achieved_minimum_mean_ratio")
assert not hasattr(plan, "boundary_facets")

mesh = eqiora.meshing.generate(plan)
assert mesh.dimension == 2
assert mesh.vertex_count == 12
assert mesh.cell_count == 6
assert mesh.coordinates.shape == (12, 2)
assert mesh.cells.shape == (6, 4)
assert mesh.source_digest == mesh.realized_geometry_digest == source.digest
assert mesh.selection_entity_count("region") == 6
assert mesh.selection_entity_count("left") == 3
assert mesh.selection_entity_count("right") == 3
assert mesh.selection_entity_count("bottom") == 2
assert mesh.selection_entity_count("top") == 2
lineage = json.loads(mesh.production_lineage_bytes)
assert lineage["provider"] == {
    "identity": "eqiora.structured-cartesian", "version": "2"
}
assert lineage["effective_policy"] == {
    "kind": "cartesian-cells", "cells": [2, 3]
}

interval_graph = eqiora.geometry.GeometryGraph()
interval_operation = interval_graph.interval(bounds=(-1.0, 2.0))
interval_geometry = interval_graph.build(interval_operation, named_topology={
    "body": interval_operation.region,
    "left": interval_operation.boundaries[0],
    "right": interval_operation.boundaries[1],
})
interval_provider = eqiora.meshing.CartesianMesher(cells=(3,))
assert interval_provider.cells == (3,)
assert repr(interval_provider) == "CartesianMesher(cells=(3,))"
interval_mesh = eqiora.meshing.generate(
    eqiora.meshing.resolve(interval_geometry, interval_provider)
)
assert interval_geometry.dimension == interval_mesh.dimension == 1
assert interval_geometry.bounds == ((-1.0, 2.0),)
assert interval_geometry.selection("body").dimension == 1
assert interval_geometry.selection("left").dimension == 0
assert interval_mesh.coordinates.tolist() == [[-1.0], [0.0], [1.0], [2.0]]
assert interval_mesh.cells.tolist() == [[0, 1], [1, 2], [2, 3]]
assert interval_mesh.selection_entity_count(interval_geometry.selection("body")) == 3
assert interval_mesh.selection_entity_count(interval_geometry.selection("left")) == 1
assert interval_mesh.selection_entity_count(interval_geometry.selection("right")) == 1
replayed_interval_mesh = eqiora.meshing.Mesh.from_bytes(interval_mesh.to_bytes())
assert replayed_interval_mesh.digest == interval_mesh.digest
assert replayed_interval_mesh.source_digest == interval_geometry.digest
for geometry, mismatched in (
    (interval_geometry, eqiora.meshing.CartesianMesher(cells=(2, 3))),
    (source, interval_provider),
):
    try:
        eqiora.meshing.resolve(geometry, mismatched)
    except eqiora.ValidationError:
        pass
    else:
        raise AssertionError("dimension-mismatched Cartesian policy was admitted")

for cells in ((0, 3), (2, 0), (), (1, 1, 1, 1), (4_000_001,), (True, 3)):
    try:
        eqiora.meshing.CartesianMesher(cells=cells)
    except (eqiora.ValidationError, TypeError):
        pass
    else:
        raise AssertionError("invalid Cartesian policy was admitted")

try:
    eqiora.meshing.generate(rectangle(3.0), plan=plan)
except TypeError:
    pass
else:
    raise AssertionError("displaced Geometry argument was admitted")
for args, kwargs in (((), {"plan": plan}), ((object(),), {})):
    try:
        eqiora.meshing.generate(*args, **kwargs)
    except TypeError:
        pass
    else:
        raise AssertionError("generate admitted a non-positional or non-Plan request")

graph = eqiora.geometry.GeometryGraph()
outer = graph.rectangle(x_bounds=(0.0, 2.0), y_bounds=(-1.0, 2.0))
hole = graph.circle(center=(1.0, 0.0), radius=0.1)
cut = graph.subtract(outer, hole)
not_rectangle = graph.build(cut, named_topology={
    "region": cut.region,
    "left": outer.boundaries[0],
    "right": outer.boundaries[1],
    "walls": outer.boundaries[2:4],
    "hole": hole.boundaries[0],
})
try:
    eqiora.meshing.resolve(not_rectangle, request)
except eqiora.ValidationError:
    pass
else:
    raise AssertionError("non-rectangle Geometry reached Cartesian production")
"#
            ),
            None,
            Some(&locals),
        )
    })
}
