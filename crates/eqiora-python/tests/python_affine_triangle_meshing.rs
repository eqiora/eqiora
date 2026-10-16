mod support;
use support::public_module;

use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[test]
fn python_rectangle_affine_triangle_meshing_owns_exact_common_resources() -> PyResult<()> {
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

provider = eqiora.meshing.AffineTriangleMesher(cells=(2, 3))
assert provider.cells == (2, 3)
assert provider.diagonal == "lower-left-to-upper-right"
assert repr(provider) == "AffineTriangleMesher(cells=(2, 3))"
request = provider
source = rectangle()
plan = eqiora.meshing.resolve(source, request)
assert plan.provider == provider
assert not hasattr(plan, "achieved_minimum_mean_ratio")
assert not hasattr(plan, "boundary_facets")

mesh = eqiora.meshing.generate(plan)
assert mesh.dimension == 2
assert mesh.vertex_count == 12
assert mesh.cell_count == 12
assert mesh.coordinates.shape == (12, 2)
assert mesh.cells.shape == (12, 3)
assert mesh.coordinates.flags.writeable is False
assert mesh.cells.flags.writeable is False
assert mesh.coordinates[[0, -1]].tolist() == [[0.0, -1.0], [2.0, 2.0]]
assert mesh.cells.tolist() == [
    [0, 4, 5], [0, 5, 1],
    [1, 5, 6], [1, 6, 2],
    [2, 6, 7], [2, 7, 3],
    [4, 8, 9], [4, 9, 5],
    [5, 9, 10], [5, 10, 6],
    [6, 10, 11], [6, 11, 7],
]
assert mesh.source_digest == mesh.realized_geometry_digest == source.digest
assert mesh.selection_entity_count("region") == 12
assert mesh.selection_entity_count("left") == 3
assert mesh.selection_entity_count("right") == 3
assert mesh.selection_entity_count("bottom") == 2
assert mesh.selection_entity_count("top") == 2
lineage = json.loads(mesh.production_lineage_bytes)
assert lineage["provider"] == {
    "identity": "eqiora.affine-triangle", "version": "2"
}
assert lineage["effective_policy"] == {
    "kind": "affine-triangle",
    "cells": [2, 3],
    "diagonal": "lower-left-to-upper-right",
    "refinements": [],
}

for cells in ((0, 3), (2, 0), (2,), (True, 3)):
    try:
        eqiora.meshing.AffineTriangleMesher(cells=cells)
    except (eqiora.ValidationError, TypeError):
        pass
    else:
        raise AssertionError("invalid affine-triangle policy was admitted")

try:
    eqiora.meshing.generate(rectangle(3.0), plan=plan)
except TypeError:
    pass
else:
    raise AssertionError("displaced Geometry argument was admitted")

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
    raise AssertionError("non-rectangle Geometry reached affine-triangle production")
"#
            ),
            None,
            Some(&locals),
        )
    })
}
