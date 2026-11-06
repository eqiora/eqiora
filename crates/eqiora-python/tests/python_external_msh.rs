mod support;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use support::public_module;

#[test]
fn external_msh_python_delegates_replay_and_requires_explicit_supported_units() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let module = public_module(py)?;
        let locals = PyDict::new(py);
        locals.set_item("eqiora", module)?;
        py.run(c_str!(r#"
import json
import eqiora
source = b'''$MeshFormat
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
'''
graph = eqiora.geometry.GeometryGraph()
face = graph.rectangle(x_bounds=(0.,1.), y_bounds=(0.,1.))
geometry = graph.build(face, named_topology={"body":face.region, "left":face.boundaries[0], "right":face.boundaries[1], "bottom":face.boundaries[2], "top":face.boundaries[3]})
mesh = eqiora.meshing.Mesh.import_msh(geometry, source, coordinate_unit="m")
assert mesh.dimension == 2 and mesh.cell_count == 2 and mesh.vertex_count == 4
assert mesh.cells.tolist() == [[0,1,2], [0,2,3]]
assert mesh.coordinates.tolist() == [[0.,0.], [1.,0.], [1.,1.], [0.,1.]]
assert mesh.selection_entity_count("body") == 2
for side in ("left","right","bottom","top"):
    assert mesh.selection_entity_count(side) == 1
    assert mesh.selection_entity_count(geometry.selection(side)) == 1
assert mesh.source_digest == geometry.digest
lineage = json.loads(mesh.production_lineage_bytes)
assert lineage["provider"] == {"identity":"eqiora.external-msh", "version":"4.1"}
assert lineage["effective_policy"]["coordinate_unit"] == "m"
assert "maximum_target_size_m" not in lineage["effective_policy"]
assert eqiora.meshing.Mesh.from_bytes(mesh.to_bytes()).to_bytes() == mesh.to_bytes()
try:
    eqiora.meshing.Mesh.import_msh(geometry, source)
except TypeError:
    pass
else:
    raise AssertionError("missing unit accepted")
for unit in ("mm", "s", "", "metres"):
    try:
        eqiora.meshing.Mesh.import_msh(geometry, source, coordinate_unit=unit)
    except eqiora.ValidationError as error:
        assert "unit" in str(error)
    else:
        raise AssertionError("unsupported unit accepted")
for override in ({"maximum_boundary_facets":3}, {"maximum_source_bytes":len(source)-1}, {"minimum_mean_ratio":0.}, {"minimum_mean_ratio":float("nan")}):
    try:
        eqiora.meshing.Mesh.import_msh(geometry, source, coordinate_unit="m", **override)
    except eqiora.ValidationError:
        pass
    else:
        raise AssertionError("invalid policy accepted")
"#), None, Some(&locals))
    })
}
