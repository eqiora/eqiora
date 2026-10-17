"""Installed export boundary: XML meaning and failure-safe sidecar publication."""
import hashlib
import json
import xml.etree.ElementTree as ET

import pytest
import eqiora as q

from test_discretization_error import solve


def document(path):
    xml = ET.parse(path)
    companions = set()
    for item in xml.findall(".//DataItem"):
        assert item.attrib["Format"] == "HDF"
        locator, dataset = item.text.strip().split(":", 1)
        assert dataset.startswith("/")
        companion = path.parent / locator
        data = companion.read_bytes()
        assert data.startswith(b"\x89HDF\r\n\x1a\n")
        assert locator == f"eqiora-{hashlib.sha256(data).hexdigest()}.h5"
        companions.add(companion)
    assert len(companions) == 1
    return xml, companions.pop()


def information(element):
    return {item.attrib["Name"]: item.attrib["Value"] for item in element.findall("Information")}


def test_static_export_keeps_shape_units_lineage_and_exact_source(tmp_path):
    _, _, plan, result = solve(2)
    before = result.to_bytes()
    path = tmp_path / "temperature.xdmf"
    result.export_xdmf(path)
    xml, companion = document(path)
    frame = xml.find("./Domain/Grid")
    assert frame.attrib["GridType"] == "Uniform"
    assert frame.find("Time") is None
    assert frame.find("Topology").attrib["TopologyType"] == "Triangle"
    assert frame.find("Geometry").attrib["GeometryType"] == "XY"
    assert information(frame)["eqiora.result"] == json.loads(before)["identity"]
    fields = frame.findall("Attribute")
    assert len(fields) == 1
    assert fields[0].attrib["Center"] == "Node"
    assert fields[0].attrib["AttributeType"] == "Scalar"
    assert fields[0].find("DataItem").attrib["Dimensions"] == "9"
    meaning = information(fields[0])
    assert meaning["eqiora.units"] == "coherent-SI"
    assert meaning["eqiora.dimension"] == "Θ"
    assert meaning["eqiora.representation"] == "nodal-values"
    metadata, bulk = path.read_bytes(), companion.read_bytes()
    q.Result.from_bytes(q.Plan.from_bytes(plan.to_bytes()), before).export_xdmf(path)
    assert path.read_bytes() == metadata and companion.read_bytes() == bulk
    assert result.to_bytes() == before
    assert not list(tmp_path.glob(".eqiora-*.tmp"))


def test_failed_companion_publication_preserves_existing_pair(tmp_path):
    _, _, _, original = solve(2)
    _, _, _, changed = solve(2, boundary="1[K]")
    path = tmp_path / "result.xdmf"
    original.export_xdmf(path)
    _, companion = document(path)
    metadata, bulk = path.read_bytes(), companion.read_bytes()
    staging = tmp_path / "another"
    staging.mkdir()
    changed.export_xdmf(staging / "result.xdmf")
    _, replacement = document(staging / "result.xdmf")
    assert replacement.name != companion.name
    collision = tmp_path / replacement.name
    collision.write_bytes(b"existing unrelated data")
    with pytest.raises(q.CompatibilityError, match="different bytes"):
        changed.export_xdmf(path)
    assert path.read_bytes() == metadata and companion.read_bytes() == bulk
    assert collision.read_bytes() == b"existing unrelated data"
    collision.unlink()
    collision.symlink_to(companion.name)
    with pytest.raises(q.CompatibilityError, match="regular file"):
        changed.export_xdmf(path)
    assert collision.is_symlink()
    assert path.read_bytes() == metadata and companion.read_bytes() == bulk
    with pytest.raises(q.CompatibilityError, match="suffix"):
        original.export_xdmf(tmp_path / "wrong.csv")
    assert not list(tmp_path.glob(".eqiora-*.tmp"))


def test_unsupported_mesh_rejects_before_destination_changes(tmp_path):
    _, _, _, result = solve(2, simplex=False)
    path = tmp_path / "result.xdmf"
    path.write_bytes(b"existing metadata")
    with pytest.raises(q.ValidationError, match="simplicial"):
        result.export_xdmf(path)
    assert path.read_bytes() == b"existing metadata"
    assert list(tmp_path.iterdir()) == [path]


def test_fixed_mesh_trajectory_exports_each_retained_time(tmp_path):
    graph = q.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0., 1.), y_bounds=(0., 1.))
    sides = ("left", "right", "bottom", "top")
    geometry = graph.build(rectangle, named_topology={
        "body": rectangle.region, **dict(zip(sides, rectangle.boundaries, strict=True)),
    })
    model = q.compile(source="""public component Ramp(
        support body:volume(ambient_dimension=2),
        support left:boundary(parent=body), support right:boundary(parent=body),
        support bottom:boundary(parent=body), support top:boundary(parent=body)) {
        state u:K on body in smooth;
        initial {u=0[K];}
        law balance on body {storage 1[s/m^2]*u; flux -grad(u); source 1[K/m^2];}
        relation l on left {normal(grad(u))=0;}
        relation r on right {normal(grad(u))=0;}
        relation b on bottom {normal(grad(u))=0;}
        relation t on top {normal(grad(u))=0;}
    }""", entry="Ramp", geometry=geometry, bindings={
        "body": geometry.selection("body"),
        **{name: (geometry.selection(name), geometry.selection("body")) for name in sides},
    })
    mesh = q.meshing.generate(q.meshing.resolve(geometry, q.meshing.AffineTriangleMesher(cells=(2, 2))))
    plan = q.resolve(model, mesh=mesh, spatial=(q.fem.P1().at(model.domain("body")),),
        temporal=q.time.BackwardEuler(0.25), solve=q.solve.Linear(
            algorithm=q.solve.LinearSolver.SparseLu, preconditioner=q.solve.Preconditioner.Identity,
            reduction=q.solve.Reduction.Fast, provider=q.solve.SolverProvider.faer(),
            relative_tolerance=1e-12, absolute_tolerance=1e-14, maximum_iterations=100))
    result = q.run(plan, state=q.State.initial(plan), steps=3, output_steps=(1, 2, 3))
    before = result.to_bytes()
    for index, candidate in enumerate((result, result.trajectory)):
        path = tmp_path / f"ramp{index}.xdmf"
        candidate.export_xdmf(path)
        xml, _ = document(path)
        frames = xml.findall("./Domain/Grid/Grid")
        assert [float(frame.find("Time").attrib["Value"]) for frame in frames] == [0.25, 0.5, 0.75]
        assert all(information(frame)["eqiora.time-unit"] == "s" for frame in frames)
        assert len({information(frame)["eqiora.mesh"] for frame in frames}) == 1
        assert len({information(frame)["eqiora.state"] for frame in frames}) == 3
        assert all(frame.find("Attribute").attrib["Center"] == "Node" for frame in frames)
    assert result.to_bytes() == before


def test_vector_result_keeps_spatial_components(tmp_path):
    import numpy as np
    graph = q.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0., 1.), y_bounds=(0., 1.))
    sides = ("left", "right", "bottom", "top")
    geometry = graph.build(rectangle, named_topology={
        "body": rectangle.region, **dict(zip(sides, rectangle.boundaries, strict=True)),
    })
    model = q.compile(source="""public component Flow(
        support body:volume(ambient_dimension=2),
        support left:boundary(parent=body), support right:boundary(parent=body),
        support bottom:boundary(parent=body), support top:boundary(parent=body)) {
        variable u:vector<m/s,2> on body in smooth;
        variable potential:m^2/s on body in smooth;
        relation prescribed on body {potential=1[m/s]*coordinate(0)+2[m/s]*coordinate(1);}
        relation momentum on body {-div(2[kg/(m*s)]*symmetric_part(grad(u)))=0;}
        relation l on left {trace(u)=trace(grad(potential));}
        relation r on right {normal(2[kg/(m*s)]*symmetric_part(grad(u)))=0;}
        relation b on bottom {trace(u)=trace(grad(potential));}
        relation t on top {trace(u)=trace(grad(potential));}
    }""", entry="Flow", geometry=geometry, bindings={
        "body": geometry.selection("body"),
        **{name: (geometry.selection(name), geometry.selection("body")) for name in sides},
    })
    mesh = q.meshing.generate(q.meshing.resolve(geometry, q.meshing.AffineTriangleMesher(cells=(2, 2))))
    plan = q.resolve(model, mesh=mesh, spatial=(q.fem.P1().at(model.domain("body")),),
        solve=q.solve.Linear(algorithm=q.solve.LinearSolver.SparseLu,
            preconditioner=q.solve.Preconditioner.Identity, reduction=q.solve.Reduction.Fast,
            provider=q.solve.SolverProvider.faer(), relative_tolerance=1e-12,
            absolute_tolerance=1e-14, maximum_iterations=100))
    result = q.run(plan)
    # Constant [1,2] has zero strain and exactly satisfies all boundary data.
    output = result.output(model.field("definition.u"))
    assert output.value_type.shape == [2]
    values = np.asarray(output.values("vertex")).reshape(9, 2)
    np.testing.assert_allclose(values, np.tile([1., 2.], (9, 1)), rtol=0, atol=1e-12)
    path = tmp_path / "velocity.xdmf"
    result.export_xdmf(path)
    xml, _ = document(path)
    field = xml.find("./Domain/Grid/Attribute")
    assert field.attrib["AttributeType"] == "Vector"
    assert field.attrib["Center"] == "Node"
    assert field.find("DataItem").attrib["Dimensions"] == "9 2"
    assert information(field)["eqiora.value-frame"] == "spatial-cartesian"


def test_mini_coefficients_are_not_silently_published_as_nodal_values(tmp_path):
    from test_spatial_integrals import mini_result
    model, _, candidate = mini_result()
    velocity = candidate.output(model.field("u"))
    assert set(velocity.associations) == {"vertex", "cell-bubble"}
    before = candidate.to_bytes()
    _, _, _, original = solve(2)
    path = tmp_path / "result.xdmf"
    original.export_xdmf(path)
    _, companion = document(path)
    metadata, bulk = path.read_bytes(), companion.read_bytes()
    paths = set(tmp_path.iterdir())
    with pytest.raises(q.ValidationError, match="multiple coefficient blocks"):
        candidate.export_xdmf(path)
    assert path.read_bytes() == metadata and companion.read_bytes() == bulk
    assert candidate.to_bytes() == before and set(tmp_path.iterdir()) == paths
