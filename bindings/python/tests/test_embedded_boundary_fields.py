"""Boundary Fields use the existing source, Geometry and Model owners."""
import pytest
import eqiora

LINE = """model Line() {
  domain body=box(0,2,0,3);
  domain wall=boundary(body,axis=0,side=lower);
  variable density:kg/m on wall;
  relation retain on wall {density=1[kg/m];}
  observable mass:kg=integral(density,measure(wall));
}"""
GEOMETRY_LINE = """model Line(support body:volume(ambient_dimension=2),
                            support wall:boundary(parent=body)) {
  variable density:kg/m on wall;
  relation retain on wall {density=1[kg/m];}
  observable mass:kg=integral(density,measure(wall));
}"""


def rectangle(upper=2.0):
    graph = eqiora.geometry.GeometryGraph()
    region = graph.rectangle(x_bounds=(0.0, upper), y_bounds=(0.0, 3.0))
    return graph.build(region, named_topology={
        "body": region.region, "wall": region.boundaries[0],
        "right": region.boundaries[1], "bottom": region.boundaries[2],
        "top": region.boundaries[3],
    })


def test_source_boundary_field_preserves_model_and_line_measure_units():
    model = eqiora.compile(source=LINE)
    assert eqiora.Model.from_bytes(model.to_bytes()).to_bytes() == model.to_bytes()
    with pytest.raises(eqiora.ValidationError, match="expression and measure"):
        eqiora.compile(source=LINE.replace("mass:kg", "mass:kg*m"))


def test_geometry_boundary_field_retains_exact_selection_and_artifact_reference():
    geometry = rectangle()
    parent = geometry.selection("body")
    values = {"body": parent, "wall": (geometry.selection("wall"), parent)}
    model = eqiora.compile(source=GEOMETRY_LINE, entry="Line", geometry=geometry,
                           bindings=values)
    # Canonical reference replay does not manufacture a mesh or Geometry authority.
    assert eqiora.Model.from_bytes(model.to_bytes()).to_bytes() == model.to_bytes()
    foreign = rectangle(4.0)
    values["wall"] = (foreign.selection("wall"), parent)
    with pytest.raises(ValueError, match="different exact Geometry revision"):
        eqiora.compile(source=GEOMETRY_LINE, entry="Line", geometry=geometry,
                       bindings=values)
