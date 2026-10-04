"""Physical Geometry factors and mathematical intervals share the ordinary Model path."""
import pytest
import eqiora


@pytest.mark.parametrize("axes", [1, 2])
def test_geometry_velocity_product_retains_exact_units_and_reference(axes):
    graph = eqiora.geometry.GeometryGraph()
    region = (graph.interval(bounds=(0.0, 2.0)) if axes == 1 else
              graph.rectangle(x_bounds=(0.0, 2.0), y_bounds=(0.0, 3.0)))
    geometry = graph.build(region, named_topology={
        "position": region.region,
        **{f"side{i}": boundary for i, boundary in enumerate(region.boundaries)},
    })
    source = f"""model Distribution(support position:volume(ambient_dimension={axes}),
                                    support velocity:interval(m/s)) {{
        support phase:product(position,velocity);
        variable f:s/m^{axes+1} on phase;
        relation retain on phase {{f=0[s/m^{axes+1}];}}
        observable count:1=integral(f,measure(phase));
    }}"""
    bindings = {"position": geometry.selection("position"),
                "velocity": eqiora.CoordinateInterval(-2, 2, dimension=eqiora.Dimension(length=1, time=-1))}
    model = eqiora.compile(source=source, entry="Distribution", geometry=geometry, bindings=bindings)
    # Reference replay retains the Geometry identity; it does not produce a chart or mesh.
    assert eqiora.Model.from_bytes(model.to_bytes()).to_bytes() == model.to_bytes()
    with pytest.raises(eqiora.ValidationError, match="expression and measure"):
        eqiora.compile(source=source.replace("count:1", "count:m"), entry="Distribution",
                       geometry=geometry, bindings=bindings)
