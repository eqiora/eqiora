"""Public Python table authoring executes the exact offline package specimen."""
import eqiora
import pytest

q = eqiora.lang
u = eqiora.units


@pytest.mark.parametrize("scale", [1.0, 1e-6])
def test_public_python_table_package_and_model_replay(tmp_path, scale):
    source = eqiora.Module("main", package="org.example.PythonTable")
    temperature = eqiora.ValueType.real(eqiora.Dimension(temperature=1))
    conductivity = eqiora.ValueType.real(eqiora.Dimension(mass=1, length=1, time=-3, temperature=-1))
    contract = source.property_contract("Conductivity", value_type=conductivity,
                                       inputs={"temperature": temperature}, derivatives="first_open_intervals")
    samples = source.property_table_release(
        "Samples", implements=contract, data="conductivity_samples",
        axis_unit=u.K, source_unit=u.kg*u.m/u.s**3/u.K,
        validity=(q.quantity(300,u.K),q.quantity(360,u.K)),
        citation="synthetic_definition", license="repository_license")
    consumers = []
    for name,kind,factor in [
        ("FourierFlux",eqiora.ValueType.real(eqiora.Dimension(mass=1,time=-3)),q.quantity(-2,u.K/u.m)),
        ("SlabConductance",eqiora.ValueType.real(eqiora.Dimension(mass=1,length=2,time=-3,temperature=-1)),q.quantity(0.1,u.m)),
    ]:
        component = source.component(name)
        tick = component.clock_requirement("tick")
        prop = component.property("conductivity",contract=contract)
        point = component.parameter("temperature",value_type=temperature)
        output = component.output("result",value_type=kind,at=tick)
        component.relation("law",q.equation(output,prop(temperature=point)*factor),at=tick)
        consumers.append((component,kind))
    root = source.model("Main")
    clock = root.clock("tick",period_s=1)
    for (component,kind),prefix in zip(consumers,("flux","conductance")):
        for suffix,point in [("cool",310),("warm",340)]:
            name = f"{prefix}_{suffix}"
            instance = root.instance(name,component=component,bindings={"tick":clock,"conductivity":samples,"temperature":q.quantity(point,u.K)})
            output = root.output(f"result_{name}",value_type=kind,at=clock)
            root.relation(f"observe_{name}",q.equation(output,instance["result"]),at=clock)
    project = tmp_path / "project"
    for path in ["src","data","docs"]:
        (project/path).mkdir(parents=True)
    source.write_eqi(project/"src/main.eqi")
    (project/"eqiora.toml").write_text('[package]\nname="org.example.PythonTable"\nversion="1.0.0"\nentry="main"\n')
    array = eqiora.data.ResolvedArray.from_f64((3, 2), [300., 10.*scale, 320., 14.*scale, 360., 18.*scale])
    (project/"data/conductivity_samples.json").write_bytes(array.to_bytes())
    (project/"docs/synthetic_definition.md").write_text(f"Synthetic conductivity points (300,10), (320,14), (360,18), with conductivity multiplied by {scale}.")
    (project/"docs/repository_license.md").write_text("CC0-1.0 synthetic specimen.")
    store = tmp_path/"store"
    store.mkdir()
    resolution = eqiora.resolve_local_project(project,store)
    model = eqiora.compile_package(store,resolution,entry="Main")
    reopened = eqiora.Model.from_bytes(model.to_bytes())
    assert reopened.to_bytes() == model.to_bytes()
    assert len(model.property_bindings) == 4
    assert all(binding.derivatives == "first_open_intervals" for binding in model.property_bindings)
    session = model.execution_session(end_time_s=1,max_step_s=0.1,inputs={})
    assert session.advance_ticks(1) == 1
    for name,expected in [("flux_cool",-24.),("flux_warm",-32.),("conductance_cool",1.2),("conductance_warm",1.6)]:
        assert abs(session.output(f"result_{name}",0)[1]-expected*scale) < 1e-12*scale


def test_property_metadata_reopens_without_admitting_external_geometry():
    from test_external_geometry_round_trip import interval
    source = eqiora.Module("main")
    contract = source.property_contract("Coefficient",value_type=eqiora.ValueType.real())
    release = source.property_release("UnitCoefficient",implements=contract,value=1,
                                      source_unit=u.one,source_scale=1,
                                      citation="synthetic.definition",license="spdx.CC0_1_0")
    component = source.component("Consumer")
    region = component.volume("region",dimensions=1)
    coefficient = component.property("coefficient",contract=contract)
    field = component.field("value",on=region,role=eqiora.FieldRole.Variable,value_type=eqiora.ValueType.real())
    component.relation("law",q.equation(field,coefficient),on=region)
    root = source.model("Main")
    body = root.volume("body",dimensions=1)
    root.instance("consumer",component=component,bindings={"region":body,"coefficient":release})
    geometry = interval()
    direct = eqiora.compile(source=source,entry="Main",geometry=geometry,bindings={"body":geometry.selection("body")})
    reopened = eqiora.Model.from_bytes(direct.to_bytes())
    assert reopened.to_bytes() == direct.to_bytes()
    assert len(reopened.property_bindings) == 1
    binding = reopened.property_bindings[0]
    assert binding.release == direct.property_bindings[0].release
    assert binding.normalized_value == direct.property_bindings[0].normalized_value
