from pathlib import Path
from fractions import Fraction

import pytest

import eqiora
from _signature_bindings import support_bindings


q = eqiora.lang
u = eqiora.units


def test_source_power_preserves_python_negative_base_grouping():
    source = eqiora.Module("main")
    component = source.component("Powers")
    body = component.volume("body", dimensions=1)
    x = component.parameter("x", value_type=eqiora.ValueType.real(eqiora.Dimension()))
    cases = [
        ("negative_power", -(x**2), "-x ^ 2 = 0;"),
        ("negative_base", (-x)**2, "(-x) ^ 2 = 0;"),
        ("signed_power", x**-2, "x ^ -2 = 0;"),
        ("quantity_base", q.quantity(-2, u.one)**2, "(-2 [1]) ^ 2 = 0;"),
    ]
    for name, expression, _ in cases:
        component.relation(name, q.equation(expression, 0), on=body)
    emitted = source.to_eqi()
    for _, _, text in cases:
        assert text in emitted


def test_documentation_emits_attached_paragraphs_with_utf8_byte_bound():
    source = eqiora.Module("main")
    source.component("Documented", doc="Summary.\n\nFurther **prose**.")
    assert "/// Summary.\n///\n/// Further **prose**.\n" in source.to_eqi()
    bounded = eqiora.Module("main")
    bounded.component("Bounded", doc="é" * 8192)
    assert "/// " + "é" * 8192 in bounded.to_eqi()
    with pytest.raises(q.ModuleError, match="16384-byte limit"):
        eqiora.Module("main").component("TooLong", doc="é" * 8193)


@pytest.mark.parametrize("value, syntax", [
    (0, "0"),
    (q.quantity(2, u.one / u.m**2), "2 [1 / m ^ 2]"),
    (q.quantity(-2, u.one / u.m**2), "-2 [1 / m ^ 2]"),
])
def test_component_parameter_bindings_emit_explicit_units(value, syntax):
    source = scalar_property_source(binding=value)
    assert f"source_scale = {syntax}" in source.to_eqi()


def test_rational_unit_exponents_preserve_exact_source_spelling():
    source = eqiora.Module("main")
    component = source.component("Wave")
    component.parameter("amplitude", value_type=eqiora.ValueType.real(eqiora.Dimension(length=Fraction(-1, 2))))
    text = source.to_eqi()
    assert "m ^ (-1 / 2)" in text
    for exponent in [True, 0.5, 1.0]:
        with pytest.raises(TypeError):
            u.m ** exponent
    for exponent in [2147483648, -2147483648, Fraction(1, 2147483648)]:
        with pytest.raises(ValueError):
            u.m ** exponent


def test_input_quantities_compile_from_python_and_emitted_source(tmp_path: Path):
    source = eqiora.Module("main")
    law = source.component("RootLength")
    region = law.volume("region", dimensions=2)
    length = law.field("length", role=eqiora.FieldRole.Variable, on=region, value_type=eqiora.ValueType.real(eqiora.Dimension(length=1)))
    law.relation("balance", q.equation(length - q.math.sqrt(q.quantity(4, u.m.prefixed("m") ** 2)), 0), on=region)
    assert "4 [mm ^ 2]" in source.to_eqi()
    direct = eqiora.compile(source=source, geometry=(_binding_geometry := rectangle_geometry()), entry='RootLength', bindings={**support_bindings(_binding_geometry, ['region'], []), **{}})
    path = tmp_path / "quantity.eqi"
    source.write_eqi(path)
    emitted = eqiora.compile(path=path, geometry=(_binding_geometry := rectangle_geometry()), entry='RootLength', bindings={**support_bindings(_binding_geometry, ['region'], []), **{}})
    assert direct.digest == emitted.digest
    assert eqiora.Model.from_bytes(direct.to_bytes()).digest == direct.digest
    for unit in [u.kg, u.one, u.s.prefixed("m"), u.m / u.s]:
        with pytest.raises(ValueError):
            unit.prefixed("m")
    for prefix in ["MILLI", "µ", "", "kk"]:
        with pytest.raises(ValueError):
            u.s.prefixed(prefix)
    for value in [True, float("inf"), float("nan")]:
        with pytest.raises((TypeError, ValueError)):
            q.quantity(value, u.s)
    with pytest.raises(TypeError):
        q.quantity(1, "s")


def cylinder_source(
    *,
    doc: str = "Equations-only steady incompressible flow component.",
    velocity_type=eqiora.ValueType.vector(eqiora.ValueType.real(eqiora.Dimension(length=1, time=-1)), 2),
):
    source = eqiora.Module("main")
    stokes = source.component("SteadyFlowPastCylinder", doc=doc)
    fluid = stokes.volume("fluid", dimensions=2)
    inlet = stokes.boundary("inlet", parent=fluid)
    outlet = stokes.boundary("outlet", parent=fluid)
    walls = stokes.boundary("walls", parent=fluid)
    cylinder = stokes.boundary("cylinder", parent=fluid)

    dynamic_viscosity = stokes.parameter("dynamic_viscosity", value_type=eqiora.ValueType.real(eqiora.Dimension(mass=1, length=-1, time=-1)))
    zero_pressure = stokes.parameter("zero_pressure", value_type=eqiora.ValueType.real(eqiora.Dimension(mass=1, length=-1, time=-2)))
    inlet_speed = stokes.parameter("inlet_speed", value_type=eqiora.ValueType.real(eqiora.Dimension(length=1, time=-1)))
    channel_height = stokes.parameter("channel_height", value_type=eqiora.ValueType.real(eqiora.Dimension(length=1)))

    velocity = stokes.field("velocity", role=eqiora.FieldRole.Variable, on=fluid, value_type=velocity_type)
    pressure = stokes.field("pressure", role=eqiora.FieldRole.Variable, on=fluid, value_type=eqiora.ValueType.real(eqiora.Dimension(mass=1, length=-1, time=-2)))
    force_potential = stokes.field(
        "force_potential", role=eqiora.FieldRole.Variable, on=fluid, value_type=eqiora.ValueType.real(eqiora.Dimension(mass=1, length=-1, time=-2))
    )
    inlet_profile = stokes.field("inlet_profile", role=eqiora.FieldRole.Variable, on=fluid, value_type=eqiora.ValueType.real(eqiora.Dimension(length=1, time=-1)))

    stokes.relation("force_definition", q.equation(force_potential, zero_pressure), on=fluid)
    stokes.relation(
        "inlet_profile_definition",
        q.equation(
            inlet_profile,
            4
            * inlet_speed
            * q.coordinate(1)
            * (channel_height - q.coordinate(1))
            / channel_height**2,
        ),
        on=fluid,
    )
    stress = 2 * dynamic_viscosity * q.symmetric_part(
        q.grad(velocity)
    ) - q.isotropic_lift(pressure)
    stokes.relation("momentum", q.equation(-q.div(stress) - q.grad(force_potential), 0), on=fluid, doc="Steady Stokes momentum balance.")
    stokes.relation("incompressibility", q.equation(q.div(velocity), 0), on=fluid)
    stokes.relation("inlet_velocity", q.equation(q.trace(velocity) + q.normal(q.isotropic_lift(inlet_profile)), 0), on=inlet)
    stokes.relation("outlet_traction", q.equation(q.normal(stress), 0), on=outlet)
    stokes.relation("wall_velocity", q.equation(q.trace(velocity), 0), on=walls)
    stokes.relation("cylinder_velocity", q.equation(q.trace(velocity), 0), on=cylinder)
    return source


def test_relation_requires_ordered_left_and_right_including_zero() -> None:
    source = eqiora.Module("main")
    component = source.component("NaturalEquation")
    body = component.volume("body", dimensions=2)
    value = component.field("value", role=eqiora.FieldRole.Variable, on=body, value_type=eqiora.ValueType.real())
    source_scale = component.parameter("source_scale", value_type=eqiora.ValueType.real())

    natural = component.relation("natural", q.equation(q.div(q.grad(value)), -source_scale), on=body, doc="Natural equation.")
    residual = component.relation("residual", q.equation(value, 0), on=body)
    assert isinstance(natural, q.Relation)
    assert isinstance(residual, q.Relation)
    with pytest.raises(TypeError):
        q.Relation()
    with pytest.raises(AttributeError):
        natural.name = "other"

    text = source.to_eqi()
    assert "/// Natural equation." in text
    assert "div(grad(value)) = -source_scale;" in text
    assert "value = 0;" in text


def test_relation_rejects_mixed_incomplete_foreign_and_nonfinite_equations() -> None:
    source = eqiora.Module("main")
    component = source.component("Relations")
    body = component.volume("body", dimensions=2)
    value = component.field("value", role=eqiora.FieldRole.Variable, on=body, value_type=eqiora.ValueType.real())
    foreign = eqiora.Module("main")
    foreign_component = foreign.component("Foreign")
    foreign_body = foreign_component.volume("body", dimensions=2)
    foreign_value = foreign_component.field("value", role=eqiora.FieldRole.Variable, on=foreign_body, value_type=eqiora.ValueType.real())

    invalid = (
        lambda: component.relation("missing", on=body),
        lambda: component.relation("left_only", q.equation(value), on=body),
        lambda: component.relation("right_only", q.equation(rhs=value), on=body),
        lambda: component.relation("mixed", q.equation(value, 0), on=body, residual=value),
        lambda: component.relation("foreign", q.equation(value, foreign_value), on=body),
        lambda: component.relation("nonfinite", q.equation(value, float("nan")), on=body),
    )
    for operation in invalid:
        with pytest.raises((q.ModuleError, TypeError)):
            operation()


def cylinder_geometry():
    graph = eqiora.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0.0, 2.2), y_bounds=(0.0, 0.41))
    circle = graph.circle(center=(0.2, 0.2), radius=0.05)
    fluid = graph.subtract(rectangle, circle)
    geometry = graph.build(
        fluid,
        named_topology={
            "fluid": fluid.region,
            "inlet": rectangle.boundaries[0],
            "outlet": rectangle.boundaries[1],
            "walls": rectangle.boundaries[2:],
            "cylinder": circle.boundaries[0],
        },
    )
    return geometry


PARAMETERS = {
    "dynamic_viscosity": 0.001,
    "zero_pressure": 0.0,
    "inlet_speed": 0.3,
    "channel_height": 0.41,
}


def scalar_property_source(*, doc: str = "Reference scalar diffusivity release.", binding=None):
    source = eqiora.Module("main")
    contract = source.property_contract("Diffusivity", value_type=eqiora.ValueType.real())
    release = source.property_release(
        "ReferenceDiffusivity",
        implements=contract,
        value=25,
        source_unit=u.one,
        source_scale=0.001,
        citation="org.example.measurement",
        license="spdx.CC0_1_0",
        doc=doc,
    )

    law = source.component("PoissonLaw")
    law_region = law.volume("region", dimensions=2)
    law_left = law.boundary("left", parent=law_region)
    law_right = law.boundary("right", parent=law_region)
    law_bottom = law.boundary("bottom", parent=law_region)
    law_top = law.boundary("top", parent=law_region)
    law_source_scale = law.parameter("source_scale", value_type=eqiora.ValueType.real(eqiora.Dimension(length=-2)))
    diffusivity = law.property("diffusivity", contract=contract)
    potential = law.field("potential", role=eqiora.FieldRole.Variable, on=law_region, value_type=eqiora.ValueType.real())
    law.relation(
        "balance",
        q.equation(-q.div(diffusivity * q.grad(potential)) - law_source_scale, 0),
        on=law_region,
    )
    law.relation("left_value", q.equation(q.trace(potential), 0), on=law_left)
    law.relation("right_value", q.equation(q.trace(potential), 0), on=law_right)
    law.relation("bottom_value", q.equation(q.trace(potential), 0), on=law_bottom)
    law.relation("top_value", q.equation(q.trace(potential), 0), on=law_top)

    root = source.component("PoissonRectangle")
    root_region = root.volume("region", dimensions=2)
    root_left = root.boundary("left", parent=root_region)
    root_right = root.boundary("right", parent=root_region)
    root_bottom = root.boundary("bottom", parent=root_region)
    root_top = root.boundary("top", parent=root_region)
    root_source_scale = root.parameter("source_scale", value_type=eqiora.ValueType.real(eqiora.Dimension(length=-2)))
    root.instance('equation', component=law, bindings={'region': root_region, 'left': root_left, 'right': root_right, 'bottom': root_bottom, 'top': root_top, 'source_scale': root_source_scale if binding is None else binding, 'diffusivity': release})
    return source


def material_composition_source() -> eqiora.Module:
    source = eqiora.Module("main")
    conductivity_contract = source.property_contract("Conductivity", value_type=eqiora.ValueType.real())
    capacity_contract = source.property_contract("Capacity", value_type=eqiora.ValueType.real())
    conductivity_release = source.property_release(
        "ConductivityA",
        implements=conductivity_contract,
        value=2,
        source_unit=u.one,
        source_scale=1,
        citation="org.example.a",
        license="spdx.CC0_1_0",
    )
    capacity_release = source.property_release(
        "CapacityA",
        implements=capacity_contract,
        value=4,
        source_unit=u.one,
        source_scale=1,
        citation="org.example.a",
        license="spdx.CC0_1_0",
    )
    law = source.component("DiffusionLaw")
    region = law.volume("region", dimensions=2)
    conductivity = law.property("conductivity", contract=conductivity_contract)
    capacity = law.property("capacity", contract=capacity_contract)
    law.relation("law", q.equation(conductivity / capacity, 0), on=region)
    material = source.material_composition(
        "MaterialA",
        properties={
            "conductivity": conductivity_release,
            "capacity": capacity_release,
        },
    )
    root = source.component("UseMaterial")
    root_region = root.volume("region", dimensions=2)
    root.instance('law', component=law, bindings={'region': root_region, 'conductivity': material['conductivity'], 'capacity': material['capacity']})
    return source


def test_removed_source_choice_keywords_are_unexpected() -> None:
    source = eqiora.Module("main")
    with pytest.raises(TypeError, match="unexpected keyword argument 'public'"):
        source.component("Component", public=True)

    contract_source = eqiora.Module("main")
    with pytest.raises(TypeError, match="unexpected keyword argument 'public'"):
        contract_source.property_contract("Diffusivity", value_type=eqiora.ValueType.real(), public=True)
    contract = contract_source.property_contract("Diffusivity", value_type=eqiora.ValueType.real())
    release_arguments = {
        "implements": contract,
        "value": 25,
        "source_unit": u.one,
        "source_scale": 0.001,
        "citation": "org.example.measurement",
        "license": "spdx.CC0_1_0",
    }
    with pytest.raises(TypeError, match="unexpected keyword argument 'public'"):
        contract_source.property_release(
            "ReferenceDiffusivity", **release_arguments, public=True
        )
    with pytest.raises(TypeError, match="numeric literals must be finite int or float"):
        contract_source.property_release(
            "ReferenceDiffusivity",
            **release_arguments,
            validity="unconditional",
        )

    component = contract_source.component("Component")
    volume = component.volume("volume", dimensions=2)
    for operation in (
        lambda: component.volume("other_volume", dimensions=2, public=True),
        lambda: component.boundary("boundary", parent=volume, public=True),
        lambda: component.parameter("parameter", value_type=eqiora.ValueType.real(), public=True),
        lambda: component.property("property", contract=contract, public=True),
    ):
        with pytest.raises(TypeError, match="unexpected keyword argument 'public'"):
            operation()


def rectangle_geometry():
    graph = eqiora.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    return graph.build(
        rectangle,
        named_topology={
            "region": rectangle.region,
            "left": rectangle.boundaries[0],
            "right": rectangle.boundaries[1],
            "bottom": rectangle.boundaries[2],
            "top": rectangle.boundaries[3],
        },
    )


def scalar_primal_source():
    source = eqiora.Module("main")
    law = source.component("ScalarDiffusion")
    region = law.volume("region", dimensions=2)
    diffusion = law.parameter("diffusion", value_type=eqiora.ValueType.real())
    wave_number = law.parameter("wave_number", value_type=eqiora.ValueType.real(eqiora.Dimension(length=-1)))
    source_scale = law.parameter("source_scale", value_type=eqiora.ValueType.real(eqiora.Dimension(length=-2)))
    potential = law.field("potential", role=eqiora.FieldRole.Variable, on=region, value_type=eqiora.ValueType.real())
    balance = law.relation(
        "balance",
        q.equation(
            -q.div(diffusion * q.grad(potential))
            - source_scale * q.math.sin(q.math.pi * wave_number * q.coordinate(0)),
            0,
        ),
        on=region,
    )
    left_boundary = law.boundary("left", parent=region)
    w = law.test("w", for_=potential, zero_on=left_boundary)
    law.weak_form(
        "weak", [balance],
        equations=[(q.integrate(
            region,
            q.dot(q.grad(w), diffusion * q.grad(potential)),
        ),
        q.integrate(
            region,
            w
            * source_scale
            * q.math.sin(q.math.pi * wave_number * q.coordinate(0)),
        ),
        )],
        doc="Authored scalar primal form.",
    )
    return source


def test_python_source_emits_and_fresh_compile_inspects_scalar_primal_form(
    tmp_path: Path,
) -> None:
    source = scalar_primal_source()
    text = source.to_eqi()
    assert "form weak for balance" in text
    assert "/// Authored scalar primal form." in text
    assert text.count("math.pi") == 2
    assert text.count("math.sin") == 2
    assert not hasattr(q, "sin")

    model = eqiora.compile(source=source, geometry=(_binding_geometry := rectangle_geometry()), entry='ScalarDiffusion', bindings={**support_bindings(_binding_geometry, ['region'], [('left', 'region')]), **{'diffusion': 1.0, 'wave_number': 2.0, 'source_scale': 2.0}})
    assert len(model.authored_formulations) == 1
    form = model.authored_formulations[0]
    assert form.kind == "primal"
    assert len(form.source_identity) == 64
    assert form.filename == "<module>"
    assert form.trial_field_ids[0] in model.field_ids

    path = tmp_path / "scalar-primal.eqi"
    source.write_eqi(path)
    emitted = eqiora.compile(path=path, geometry=(_binding_geometry := rectangle_geometry()), entry='ScalarDiffusion', bindings={**support_bindings(_binding_geometry, ['region'], [('left', 'region')]), **{'diffusion': 1.0, 'wave_number': 2.0, 'source_scale': 2.0}})
    assert emitted.digest == model.digest

    replayed = eqiora.Model.from_bytes(model.to_bytes())
    assert replayed.authored_formulations == ()


def test_uninitialized_scalar_field_compiles_from_source_and_emitted_file(
    tmp_path: Path,
) -> None:
    source = eqiora.Module("main")
    law = source.component("AlgebraicField")
    region = law.volume("region", dimensions=2)
    potential = law.field("potential", role=eqiora.FieldRole.Variable, on=region, value_type=eqiora.ValueType.real())
    law.relation("balance", q.equation(potential, 0), on=region)

    text = source.to_eqi()
    assert "variable potential: 1 on region;" in text
    assert "variable potential: 1 =" not in text

    direct = eqiora.compile(source=source, geometry=(_binding_geometry := rectangle_geometry()), entry='AlgebraicField', bindings={**support_bindings(_binding_geometry, ['region'], []), **{}})
    path = tmp_path / "uninitialized-scalar.eqi"
    source.write_eqi(path)
    emitted = eqiora.compile(path=path, geometry=(_binding_geometry := rectangle_geometry()), entry='AlgebraicField', bindings={**support_bindings(_binding_geometry, ['region'], []), **{}})
    assert direct.digest == emitted.digest


def test_source_is_deterministic_and_direct_file_compilation_has_one_identity(
    tmp_path: Path,
) -> None:
    first = cylinder_source()
    second = cylinder_source()
    assert first.to_eqi() == second.to_eqi()
    assert "/// Equations-only steady incompressible flow component." in first.to_eqi()

    geometry = cylinder_geometry()
    direct = eqiora.compile(source=first, geometry=geometry, entry='SteadyFlowPastCylinder', bindings={**support_bindings(geometry, ['fluid'], [('inlet', 'fluid'), ('outlet', 'fluid'), ('walls', 'fluid'), ('cylinder', 'fluid')]), **PARAMETERS})
    path = tmp_path / "steady-flow-past-cylinder.eqi"
    first.write_eqi(path)
    assert path.read_text(encoding="utf-8") == first.to_eqi()
    emitted = eqiora.compile(path=path, geometry=geometry, entry='SteadyFlowPastCylinder', bindings={**support_bindings(geometry, ['fluid'], [('inlet', 'fluid'), ('outlet', 'fluid'), ('walls', 'fluid'), ('cylinder', 'fluid')]), **PARAMETERS})
    other_comments = eqiora.compile(source=cylinder_source(doc='Different presentation-only documentation.'), geometry=geometry, entry='SteadyFlowPastCylinder', bindings={**support_bindings(geometry, ['fluid'], [('inlet', 'fluid'), ('outlet', 'fluid'), ('walls', 'fluid'), ('cylinder', 'fluid')]), **PARAMETERS})
    assert direct.digest == emitted.digest == other_comments.digest


def test_scalar_property_source_compiles_with_same_source_release(
    tmp_path: Path,
) -> None:
    first = scalar_property_source()
    second = scalar_property_source()
    assert first.to_eqi() == second.to_eqi()
    assert "public property contract Diffusivity" in first.to_eqi()
    assert "diffusivity = ReferenceDiffusivity" in first.to_eqi()

    geometry = rectangle_geometry()
    bindings = {
        **support_bindings(geometry, ["region"], [(side, "region") for side in ("left", "right", "bottom", "top")]),
        "source_scale": 1.0,
    }
    direct = eqiora.compile(source=first, geometry=geometry, entry="PoissonRectangle", bindings=bindings)
    path = tmp_path / "property-poisson.eqi"
    first.write_eqi(path)
    assert path.read_text(encoding="utf-8") == first.to_eqi()
    emitted = eqiora.compile(path=path, geometry=geometry, entry="PoissonRectangle", bindings=bindings)
    assert direct.digest == emitted.digest
    assert scalar_property_source(doc="Different release documentation.").to_eqi().replace(
        "/// Different release documentation.\n", ""
    ) == first.to_eqi().replace("/// Reference scalar diffusivity release.\n", "")


def test_material_composition_emits_one_ordered_typed_binding_set() -> None:
    text = material_composition_source().to_eqi()
    assert "public material composition MaterialA" in text
    assert text.index("property capacity = CapacityA") < text.index(
        "property conductivity = ConductivityA"
    )
    assert "conductivity = MaterialA.conductivity" in text
    assert "capacity = MaterialA.capacity" in text
    with pytest.raises(TypeError):
        q.MaterialComposition()


def test_scalar_property_source_owns_exact_handles_and_complete_binding() -> None:
    source = eqiora.Module("main")
    contract = source.property_contract("Diffusivity", value_type=eqiora.ValueType.real())
    with pytest.raises(TypeError):
        q.PropertyContract()
    with pytest.raises(TypeError):
        q.PropertyRelease()
    with pytest.raises(q.ModuleError, match="strictly positive"):
        source.property_release(
            "ReferenceDiffusivity",
            implements=contract,
            value=25,
            source_unit=u.one,
            source_scale=0,
            citation="org.example.measurement",
            license="spdx.CC0_1_0",
        )
    with pytest.raises(q.ModuleError, match="citation identity"):
        source.property_release(
            "ReferenceDiffusivity",
            implements=contract,
            value=25,
            source_unit=u.one,
            source_scale=0.001,
            citation="not/a/name/path",
            license="spdx.CC0_1_0",
        )
    release = source.property_release(
        "ReferenceDiffusivity",
        implements=contract,
        value=25,
        source_unit=u.one,
        source_scale=0.001,
        citation="org.example.measurement",
        license="spdx.CC0_1_0",
    )
    with pytest.raises(AttributeError):
        contract.name = "Other"
    with pytest.raises(AttributeError):
        release.value = 1

    foreign = eqiora.Module("main")
    foreign_contract = foreign.property_contract("Diffusivity", value_type=eqiora.ValueType.real())
    foreign_release = foreign.property_release(
        "ReferenceDiffusivity", implements=foreign_contract, value=25,
        source_unit=u.one, source_scale=0.001,
        citation="org.example.measurement", license="spdx.CC0_1_0",
    )
    foreign_component = foreign.component("Foreign")
    with pytest.raises(q.ModuleError, match="belong to this Module"):
        foreign_component.property("diffusivity", contract=contract)

    consumer = source.component("Consumer")
    requirement = consumer.property("diffusivity", contract=contract)
    root = source.component("Root")
    with pytest.raises(q.ModuleError, match="bindings must satisfy the exact required signature"):
        root.instance('equation', component=consumer, bindings={})
    with pytest.raises(q.ModuleError, match="Module|contract"):
        root.instance("equation", component=consumer, bindings={'diffusivity': foreign_release})
    root.instance('equation', component=consumer, bindings={'diffusivity': release})
    assert "diffusivity = ReferenceDiffusivity" in source.to_eqi()
    with pytest.raises(q.ModuleError):
        source.component("Third")


def test_source_owns_handles_limits_and_atomic_output(tmp_path: Path) -> None:
    left = eqiora.Module("main")
    left_component = left.component("Left")
    left_volume = left_component.volume("left", dimensions=2)
    left_value = left_component.field("value", role=eqiora.FieldRole.Variable, on=left_volume, value_type=eqiora.ValueType.real(eqiora.Dimension(length=1)))
    right = eqiora.Module("main")
    right_component = right.component("Right")
    right_volume = right_component.volume("right", dimensions=2)
    right_value = right_component.field("value", role=eqiora.FieldRole.Variable, on=right_volume, value_type=eqiora.ValueType.real(eqiora.Dimension(length=1)))

    invalid = (
        lambda: left_value + right_value,
        lambda: left_component.boundary("foreign_parent", parent=right_volume),
        lambda: left_component.field("wrong_support", role=eqiora.FieldRole.Variable, on=right_volume, value_type=eqiora.ValueType.real(eqiora.Dimension(length=1))),
        lambda: left_component.field("value", role=eqiora.FieldRole.Variable, on=left_volume, value_type=eqiora.ValueType.real(eqiora.Dimension(length=1))),
        lambda: left_value + float("nan"),
    )
    for operation in invalid:
        with pytest.raises(q.ModuleError):
            operation()

    assert isinstance(q.math.pi, q.Expression)
    assert q.math.pi is q.math.pi
    with pytest.raises(TypeError):
        float(q.math.pi)
    with pytest.raises(AttributeError):
        q.math.pi = left_value
    with pytest.raises(q.ModuleError, match="different Module"):
        q.math.pi + left_value + right_value

    with pytest.raises(q.ModuleError):
        eqiora.Module("main").component("not-valid")
    with pytest.raises(q.ModuleError):
        eqiora.Module("main").component("a" * 1025)
    with pytest.raises(q.ModuleError):
        left_value + (1 << 1025)

    deep = q.math.pi
    with pytest.raises(q.ModuleError):
        for _ in range(100):
            deep = q.grad(deep)

    wide = q.coordinate(0)
    with pytest.raises(q.ModuleError):
        for _ in range(20):
            wide = wide + wide

    bounded = eqiora.Module("main").component("Bounded")
    bounded_volume = bounded.volume("volume", dimensions=2)
    with pytest.raises(q.ModuleError):
        for index in range(300):
            bounded.field(f"value_{index}", role=eqiora.FieldRole.Variable, on=bounded_volume, value_type=eqiora.ValueType.real(eqiora.Dimension(length=1)))

    target = tmp_path / "target.eqi"
    target.write_text("preserved", encoding="utf-8")
    link = tmp_path / "link.eqi"
    link.symlink_to(target)
    with pytest.raises(ValueError):
        cylinder_source().write_eqi(link)
    assert target.read_text(encoding="utf-8") == "preserved"


def test_canonical_compiler_owns_expression_shape_diagnostics() -> None:
    with pytest.raises(eqiora.ValidationError) as error:
        eqiora.compile(source=cylinder_source(velocity_type=eqiora.ValueType.real(eqiora.Dimension(length=1, time=-1))), geometry=(_binding_geometry := cylinder_geometry()), entry='SteadyFlowPastCylinder', bindings={**support_bindings(_binding_geometry, ['fluid'], [('inlet', 'fluid'), ('outlet', 'fluid'), ('walls', 'fluid'), ('cylinder', 'fluid')]), **PARAMETERS})
    assert error.value.diagnostics
    assert any(
        diagnostic.source_span is None
        and diagnostic.graph_path is not None
        for diagnostic in error.value.diagnostics
    )


def test_static_alias_authoring_emits_private_typed_immutable_expressions():
    source = eqiora.Module("main")
    component = source.component("Aliases")
    pressure = eqiora.ValueType.real(eqiora.Dimension(mass=1, length=-1, time=-2))
    supplied = component.parameter("supplied", value_type=pressure)
    doubled = component.let_alias("doubled", supplied * 2, value_type=pressure,
                                  doc="Twice the supplied pressure.")
    component.let_alias("opposite", -doubled)
    assert type(doubled) is q.Expression
    with pytest.raises(AttributeError):
        doubled.name = "changed"
    text = source.to_eqi()
    assert f"/// Twice the supplied pressure.\n  let doubled: {pressure.to_eqi()} = supplied * 2;" in text
    assert "let opposite = -doubled;" in text
    assert "public parameter doubled" not in text
    with pytest.raises(q.ModuleError, match="frozen"):
        component.let_alias("late", 1)


def test_static_alias_authoring_rejects_foreign_values_and_invalid_assertions():
    component = eqiora.Module("main").component("Owner")
    foreign = eqiora.Module("main").component("Foreign").parameter("input", value_type=eqiora.ValueType.real())
    with pytest.raises(q.ModuleError, match="this Component"):
        component.let_alias("foreign", foreign + 1)
    with pytest.raises(TypeError, match="eqiora.ValueType"):
        component.let_alias("invalid", 1, value_type=u.one)
    with pytest.raises(q.ModuleError, match="finite"):
        component.let_alias("invalid", float("nan"))
    component.let_alias("invalid", 1)
    with pytest.raises(q.ModuleError, match="duplicate"):
        component.parameter("invalid", value_type=eqiora.ValueType.real())


def test_static_alias_authoring_bounds_total_expression_nodes():
    component = eqiora.Module("main").component("BoundedAliases")
    expression = q.math.pi
    for _ in range(11):
        expression = expression + expression
    component.let_alias("large", expression)
    component.let_alias("last", 0)
    with pytest.raises(q.ModuleError, match="alias expressions exceed the 4096-node limit"):
        component.let_alias("overflow", 0)


def test_static_alias_compiles_without_required_binding_or_edit_target(tmp_path):
    source = eqiora.Module("main")
    component = source.component("StaticAlias")
    region = component.volume("region", dimensions=2)
    supplied = component.parameter("supplied", value_type=eqiora.ValueType.real())
    doubled = component.let_alias("doubled", supplied * 2)
    value = component.field("value", on=region, role=eqiora.FieldRole.Variable,
                            value_type=eqiora.ValueType.real())
    component.relation("balance", q.equation(value, doubled), on=region)
    model = eqiora.compile(source=source, geometry=(_binding_geometry := rectangle_geometry()), entry='StaticAlias', bindings={**support_bindings(_binding_geometry, ['region'], []), **{'supplied': 3.0}})
    assert len(model.parameter_ids) == 1
    assert len(model.field_ids) == 1
    with pytest.raises(eqiora.EqioraError, match="Parameter"):
        model.preview_value_edit("doubled", 9.0)
    path = tmp_path / "static-alias.eqi"
    source.write_eqi(path)
    replay = eqiora.compile(path=path, geometry=(_binding_geometry := rectangle_geometry()), entry='StaticAlias', bindings={**support_bindings(_binding_geometry, ['region'], []), **{'supplied': 3.0}})
    assert replay.structural_fingerprint == model.structural_fingerprint


def test_static_alias_nested_parameter_binding_uses_expression_rhs():
    source = eqiora.Module("main")
    child = source.component("Child")
    child_region = child.volume("region", dimensions=2)
    required = child.parameter("source_scale", value_type=eqiora.ValueType.real())
    private = child.let_alias("child_source", required * 3)
    value = child.field("value", on=child_region, role=eqiora.FieldRole.Variable,
                        value_type=eqiora.ValueType.real())
    child.relation("balance", q.equation(value, private), on=child_region)
    parent = source.component("Parent")
    parent_region = parent.volume("region", dimensions=2)
    supplied = parent.parameter("source_scale", value_type=eqiora.ValueType.real())
    adjusted = parent.let_alias("adjusted_source", supplied * 2)
    for parameters in ({}, {'source_scale': adjusted, 'child_source': 1}):
        with pytest.raises(q.ModuleError, match="bindings must satisfy the exact required signature"):
            parent.instance('child', component=child, bindings={'region': parent_region, **parameters})
    parent.instance('child', component=child, bindings={'region': parent_region, 'source_scale': adjusted})
    text = source.to_eqi()
    assert "let adjusted_source = source_scale * 2;" in text
    assert "source_scale = adjusted_source" in text
    model = eqiora.compile(source=source, geometry=(_binding_geometry := rectangle_geometry()), entry='Parent', bindings={**support_bindings(_binding_geometry, ['region'], []), **{'source_scale': 3.0}})
    for name in ("adjusted_source", "child.child_source"):
        with pytest.raises(eqiora.EqioraError, match="Parameter"):
            model.preview_value_edit(name, 9.0)


def test_static_alias_authoring_cannot_bind_private_alias_as_parameter():
    source = eqiora.Module("main")
    child = source.component("Child")
    required = child.parameter("required", value_type=eqiora.ValueType.real())
    private = child.let_alias("private", required * 2)
    root = source.component("Root")
    rhs = root.let_alias("rhs", 3)
    with pytest.raises(q.ModuleError, match="bindings must satisfy the exact required signature"):
        root.instance('child', component=child, bindings={'required': rhs, 'private': 1})
    root.instance('child', component=child, bindings={'required': rhs})


@pytest.mark.parametrize("kind", ["parameter", "alias", "compound", "trace", "property"])
def test_static_alias_authoring_rejects_same_source_sibling_capture(kind):
    source = eqiora.Module("main")
    contract = source.property_contract("Scalar", value_type=eqiora.ValueType.real())
    release = source.property_release(
        "Unit", implements=contract, value=1, source_unit=u.one, source_scale=1,
        citation="org.example.unit", license="spdx.CC0_1_0",
    )
    left = source.component("Left")
    left_region = left.volume("region", dimensions=2)
    left_parameter = left.parameter("supplied", value_type=eqiora.ValueType.real())
    left_alias = left.let_alias("derived", left_parameter * 2)
    left_field = left.field("value", on=left_region, role=eqiora.FieldRole.Variable,
                           value_type=eqiora.ValueType.real())
    left_property = left.property("coefficient", contract=contract)
    right = source.component("Right")
    right_region = right.volume("region", dimensions=2)
    right_parameter = right.parameter("supplied", value_type=eqiora.ValueType.real())
    right.let_alias("derived", right_parameter * 3)
    right.property("coefficient", contract=contract)
    right_relation = right.relation("own", q.equation(right_parameter, 0), on=right_region)
    foreign = {
        "parameter": left_parameter,
        "alias": left_alias,
        "compound": -(left_alias + 1) * 2,
        "trace": q.trace(left_field),
        "property": left_property,
    }[kind]
    for operation in (
        lambda: right.let_alias("captured", foreign),
        lambda: right.relation("captured", q.equation(foreign, 0), on=right_region),
        lambda: q.integrate(right_region, foreign),
        lambda: foreign + right_parameter,
        lambda: right.weak_form("weak", [right_relation], equations=[(q.integrate(left_region, foreign),
                                  q.integrate(right_region, right_parameter))]),
        lambda: right.instance('child', component=left, bindings={'region': right_region, 'supplied': foreign, 'coefficient': release}),
    ):
        with pytest.raises(q.ModuleError, match="Component"):
            operation()


def test_component_hierarchy_uses_shared_top_level_bound_and_freezes():
    source = eqiora.Module("main")
    for index in range(256):
        source.component(f"Component{index}")
    with pytest.raises(q.ModuleError, match="256-declaration limit"):
        source.component("Overflow")
    assert source.to_eqi().count("public component ") == 256
    with pytest.raises(q.ModuleError, match="frozen"):
        source.component("Late")


def test_component_hierarchy_retains_total_output_byte_bound():
    source = eqiora.Module("main")
    doc = "x" * 16_384
    for index in range(130):
        component = source.component(f"Component{index}", doc=doc)
        for alias_index in range(3):
            component.let_alias(f"alias{alias_index}", 1, doc=doc)
    with pytest.raises(q.ModuleError, match="8388608-byte limit"):
        source.to_eqi()
    source.component("StillOpenAfterRejectedEmission")


def test_property_hierarchy_allows_unselected_local_definitions():
    source = scalar_property_source()
    source.component("Additional")
    geometry = rectangle_geometry()
    model = eqiora.compile(source=source, geometry=geometry, entry="PoissonRectangle", bindings={
        **support_bindings(geometry, ["region"], [(side, "region") for side in ("left", "right", "bottom", "top")]),
        "source_scale": 1.0,
    })
    assert model.digest


def runtime_arithmetic_alias_source(*, aliases=True):
    source = eqiora.Module("main")
    component = source.component("RuntimeArithmetic")
    region = component.volume("region", dimensions=2)
    supplied = component.parameter("supplied", value_type=eqiora.ValueType.real())
    value = component.field("value", on=region, role=eqiora.FieldRole.Variable,
                            value_type=eqiora.ValueType.real())
    doubled = component.let_alias("doubled", value * 2) if aliases else value * 2
    shifted = component.let_alias("shifted", doubled + supplied) if aliases else doubled + supplied
    component.relation("balance", q.equation(shifted, 0), on=region)
    return source


def runtime_heatflux_alias_source(*, aliases=True, assert_support=False):
    source = eqiora.Module("main")
    component = source.component("RuntimeHeatFlux")
    region = component.volume("region", dimensions=2)
    left = component.boundary("left", parent=region)
    coefficient = component.parameter("coefficient", value_type=eqiora.ValueType.real())
    forcing = component.parameter("forcing", value_type=eqiora.ValueType.real(eqiora.Dimension(length=-2)))
    potential = component.field("potential", on=region, role=eqiora.FieldRole.Variable,
                                value_type=eqiora.ValueType.real())
    flux = coefficient * q.grad(potential)
    if aliases:
        flux = component.let_alias("heatflux", flux, on=region if assert_support else None,
                                   value_type=eqiora.ValueType.vector(
                                       eqiora.ValueType.real(eqiora.Dimension(length=-1)), 2))
    component.relation("balance", q.equation(-q.div(flux) - forcing, 0), on=region)
    component.relation("left_value", q.equation(q.trace(potential), 0), on=left)
    return source


def test_runtime_alias_authoring_preserves_field_and_gradient_expressions():
    arithmetic = runtime_arithmetic_alias_source().to_eqi()
    assert "let doubled = value * 2;" in arithmetic
    assert "let shifted = doubled + supplied;" in arithmetic
    assert "shifted = 0;" in arithmetic
    heatflux = runtime_heatflux_alias_source().to_eqi()
    assert "= coefficient * grad(potential);" in heatflux
    assert "-div(heatflux) - forcing = 0;" in heatflux
    assert "trace(potential) = 0;" in heatflux
    assert "let trace" not in heatflux
    asserted = runtime_heatflux_alias_source(assert_support=True).to_eqi()
    assert " on region = coefficient * grad(potential);" in asserted
    assert "-div(heatflux) - forcing = 0;" in asserted


@pytest.mark.parametrize("factory, parameters, alias_names, entry, boundaries", [
    (runtime_arithmetic_alias_source, {"supplied": 3.0}, ("doubled", "shifted"), "RuntimeArithmetic", []),
    (runtime_heatflux_alias_source, {"coefficient": 2.0, "forcing": 3.0}, ("heatflux",), "RuntimeHeatFlux", [("left", "region")]),
])
def test_runtime_aliases_compile_like_expanded_expressions_without_storage(
    tmp_path, factory, parameters, alias_names, entry, boundaries,
):
    source = factory()
    model = eqiora.compile(source=source, geometry=(_binding_geometry := rectangle_geometry()), entry=entry, bindings={**support_bindings(_binding_geometry, ['region'], boundaries), **parameters})
    path = tmp_path / "runtime-alias.eqi"
    source.write_eqi(path)
    from_file = eqiora.compile(path=path, geometry=(_binding_geometry := rectangle_geometry()), entry=entry, bindings={**support_bindings(_binding_geometry, ['region'], boundaries), **parameters})
    expanded = eqiora.compile(source=factory(aliases=False), geometry=(_binding_geometry := rectangle_geometry()), entry=entry, bindings={**support_bindings(_binding_geometry, ['region'], boundaries), **parameters})
    assert model.structural_fingerprint == from_file.structural_fingerprint
    assert model.structural_fingerprint == expanded.structural_fingerprint
    assert len(model.field_ids) == len(expanded.field_ids) == 1
    assert len(model.parameter_ids) == len(expanded.parameter_ids) == len(parameters)
    for name in alias_names:
        with pytest.raises(eqiora.EqioraError, match="Parameter"):
            model.preview_value_edit(name, 9.0)


def test_transitive_runtime_alias_cannot_supply_child_parameter_binding():
    source = eqiora.Module("main")
    child = source.component("Child")
    child_region = child.volume("region", dimensions=2)
    required = child.parameter("required", value_type=eqiora.ValueType.real())
    output = child.field("output", on=child_region, role=eqiora.FieldRole.Variable,
                         value_type=eqiora.ValueType.real())
    child.relation("balance", q.equation(output - required, 0), on=child_region)
    parent = source.component("Parent")
    region = parent.volume("region", dimensions=2)
    value = parent.field("value", on=region, role=eqiora.FieldRole.Variable,
                         value_type=eqiora.ValueType.real())
    first = parent.let_alias("first", value * 2)
    transitive = parent.let_alias("transitive", first + 1)
    parent.relation("balance", q.equation(value - 1, 0), on=region)
    parent.instance('child', component=child, bindings={'region': region, 'required': transitive})
    with pytest.raises(eqiora.ValidationError, match="static|runtime|[Pp]arameter"):
        eqiora.compile(source=source, geometry=(_binding_geometry := rectangle_geometry()), entry='Parent', bindings={**support_bindings(_binding_geometry, ['region'], []), **{}})


def test_alias_support_authoring_preserves_type_documentation_and_freeze():
    source = eqiora.Module("main")
    component = source.component("Supported")
    region = component.volume("region", dimensions=2)
    value = component.field("value", on=region, role=eqiora.FieldRole.Variable,
                            value_type=eqiora.ValueType.real())
    kind = eqiora.ValueType.real()
    alias = component.let_alias("local", value * 2, value_type=kind, on=region,
                                doc="Exact nominal support.")
    component.let_alias("untyped", alias + value, on=region)
    text = source.to_eqi()
    assert f"  /// Exact nominal support.\n  let local: {kind.to_eqi()} on region = value * 2;" in text
    assert "let untyped on region = local + value;" in text
    with pytest.raises(AttributeError):
        alias._ast = None
    with pytest.raises(q.ModuleError, match="frozen"):
        component.let_alias("late", value, on=region)


@pytest.mark.parametrize("foreign_kind", ["sibling", "source", "invalid"])
def test_alias_support_authoring_rejects_foreign_support_before_mutation(foreign_kind):
    source = eqiora.Module("main")
    component = source.component("Owner")
    region = component.volume("region", dimensions=2)
    value = component.field("value", on=region, role=eqiora.FieldRole.Variable,
                            value_type=eqiora.ValueType.real())
    other_source = eqiora.Module("main") if foreign_kind == "source" else source
    other = other_source.component("Other")
    foreign = other.volume("region", dimensions=2) if foreign_kind != "invalid" else "region"
    with pytest.raises(q.ModuleError, match="Component"):
        component.let_alias("local", value, on=foreign)
    component.let_alias("local", value, on=region)
    assert source.to_eqi().count("let local on region = value;") == 1


def test_alias_support_heatflux_compiles_like_expanded_expression(tmp_path):
    source = runtime_heatflux_alias_source(assert_support=True)
    parameters = {"coefficient": 2.0, "forcing": 3.0}
    model = eqiora.compile(source=source, geometry=(_binding_geometry := rectangle_geometry()), entry='RuntimeHeatFlux', bindings={**support_bindings(_binding_geometry, ['region'], [('left', 'region')]), **parameters})
    path = tmp_path / "supported-heatflux.eqi"
    source.write_eqi(path)
    from_file = eqiora.compile(path=path, geometry=(_binding_geometry := rectangle_geometry()), entry='RuntimeHeatFlux', bindings={**support_bindings(_binding_geometry, ['region'], [('left', 'region')]), **parameters})
    expanded = eqiora.compile(source=runtime_heatflux_alias_source(aliases=False), geometry=(_binding_geometry := rectangle_geometry()), entry='RuntimeHeatFlux', bindings={**support_bindings(_binding_geometry, ['region'], [('left', 'region')]), **parameters})
    assert model.structural_fingerprint == from_file.structural_fingerprint
    assert model.structural_fingerprint == expanded.structural_fingerprint
    assert len(model.field_ids) == len(expanded.field_ids) == 1
    assert len(model.parameter_ids) == len(expanded.parameter_ids) == 2
    with pytest.raises(eqiora.EqioraError, match="Parameter"):
        model.preview_value_edit("heatflux", 9.0)


@pytest.mark.parametrize("kind", ["wrong_region", "constant", "boundary_trace"])
def test_alias_support_assertion_rejects_inference_or_context_changes(kind):
    source = eqiora.Module("main")
    component = source.component("InvalidSupport")
    region = component.volume("region", dimensions=2)
    value = component.field("value", on=region, role=eqiora.FieldRole.Variable,
                            value_type=eqiora.ValueType.real())
    support = region
    expression = value * 2
    geometry = rectangle_geometry()
    entry = "InvalidSupport"
    if kind == "wrong_region":
        support = component.volume("other", dimensions=2)
        # All definitions must be checked, including unused Components whose
        # abstract supports do not need bindings in the selected Geometry.
        entry = "Selected"
        selected = source.component(entry)
        selected_region = selected.volume("region", dimensions=2)
        selected_value = selected.field("value", on=selected_region,
                                        role=eqiora.FieldRole.Variable,
                                        value_type=eqiora.ValueType.real())
        selected.relation("balance", q.equation(selected_value, 0), on=selected_region)
    elif kind == "constant":
        expression = 2
    else:
        support = component.boundary("left", parent=region)
        component.boundary("right", parent=region)
        expression = q.trace(value)
    component.let_alias("invalid", expression, on=support)
    component.relation("balance", q.equation(value, 0), on=region)
    with pytest.raises(eqiora.ValidationError, match="support|scope|context|trace"):
        eqiora.compile(source=source, geometry=geometry, entry=entry, bindings={**support_bindings(geometry, ['region'], [('left', 'region'), ('right', 'region')] if kind == 'boundary_trace' else []), **{}})


def clocked_alias_source(*, aliases=True, wrong_clock=False):
    source = eqiora.Module("main")
    component = source.component("Clocked")
    region = component.volume("region", dimensions=2)
    tick = component.clock("tick", period_s=Fraction(1, 10), doc="Exact sampling clock.")
    asserted = component.clock("other", period_s=Fraction(1, 10)) if wrong_clock else tick
    state = component.field("memory", on=region, at=tick, role=eqiora.FieldRole.State,
                            value_type=eqiora.ValueType.real())
    observer = component.field("observer", on=region, role=eqiora.FieldRole.Variable,
                               value_type=eqiora.ValueType.real())
    component.initial((q.pre(state) - 1, 0), doc="Fresh pre-tick memory.")
    component.relation("hold", q.equation(q.next(state), q.pre(state)), on=region, at=tick)
    current = component.let_alias("current", 2 * state, on=region, at=asserted,
                                  value_type=eqiora.ValueType.real()) if aliases else 2 * state
    component.relation("observe", q.equation(observer, current), on=region)
    return source


def test_clock_authoring_emits_exact_activation_and_simultaneous_initial():
    text = clocked_alias_source().to_eqi()
    assert "/// Exact sampling clock.\n  clock tick = periodic(1 [s] / 10 [1], phase = 0 [s] / 1 [1]);" in text
    assert "state memory: 1 on region at tick;" in text
    assert "let current: 1 on region at tick = 2 * memory;" in text
    assert "/// Fresh pre-tick memory.\n  initial {\n    pre(memory) - 1 = 0;\n  }" in text
    assert "relation hold on region at tick {\n    next(memory) = pre(memory);" in text
    assert "relation observe on region {" in text
    simultaneous = eqiora.Module("main")
    component = simultaneous.component("Simultaneous")
    component.initial((1, 0), (2, 0))
    assert "initial {\n    1 = 0;\n    2 = 0;\n  }" in simultaneous.to_eqi()


def test_clock_authoring_normalizes_rationals_and_preserves_nominal_immutability():
    source = eqiora.Module("main")
    component = source.component("Clocks")
    tick = component.clock("tick", period_s=Fraction(6, 8), phase_s=Fraction(2, 8))
    equal = component.clock("equal", period_s=Fraction(3, 4), phase_s=Fraction(1, 4))
    component.clock("maximum", period_s=(1 << 64) - 1, phase_s=Fraction(1, (1 << 64) - 1))
    assert tick is not equal
    with pytest.raises(TypeError, match="Component.clock"):
        q.Clock()
    with pytest.raises(AttributeError, match="immutable"):
        tick._name = "changed"
    assert source.to_eqi().count("periodic(3 [s] / 4 [1], phase = 1 [s] / 4 [1]") == 2
    with pytest.raises(q.ModuleError, match="frozen"):
        component.clock("late", period_s=1)
    with pytest.raises(q.ModuleError, match="frozen"):
        component.initial((0, 0))


@pytest.mark.parametrize("keyword, value, error", [
    ("period_s", 0, q.ModuleError), ("period_s", -1, q.ModuleError),
    ("phase_s", -1, q.ModuleError), ("period_s", 0.1, TypeError),
    ("phase_s", 0.0, TypeError), ("period_s", True, TypeError),
    ("phase_s", False, TypeError), ("period_s", "1/10", TypeError),
    ("period_s", 1 << 64, q.ModuleError),
    ("phase_s", Fraction(1, 1 << 64), q.ModuleError),
])
def test_clock_authoring_rejects_invalid_seconds_before_reserving_name(keyword, value, error):
    component = eqiora.Module("main").component("Clocks")
    arguments = {"period_s": 1, keyword: value}
    with pytest.raises(error):
        component.clock("tick", **arguments)
    component.clock("tick", period_s=1)


@pytest.mark.parametrize("consumer", ["field", "relation", "alias"])
@pytest.mark.parametrize("owner", ["sibling", "source", "string"])
def test_clock_authoring_rejects_foreign_activation_before_mutation(consumer, owner):
    source = eqiora.Module("main")
    component = source.component("Owner")
    region = component.volume("region", dimensions=2)
    tick = component.clock("tick", period_s=1)
    other = (eqiora.Module("main") if owner == "source" else source).component("Other")
    foreign = "tick" if owner == "string" else other.clock("tick", period_s=1)

    def declare(clock):
        if consumer == "field":
            component.field("candidate", on=region, at=clock, role=eqiora.FieldRole.State,
                            value_type=eqiora.ValueType.real())
        elif consumer == "relation":
            component.relation("candidate", q.equation(0, 0), on=region, at=clock)
        else:
            component.let_alias("candidate", 0, at=clock)

    with pytest.raises(q.ModuleError, match="clock must belong to this Component"):
        declare(foreign)
    declare(tick)


def test_clock_authoring_initial_and_tick_operators_preserve_lexical_expression_ownership():
    source = eqiora.Module("main")
    owner = source.component("Owner")
    sibling = source.component("Sibling")
    region = sibling.volume("region", dimensions=2)
    tick = sibling.clock("tick", period_s=1)
    state = sibling.field("memory", on=region, at=tick, role=eqiora.FieldRole.State,
                          value_type=eqiora.ValueType.real())
    for expression in (q.pre(state), q.next(state) + 1):
        with pytest.raises(q.ModuleError, match="this Component"):
            owner.initial((expression, 0))
        with pytest.raises(q.ModuleError, match="this Component"):
            owner.let_alias("foreign", expression)
    owner.initial((0, 0))


def test_clock_authoring_initial_uses_existing_declaration_and_expression_bounds():
    source = eqiora.Module("main")
    component = source.component("Bounded")
    component.clock("tick", period_s=1)
    for _ in range(255):
        component.initial((0, 0))
    with pytest.raises(q.ModuleError, match="256-declaration limit"):
        component.initial((0, 0))
    with pytest.raises(q.ModuleError, match="256-declaration limit"):
        component.clock("excess", period_s=1)
    bounded_source = eqiora.Module("main")
    bounded = bounded_source.component("Expressions")
    expression = q.math.pi
    for _ in range(11):
        expression = expression + expression
    explicit_source = eqiora.Module("main")
    explicit = explicit_source.component("ExplicitExpressions")
    # Both authored equality sides count, including an explicitly supplied zero.
    explicit.initial(left=expression, right=0)
    with pytest.raises(q.ModuleError, match="initial expressions exceed the 4096-node limit"):
        explicit.initial(left=0, right=0)
    assert explicit_source.to_eqi().count("initial {") == 1
    bounded.initial((expression, 0))
    with pytest.raises(q.ModuleError, match="initial expressions exceed the 4096-node limit"):
        bounded.initial((0, 0))
    # A rejected equation batch cannot consume a declaration or its output.
    assert bounded_source.to_eqi().count("initial {") == 1


def test_clocked_alias_source_compiles_like_expanded_current_read(tmp_path):
    source = clocked_alias_source()
    model = eqiora.compile(source=source, geometry=(_binding_geometry := rectangle_geometry()), entry='Clocked', bindings={**support_bindings(_binding_geometry, ['region'], []), **{}})
    path = tmp_path / "clocked-alias.eqi"
    source.write_eqi(path)
    from_file = eqiora.compile(path=path, geometry=(_binding_geometry := rectangle_geometry()), entry='Clocked', bindings={**support_bindings(_binding_geometry, ['region'], []), **{}})
    expanded = eqiora.compile(source=clocked_alias_source(aliases=False), geometry=(_binding_geometry := rectangle_geometry()), entry='Clocked', bindings={**support_bindings(_binding_geometry, ['region'], []), **{}})
    assert model.structural_fingerprint == from_file.structural_fingerprint
    assert model.structural_fingerprint == expanded.structural_fingerprint
    assert len(model.field_ids) == len(expanded.field_ids) == 2
    assert len(model.parameter_ids) == len(expanded.parameter_ids) == 0
    with pytest.raises(eqiora.EqioraError, match="Parameter"):
        model.preview_value_edit("current", 9.0)


def test_clocked_alias_rejects_distinct_equal_period_clock_assertion():
    with pytest.raises(eqiora.ValidationError, match="clock|activation"):
        eqiora.compile(source=clocked_alias_source(wrong_clock=True), geometry=(_binding_geometry := rectangle_geometry()), entry='Clocked', bindings={**support_bindings(_binding_geometry, ['region'], []), **{}})


def test_clock_authoring_tick_expressions_retain_depth_bound_and_doc_validation():
    component = eqiora.Module("main").component("Bounded")
    with pytest.raises(q.ModuleError, match="doc"):
        component.clock("tick", period_s=1, doc="x" * 16_385)
    component.clock("tick", period_s=1)
    with pytest.raises(q.ModuleError, match="doc"):
        component.initial((0, 0), doc="x" * 16_385)
    expression = q.math.pi
    for _ in range(63):
        expression = q.pre(expression)
    with pytest.raises(q.ModuleError, match="depth"):
        q.next(expression)


def test_typed_value_authoring_preserves_complex_and_nested_channel_expressions():
    source = eqiora.Module("main")
    component = source.component("Typed")
    supplied = component.parameter("supplied", value_type=eqiora.ValueType.complex())
    component.let_alias("imaginary", q.math.i)
    component.let_alias("phasor", q.math.complex(2, -3))
    component.let_alias("channels", q.array([[supplied, 2j], [3, 4]]))
    component.let_alias("selected", (q.array([supplied, 1]) + q.array([2, 3]))[1])
    text = source.to_eqi()
    assert "let imaginary = math.i;" in text
    assert "let phasor = math.complex(2, -3);" in text
    assert "[[supplied, math.complex(0, 2)], [3, 4]]" in text
    assert "([supplied, 1] + [2, 3])[1]" in text


def test_typed_value_authoring_rejects_foreign_array_and_complex_operands():
    source = eqiora.Module("main")
    left = source.component("Left")
    right = source.component("Right")
    local = left.parameter("value", value_type=eqiora.ValueType.real())
    foreign = right.parameter("value", value_type=eqiora.ValueType.real())
    for make in (lambda: q.array([local, foreign]), lambda: q.math.complex(local, foreign)):
        with pytest.raises(q.ModuleError, match="Component"):
            make()
    with pytest.raises(q.ModuleError, match="Component"):
        right.let_alias("captured", q.array([local])[0])


def test_typed_value_authoring_retains_bounds_and_explicit_array_admission():
    with pytest.raises(q.ModuleError, match="nonempty"):
        q.array([])
    with pytest.raises(TypeError, match="sequence"):
        q.array("123")
    with pytest.raises(q.ModuleError, match="4096"):
        q.array([1] * 4096)
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(q.ModuleError, match="depth"):
        q.array(cyclic)
    for index in (True, 1.5, "0"):
        with pytest.raises(TypeError, match="indices"):
            q.array([1])[index]
    with pytest.raises(q.ModuleError, match="nonnegative"):
        q.array([1])[-1]
    with pytest.raises(q.ModuleError, match="finite"):
        q.math.complex(1, float("inf"))


def test_typed_property_authoring_emits_complete_contract_and_literal_components():
    source = eqiora.Module("main")
    kind = eqiora.ValueType.array(eqiora.ValueType.complex(), 2)
    contract = source.property_contract("Response", value_type=kind)
    source.property_release("Reference", implements=contract, value=[1 + 2j, 3 - 4j],
                            source_unit=u.one, source_scale=2, citation="test.reference", license="CC0")
    source.component("Consumer").property("response", contract=contract)
    text = source.to_eqi()
    assert f"property contract Response(): {kind.to_eqi()} {{" in text
    assert "derivatives value_only;" in text
    assert "value = [math.complex(1, 2), math.complex(3, -4)];" in text
    assert "source_unit: 1 = 2;" in text
    assert not hasattr(source, "scalar_property_contract")
    assert not hasattr(source, "scalar_property_release")


def test_typed_parameter_geometry_input_preserves_complex_channel_index(tmp_path):
    source = eqiora.Module("main")
    component = source.component("TypedInputs")
    region = component.volume("region", dimensions=2)
    kind = eqiora.ValueType.array(eqiora.ValueType.complex(), 2)
    parameter = component.parameter("coefficients", value_type=kind)
    field = component.field("value", on=region, role=eqiora.FieldRole.Variable,
                            value_type=eqiora.ValueType.complex())
    component.relation("law", q.equation(field, parameter[1]), on=region)
    parameters = {"coefficients": [1 + 2j, 3 - 4j]}
    model = eqiora.compile(source=source, geometry=(_binding_geometry := rectangle_geometry()), entry='TypedInputs', bindings={**support_bindings(_binding_geometry, ['region'], []), **parameters})
    path = tmp_path / "typed-inputs.eqi"
    source.write_eqi(path)
    from_file = eqiora.compile(path=path, geometry=(_binding_geometry := rectangle_geometry()), entry='TypedInputs', bindings={**support_bindings(_binding_geometry, ['region'], []), **parameters})
    assert model.structural_fingerprint == from_file.structural_fingerprint
    changed = model.commit(model.preview_value_edit("coefficients", [1 + 2j, 3 - 7j]))
    assert changed.digest != model.digest
    assert eqiora.Model.from_bytes(changed.to_bytes()).digest == changed.digest
    with pytest.raises(eqiora.ValidationError):
        eqiora.compile(source=source, geometry=(_binding_geometry := rectangle_geometry()), entry='TypedInputs', bindings={**support_bindings(_binding_geometry, ['region'], []), **{'coefficients': [1 + 2j]}})


def sampled_source():
    source = eqiora.Module("main")
    model = source.model("Sampled")
    tick = model.clock_requirement("tick", doc="Caller-owned nominal clock.")
    drive = model.input("drive", value_type=eqiora.ValueType.real(), at=tick)
    observed = model.output("observed", value_type=eqiora.ValueType.real(), at=tick)
    memory = model.field("memory", value_type=eqiora.ValueType.real(), role=eqiora.FieldRole.State, at=tick)
    model.initial((q.pre(memory), 0))
    model.relation("update", q.equation(q.next(memory), q.pre(memory) + drive), at=tick)
    model.relation("observe", q.equation(observed, q.pre(memory)), at=tick)
    return source


def test_source_model_sampled_signature_and_exact_snapshot_resume(tmp_path):
    source = sampled_source()
    text = source.to_eqi()
    assert "public model Sampled(" in text
    assert "clock tick: periodic," in text
    assert "input drive: 1 at tick," in text
    assert "output observed: 1 at tick," in text
    assert "state memory: 1 at tick;" in text
    clock = eqiora.ClockDomain(period_s=Fraction(1, 10))
    model = eqiora.compile(source=source, entry="Sampled", bindings={"tick": clock})
    path = tmp_path / "sampled.eqi"
    source.write_eqi(path)
    emitted = eqiora.compile(path=path, entry="Sampled", bindings={"tick": clock})
    assert emitted.digest == model.digest
    session = model.execution_session(end_time_s=0.2, max_step_s=0.01,
                                    inputs={"drive": ("tick", [1.0, 2.0, 3.0])})
    assert session.output("observed", 0) is None
    assert session.next_tick == Fraction(0)
    assert session.advance_ticks(1) == 1
    assert session.output("observed", 0) == (Fraction(0), 0.0)
    assert session.field("memory") == 1.0
    snapshot = session.checkpoint()
    resumed = emitted.resume_execution(snapshot)
    assert session.advance_ticks(2) == resumed.advance_ticks(2) == 2
    assert resumed.field("memory") == session.field("memory") == 6.0
    assert resumed.output("observed", 1) == (Fraction(1, 10), 1.0)
    assert resumed.output("observed", 2) == (Fraction(1, 5), 3.0)
    assert resumed.output("observed", 3) is None
    changed = eqiora.compile(source=text, entry="Sampled",
                             bindings={"tick": eqiora.ClockDomain(period_s=Fraction(1, 10))})
    with pytest.raises(eqiora.EqioraError):
        changed.resume_execution(snapshot)


def test_source_signature_borrowing_and_forward_defaults_use_exact_handles():
    source = eqiora.Module("main")
    child = source.component("Increment")
    tick = child.clock_requirement("tick")
    borrowed = child.field_requirement("memory", value_type=eqiora.ValueType.real(),
                                      role=eqiora.FieldRole.State, at=tick)
    amount = child.parameter("amount", value_type=eqiora.ValueType.real())
    factor = child.parameter("factor", value_type=eqiora.ValueType.real())
    child.set_default(amount, factor * 2)
    child.set_default(factor, 1)
    child.relation("increment", q.equation(q.next(borrowed), q.pre(borrowed) + amount), at=tick)
    root = source.model("Root")
    root_tick = root.clock("tick", period_s=Fraction(1, 10))
    memory = root.field("memory", value_type=eqiora.ValueType.real(),
                        role=eqiora.FieldRole.State, at=root_tick)
    root.initial((q.pre(memory), 0))
    with pytest.raises(q.ModuleError, match="exact required signature"):
        root.instance("child", component=child, bindings={'tick': root_tick})
    with pytest.raises(q.ModuleError, match="enclosing Field"):
        root.instance("child", component=child, bindings={'tick': root_tick, 'memory': borrowed})
    root.instance("child", component=child, bindings={'tick': root_tick, 'memory': memory})
    text = source.to_eqi()
    assert "parameter amount: 1 = factor * 2," in text
    assert "memory = memory" in text
    assert "field memory =" not in text
    compiled = eqiora.compile(source=source, entry="Root")
    assert len(compiled.field_ids) == 1
    session = compiled.execution_session(end_time_s=0.1, max_step_s=0.01, inputs={})
    assert session.advance_ticks(2) == 2
    assert session.field("memory") == 4.0


def test_clock_domain_identity_is_nominal_and_exact():
    first = eqiora.ClockDomain(period_s=Fraction(2, 6), phase_s=Fraction(1, 7))
    second = eqiora.ClockDomain(period_s=Fraction(1, 3), phase_s=Fraction(1, 7))
    assert first.period_s == second.period_s == Fraction(1, 3)
    assert first.phase_s == Fraction(1, 7)
    assert first != second
    assert first.id != second.id
    clocks = {first: "first", second: "second"}
    assert clocks[first] == "first"
    assert len(clocks) == 2
    assert first.id in repr(first)
    assert "period_s=Fraction(1, 3)" in repr(first)
    for invalid in (True, 0.1, -1, 0):
        with pytest.raises((TypeError, ValueError, OverflowError)):
            eqiora.ClockDomain(period_s=invalid)
    for invalid in (None, True, 0.1, -1):
        with pytest.raises((TypeError, ValueError, OverflowError)):
            eqiora.ClockDomain(period_s=1, phase_s=invalid)
    with pytest.raises(AttributeError):
        first.period_s = 1


def test_signature_authoring_preserves_output_ownership_and_forward_default_bounds():
    source = eqiora.Module("main")
    child = source.component("Child")
    input_value = child.input("supplied", value_type=eqiora.ValueType.real())
    output = child.output("result", value_type=eqiora.ValueType.real())
    child.relation("evaluate", q.equation(output, input_value * 2))
    parent = source.model("Parent")
    required = parent.parameter("required", value_type=eqiora.ValueType.real())
    later = parent.parameter("later", value_type=eqiora.ValueType.real())
    parent.set_default(required, later + 1)
    parent.set_default(later, 2)
    instance = parent.instance("child", component=child, bindings={'supplied': required})
    observed = parent.output("observed", value_type=eqiora.ValueType.real())
    parent.relation("observe", q.equation(observed, instance['result']))
    with pytest.raises(TypeError):
        instance['result'] = required
    with pytest.raises(q.ModuleError, match="Component.*owners"):
        child.relation("foreign", q.equation(output, instance['result']))
    with pytest.raises(q.ModuleError, match="Parameter requirement"):
        child.set_default(required, 1)
    text = source.to_eqi()
    assert "parameter required: 1 = later + 1," in text
    assert "observed = child.result;" in text
    assert "variable result" not in text
    assert "input supplied: 1," in text
    assert "output observed: 1," in text
    with pytest.raises(q.ModuleError, match="frozen"):
        parent.set_default(later, 3)


def test_external_clock_alias_assertion_compares_nominal_identity():
    source = eqiora.Module("main")
    owner = source.model("Clocks")
    first = owner.clock_requirement("first")
    second = owner.clock_requirement("second")
    memory = owner.field("memory", value_type=eqiora.ValueType.real(), role=eqiora.FieldRole.State, at=first)
    observed = owner.output("observed", value_type=eqiora.ValueType.real(), at=first)
    owner.initial((q.pre(memory) - 1, 0))
    owner.relation("hold", q.equation(q.next(memory), q.pre(memory)), at=first)
    alias = owner.let_alias("current", memory, at=second)
    owner.relation("observe", q.equation(observed, alias), at=first)
    shared = eqiora.ClockDomain(period_s=Fraction(1, 10))
    assert eqiora.compile(source=source, entry="Clocks", bindings={"first": shared, "second": shared}).digest
    with pytest.raises(eqiora.EqioraError, match="clock|activation"):
        eqiora.compile(source=source, entry="Clocks", bindings={
            "first": shared, "second": eqiora.ClockDomain(period_s=Fraction(1, 10)),
        })


def test_named_test_cannot_be_silently_dropped_without_its_form():
    module = eqiora.Module("main")
    component = module.component("C")
    body = component.volume("body", dimensions=2)
    surface = component.complete_exterior("surface", parent=body)
    value = component.field("value", value_type=eqiora.ValueType.real(), role=eqiora.FieldRole.Variable, on=body)
    component.test("w", for_=value, zero_on=surface)
    with pytest.raises(q.ModuleError, match="owning weak form"):
        module.to_eqi()


def test_plural_weak_form_authoring_preserves_bindings_and_equation_order():
    module = eqiora.Module("main")
    component = module.component("Mixed")
    body = component.volume("body", dimensions=2)
    exterior = component.complete_exterior("exterior", parent=body)
    velocity = component.field("velocity", value_type=eqiora.ValueType.real(),
                               role=eqiora.FieldRole.Variable, on=body)
    pressure = component.field("pressure", value_type=eqiora.ValueType.real(),
                               role=eqiora.FieldRole.Variable, on=body)
    momentum = component.relation("momentum", q.equation(velocity, 0), on=body)
    continuity = component.relation("continuity", q.equation(pressure, 0), on=body)
    v = component.test("v", for_=velocity, zero_on=exterior)
    p = component.test("p", for_=pressure)
    with pytest.raises(q.ModuleError, match="only one test"):
        component.test("other", for_=velocity)
    with pytest.raises(q.ModuleError, match="distinct"):
        component.weak_form("invalid", [momentum, momentum], equations=[(v, p)])
    with pytest.raises(q.ModuleError, match="1 and 8 equations"):
        component.weak_form("invalid", [momentum], equations=[(v, p)] * 9)
    component.weak_form("mixed", [momentum, continuity],
                        equations=[(v * velocity, v * 0), (p * pressure, p * 0)])
    with pytest.raises(q.ModuleError, match="one named weak form"):
        component.weak_form("second", [momentum], equations=[(v, p)])
    text = module.to_eqi()
    assert "test v: 1 for velocity zero_on exterior;" in text
    assert "test p: 1 for pressure;" in text
    assert "form mixed for momentum, continuity" in text
    assert text.index("v * velocity") < text.index("p * pressure")


def test_weak_form_test_names_remain_component_unique():
    module = eqiora.Module("main")
    component = module.component("C")
    body = component.volume("body", dimensions=2)
    first = component.field("first", value_type=eqiora.ValueType.real(),
                            role=eqiora.FieldRole.Variable, on=body)
    second = component.field("second", value_type=eqiora.ValueType.real(),
                             role=eqiora.FieldRole.Variable, on=body)
    component.test("w", for_=first)
    with pytest.raises(q.ModuleError):
        component.test("w", for_=second)
