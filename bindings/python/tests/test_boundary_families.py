"""Exact field port families preserve separate Geometry and source identities."""

import pytest

import eqiora

q = eqiora.lang
DISPLACEMENT = eqiora.ValueType.real(eqiora.Dimension(length=1))
TRACTION = eqiora.ValueType.real(eqiora.Dimension(mass=1, length=-1, time=-2))
NAMES = ("left", "right", "bottom", "top")


def rectangle(*, x_upper=1.0):
    graph = eqiora.geometry.GeometryGraph()
    box = graph.rectangle(x_bounds=(0.0, x_upper), y_bounds=(0.0, 1.0))
    return graph.build(box, named_topology={
        "body": box.region, **dict(zip(NAMES, box.boundaries)),
    })


def bindings(geometry):
    parent = geometry.selection("body")
    return {"body": parent,
            **{name: (geometry.selection(name), parent) for name in NAMES}}


def authored_exterior(*, incomplete=False, distinct=False, library=False, package="eqiora.local_project"):
    module = eqiora.Module("parts" if library else "main", package=package)
    connector = module.field_connector("MechanicalBoundary", trace=("displacement", DISPLACEMENT),
                                       flux=("traction", TRACTION), spatial_vector=True)
    other = (module.field_connector("OtherBoundary", trace=("displacement", DISPLACEMENT),
                                    flux=("traction", TRACTION), spatial_vector=True)
             if distinct else connector)
    law = module.component("ExteriorLaw")
    body = law.volume("body", dimensions=2)
    exterior = law.complete_exterior("exterior", parent=body)
    member = exterior.member("boundary")
    port = law.port("mechanical", connector=connector, on=member)
    selected = port[member]
    law.relation("law", q.equation(selected.displacement - selected.displacement, 0),
                 q.equation(selected.traction - selected.traction, 0), on=member)

    wrapper = module.component("ExteriorWrapper")
    wrapper_body = wrapper.volume("body", dimensions=2)
    wrapper_exterior = wrapper.complete_exterior("exterior", parent=wrapper_body)
    wrapper_member = wrapper_exterior.member("boundary")
    wrapper_port = wrapper.port("mechanical", connector=connector, on=wrapper_member)
    child = wrapper.instance("child", component=law,
                             bindings={"body": wrapper_body, "exterior": wrapper_exterior})
    wrapper.connect(child["mechanical"][wrapper_member], wrapper_port[wrapper_member],
                     over=wrapper_member)

    terminal = module.component("BoundaryTerminal")
    terminal_body = terminal.volume("body", dimensions=2)
    face = terminal.boundary("face", parent=terminal_body)
    mechanical = terminal.port("mechanical", connector=other, on=face)
    terminal.relation("law", q.equation(mechanical.displacement - mechanical.displacement, 0),
                      q.equation(mechanical.traction - mechanical.traction, 0), on=face)

    if library:
        return module

    root = module.model("Main")
    root_body = root.volume("body", dimensions=2)
    faces = {name: root.boundary(name, parent=root_body) for name in NAMES}
    selected_faces = tuple(faces.values())[:-1] if incomplete else tuple(faces.values())
    wrapped = root.instance("wrapped", component=wrapper,
                            bindings={"body": root_body,
                                      "exterior": root.boundaries(*selected_faces)})
    for name, face in faces.items():
        if incomplete and name == "top":
            continue
        end = root.instance(name + "_terminal", component=terminal,
                             bindings={"body": root_body, "face": face})
        root.connect(wrapped["mechanical"][face], end["mechanical"])
    return module


def test_complete_exterior_keeps_geometry_separate_across_source_and_artifacts(tmp_path):
    module = authored_exterior()
    geometry = rectangle()
    text = module.to_eqi()
    assert "complete_exterior(parent = body)" in text
    assert "connect [boundary in exterior]" in text
    assert "mechanical[boundary = boundary].traction" in text
    assert "domain " not in text
    direct = eqiora.compile(source=module, geometry=geometry, entry="Main", bindings=bindings(geometry))
    path = tmp_path / "exterior.eqi"
    module.write_eqi(path)
    emitted = eqiora.compile(path=path, geometry=geometry, entry="Main", bindings=bindings(geometry))
    assert direct.to_bytes() == emitted.to_bytes()
    assert eqiora.Model.from_bytes(direct.to_bytes()).digest == direct.digest


@pytest.mark.parametrize("route", ("module", "source"))
@pytest.mark.parametrize("mutant, expected", (("incomplete", "missing Cartesian side"), ("distinct", "connector")))
def test_boundary_family_mutants_reach_the_exact_gate(route, mutant, expected):
    module = authored_exterior(**{mutant: True})
    geometry = rectangle()
    source = module.to_eqi() if route == "source" else module
    with pytest.raises(eqiora.ValidationError) as error:
        eqiora.compile(source=source, geometry=geometry, entry="Main", bindings=bindings(geometry))
    assert any(expected.lower() in diagnostic.message.lower() for diagnostic in error.value.diagnostics)
    assert all("unresolved Cartesian" not in diagnostic.message for diagnostic in error.value.diagnostics)


def test_boundary_families_cannot_reuse_selections_from_another_geometry_revision():
    module = authored_exterior()
    geometry = rectangle()
    foreign = rectangle(x_upper=2.0)
    values = bindings(geometry)
    values["left"] = (foreign.selection("left"), geometry.selection("body"))
    with pytest.raises(ValueError, match="different exact Geometry revision"):
        eqiora.compile(source=module, geometry=geometry, entry="Main", bindings=values)


def test_boundary_member_scopes_and_parent_identity_reject_before_mutation():
    module = eqiora.Module("main")
    connector = module.field_connector("Boundary", trace=("u", DISPLACEMENT), flux=("f", TRACTION))
    component = module.component("Law")
    body = component.volume("body", dimensions=2)
    exterior = component.complete_exterior("exterior", parent=body)
    other = component.complete_exterior("other", parent=body)
    member = exterior.member("boundary")
    wrong = other.member("boundary")
    port = component.port("port", connector=connector, on=member)
    with pytest.raises(q.ModuleError, match="different complete exterior"):
        port[wrong]
    with pytest.raises(q.ModuleError, match="select an exact boundary"):
        _ = port.u
    for support in (None, wrong):
        with pytest.raises(q.ModuleError, match="unbound boundary"):
            component.relation("law", q.equation(port[member].u, 0), on=support)
    component.relation("law", q.equation(port[member].u - port[member].u, 0), on=member)
    face = component.boundary("face", parent=body)
    foreign_body = component.volume("foreign_body", dimensions=2)
    foreign_face = component.boundary("foreign_face", parent=foreign_body)
    for values, message in (((face, face), "duplicate"), ((face, foreign_face), "exact parent")):
        with pytest.raises(q.ModuleError, match=message):
            component.boundaries(*values)
    with pytest.raises(q.ModuleError, match="different exact parent"):
        port[foreign_face]
    for handle in (exterior, member, component.boundaries(face), port[member]):
        with pytest.raises(AttributeError):
            handle._name = "changed"


def test_field_connector_derives_shape_and_rejects_conflicting_quantity_types():
    module = eqiora.Module("main")
    vector = eqiora.ValueType.vector(DISPLACEMENT, 2)
    vector_flux = eqiora.ValueType.vector(TRACTION, 2)
    with pytest.raises(q.ModuleError, match="identical shape"):
        module.field_connector("Boundary", trace=("u", vector), flux=("f", TRACTION))
    with pytest.raises(q.ModuleError, match="requires scalar"):
        module.field_connector("Boundary", trace=("u", vector), flux=("f", vector_flux), spatial_vector=True)
    module.field_connector("Boundary", trace=("u", vector), flux=("f", vector_flux))
    text = module.to_eqi()
    assert "shape [2];" in text
    assert "frame spatial;" in text
    assert "orientation parent_outward;" in text


def test_periodic_topology_is_explicit_and_model_scoped():
    module = eqiora.Module("main")
    connector = module.field_connector("Periodic", trace=("u", DISPLACEMENT), flux=("f", TRACTION))
    law = module.component("Law")
    body = law.volume("body", dimensions=2)
    lower = law.boundary("lower", parent=body)
    upper = law.boundary("upper", parent=body)
    first = law.port("first", connector=connector, on=lower)
    second = law.port("second", connector=connector, on=upper)
    with pytest.raises(q.ModuleError, match="only to a Model"):
        law.connect_periodic(first, second)
    root = module.model("Main")
    body = root.volume("body", dimensions=2)
    lower = root.boundary("lower", parent=body)
    upper = root.boundary("upper", parent=body)
    first = root.port("first", connector=connector, on=lower)
    second = root.port("second", connector=connector, on=upper)
    root.connect_periodic(first, second)
    assert "connect periodic first, second;" in module.to_eqi()


def selected_exterior(*, component=False):
    module = eqiora.Module("main")
    root = module.component("Selected") if component else module.model("Selected")
    body = root.volume("body", dimensions=2)
    exterior = root.complete_exterior("exterior", parent=body)
    member = exterior.member("face")
    value = root.field("value", spatial_regularity=eqiora.SpatialRegularity.H1, value_type=DISPLACEMENT,
                       role=eqiora.FieldRole.Variable, on=body)
    root.relation("interior", q.equation(value - value, 0), on=body)
    root.relation("exterior_law", q.equation(q.trace(value) - q.trace(value), 0), on=member)
    return module


def exterior_bindings(geometry, names=NAMES):
    parent = geometry.selection("body")
    return {"body": parent,
            "exterior": (tuple(geometry.selection(name) for name in names), parent)}


@pytest.mark.parametrize("component", (False, True))
def test_selected_root_complete_exterior_retains_exact_members_on_artifact_replay(component, tmp_path):
    module = selected_exterior(component=component)
    geometry = rectangle()
    values = exterior_bindings(geometry)
    direct = eqiora.compile(source=module, geometry=geometry, entry="Selected", bindings=values)
    path = tmp_path / "selected.eqi"
    module.write_eqi(path)
    emitted = eqiora.compile(path=path, geometry=geometry, entry="Selected", bindings=values)
    reordered = eqiora.compile(source=module, geometry=geometry, entry="Selected",
                               bindings=exterior_bindings(geometry, tuple(reversed(NAMES))))
    assert direct.to_bytes() == emitted.to_bytes() == reordered.to_bytes()
    assert eqiora.Model.from_bytes(direct.to_bytes()).digest == direct.digest


@pytest.mark.parametrize("names, message", ((NAMES[:-1], "missing Cartesian side"),
                                           (("left", "left", "bottom", "top"), "more than once"),
                                           (("body", "right", "bottom", "top"), "boundary")))
def test_selected_root_exterior_rejects_incomplete_duplicate_and_volume_members(names, message):
    geometry = rectangle()
    with pytest.raises(eqiora.ValidationError) as error:
        eqiora.compile(source=selected_exterior(), geometry=geometry, entry="Selected",
                       bindings=exterior_bindings(geometry, names))
    assert any(message.lower() in diagnostic.message.lower() for diagnostic in error.value.diagnostics)


def test_selected_root_exterior_rejects_stale_members_and_parent_before_compilation():
    geometry = rectangle()
    other = rectangle(x_upper=2)
    values = exterior_bindings(geometry)
    members, parent = values["exterior"]
    for binding in (((other.selection("left"), *members[1:]), parent),
                    (members, other.selection("body"))):
        values["exterior"] = binding
        with pytest.raises(ValueError, match="different exact Geometry revision"):
            eqiora.compile(source=selected_exterior(), geometry=geometry, entry="Selected", bindings=values)


@pytest.mark.parametrize("count", (0, 257))
def test_selected_root_exterior_adapter_bounds_explicit_member_tuple(count):
    geometry = rectangle()
    parent = geometry.selection("body")
    values = {"body": parent, "exterior": ((geometry.selection("left"),) * count, parent)}
    with pytest.raises(ValueError, match="between 1 and 256 exact boundary selections"):
        eqiora.compile(source=selected_exterior(), geometry=geometry, entry="Selected", bindings=values)


def test_local_support_binding_preserves_parent_and_dimension_before_reserving_instance_name():
    module = eqiora.Module("main")
    law = module.component("Law")
    law_body = law.volume("body", dimensions=2)
    law.boundary("face", parent=law_body)
    root = module.model("Main")
    body = root.volume("body", dimensions=2)
    other = root.volume("other", dimensions=2)
    volume3 = root.volume("volume3", dimensions=3)
    face = root.boundary("face", parent=body)
    for parent, message in ((other, "exact bound parent"), (volume3, "ambient dimension")):
        with pytest.raises(q.ModuleError, match=message):
            root.instance("child", component=law, bindings={"body": parent, "face": face})
    root.instance("child", component=law, bindings={"body": body, "face": face})
