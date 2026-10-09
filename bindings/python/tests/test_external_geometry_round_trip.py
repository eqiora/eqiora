"""Abstract Module supports retain caller Geometry through execution and replay."""

import numpy as np
import pytest

import eqiora


def interval(*, upper=1.0, label="body"):
    graph = eqiora.geometry.GeometryGraph()
    domain = graph.interval(bounds=(0.0, upper))
    return graph.build(domain, named_topology={
        label: domain.region,
        "left": domain.boundaries[0],
        "right": domain.boundaries[1],
    })


def bindings(geometry, *, label="body"):
    parent = geometry.selection(label)
    return {
        "body": parent,
        "left": (geometry.selection("left"), parent),
        "right": (geometry.selection("right"), parent),
    }


def module(*, doc="A caller-bound diffusion interval."):
    q = eqiora.lang
    source = eqiora.Module("main")
    owner = source.component("Diffusion", doc=doc)
    body = owner.volume("body", dimensions=1)
    left = owner.boundary("left", parent=body)
    right = owner.boundary("right", parent=body)
    value = owner.field("u", spatial_regularity=eqiora.SpatialRegularity.H1, on=body, role=eqiora.FieldRole.Variable,
                        value_type=eqiora.ValueType.real())
    owner.relation("balance", q.equation(
        -q.div(q.grad(value)), q.quantity(2, eqiora.units.one / eqiora.units.m**2),
    ), on=body)
    for name, boundary in (("left_value", left), ("right_value", right)):
        owner.relation(name, q.equation(q.trace(value), 0), on=boundary)
    return source


def execute(model, mesh, field_id):
    plan = eqiora.resolve(
        model, mesh=mesh, spatial=eqiora.fem.Q1(),
        solve=eqiora.solve.Linear(
            algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
            preconditioner=eqiora.solve.Preconditioner.Identity,
            reduction=eqiora.solve.Reduction.Reproducible,
            provider=eqiora.solve.SolverProvider.reference(),
            relative_tolerance=1e-12,
            absolute_tolerance=1e-14,
            maximum_iterations=100,
        ),
    )
    assert plan.geometry_digest == mesh.source_digest
    reopened_plan = eqiora.Plan.from_bytes(plan.to_bytes())
    assert reopened_plan.geometry_digest == plan.geometry_digest
    result = eqiora.run(reopened_plan)
    return result.output(model.field(field_id)).values("vertex").numpy().reshape(-1)


def test_external_geometry_survives_module_source_move_and_model_replay(tmp_path):
    geometry = interval(label="caller_region")
    selected = bindings(geometry, label="caller_region")
    authored = module()
    text = authored.to_eqi()
    assert "support body: volume" in text
    assert "support left: boundary(parent = body)" in text
    assert geometry.digest not in text
    assert "caller_region" not in text
    direct = eqiora.compile(source=authored, geometry=geometry,
                            entry="Diffusion", bindings=selected)
    field_id = direct.field("definition.u").id
    path = tmp_path / "original" / "diffusion.eqi"
    path.parent.mkdir()
    authored.write_eqi(path)
    moved = tmp_path / "moved" / "diffusion.eqi"
    moved.parent.mkdir()
    path.rename(moved)
    emitted = eqiora.compile(path=moved, geometry=geometry,
                             entry="Diffusion", bindings=selected)
    changed_comments = eqiora.compile(source=module(doc="Moved workflow notes."),
                                      geometry=geometry, entry="Diffusion", bindings=selected)
    reopened = eqiora.Model.from_bytes(direct.to_bytes())
    models = (direct, emitted, changed_comments, reopened)
    assert all(model.digest == direct.digest for model in models)
    mesh = eqiora.meshing.generate(eqiora.meshing.resolve(
        geometry, eqiora.meshing.CartesianMesher(cells=(4,)),
    ))
    assert mesh.source_digest == geometry.digest
    # -u''=2 m^-2, u(0)=u(1 m)=0 gives u=x(1-x) in metre coordinates.
    # Uniform Q1 stiffness and constant-load integration reproduce these nodal
    # values exactly; only floating-point solve error enters this comparison.
    x = mesh.coordinates[:, 0]
    expected = x * (1.0 - x)
    # Model artifacts retain exact Field identity; source lookup aliases remain
    # compilation metadata and are not reconstructed from presentation names.
    values = [execute(model, mesh, field_id) for model in models]
    for actual in values:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(actual[[0, -1]], [0.0, 0.0])
    for actual in values[1:]:
        np.testing.assert_array_equal(actual, values[0])


@pytest.mark.parametrize("route", ("module", "source"))
@pytest.mark.parametrize("mutant", ("stale", "foreign", "wrong_dimension", "wrong_parent"))
def test_binding_failures_reach_geometry_admission_on_both_routes(route, mutant):
    geometry = interval()
    selected = bindings(geometry)
    source = module()
    if route == "source":
        source = source.to_eqi()
    # Establish the ordinary positive through this same ingress before mutating
    # only the caller-owned binding. Equal coordinates do not rescue foreign
    # identity: changing the explicit naming catalog changes its Geometry.
    assert eqiora.compile(source=source, geometry=geometry,
                          entry="Diffusion", bindings=selected).digest
    if mutant in ("stale", "foreign"):
        other = interval(upper=2.0) if mutant == "stale" else interval(label="other_body")
        selected["left"] = (other.selection("left"), geometry.selection("body"))
    elif mutant == "wrong_dimension":
        selected["body"] = geometry.selection("left")
    else:
        selected["left"] = (geometry.selection("left"), geometry.selection("right"))
    if mutant in ("stale", "foreign"):
        with pytest.raises(ValueError, match="different exact Geometry revision"):
            eqiora.compile(source=source, geometry=geometry,
                           entry="Diffusion", bindings=selected)
        return
    with pytest.raises(eqiora.ValidationError) as error:
        eqiora.compile(source=source, geometry=geometry,
                       entry="Diffusion", bindings=selected)
    assert any(any(word in diagnostic.message.lower()
                   for word in ("geometry", "dimension", "parent", "boundary", "selection"))
               for diagnostic in error.value.diagnostics)


def test_deleted_region_cannot_become_an_external_support_selection():
    graph = eqiora.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0.0, 2.0), y_bounds=(0.0, 1.0))
    circle = graph.circle(center=(1.0, 0.5), radius=0.2)
    cut = graph.subtract(rectangle, circle)
    names = {"body": cut.region, "outer": rectangle.boundaries,
             "hole": circle.boundaries[0]}
    assert graph.build(cut, named_topology=names).selection("body").dimension == 2
    # The old region has a known identity but was deleted by subtraction. This
    # rejects while constructing Geometry, before a Model binding can exist.
    with pytest.raises(eqiora.ValidationError, match="deleted"):
        graph.build(cut, named_topology={**names, "body": rectangle.region})


def test_model_replay_does_not_accept_a_foreign_mesh_with_equal_coordinates():
    geometry = interval()
    direct = eqiora.compile(source=module(), geometry=geometry,
                            entry="Diffusion", bindings=bindings(geometry))
    field_id = direct.field("definition.u").id
    replayed = eqiora.Model.from_bytes(direct.to_bytes())
    meshes = [eqiora.meshing.generate(eqiora.meshing.resolve(
        owner, eqiora.meshing.CartesianMesher(cells=(4,)),
    )) for owner in (geometry, interval(label="foreign_body"))]
    np.testing.assert_array_equal(meshes[0].coordinates, meshes[1].coordinates)
    assert meshes[0].source_digest != meshes[1].source_digest
    for model in (direct, replayed):
        execute(model, meshes[0], field_id)
        with pytest.raises(eqiora.EqioraError):
            execute(model, meshes[1], field_id)


def test_ambiguous_or_incomplete_geometry_membership_never_reaches_a_model():
    graph = eqiora.geometry.GeometryGraph()
    domain = graph.interval(bounds=(0.0, 1.0))
    names = {"body": domain.region, "left": domain.boundaries[0],
             "right": domain.boundaries[1]}
    assert graph.build(domain, named_topology=names).selection("left").dimension == 0
    for invalid in (
        {"body": domain.region, "left": domain.boundaries[0]},
        {**names, "body": (domain.region, domain.boundaries[0])},
    ):
        with pytest.raises(eqiora.ValidationError):
            graph.build(domain, named_topology=invalid)
