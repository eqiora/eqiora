"""Ordinary installed-product coverage for Geometry v2 through Gmsh."""

from __future__ import annotations

import json
from importlib.resources import files

import numpy as np
import pytest

import eqiora
from _signature_bindings import support_bindings


def geometry(*, x_max: float = 2.2, center: tuple[float, float] = (0.2, 0.2)) -> eqiora.geometry.Geometry:
    graph = eqiora.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0.0, x_max), y_bounds=(0.0, 0.41))
    circle = graph.circle(center=center, radius=0.05)
    fluid = graph.subtract(rectangle, circle)
    return graph.build(
        fluid,
        named_topology={
            "fluid": fluid.region,
            "inlet": rectangle.boundaries[0],
            "outlet": rectangle.boundaries[1],
            "walls": rectangle.boundaries[2:],
            "cylinder": circle.boundaries[0],
        },
    )


def provider(**changes: float | int | None) -> eqiora.meshing.GmshMesher:
    values: dict[str, float | int | None] = {
        "maximum_boundary_error": 1.0e-4,
        "minimum_mean_ratio": 1.0e-5,
        "maximum_boundary_facets": 50,
    }
    values.update(changes)
    return eqiora.meshing.GmshMesher(**values)


def test_gmsh_plan_publishes_complete_source_owned_mesh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EQIORA_GMSH", "gmsh")
    source = geometry()
    policy = provider()
    plan = eqiora.meshing.resolve(source, policy)
    mesh = eqiora.meshing.generate(plan)

    assert plan.provider == policy
    assert plan.source_digest == source.digest == mesh.source_digest
    assert mesh.realized_geometry_digest == source.digest
    assert mesh.dimension == 2
    assert mesh.vertex_count > 0
    assert mesh.cell_count > 0
    assert mesh.selection_entity_count(source.selection("fluid")) == mesh.cell_count
    assert set(mesh.selection_names) == {"fluid", "inlet", "outlet", "walls", "cylinder"}
    assert sum(
        mesh.selection_entity_count(source.selection(name))
        for name in ("inlet", "outlet", "walls", "cylinder")
    ) > 0
    assert not hasattr(plan, "production_lineage_bytes")
    assert not hasattr(plan, "production_lineage_digest")
    lineage = json.loads(mesh.production_lineage_bytes)
    assert lineage["provider"] == {"identity": "eqiora.gmsh-cli", "version": "4.15.2"}
    assert lineage["effective_policy"] == {
        "kind": "gmsh-mesh",
        "maximum_boundary_error_m": 1.0e-4,
        "minimum_mean_ratio": 1.0e-5,
        "maximum_boundary_facets": 50,
        "maximum_target_size_m": 0.2,
        "maximum_target_size_ownership": "automatic",
    }

    coordinates = mesh.coordinates
    cells = mesh.cells
    assert coordinates.shape == (mesh.vertex_count, 2)
    assert cells.shape == (mesh.cell_count, 3)
    assert np.isfinite(coordinates).all()
    assert np.all(cells < mesh.vertex_count)
    assert not coordinates.flags.writeable
    assert not cells.flags.writeable


def test_gmsh_resolve_is_planning_only(monkeypatch: pytest.MonkeyPatch) -> None:
    source = geometry()
    monkeypatch.setenv("EQIORA_GMSH", "/provider/must/not/be-launched-by-resolve")

    plan = eqiora.meshing.resolve(source, provider())

    assert plan.source_digest == source.digest
    with pytest.raises(AttributeError):
        plan.source_digest = "mutated"
    with pytest.raises(eqiora.ValidationError):
        eqiora.meshing.generate(plan)


def test_gmsh_plan_replay_and_crosswire_are_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EQIORA_GMSH", "gmsh")
    source = geometry()
    plan = eqiora.meshing.resolve(source, provider())
    first = eqiora.meshing.generate(plan)
    second = eqiora.meshing.generate(plan)
    direct = eqiora.meshing.generate(source, provider())
    for mesh in (second, direct):
        assert first.source_digest == mesh.source_digest == source.digest
        assert first.digest == mesh.digest
        assert first.correspondence_digest == mesh.correspondence_digest
        assert first.canonical_bytes == mesh.canonical_bytes
        assert first.production_lineage_digest == mesh.production_lineage_digest
        assert first.production_lineage_bytes == mesh.production_lineage_bytes
        assert first.to_bytes() == mesh.to_bytes()

    model = eqiora.compile(
        path=files(eqiora).joinpath("examples", "steady-flow-past-cylinder.eqi"),
        geometry=source,
        entry="SteadyFlowPastCylinder",
        bindings={
            **support_bindings(source, ["fluid"], [
                (name, "fluid") for name in ("inlet", "outlet", "walls", "cylinder")
            ]),
            "dynamic_viscosity": 0.001,
            "zero_pressure": 0.0,
            "inlet_speed": 0.3,
            "channel_height": source.bounds[1][1] - source.bounds[1][0],
        },
    )
    linear = eqiora.solve.Linear(
        algorithm=eqiora.solve.LinearSolver.SparseLu,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Fast,
        provider=eqiora.solve.SolverProvider.faer(),
        relative_tolerance=1.0e-6,
        absolute_tolerance=1.0e-13,
        maximum_iterations=10_000,
    )
    explicit_plan, direct_plan = (
        eqiora.resolve(model, mesh=mesh, spatial=eqiora.fem.MiniP1(), solve=linear, scaling=None)
        for mesh in (first, direct)
    )
    assert explicit_plan.identity == direct_plan.identity

    foreign = geometry(center=(0.21, 0.2))
    with pytest.raises(TypeError):
        eqiora.meshing.generate(foreign, plan=plan)
    with pytest.raises(eqiora.ValidationError):
        first.selection_entity_count(foreign.selection("cylinder"))


def test_explicit_target_size_is_replayable_and_strictly_denser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EQIORA_GMSH", "gmsh")
    source = geometry()
    automatic_provider = provider()
    explicit_provider = provider(maximum_target_size=0.05)
    assert automatic_provider.maximum_target_size is None
    assert explicit_provider.maximum_target_size == 0.05

    automatic = eqiora.meshing.generate(
        eqiora.meshing.resolve(source, automatic_provider)
    )
    explicit_plan = eqiora.meshing.resolve(source, explicit_provider)
    explicit = eqiora.meshing.generate(explicit_plan)
    replayed = eqiora.meshing.generate(explicit_plan)

    assert explicit.cell_count > automatic.cell_count
    assert explicit.vertex_count > automatic.vertex_count
    assert replayed.digest == explicit.digest
    lineage = json.loads(explicit.production_lineage_bytes)
    assert lineage["effective_policy"]["maximum_target_size_m"] == 0.05
    assert lineage["effective_policy"]["maximum_target_size_ownership"] == "explicit"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"maximum_boundary_error": 0.0},
        {"maximum_target_size": 0.0},
        {"maximum_target_size": float("nan")},
        {"maximum_target_size": float("inf")},
        {"minimum_mean_ratio": 0.0},
        {"maximum_boundary_facets": 7},
    ],
)
def test_gmsh_policy_rejects_invalid_values(kwargs: dict[str, float | int | None]) -> None:
    with pytest.raises(eqiora.ValidationError):
        provider(**kwargs)


def test_gmsh_plan_rejects_target_smaller_than_boundary_chord() -> None:
    with pytest.raises(eqiora.ValidationError):
        eqiora.meshing.resolve(geometry(), provider(maximum_target_size=0.001))


def test_reference_and_import_products_are_absent() -> None:
    assert not hasattr(eqiora.meshing, "ReferenceMesher")
    assert not hasattr(eqiora.meshing, "GmshImport")
    assert not hasattr(eqiora.meshing, "import_gmsh")


def test_generate_rejects_ambiguous_or_invalid_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EQIORA_GMSH", "/provider/must/not/be-launched-for-invalid-input")
    source = geometry()
    policy = provider()
    plan = eqiora.meshing.resolve(source, policy)
    with pytest.raises(TypeError, match="does not accept a provider"):
        eqiora.meshing.generate(plan, policy)
    with pytest.raises(TypeError, match="requires an explicit provider"):
        eqiora.meshing.generate(source)
    with pytest.raises(TypeError):
        eqiora.meshing.generate(object(), policy)
    with pytest.raises(eqiora.ValidationError):
        eqiora.meshing.generate(source, object())
