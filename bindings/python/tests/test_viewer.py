from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from importlib import resources

import numpy as np
import pytest

import eqiora
from _signature_bindings import support_bindings
from eqiora._eqiora import _compose_view


POISSON = """
public component ViewerPoisson(
  support square: volume(ambient_dimension = 2),
  support left: boundary(parent = square),
  support right: boundary(parent = square),
  support bottom: boundary(parent = square),
  support top: boundary(parent = square),
  parameter diffusion: 1,
  parameter source_scale: 1 / m ^ 2
) {

  variable potential: 1 on square;
  relation balance on square {
    -div(diffusion * grad(potential)) - source_scale = 0;
  }
  relation left_value on left { trace(potential) = 0; }
  relation right_value on right { trace(potential) = 0; }
  relation bottom_value on bottom { trace(potential) = 0; }
  relation top_value on top { trace(potential) = 0; }
}
"""

ELASTICITY = """
public component ViewerElasticity(
  support square: volume(ambient_dimension = 2),
  support left: boundary(parent = square),
  support right: boundary(parent = square),
  support bottom: boundary(parent = square),
  support top: boundary(parent = square),
  parameter stiffness: kg / (m * s ^ 2),
  parameter lambda: kg / (m * s ^ 2),
  parameter length_scale: m
) {

  variable displacement: vector<m, 2> on square;
  variable load_potential: kg / (m * s ^ 2) on square;
  relation load on square {
    load_potential - 2 * stiffness * coordinate(0) / length_scale = 0;
  }
  relation balance on square {
    -div(2 * stiffness * symmetric_part(grad(displacement))
      + lambda * isotropic_lift(div(displacement)))
      - grad(load_potential) = 0;
  }
  relation left_value on left { trace(displacement) = 0; }
  relation right_value on right {
    normal(2 * stiffness * symmetric_part(grad(displacement))
      + lambda * isotropic_lift(div(displacement))) = 0;
  }
  relation bottom_value on bottom {
    normal(2 * stiffness * symmetric_part(grad(displacement))
      + lambda * isotropic_lift(div(displacement))) = 0;
  }
  relation top_value on top {
    normal(2 * stiffness * symmetric_part(grad(displacement))
      + lambda * isotropic_lift(div(displacement))) = 0;
  }
}
"""


def rectangle_and_mesh(
    *, x_bounds: tuple[float, float] = (0.0, 1.0)
) -> tuple[eqiora.geometry.Geometry, eqiora.meshing.Mesh]:
    graph = eqiora.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=x_bounds, y_bounds=(0.0, 1.0))
    geometry = graph.build(
        rectangle,
        named_topology={
            "square": rectangle.region,
            "left": rectangle.boundaries[0],
            "right": rectangle.boundaries[1],
            "bottom": rectangle.boundaries[2],
            "top": rectangle.boundaries[3],
        },
    )
    request = eqiora.meshing.CartesianMesher(cells=(2, 2))
    mesh = eqiora.meshing.generate(eqiora.meshing.resolve(geometry, request))
    return geometry, mesh


def scalar_output(
    spatial: eqiora.fem.Q1 | eqiora.fvm.CellCenteredTpfa,
) -> tuple[eqiora.geometry.Geometry, eqiora.meshing.Mesh, eqiora.FieldOutput]:
    geometry, mesh = rectangle_and_mesh()
    model = eqiora.compile(source=POISSON, geometry=geometry, entry='ViewerPoisson', bindings={**support_bindings(geometry, ['square'], [('left', 'square'), ('right', 'square'), ('bottom', 'square'), ('top', 'square')]), **{'diffusion': 1.0, 'source_scale': 1.0}})
    plan = eqiora.resolve(
        model,
        mesh=mesh,
        spatial=spatial,
        solve=eqiora.solve.Linear(
            algorithm=(eqiora.solve.LinearSolver.ConjugateGradient if spatial == eqiora.fvm.CellCenteredTpfa() else eqiora.solve.LinearSolver.BiConjugateGradientStabilized),
            preconditioner=eqiora.solve.Preconditioner.Identity,
            reduction=eqiora.solve.Reduction.Reproducible,
            provider=eqiora.solve.SolverProvider.reference(),
            relative_tolerance=1.0e-10,
            absolute_tolerance=1.0e-12,
            maximum_iterations=1_000,
        ),
    )
    result = eqiora.run(plan)
    return geometry, mesh, result.output(plan.capability.fields[0])


def document(*values: object) -> tuple[dict[str, object], tuple[bytes, ...]]:
    scene = _compose_view(values)
    return json.loads(scene.metadata_json), scene.buffers


def test_private_scene_composes_current_geometry_mesh_and_exact_selections() -> None:
    geometry, mesh = rectangle_and_mesh()
    metadata, buffers = document(geometry, mesh)
    layers = metadata["layers"]
    assert isinstance(layers, list)

    geometry_layer = next(layer for layer in layers if layer["kind"] == "geometry")
    mesh_layer = next(layer for layer in layers if layer["kind"] == "mesh")
    selections = [layer for layer in layers if layer["kind"] == "selection"]
    assert metadata["schema"] == "eqiora.viewer.scene/v0-private"
    assert metadata["presentation"] == {
        "camera": "disposable",
        "state_is_scientific": False,
    }
    assert metadata["reserved_layer_kinds"] == [
        "vector-field",
        "tensor-field",
        "trajectory",
    ]
    assert geometry_layer["owner_digest"] == geometry.digest
    assert mesh_layer["owner_digest"] == mesh.digest
    assert mesh_layer["source_digest"] == geometry.digest
    assert mesh_layer["cell_kind"] == "quadrilateral"
    assert {layer["name"] for layer in selections} == {
        "square",
        "left",
        "right",
        "bottom",
        "top",
    }
    assert any(
        layer["target_layer"] == geometry_layer["id"]
        and layer["name"] == "square"
        and layer["available"] is False
        and "exact face primitive" in layer["unavailable_reason"]
        for layer in selections
    )
    assert any(
        layer["target_layer"] == mesh_layer["id"]
        and layer["name"] == "square"
        and layer["available"] is True
        and layer["correspondence_digest"] == mesh.correspondence_digest
        for layer in selections
    )

    geometry_left = next(
        layer
        for layer in selections
        if layer["target_layer"] == geometry_layer["id"]
        and layer["name"] == "left"
    )
    geometry_positions = np.frombuffer(
        buffers[geometry_layer["positions"]["buffer"]], dtype="<f8"
    ).reshape((-1, 2))
    geometry_segments = np.frombuffer(
        buffers[geometry_layer["segments"]["buffer"]], dtype="<u4"
    ).reshape((-1, 2))
    left_primitives = np.frombuffer(
        buffers[geometry_left["entity_indices"]["buffer"]], dtype="<u4"
    )
    assert left_primitives.size == 1
    np.testing.assert_array_equal(
        geometry_positions[geometry_segments[left_primitives]][:, :, 0],
        np.zeros((1, 2)),
    )

    descriptors = metadata["buffers"]
    assert isinstance(descriptors, list)
    assert len(buffers) == len(descriptors)
    for index, (payload, descriptor) in enumerate(zip(buffers, descriptors, strict=True)):
        assert type(payload) is bytes
        assert descriptor["index"] == index
        assert descriptor["byte_length"] == len(payload)
        assert descriptor["sha256"] == hashlib.sha256(payload).hexdigest()

    coordinates = np.frombuffer(
        buffers[mesh_layer["coordinates"]["buffer"]], dtype="<f8"
    ).reshape(mesh.coordinates.shape)
    connectivity = np.frombuffer(
        buffers[mesh_layer["connectivity"]["buffer"]], dtype="<u4"
    ).reshape(mesh.cells.shape)
    np.testing.assert_array_equal(coordinates, mesh.coordinates)
    np.testing.assert_array_equal(connectivity, mesh.cells)
    assert not coordinates.flags.writeable
    assert not connectivity.flags.writeable


@pytest.mark.parametrize(
    ("spatial", "association"),
    [
        (eqiora.fem.Q1(), "vertex"),
        (eqiora.fvm.CellCenteredTpfa(), "cell"),
    ],
)
def test_scalar_field_preserves_owner_association_unit_and_accepted_values(
    spatial: eqiora.fem.Q1 | eqiora.fvm.CellCenteredTpfa,
    association: str,
) -> None:
    geometry, mesh, output = scalar_output(spatial)
    metadata, buffers = document(geometry, mesh, output)
    fields = [layer for layer in metadata["layers"] if layer["kind"] == "scalar-field"]
    assert len(fields) == 1
    field = fields[0]
    assert field["mesh_digest"] == mesh.digest
    assert field["model_digest"] == output.field.model_digest
    assert field["field_id"] == output.field.id
    assert field["observation_digest"] is None
    assert field["operator"] is None
    assert field["association"] == association
    assert field["component_shape"] == []
    assert field["unit"] == "coherent-si"
    assert field["dimension"] == [[value.numerator, value.denominator] for value in output.dimension]
    assert field["frame"] == "scalar"
    assert field["space"] == output.space
    assert field["scale"]["provenance"] == (
        "presentation-linear-range-from-accepted-values/v0"
    )
    values = np.frombuffer(buffers[field["values"]["buffer"]], dtype="<f8")
    np.testing.assert_array_equal(values, output.values(association))


def test_scalar_field_rejects_a_shape_matching_foreign_mesh() -> None:
    _, _, output = scalar_output(eqiora.fem.Q1())
    foreign_geometry, foreign_mesh = rectangle_and_mesh(x_bounds=(0.0, 2.0))
    with pytest.raises(eqiora.ValidationError, match="exact MeshLayer"):
        document(foreign_geometry, foreign_mesh, output)


def test_v2_rejects_vector_fields_explicitly() -> None:
    geometry, mesh = rectangle_and_mesh()
    model = eqiora.compile(source=ELASTICITY, geometry=geometry, entry='ViewerElasticity', bindings={**support_bindings(geometry, ['square'], [('left', 'square'), ('right', 'square'), ('bottom', 'square'), ('top', 'square')]), **{'stiffness': 1.0, 'lambda': 0.0, 'length_scale': 1.0}})
    plan = eqiora.resolve(
        model,
        mesh=mesh,
        spatial=eqiora.fem.Q1(),
        solve=eqiora.solve.Linear(
            algorithm=eqiora.solve.LinearSolver.ConjugateGradient,
            preconditioner=eqiora.solve.Preconditioner.Identity,
            reduction=eqiora.solve.Reduction.Reproducible,
            provider=eqiora.solve.SolverProvider.reference(),
            relative_tolerance=1.0e-10,
            absolute_tolerance=1.0e-12,
            maximum_iterations=1_000,
        ),
    )
    output = eqiora.run(plan).output(plan.capability.displacement)
    assert output.value_shape == (2,)
    with pytest.raises(eqiora.CapabilityError, match="scalar FieldOutput"):
        document(geometry, mesh, output)


def test_view_has_deterministic_text_fallback_and_explicit_lifecycle() -> None:
    geometry, mesh = rectangle_and_mesh()
    view = eqiora.View().add(geometry).add(mesh)
    expected = "View(layers=[Geometry, Mesh], closed=False)"
    assert repr(view) == expected
    assert view._repr_mimebundle_(include={"text/plain"}) == {"text/plain": expected}
    assert view._repr_mimebundle_(include={"image/png"}) == {}

    view.close()
    assert repr(view) == "View(layers=[], closed=True)"
    assert view._repr_mimebundle_() == {
        "text/plain": "View(layers=[], closed=True)\nViewer unavailable: this View is closed."
    }
    view.close()
    with pytest.raises(RuntimeError, match="closed"):
        view.add(mesh)


def test_base_import_does_not_load_notebook_frameworks() -> None:
    observed = subprocess.check_output(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import json, sys; import eqiora; "
                "print(json.dumps([name for name in "
                "('anywidget', 'traitlets', 'ipywidgets') if name in sys.modules]))"
            ),
        ],
        text=True,
    )
    assert json.loads(observed) == []


def test_installed_wheel_carries_viewer_assets_and_threejs_notice() -> None:
    package = resources.files("eqiora._viewer")
    assert len(package.joinpath("static/viewer.mjs").read_bytes()) > 100_000
    assert len(package.joinpath("static/viewer.css").read_bytes()) > 1_000
    notice = package.joinpath("THIRD_PARTY_NOTICES.txt").read_text(encoding="utf-8")
    assert "Three.js 0.185.1" in notice
    assert "The MIT License" in notice


def test_installed_base_distribution_emits_immutable_anywidget_payload() -> None:
    import anywidget  # noqa: F401
    import traitlets
    geometry, mesh = rectangle_and_mesh()
    view = eqiora.View().add(geometry).add(mesh)
    bundle = view._repr_mimebundle_()
    data = bundle[0] if isinstance(bundle, tuple) else bundle
    assert "application/vnd.jupyter.widget-view+json" in data
    delegate = view._delegate
    assert delegate is not None
    assert delegate.scene_metadata.startswith('{"schema":"eqiora.viewer.scene/v0-private"')
    assert delegate.buffers
    with pytest.raises(traitlets.TraitError, match="immutable"):
        delegate.scene_metadata = "{}"
    view.close()
    assert view._delegate is None
