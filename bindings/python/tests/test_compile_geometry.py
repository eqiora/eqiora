"""Focused installed-Python checks for the single root compile ingress."""

from importlib.resources import files
from pathlib import Path

import pytest

import eqiora
from _signature_bindings import support_bindings


def geometry() -> eqiora.geometry.Geometry:
    graph = eqiora.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0.0, 2.2), y_bounds=(0.0, 0.41))
    circle = graph.circle(center=(0.2, 0.2), radius=0.05)
    fluid = graph.subtract(rectangle, circle)
    return graph.build(fluid, named_topology={
        "fluid": fluid.region,
        "inlet": rectangle.boundaries[0],
        "outlet": rectangle.boundaries[1],
        "walls": rectangle.boundaries[2:],
        "cylinder": circle.boundaries[0],
    })


def parameters(source: eqiora.geometry.Geometry) -> dict[str, float]:
    return {
        **support_bindings(source, ['fluid'], [(n, 'fluid') for n in ['inlet', 'outlet', 'walls', 'cylinder']]),
        "dynamic_viscosity": 0.001,
        "zero_pressure": 0.0,
        "inlet_speed": 0.3,
        "channel_height": source.bounds[1][1] - source.bounds[1][0],
    }


def test_installed_path_and_loaded_source_have_one_model_meaning() -> None:
    source_path = files(eqiora).joinpath("examples", "steady-flow-past-cylinder.eqi")
    assert source_path.is_file()
    assert not files(eqiora).joinpath("examples", "steady-flow-past-cylinder.model.json").is_file()
    source = geometry()
    from_path = eqiora.compile(path=source_path, geometry=source, entry="SteadyFlowPastCylinder", bindings=parameters(source))
    from_text = eqiora.compile(
        source=source_path.read_text(encoding="utf-8"),
        filename="logical/cylinder.eqi",
        geometry=source,
        entry="SteadyFlowPastCylinder",
        bindings=parameters(source),
    )
    assert from_path.digest == from_text.digest
    assert eqiora.Model.from_bytes(from_path.to_bytes()).digest == from_path.digest


def test_source_shape_and_argument_admission_fail_closed(tmp_path: Path) -> None:
    root_source = """
model Main() {
  variable x: 1;
  relation balance { x - 1 = 0; }
}
"""
    assert eqiora.compile(source=root_source).digest
    with pytest.raises(TypeError):
        eqiora.compile(root_source)  # type: ignore[call-arg]
    with pytest.raises(eqiora.ValidationError, match="exactly one"):
        eqiora.compile()
    with pytest.raises(eqiora.ValidationError, match="exactly one"):
        eqiora.compile(path=tmp_path / "x.eqi", source=root_source)
    with pytest.raises(eqiora.ValidationError, match="filename"):
        eqiora.compile(path=tmp_path / "x.eqi", filename="logical.eqi")
    with pytest.raises(eqiora.ValidationError, match="explicit entry"):
        eqiora.compile(source=root_source, bindings={})

    invalid_utf8 = tmp_path / "invalid.eqi"
    invalid_utf8.write_bytes(b"\xff")
    with pytest.raises(eqiora.ValidationError, match="UTF-8"):
        eqiora.compile(path=invalid_utf8)

    oversized = tmp_path / "oversized.eqi"
    oversized.write_bytes(b" " * 8_388_609)
    with pytest.raises(eqiora.ValidationError, match="8388608-byte"):
        eqiora.compile(path=oversized)
    with pytest.raises(TypeError, match="Unicode"):
        eqiora.compile(path=b"invalid.eqi")


def test_component_and_parameter_inventory_are_source_owned() -> None:
    source_path = files(eqiora).joinpath("examples", "steady-flow-past-cylinder.eqi")
    source_text = source_path.read_text(encoding="utf-8")
    authored = geometry()
    values = parameters(authored)
    for invalid in (
        {name: value for name, value in values.items() if name != "channel_height"},
        {**values, "extra": 1.0},
    ):
        with pytest.raises(eqiora.ValidationError):
            eqiora.compile(source=source_text, geometry=authored, entry="SteadyFlowPastCylinder", bindings=invalid)
    # Boolean is a valid authored value, but cannot become a real velocity.
    with pytest.raises(eqiora.ValidationError) as mismatch:
        eqiora.compile(
            source=source_text,
            geometry=authored,
            entry="SteadyFlowPastCylinder",
            bindings={**values, "inlet_speed": True},
        )
    assert mismatch.value.diagnostics[0].code == "EQ0603"
    with pytest.raises(TypeError):
        eqiora.compile(
            source=source_text,
            geometry=authored,
            entry="SteadyFlowPastCylinder",
            bindings={**values, "inlet_speed": object()},  # type: ignore[dict-item]
        )
    for invalid in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            eqiora.compile(
                source=source_text,
                geometry=authored,
                entry="SteadyFlowPastCylinder",
                bindings={**values, "inlet_speed": invalid},
            )

    ambiguous = source_text + source_text.replace(
        "SteadyFlowPastCylinder", "OtherSteadyFlow", 1
    )
    with pytest.raises(eqiora.ValidationError, match="entry="):
        eqiora.compile(source=ambiguous, geometry=authored, bindings=values)
    selected = eqiora.compile(
        source=ambiguous,
        geometry=authored,
        bindings=values,
        entry="OtherSteadyFlow",
    )
    assert selected.digest


def test_source_named_boundary_parent_has_one_model_and_plan() -> None:
    from test_mixed_boundary_elasticity import geometry_and_mesh

    authored, mesh = geometry_and_mesh(cells=2)
    source_path = files(eqiora).joinpath("examples", "mixed-boundary-elasticity.eqi")
    explicit = {
        **support_bindings(authored, ["body"], [(name, "body") for name in
            ["x_lower", "x_upper", "y_lower", "y_upper"]]),
        "mu": 3.0, "lambda": 0.0, "length_scale": 1.0,
    }
    direct = {name: value[0] if isinstance(value, tuple) else value
              for name, value in explicit.items()}
    models = []
    plans = []
    for bindings in (explicit, direct):
        for source in (
            {"path": source_path},
            {"source": source_path.read_text(encoding="utf-8"), "filename": "logical/elasticity.eqi"},
        ):
            model = eqiora.compile(**source, geometry=authored,
                entry="MixedBoundaryElasticity2d", bindings=bindings)
            models.append(model)
            plans.append(eqiora.resolve(model, mesh=mesh, spatial=eqiora.fem.Q1(),
                solve=eqiora.solve.Linear(
                    algorithm=eqiora.solve.LinearSolver.ConjugateGradient,
                    preconditioner=eqiora.solve.Preconditioner.Identity,
                    reduction=eqiora.solve.Reduction.Reproducible,
                    provider=eqiora.solve.SolverProvider.reference(),
                    relative_tolerance=1e-10, absolute_tolerance=1e-12,
                    maximum_iterations=100)))
    assert len({model.digest for model in models}) == 1
    assert all(plan.to_bytes() == plans[0].to_bytes() for plan in plans)

    foreign = geometry()
    for invalid in (
        {name: value for name, value in direct.items() if name != "body"},
        {**direct, "body": authored.selection("x_upper")},
        {**direct, "body": foreign.selection("fluid")},
        {**direct, "x_lower": (authored.selection("x_lower"), authored.selection("x_upper"))},
    ):
        with pytest.raises((eqiora.ValidationError, ValueError)):
            eqiora.compile(path=source_path, geometry=authored,
                entry="MixedBoundaryElasticity2d", bindings=invalid)
