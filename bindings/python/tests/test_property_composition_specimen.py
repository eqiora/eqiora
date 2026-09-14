"""The maintained local package executes the independently derived two-consumer law."""
import json
from pathlib import Path
import shutil

import pytest

import eqiora


SPECIMEN = Path(__file__).resolve().parents[3] / "examples/property-composition"


def assert_values(model, identities=None):
    identities = identities or {name: name for name in ("temperature", "tick", "heat_flux", "conductance")}
    # Slopes (14-10)/(320-300)=.2 and (18-14)/(360-320)=.1.
    # Fourier q=-2*k and slab G=(.01/.1)*k; endpoint values are admitted.
    temperatures = [300.0, 310.0, 320.0, 340.0, 360.0]
    session = model.execution_session(
        end_time_s=4, max_step_s=1,
        inputs={identities["temperature"]: (identities["tick"], temperatures)},
    )
    assert session.advance_ticks(5) == 5
    for index, (flux, conductance) in enumerate(
        [(-20.0, 1.0), (-24.0, 1.2), (-28.0, 1.4), (-32.0, 1.6), (-36.0, 1.8)]
    ):
        assert session.output(identities["heat_flux"], index)[1] == pytest.approx(flux, abs=1e-12, rel=0)
        assert session.output(identities["conductance"], index)[1] == pytest.approx(conductance, abs=1e-12, rel=0)


def test_maintained_property_composition_moves_and_replays_offline(tmp_path):
    project = tmp_path / "project"
    shutil.copytree(SPECIMEN, project)
    store = tmp_path / "store"
    store.mkdir()
    lock = eqiora.resolve_local_project(project, store)
    assert {node["identity"]["name"] for node in json.loads(lock)["nodes"]} == {
        "org.example.PropertyComposition"
    }
    model = eqiora.compile_package(store, lock, entry="PropertyConsumers")
    assert len(model.property_bindings) == 2
    assert len({binding.release for binding in model.property_bindings}) == 1
    assert_values(model)
    reopened = eqiora.Model.from_bytes(model.to_bytes())
    assert reopened.to_bytes() == model.to_bytes()
    identities = {label.selector: label.graph_id for label in model.notation_labels()}
    # Model bytes retain exact execution identities, not source aliases. This root
    # declares one clock, shared by both consumers; recover that retained identity.
    clocks = [node["id"]["ulid"] for node in json.loads(model.to_bytes())["nodes"]
              if node["definition"]["kind"] == "clock-domain"]
    assert len(clocks) == 1
    identities["tick"] = clocks[0]
    assert_values(reopened, identities)
    for bad in [model.model_id, identities["tick"], "01ARZ3NDEKTSV4RRFFQ69G5FAV"]:
        with pytest.raises(ValueError, match="not an exact Port"):
            reopened.execution_session(end_time_s=0, max_step_s=1,
                inputs={bad: (identities["tick"], [310.0])})
    with pytest.raises(ValueError, match="not an exact ClockDomain"):
        reopened.execution_session(end_time_s=0, max_step_s=1,
            inputs={identities["temperature"]: (identities["heat_flux"], [310.0])})
    empty = reopened.execution_session(end_time_s=0, max_step_s=1,
        inputs={identities["temperature"]: (identities["tick"], [310.0])})
    with pytest.raises(ValueError, match="not an exact Port"):
        empty.output(identities["tick"], 0)
    vendor = project / "vendor"
    vendor.mkdir()
    assert eqiora.vendor_project(project, store, vendor) == lock
    shutil.rmtree(store)
    moved = tmp_path / "moved"
    project.rename(moved)
    assert eqiora.open_project(moved, moved / "vendor") == lock
    offline = eqiora.compile_package(moved / "vendor", lock, entry="PropertyConsumers")
    assert offline.digest == model.digest
    assert offline.package_compilation_digest == model.package_compilation_digest
    assert_values(offline)
    for temperature in [299.0, 361.0]:
        with pytest.raises(eqiora.ExecutionError, match="required expression domain condition is false"):
            session = offline.execution_session(
                end_time_s=0, max_step_s=1,
                inputs={"temperature": ("tick", [temperature])},
            )
            session.advance_ticks(1)
