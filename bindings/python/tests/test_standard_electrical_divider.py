"""One maintained divider through exact packages, common Plans and unique session APIs."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest
import eqiora

ROOT = Path(__file__).resolve().parents[3]
SPECIMEN = ROOT / "examples/voltage-divider"
SOURCE = (SPECIMEN / "src/main.eqi").read_text()
spec = importlib.util.spec_from_file_location(
    "divider_example", SPECIMEN / "run.py"
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def project(tmp_path, source=SOURCE):
    application = tmp_path / "divider"
    shutil.copytree(SPECIMEN, application)
    (application / "src/main.eqi").write_text(source)
    store = tmp_path / "store"
    store.mkdir()
    resolution = eqiora.add_bundled_dependency(
        application, store, "Eqiora.Electrical.Basic", version="0.1.0")
    return application, store, resolution


def direct_model(source=SOURCE):
    basic = (
        ROOT / "crates/eqiora-api/packages/Eqiora.Electrical.Basic/src/basic.eqi"
    ).read_text()
    direct = source.replace(
        "import Eqiora.Electrical.Basic.basic as electrical;", ""
    ).replace("electrical.", "")
    return eqiora.compile(source=basic + "\n" + direct, entry="VoltageDivider")


def assert_result(model, result, supply=12):
    # Ohm/Kirchhoff: I=V/3000, midpoint=2000*I. Into-component power sign
    # gives I²*1000, I²*2000, and -V*I. No solver-produced expectation.
    current = supply / 3000
    expected = {"current": current, "midpoint": 2000*current,
                "upper_power": current**2*1000, "lower_power": current**2*2000,
                "source_power": -supply*current}
    observed = {
        name: result.observe(model.observable(name)).value for name in expected
    }
    # Fixed SI problem and requested absolute/relative residual bounds are 1e-14/1e-12.
    # Voltage error is allowed 1e-10 V; current/power 1e-12 A/W.
    for name, value in expected.items():
        tolerance = 1e-10 if name == "midpoint" else 1e-12
        assert observed[name] == pytest.approx(value, abs=tolerance, rel=0)
    powers = ("upper_power", "lower_power", "source_power")
    assert sum(observed[name] for name in powers) == pytest.approx(
        0, abs=3e-12, rel=0
    )


def test_installed_divider_common_lifecycle_and_moved_offline_package(tmp_path):
    application, store, resolution = project(tmp_path)
    assert {node["identity"]["name"] for node in json.loads(resolution)["nodes"]} == {
        "org.example.Divider", "Eqiora.Electrical.Basic"}
    model = eqiora.compile_package(store, resolution, entry="VoltageDivider")
    plan = example.resolve(model)
    assert plan.mesh is None and plan.temporal is None
    state = eqiora.State.initial(plan)
    state = eqiora.State.from_bytes(plan, state.to_bytes())
    result = eqiora.run(plan, state=state)
    assert_result(model, result)
    restored = eqiora.Plan.from_bytes(plan.to_bytes())
    assert restored.to_bytes() == plan.to_bytes()
    assert_result(model, eqiora.run(restored, state=state))
    reopened_result = eqiora.Result.from_bytes(restored, result.to_bytes())
    assert reopened_result.to_bytes() == result.to_bytes()
    direct = direct_model()
    assert direct.structural_fingerprint == model.structural_fingerprint
    assert direct.package_compilation_digest is None
    assert model.package_compilation_digest is not None
    direct_plan = example.resolve(direct)
    direct_result = eqiora.run(direct_plan, state=eqiora.State.initial(direct_plan))
    assert_result(direct, direct_result)
    with pytest.raises(eqiora.ValidationError):
        eqiora.Result.from_bytes(direct_plan, result.to_bytes())
    with pytest.raises(ValueError, match="different exact Model artifact"):
        result.observe(direct.observable("midpoint"))
    changed = direct_model(SOURCE.replace("12[V]", "24[V]"))
    changed_plan = example.resolve(changed)
    changed_result = eqiora.run(changed_plan, state=eqiora.State.initial(changed_plan))
    assert_result(changed, changed_result, supply=24)
    with pytest.raises(eqiora.ValidationError):
        eqiora.Result.from_bytes(changed_plan, result.to_bytes())
    vendor = application / "vendor"
    vendor.mkdir()
    assert eqiora.vendor_project(application, store, vendor) == resolution
    shutil.rmtree(store)
    moved = tmp_path / "moved"
    application.rename(moved)
    assert eqiora.open_project(moved, moved / "vendor") == resolution
    offline = eqiora.compile_package(moved / "vendor", resolution, entry="VoltageDivider")
    assert offline.digest == model.digest
    assert offline.package_compilation_digest == model.package_compilation_digest
    offline_plan = example.resolve(offline)
    offline_result = eqiora.run(offline_plan, state=eqiora.State.initial(offline_plan))
    assert_result(offline, offline_result)


def test_session_observes_retained_exact_ports_and_advances(tmp_path):
    _, store, resolution = project(tmp_path)
    model = eqiora.compile_package(store, resolution, entry="VoltageDivider")
    ports = {label.selector: label.graph_id for label in model.notation_labels()}
    reopened = eqiora.Model.from_bytes(model.to_bytes())
    session = reopened.execution_session(end_time_s=1, max_step_s=1, inputs={})
    upper, lower = ports["upper.positive.voltage"], ports["lower.positive.voltage"]
    assert session.through(upper) == pytest.approx(0.004, abs=1e-12, rel=0)
    assert session.across(lower) == pytest.approx(8.0, abs=1e-10, rel=0)
    while session.advance():
        assert session.through(upper) == pytest.approx(0.004, abs=1e-12, rel=0)
        assert session.across(lower) == pytest.approx(8.0, abs=1e-10, rel=0)
    direct_ports = {
        label.selector: label.graph_id for label in direct_model().notation_labels()
    }
    foreign = direct_ports["upper.positive.voltage"]
    for invalid in ("missing", model.model_id, foreign, "01ARZ3NDEKTSV4RRFFQ69G5FAV"):
        for observe in (session.across, session.through):
            with pytest.raises(ValueError, match="exact scalar physical Port"):
                observe(invalid)


def test_installed_divider_rejects_missing_ground(tmp_path):
    floating = SOURCE.replace(
        "  instance ground: electrical.Ground();\n", ""
    ).replace(", ground.terminal", "")
    assert floating != SOURCE
    _, store, resolution = project(tmp_path, floating)
    model = eqiora.compile_package(store, resolution, entry="VoltageDivider")
    with pytest.raises(eqiora.ValidationError, match="unreferenced uniform shift"):
        example.resolve(model)
