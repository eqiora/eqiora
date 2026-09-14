"""Installed bundled controls execute the same physical example and exact lock."""
from fractions import Fraction
import json
import shutil
from pathlib import Path

import pytest

import eqiora


ROOT = Path(__file__).resolve().parents[3]
SOURCE = (ROOT / "examples/standard-sampled-components/src/main.eqi").read_text(encoding="utf-8")
CONTROLS = (ROOT / "crates/eqiora-api/packages/Eqiora.Controls.Sampled/src/sampled.eqi").read_text(encoding="utf-8")



# Independent physical recurrence: no standard declarations, normalization
# adapters or component expansion. Memory carries volts/metres directly.
EXPLICIT = """
model Main(
  input voltage: V at tick, input slew: V/s at tick,
  input position: m at tick, input velocity: m/s at tick,
  output delayed_voltage: V at tick, output voltage_before: V at tick,
  output voltage_after: V at tick, output delayed_position: m at tick,
  output position_before: m at tick, output position_after: m at tick
) {
  clock tick = periodic(0.25[s]);
  state previous_voltage: V at tick;
  state integrated_voltage: V at tick;
  state previous_position: m at tick;
  state integrated_position: m at tick;
  initial {
    pre(previous_voltage) = 10[V];
    pre(integrated_voltage) = 6[V];
    pre(previous_position) = 2.5[m];
    pre(integrated_position) = 1.5[m];
  }
  relation recurrence at tick {
    delayed_voltage = pre(previous_voltage);
    next(previous_voltage) = voltage;
    voltage_before = pre(integrated_voltage);
    next(integrated_voltage) = pre(integrated_voltage) + period(tick)*slew;
    voltage_after = next(integrated_voltage);
    delayed_position = pre(previous_position);
    next(previous_position) = position;
    position_before = pre(integrated_position);
    next(integrated_position) = pre(integrated_position) + period(tick)*velocity;
    position_after = next(integrated_position);
  }
}
"""

def replace_exact(source: str, needle: str, replacement: str, count: int) -> str:
    assert source.count(needle) == count, needle
    result = source.replace(needle, replacement)
    assert result != source
    return result


def direct_source(source: str) -> str:
    root = replace_exact(source, "import Eqiora.Controls.Sampled.sampled as controls;", "", 1)
    return CONTROLS + "\n" + replace_exact(root, "controls.", "", 4)


def project(tmp_path: Path, source: str) -> tuple[Path, Path, bytes]:
    application = tmp_path / "application"
    (application / "src").mkdir(parents=True)
    (application / "src/main.eqi").write_text(source, encoding="utf-8")
    (application / "eqiora.toml").write_text(
        '[package]\nname = "org.example.SampledControls"\nversion = "0.1.0"\nentry = "main"\n',
        encoding="utf-8",
    )
    store = tmp_path / "store"
    store.mkdir()
    resolution = eqiora.add_bundled_dependency(
        application, store, "Eqiora.Controls.Sampled", version="0.1.0"
    )
    # The root plus one dependency: controls must not accidentally pull in the
    # mechanics dependency needed by unrelated bundled continuum packages.
    assert {node["identity"]["name"] for node in json.loads(resolution)["nodes"]} == {
        "org.example.SampledControls", "Eqiora.Controls.Sampled"
    }
    return application, store, resolution


@pytest.mark.parametrize(
    ("denominator", "before", "after", "position_before", "position_after"),
    [(4, [3.0, 3.5, 3.25], [3.5, 3.25, 4.0], [3.0, 2.75, 3.75], [2.75, 3.75, 4.25]),
     (2, [3.0, 4.0, 3.5], [4.0, 3.5, 5.0], [3.0, 2.5, 4.5], [2.5, 4.5, 5.5])],
)
@pytest.mark.parametrize("transport", ["source", "locked", "vendored", "explicit"])
def test_installed_controls_execute_typed_samples_and_locked_restart(
    tmp_path: Path, denominator: int, before: list[float], after: list[float],
    position_before: list[float], position_after: list[float], transport: str,
) -> None:
    assert SOURCE.count("clock tick = periodic(0.25[s]);") == 1
    source = SOURCE if denominator == 4 else replace_exact(
        SOURCE, "clock tick = periodic(0.25[s]);", "clock tick = periodic(0.5[s]);", 1
    )
    if transport in {"locked", "vendored"}:
        application, store, resolution = project(tmp_path, source)
        assert eqiora.open_project(application, store) == resolution
        model = eqiora.compile_package(store, resolution, entry="Main")
        if transport == "vendored":
            vendor = application / "vendor"
            vendor.mkdir()
            assert eqiora.vendor_project(application, store, vendor) == resolution
            shutil.rmtree(store)
            moved = tmp_path / "moved-application"
            application.rename(moved)
            application, store = moved, moved / "vendor"
        reopened = eqiora.open_project(application, store)
        assert reopened == resolution
        replay = eqiora.compile_package(store, reopened, entry="Main")
        assert model.package_compilation_digest is not None
    elif transport == "explicit":
        explicit = EXPLICIT.replace("0.25[s]", f"{1 / denominator}[s]")
        model = eqiora.compile(source=explicit, entry="Main")
        replay = eqiora.compile(source=explicit, entry="Main")
        assert model.package_compilation_digest is None
    else:
        model = eqiora.compile(source=direct_source(source), entry="Main")
        replay = eqiora.compile(source=direct_source(source), entry="Main")
        assert model.package_compilation_digest is None
    assert model.digest == replay.digest
    assert model.package_compilation_digest == replay.package_compilation_digest
    # Start from the reopened package, including the first tick after relocation.
    session = replay.execution_session(
        end_time_s=2 / denominator, max_step_s=1 / denominator,
        inputs={
            "voltage": ("tick", [4.0, -2.0, 6.0]),
            "slew": ("tick", [4.0, -2.0, 6.0]),
            "position": ("tick", [-0.5, 2.0, 1.0]),
            "velocity": ("tick", [-0.5, 2.0, 1.0]),
        },
    )
    assert session.next_tick == Fraction(0)
    assert session.output("delayed_voltage", 0) is None
    assert session.output("voltage_before", 0) is None
    assert session.output("voltage_after", 0) is None
    assert session.advance_ticks(1) == 1
    resumed = model.resume_execution(session.checkpoint())
    assert session.advance_ticks(2) == resumed.advance_ticks(2) == 2
    for suffix, scale, delayed, old, updated in [
        ("voltage", 2.0, [5.0, 2.0, -1.0], before, after),
        ("position", 0.5, [5.0, -1.0, 4.0], position_before, position_after),
    ]:
        for name, expected in [
            (f"delayed_{suffix}", delayed),
            (f"{suffix}_before", old),
            (f"{suffix}_after", updated),
        ]:
            for index, value in enumerate(expected):
                sample = session.output(name, index)
                assert sample == resumed.output(name, index)
                assert sample is not None
                assert sample[0] == Fraction(index, denominator)
                # Default reference tolerances are absolute=relative=1e-10.
                # Values in this fixed problem are <=10, with <32 dependency
                # equations, <=2 scale factors and three ticks: this bound is
                # derived from residual acceptance, never observed output.
                assert sample[1] == pytest.approx(scale * value, abs=256 * 11e-10, rel=0)
            assert session.output(name, 3) is None


@pytest.mark.parametrize("mutation", ["rate-unit", "initial-value", "foreign-clock"])
def test_installed_controls_reject_invalid_component_wiring(tmp_path: Path, mutation: str) -> None:
    if mutation == "rate-unit":
        source = replace_exact(SOURCE, "input velocity: m / s", "input velocity: m", 2)
        fragment = "dimension"
    elif mutation == "initial-value":
        source = replace_exact(SOURCE, ", initial_value = 10[V] / 2[V]", "", 1)
        fragment = "required Parameter `initial_value`"
    else:
        source = replace_exact(SOURCE,
            "clock tick = periodic(0.25[s]);",
            "clock tick = periodic(0.25[s]); clock other = periodic(0.25[s]);",
            1,
        )
        source = replace_exact(source,
            "controls.DiscreteIntegrator(tick = tick",
            "controls.DiscreteIntegrator(tick = other",
            2,
        )
        fragment = "activation"
    with pytest.raises(eqiora.ValidationError) as failure:
        eqiora.compile(source=direct_source(source), entry="Main")
    assert any(diagnostic.code == "EQ0603" and fragment in diagnostic.message
               for diagnostic in failure.value.diagnostics)
    # Package preparation deliberately summarizes source diagnostics. The direct
    # probe above names the typed cause; here require that preparation stage,
    # not an arbitrary failure opening the store or executing a Model.
    with pytest.raises(eqiora.CompatibilityError) as failure:
        project(tmp_path, source)
    assert any(diagnostic.code == "EQ0901"
               and "package source analysis produced" in diagnostic.message
               for diagnostic in failure.value.diagnostics)
