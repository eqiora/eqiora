from __future__ import annotations

import shutil
import json
from pathlib import Path

import pytest

import eqiora


FLUID_MODEL = """import Eqiora.Fluid.fluid as fluid;

model Main() {
  domain body = box(0, 4, 0, 2);
  domain x_lower = boundary(body, axis = 0, side = lower);
  domain x_upper = boundary(body, axis = 0, side = upper);
  domain y_lower = boundary(body, axis = 1, side = lower);
  domain y_upper = boundary(body, axis = 1, side = upper);

  variable velocity: vector<m / s, 2> on body;
  variable pressure: kg / (m * s ^ 2) on body;
  parameter dynamic_viscosity: kg / (m * s) = 2;
  instance governing: fluid.SteadyStokesBalance2d(
    body = body,
    velocity = velocity,
    pressure = pressure,
    dynamic_viscosity = dynamic_viscosity
  );
}
"""


def project(root: Path) -> tuple[Path, Path]:
    application = root / "application"
    (application / "src").mkdir(parents=True)
    (application / "src/main.eqi").write_text(
        "import org.example.External.main as external;\n" + FLUID_MODEL,
        encoding="utf-8",
    )
    (application / "eqiora.toml").write_text(
        '[package]\nname = "org.example.Portable"\nversion = "1.0.0"\nentry = "main"\n'
        '[dependencies."org.example.External"]\nversion = "1.0.0"\nsources = [{ path = "../external" }]\n',
        encoding="utf-8",
    )
    store = root / "store"
    store.mkdir()
    external = root / "external"
    (external / "src").mkdir(parents=True)
    (external / "eqiora.toml").write_text(
        '[package]\nname = "org.example.External"\nversion = "1.0.0"\nentry = "main"\n',
        encoding="utf-8",
    )
    (external / "src/main.eqi").write_text('public model Shared() {}', encoding="utf-8")
    return application, store


def test_bundled_project_moves_with_one_offline_closure(tmp_path: Path) -> None:
    application, store = project(tmp_path)
    resolution = eqiora.add_bundled_dependency(
        application, store, "Eqiora.Fluid", version="0.5.0"
    )
    vendor = application / "vendor"
    vendor.mkdir()
    assert eqiora.vendor_project(application, store, vendor) == resolution
    assert eqiora.vendor_project(application, store, vendor) == resolution
    original = eqiora.compile_package(store, resolution, entry='Main')
    shutil.rmtree(store)
    shutil.rmtree(tmp_path / "external")
    moved = tmp_path / "moved"
    application.rename(moved)
    assert eqiora.open_project(moved, moved / "vendor") == resolution
    replay = eqiora.compile_package(moved / 'vendor', resolution, entry='Main')
    assert replay.revision.number == original.revision.number == 1
    assert json.loads((moved / "eqiora.lock").read_bytes())["resolution"] == json.loads(resolution)


def test_fetch_and_update_are_explicit_and_failed_add_is_atomic(tmp_path: Path) -> None:
    application, store = project(tmp_path)
    resolution = eqiora.add_bundled_dependency(
        application, store, "Eqiora.Fluid", version="0.5.0"
    )
    manifest = (application / "eqiora.toml").read_bytes()
    with pytest.raises(eqiora.CompatibilityError):
        eqiora.add_bundled_dependency(
            application, store, "Eqiora.Fluid", version="99.0.0"
        )
    assert (application / "eqiora.toml").read_bytes() == manifest
    assert json.loads((application / "eqiora.lock").read_bytes())["resolution"] == json.loads(resolution)
    second = tmp_path / "second"
    second.mkdir()
    assert eqiora.fetch_project(application, second) == resolution
    (application / "src/main.eqi").write_text('model Changed() {}', encoding="utf-8")
    with pytest.raises(eqiora.CompatibilityError):
        eqiora.open_project(application, second)
    with pytest.raises(eqiora.CompatibilityError):
        eqiora.fetch_project(application, second)
    updated = eqiora.update_project(application, second)
    assert updated != resolution
    assert eqiora.open_project(application, second) == updated
    without_fluid = eqiora.remove_local_dependency(application, second, "Eqiora.Fluid")
    assert eqiora.open_project(application, second) == without_fluid


def test_solid_is_an_ordinary_exact_bundled_dependency(tmp_path: Path) -> None:
    application, store = project(tmp_path)
    (application / "src/main.eqi").write_text(
        'model Main() { parameter gain: 1 = 2; relation law { gain - 2 = 0; } }',
        encoding="utf-8",
    )
    resolution = eqiora.add_bundled_dependency(
        application, store, "Eqiora.Solid.LinearElasticity", version="0.7.0"
    )
    assert eqiora.open_project(application, store) == resolution
    assert eqiora.compile_package(store, resolution, entry='Main').revision.number == 1


@pytest.mark.parametrize("displaced", ["Eqiora.Fluid.Incompressible", "Eqiora.Fluid.InertialStokes"])
def test_displaced_current_fluid_names_are_not_bundled_aliases(tmp_path, displaced):
    application, store = project(tmp_path)
    manifest = (application / "eqiora.toml").read_bytes()
    with pytest.raises(eqiora.CompatibilityError, match="no bundled package"):
        eqiora.add_bundled_dependency(application, store, displaced, version="0.5.0")
    assert (application / "eqiora.toml").read_bytes() == manifest
    assert not (application / "eqiora.lock").exists()
