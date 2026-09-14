"""Independent Q1 heat balance and exact offline package/Plan replay."""
import importlib.util
from pathlib import Path
import shutil

import numpy as np
import pytest
import eqiora


SPECIMEN = Path(__file__).resolve().parents[3] / "examples/heated-body"
spec = importlib.util.spec_from_file_location("heated_body_example", SPECIMEN / "run.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def check_heat(model, result, heating=1):
    field = model.field(model.authored_formulations[0].trial_field_id)
    temperatures = np.sort(result.output(field).values("vertex").numpy().reshape(-1))
    assert temperatures.shape == (9,)
    # Four h=1/2 Q1 squares: each central-hat stiffness contribution is 2/3.
    # The source integral against that hat is (1/2)^2=1/4 W/m (per depth).
    # Thus (8/3)*(T_center-300)=1/4, independently of the implementation.
    # 1e-10 K covers the requested residual divided by 8/3 and cancellation at 300 K.
    np.testing.assert_allclose(temperatures[:8], 300.0, atol=1e-10, rtol=0)
    assert temperatures[8] == pytest.approx(300 + heating*3/32, abs=1e-10, rel=0)
    assert (8/3)*(temperatures[8]-300) == pytest.approx(heating/4, abs=3e-10, rel=0)


def test_steady_heated_body_package_runs_and_moves_offline(tmp_path):
    project = tmp_path / "project"
    shutil.copytree(SPECIMEN, project)
    store = tmp_path / "store"
    store.mkdir()
    geometry, bindings = example.geometry_and_bindings()
    lock = eqiora.resolve_local_project(project, store)
    model = eqiora.compile_package(store, lock, entry="HeatedBody",
                                  geometry=geometry, bindings=bindings)
    plan = example.resolve(model, geometry)
    assert plan.formulation.requested == eqiora.FormulationSelectionMode.Authored
    assert len(model.authored_formulations[0].zero_on_domain_ids) == 4
    result = eqiora.run(plan)
    check_heat(model, result)
    restored = eqiora.Plan.from_bytes(plan.to_bytes())
    assert restored.to_bytes() == plan.to_bytes()
    check_heat(model, eqiora.run(restored))
    assert eqiora.Result.from_bytes(restored, result.to_bytes()).to_bytes() == result.to_bytes()
    direct = eqiora.compile(path=project / "src/main.eqi", entry="HeatedBody",
                            geometry=geometry, bindings=bindings)
    check_heat(direct, eqiora.run(example.resolve(direct, geometry)))
    assert direct.structural_fingerprint == model.structural_fingerprint
    assert model.package_compilation_digest is not None
    assert direct.package_compilation_digest is None
    vendor = project / "vendor"
    vendor.mkdir()
    assert eqiora.vendor_project(project, store, vendor) == lock
    shutil.rmtree(store)
    moved = tmp_path / "moved"
    project.rename(moved)
    assert eqiora.open_project(moved, moved / "vendor") == lock
    offline = eqiora.compile_package(moved / "vendor", lock, entry="HeatedBody",
                                    geometry=geometry, bindings=bindings)
    assert offline.digest == model.digest
    assert offline.package_compilation_digest == model.package_compilation_digest
    check_heat(offline, eqiora.run(example.resolve(offline, geometry)))
    stronger = eqiora.compile_package(moved / "vendor", lock, entry="HeatedBody",
                                     geometry=geometry, bindings={**bindings, "heating": 2.0})
    stronger_plan = example.resolve(stronger, geometry)
    check_heat(stronger, eqiora.run(stronger_plan), heating=2)
    assert stronger_plan.identity != plan.identity
    with pytest.raises(eqiora.ValidationError):
        eqiora.Result.from_bytes(stronger_plan, result.to_bytes())
