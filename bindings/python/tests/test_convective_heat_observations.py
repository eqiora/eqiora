"""Ordinary imported public heat observations; the canonical oracle owns all bounds."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import eqiora as q
import eqiora._eqiora as native_extension

REPOSITORY = Path(__file__).resolve().parents[3]
SPECIMEN = REPOSITORY / "examples/heated-body"
spec = importlib.util.spec_from_file_location("convective_heat_example", SPECIMEN / "run.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)
ORACLE = REPOSITORY / "verify/numerics/cartesian-robin-q1/reference/robin.py"
ORACLE_SHA = "200cc1ff5de8327d438506a75e4af59e03b188e730d4fb2b2b653ea06caa507d"
IDENTITIES = (
    ("left_heat", "left", "x_lower", "outward_conductive_heat", "W/m"),
    ("right_heat", "right", "x_upper", "outward_conductive_heat", "W/m"),
    ("bottom_heat", "bottom", "y_lower", "outward_conductive_heat", "W/m"),
    ("top_heat", "top", "y_upper", "outward_conductive_heat", "W/m"),
    ("right_temperature", "right", "x_upper", "temperature_trace", "K*m"),
    ("right_convective_heat", "right", "x_upper", "outward_convective_heat", "W/m"),
)
CASES = (
    ("insulated", (2, 1, 3, 0, 280, 320)),
    ("robin", (2, 1, 3, 2, 280, 320)),
    ("changed_h", (2, 1, 3, 5, 280, 320)),
    ("changed_geometry_material_cooling", (3, 2, 2, 4, 310, 280)),
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def persist(path, data):
    path.write_bytes(data)
    return digest(data)


@pytest.mark.parametrize("case,inputs", CASES)
def test_imported_public_heat_observations_and_offline_replay(tmp_path, case, inputs):
    assert digest(ORACLE.read_bytes()) == ORACLE_SHA
    root = Path(os.environ.get("EQIORA_ROBIN_PACKAGE_PYTHON_EVIDENCE", tmp_path))
    assert root.is_absolute() and not root.is_relative_to("/tmp")
    directory = root / case
    directory.mkdir(parents=True)
    native_path = Path(native_extension.__file__).resolve(strict=True)
    native_sha = digest(native_path.read_bytes())
    attempt = {"schema": 1, "case": case, "status": "attempt-started",
               "native_extension_path": str(native_path), "native_extension_sha256": native_sha,
               "producer_source_sha256": digest(Path(__file__).read_bytes())}
    (directory / "case-attempt.json").write_text(json.dumps(attempt, indent=2) + "\n")
    hashes = {}
    project = directory / "project"
    shutil.copytree(SPECIMEN, project)
    store = directory / "store"
    store.mkdir()
    geometry, bindings = example.convection_geometry_and_bindings(*inputs)
    lock = q.resolve_local_project(project, store)
    model = q.compile_package(store, lock, entry="ConvectiveHeatedBody",
                              geometry=geometry, bindings=bindings)
    assert model.package_compilation_digest is not None
    model_bytes = model.to_bytes()
    hashes["model.json"] = persist(directory / "model.json", model_bytes)
    restored_model = q.Model.from_bytes(model_bytes)
    assert restored_model.to_bytes() == model_bytes
    assert restored_model.digest == model.digest
    field = model.field("definition.temperature")
    assert field.model_digest == model.digest
    mesh = q.meshing.generate(q.meshing.resolve(
        geometry, q.meshing.CartesianMesher(cells=(2, 2))))
    hashes["mesh.eqmesh"] = persist(directory / "mesh.eqmesh", mesh.to_bytes())
    coordinates = np.asarray(mesh.coordinates)
    assert coordinates.shape == (9, 2) and np.isfinite(coordinates).all()
    assert len(set(map(tuple, coordinates))) == 9
    plan = example.resolve(model, geometry, mesh=mesh)
    plan_bytes = plan.to_bytes()
    hashes["plan.eqplan"] = persist(directory / "plan.eqplan", plan_bytes)
    replay_plan = q.Plan.from_bytes(plan_bytes)
    assert replay_plan.to_bytes() == plan_bytes
    assert replay_plan.identity == plan.identity
    result = q.run(replay_plan)
    result_bytes = result.to_bytes()
    hashes["result.eqresult"] = persist(directory / "result.eqresult", result_bytes)
    result_wire = json.loads(result_bytes)
    result_identity = result_wire["identity"]
    assert result_wire["content"]["plan_identity"] == replay_plan.identity
    replay = q.Result.from_bytes(replay_plan, result_bytes)
    replay_bytes = replay.to_bytes()
    assert replay_bytes == result_bytes
    replay_identity = json.loads(replay_bytes)["identity"]
    assert replay_identity == result_identity
    assert result.model_digest == replay.model_digest == model.digest
    primary_output = result.output(field)
    assert primary_output.value_type == q.ValueType.real(q.Dimension(temperature=1))
    primary = primary_output.values("vertex").numpy().reshape(-1)
    secondary = replay.output(restored_model.field("definition.temperature")).values("vertex").numpy().reshape(-1)
    assert primary.shape == secondary.shape == (9,)
    assert np.isfinite(primary).all()
    assert [float(x).hex() for x in primary] == [float(x).hex() for x in secondary]
    ordered = sorted(range(9), key=lambda index: tuple(coordinates[index]))
    records = [{"coordinates_m": [str(float(x)) for x in coordinates[index]],
                "value_binary64_hex": float(primary[index]).hex()} for index in ordered]
    observations, origin = [], []
    for name, boundary, authored_boundary, quantity, unit in IDENTITIES:
        selected = model.observable(f"definition.{name}")
        replay_selected = restored_model.observable(f"definition.{name}")
        domain = model.domain(f"definition.{authored_boundary}")
        assert selected.model_digest == replay_selected.model_digest == domain.model_digest == model.digest
        assert replay_selected.id == selected.id
        accepted = result.observe(selected, quadrature_points=2)
        reopened = replay.observe(replay_selected, quadrature_points=2)
        expected_type = (q.ValueType.real(q.Dimension(temperature=1, length=1)) if unit == "K*m"
                         else q.ValueType.real(q.Dimension(mass=1, length=1, time=-3)))
        for sample, owner_identity in ((accepted, result_identity), (reopened, replay_identity)):
            assert sample.observable_id == selected.id
            assert sample.result_identity == owner_identity
            assert sample.value_type == expected_type
            assert sample.quadratures == {domain.id: ("GaussLegendre", 1, 2)}
            assert np.isfinite(sample.value)
        observations.append({"name": name, "boundary": boundary, "quantity": quantity, "unit": unit,
            "quadrature": {"family": "gauss-legendre", "points_per_segment": 2},
            "value_binary64_hex": float(accepted.value).hex(),
            "replay_value_binary64_hex": float(reopened.value).hex()})
        origin.append({"name": name, "observable_id": selected.id, "model_digest": selected.model_digest,
                       "boundary_domain_id": domain.id, "authored_boundary": authored_boundary,
                       "result_identity": accepted.result_identity, "replay_result_identity": reopened.result_identity,
                       "value_type": repr(accepted.value_type), "quadratures": accepted.quadratures})
    data = {"schema": 1, "kind": "public-observations", "case": case,
            "order": [record["coordinates_m"] for record in records],
            "units": {"coordinates": "m", "coefficients": "K"},
            "coefficients": records, "observations": observations}
    hashes["observations.json"] = persist(directory / "observations.json", json.dumps(data, indent=2).encode())
    vendor = project / "vendor"
    vendor.mkdir()
    assert q.vendor_project(project, store, vendor) == lock
    shutil.rmtree(store)
    moved = directory / "offline-project"
    project.rename(moved)
    assert q.open_project(moved, moved / "vendor") == lock
    offline = q.compile_package(moved / "vendor", lock, entry="ConvectiveHeatedBody",
                                geometry=geometry, bindings=bindings)
    assert offline.digest == model.digest
    assert offline.package_compilation_digest == model.package_compilation_digest
    receipt = {"schema": 1, "case": case, "source_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
        "producer_source_sha256": digest(Path(__file__).read_bytes()), "oracle_sha256": ORACLE_SHA,
        "python_executable_sha256": digest(Path(sys.executable).read_bytes()),
        "native_extension_path": str(native_path), "native_extension_sha256": native_sha, "artifact_sha256": hashes,
        "package_compilation_digest": model.package_compilation_digest, "model_digest": model.digest,
        "plan_identity": plan.identity, "result_identity": result_identity, "field_id": field.id,
        "coefficient_entities": [{"entity_dimension": 0, "entity_index": index,
                                   "coordinates_m": coordinates[index].tolist()} for index in ordered],
        "observables": origin, "ordinary_public_runs": 1,
        "canonical_model_reopen": True, "canonical_plan_reopen": True, "canonical_result_reopen": True,
        "replay_coefficients_equal": True, "offline_compilation_equal": True,
        "scope": "public coefficients and observations; no full A/b or private weak reactions"}
    (directory / "producer-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    comparison = directory / "comparison.json"
    checked = subprocess.run([sys.executable, str(ORACLE), "compare-observations",
                              str(directory / "observations.json"), "--output", str(comparison)], check=False)
    report = json.loads(comparison.read_text())
    assert report["input_sha256"] == hashes["observations.json"]
    assert checked.returncode == 0 and report["accepted"], report
    assert report["counts"] == {"coefficients": 9, "free": 6, "observations": 6, "observation_evaluations": 12}
    assert all(report["checks"].values())
    assert all(all(item["checks"].values()) for item in report["observations"])
