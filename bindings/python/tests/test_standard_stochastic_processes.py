"""Installed source packages use the ordinary retained stochastic time lifecycle."""
from pathlib import Path
import json
import shutil

import eqiora as q
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "examples/stochastic-processes"
PACKAGE = ROOT / "crates/eqiora-api/packages/Eqiora.Stochastic.Processes"


@pytest.mark.parametrize("zero_diffusion", [False, True])
def test_packaged_processes_execute_and_replay_from_an_offline_store(tmp_path, zero_diffusion):
    project = tmp_path / "application"
    shutil.copytree(EXAMPLE, project)
    if zero_diffusion:
        source_path = project / "src/main.eqi"
        source = source_path.read_text(encoding="utf-8")
        source = source.replace("volatility = 0.25", "volatility = 0")
        source = source.replace("volatility = 0.5", "volatility = 0")
        source_path.write_text(source, encoding="utf-8")
    store = tmp_path / "store"
    store.mkdir()
    resolution = q.add_bundled_dependency(
        project, store, "Eqiora.Stochastic.Processes", version="0.1.0"
    )
    assert {node["identity"]["name"] for node in json.loads(resolution)["nodes"]} == {
        "org.example.StochasticProcesses", "Eqiora.Stochastic.Processes"
    }
    vendor = project / "vendor"
    vendor.mkdir()
    assert q.vendor_project(project, store, vendor) == resolution
    shutil.rmtree(store)
    assert q.open_project(project, vendor) == resolution
    model = q.compile_package(vendor, resolution, entry="Main")
    assert model.package_compilation_digest is not None
    plan = q.resolve(model, temporal=q.time.EulerMaruyama(depth=6))
    fields = [model.field("ou.value"), model.field("gbm.value")]
    controls = dict(
        state=q.State.initial(plan), until_s=1., output_times_s=(.5, 1.),
        master_seed=bytes(range(32)), sampling=q.SamplingIdentity(b"member", b"path"),
    )
    first = q.run(plan, **controls)
    replay = q.run(plan, **controls)
    for field in fields:
        np.testing.assert_array_equal(first.series(field).values, replay.series(field).values)
    if zero_diffusion:
        # Closed forms of the authored zero-noise recurrences, h=1/64.
        # At most four rounded scalar operations per step, 64 steps, and
        # magnitude/amplification bounded by four give 1024 eps. This bounds
        # arithmetic disagreement, not stochastic or discretization error.
        steps = np.array([32, 64])
        roundoff = 1024 * np.finfo(float).eps
        np.testing.assert_allclose(
            first.series(fields[0]).values, .5 + .5 * (1 - 2 / 64) ** steps,
            rtol=0., atol=roundoff,
        )
        np.testing.assert_allclose(
            first.series(fields[1]).values, (1 + .25 / 64) ** steps,
            rtol=0., atol=roundoff,
        )
    else:
        # Distinct addresses must select different realizations. This product
        # check makes no statistical independence or distribution claim.
        other = q.run(plan, **(controls | {"sampling": q.SamplingIdentity(b"other", b"path")}))
        for field in fields:
            assert not np.array_equal(first.series(field).values, other.series(field).values)


def test_stratonovich_source_cannot_silently_use_the_ito_method():
    components = (PACKAGE / "src/processes.eqi").read_text(encoding="utf-8")
    source = (EXAMPLE / "src/main.eqi").read_text(encoding="utf-8")
    source = source.replace("import Eqiora.Stochastic.Processes.processes as processes;", "")
    source = source.replace("processes.", "")
    model = q.compile(source=components.replace("calculus ito;", "calculus stratonovich;") + source)
    with pytest.raises(q.ValidationError):
        q.resolve(model, temporal=q.time.EulerMaruyama(depth=6))
