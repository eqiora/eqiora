"""Independent discrete scalar-transport oracle and ordinary lifecycle."""
import importlib.util
from pathlib import Path

import eqiora
import numpy as np
import pytest

SPECIMEN = Path(__file__).resolve().parents[3] / "examples/advection-diffusion"
spec = importlib.util.spec_from_file_location("advection_diffusion_example", SPECIMEN / "run.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def compile_model(source):
    geometry, bindings = example.geometry_and_bindings()
    model = eqiora.compile(source=source, entry="Main", geometry=geometry, bindings=bindings)
    return model, example.resolve(model, geometry)


def check(model, result):
    field = model.field("definition.concentration")
    for step, state in enumerate(result.trajectory.states, 1):
        actual = np.sort(np.asarray(state.field(field).values("vertex")).reshape(-1))
        # Independent Q1 M=1/9,K=8/3,F=1/4,C_ii=0; history ratio=1/7.
        expected = 3/32 * (1 - (1/7)**step)
        np.testing.assert_allclose(actual[:8], 0, rtol=0, atol=1e-10)
        assert actual[8] == pytest.approx(expected, rel=0, abs=1e-10)


def test_imported_conservative_transport_replay_restart_and_authored_equivalence():
    source = (SPECIMEN / "model.eqi").read_text()
    model, plan = compile_model(source)
    plan = eqiora.Plan.from_bytes(plan.to_bytes())
    initial = eqiora.State.initial(plan)
    initial = eqiora.State.from_bytes(plan, initial.to_bytes())
    result = eqiora.run(plan, state=initial, steps=3, output_steps=(1, 2, 3))
    check(model, result)
    assert eqiora.Result.from_bytes(plan, result.to_bytes()).to_bytes() == result.to_bytes()
    restart = result.trajectory.states[0]
    resumed = eqiora.run(plan, state=restart, steps=2, output_steps=(2,))
    field = model.field("definition.concentration")
    np.testing.assert_array_equal(resumed.trajectory.states[-1].field(field).values("vertex"), result.trajectory.states[-1].field(field).values("vertex"))
    start = source.index("  instance transport:")
    end = source.index("  instance west:")
    handwritten = source[:start] + """  law transport on body {
      storage concentration;
      flux concentration * (1[m/s]*grad(coordinate(0)) + 2[m/s]*grad(coordinate(1))) - 1[m^2/s]*grad(concentration);
      source 1[1/s];
    }
""" + source[end:]
    direct, direct_plan = compile_model(handwritten)
    direct_result = eqiora.run(direct_plan, state=eqiora.State.initial(direct_plan), steps=3, output_steps=(1, 2, 3))
    check(direct, direct_result)
    for imported, authored in zip(result.trajectory.states, direct_result.trajectory.states):
        np.testing.assert_allclose(imported.field(field).values("vertex"), authored.field(direct.field("definition.concentration")).values("vertex"), rtol=0, atol=1e-12)


@pytest.mark.parametrize("before,after", [
    ("diffusivity=1[m^2/s]", "diffusivity=1[m/s]"),
    ("velocity_x=1[m/s]", "velocity_x=1[1/s]"),
    ("velocity_x=1[m/s]", "velocity_x=concentration*1[m/s]"),
    ("ambient_dimension = 2", "ambient_dimension = 3"),
])
def test_transport_rejects_incompatible_bindings(before, after):
    source = (SPECIMEN / "model.eqi").read_text()
    assert before in source
    with pytest.raises(eqiora.ValidationError):
        compile_model(source.replace(before, after))
