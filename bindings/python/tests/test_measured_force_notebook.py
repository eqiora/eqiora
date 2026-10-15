import json
from pathlib import Path

import pytest


def test_measured_calibration_notebook():
    path = Path(__file__).resolve().parents[3] / "examples/python/measured_force_calibration.ipynb"
    notebook = json.loads(path.read_text())
    namespace = {}
    try:
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
        assert namespace["fitted_mass"] == pytest.approx(2., abs=1e-7, rel=0)
        assert namespace["history"][0][1] == 1.
        assert len(namespace["history"]) <= 8
        assert namespace["history"][-1][2] < 1e-12
        assert namespace["output_times"] == (.125, .5, .875, 1.)
        for index, time in enumerate(namespace["output_times"]):
            expected = time*time if time <= .5 else 2*time-time*time-.5
            observed = namespace["reopened"].observe(namespace["reopened_model"].observable(namespace["observable_ids"]["speed"]), output_index=index)
            assert observed.value == pytest.approx(expected, abs=1e-8, rel=0)
    finally:
        if "work" in namespace:
            namespace["work"].cleanup()
