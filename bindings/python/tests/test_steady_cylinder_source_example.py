"""The documented authoring example uses the installed public Python API."""
from pathlib import Path
import runpy

import eqiora

from _signature_bindings import support_bindings
from test_language_source import PARAMETERS, cylinder_geometry


def test_steady_cylinder_example_compiles_from_module_and_emitted_source():
    program = Path(__file__).resolve().parents[3] / "examples/python/steady_cylinder_source.py"
    module = runpy.run_path(str(program))["build_source"]()
    geometry = cylinder_geometry()
    bindings = {
        **support_bindings(geometry, ["fluid"],
                           [(name, "fluid") for name in ("inlet", "outlet", "walls", "cylinder")]),
        **PARAMETERS,
    }
    direct = eqiora.compile(source=module, geometry=geometry,
                            entry="SteadyFlowPastCylinder", bindings=bindings)
    emitted = eqiora.compile(source=module.to_eqi(), geometry=geometry,
                             entry="SteadyFlowPastCylinder", bindings=bindings)
    assert direct.to_bytes() == emitted.to_bytes()
    assert len(direct.field_ids) == 4
