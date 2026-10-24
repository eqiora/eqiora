"""Inputs for two ordinary installed-API workloads; no timed implementation here."""
import hashlib
import importlib
import json
from pathlib import Path

import eqiora
import numpy as np

ROOT = Path(__file__).parent
INITIAL_SHA256 = "e4dfc16fba39177d6f488c953134ed6c2344a6c46bfb83371b3ebcdae384f35e"


def cylinder(edition):
    _, plan, mesh, _ = importlib.import_module(edition + "_cylinder").prepare_plans()
    encoded = (ROOT / "cylinder-initial-data.json").read_bytes()
    if hashlib.sha256(encoded).hexdigest() != INITIAL_SHA256:
        raise ValueError("Original a7 physical initial coefficients changed")
    data = json.loads(encoded)
    fields = data["payload"]
    state = eqiora.State.initial(plan, time_s=data["time_s"], fields=(
        eqiora.InitialField(plan.capability.velocity,
            vertex_values=np.asarray(fields["velocity_vertex"]),
            cell_values=np.asarray(fields["velocity_cell"])),
        eqiora.InitialField(plan.capability.pressure,
            vertex_values=np.asarray(fields["pressure_vertex"]))))
    actual = json.loads(state.to_bytes())
    if actual["time_s"] != data["time_s"] or actual["payload"] != fields:
        raise ValueError("Physical initial data changed on binding")
    return plan, state, mesh


def decay(edition):
    model = eqiora.compile(path=ROOT / (edition + "_decay.eqi"))
    field = model.field(model.field_ids[0])
    key = field if edition == "a7" else (field, 0, 0, False)
    plan = eqiora.resolve(model, temporal=eqiora.time.Tsitouras45(
        initial_step_s=0.01, relative_tolerance=1e-9,
        absolute_tolerances={key: 1e-11}))
    return plan, eqiora.State.initial(plan), field
