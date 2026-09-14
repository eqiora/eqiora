# Exact conductivity composition

This local package evaluates two ordinary sampled Components with one exact synthetic
conductivity release. It is an executable package specimen, not a bundled standard release
or a transient heat solver. Public declarations and their docs are in `src/main.eqi`.

From the repository root with Eqiora installed:

```python
from pathlib import Path
import eqiora

project = Path("examples/property-composition")
store = Path("property-store")
store.mkdir(exist_ok=True)
lock = eqiora.resolve_local_project(project, store)
model = eqiora.compile_package(store, lock, entry="PropertyConsumers")
session = model.execution_session(
    end_time_s=1, max_step_s=1,
    inputs={"temperature": ("tick", [310.0, 340.0])},
)
session.advance_ticks(2)
print(session.output("heat_flux", 0))       # -24 W/m² at the first sample
print(session.output("conductance", 1))     # 1.6 W/K at the second sample
```

The table is (300 K, 10 W/(m K)), (320 K, 14 W/(m K)), (360 K, 18 W/(m K)).
The two line slopes are 0.2 and 0.1 W/(m K²), so k(310)=12 and k(340)=16.
Fourier's law gives q=-k·2 K/m: -24 and -32 W/m². The slab gives
G=k·0.01 m²/0.1 m: 1.2 and 1.6 W/K. These are constructed instructional values.

Value evaluation admits the endpoints and shared knot. Outside [300 K, 360 K] it rejects;
there is no extrapolation. Derivatives reject the endpoints and nonsmooth middle knot.
The explicit clock schedules independent constitutive evaluations, not thermal evolution.

The installed test `bindings/python/tests/test_property_composition_specimen.py` checks
five temperatures, both consumers, Model replay and the moved offline package.
Reopened Model artifacts retain exact Port/Clock ULIDs, not source aliases; save those
identities when selecting inputs and outputs for artifact-based execution. The existing
source table tests check derivatives and exact asset authentication. Vendor with
`eqiora.vendor_project(project, store, project / "vendor")` after creating that directory;
`eqiora.open_project` then reopens the exact lock without fetching anything.
