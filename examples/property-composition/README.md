# Exact conductivity composition

This local package evaluates two sampled Components that share one synthetic
conductivity table. It is a runnable example package, not part of the standard
library, and it does not solve transient heat conduction. The public declarations
and their documentation are in `src/main.eqi`.

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

Values can be evaluated at the endpoints and at the shared middle knot; outside
[300 K, 360 K] evaluation fails, with no extrapolation. Derivatives fail at the
endpoints and at the non-smooth middle knot. The clock only schedules independent
evaluations of the property; nothing evolves thermally.

The installed test `bindings/python/tests/test_property_composition_specimen.py` checks
five temperatures, both consumers, Model replay, and the moved offline package.
A reopened Model artifact identifies Ports and Clocks by their ULIDs, not by
source names, so record those identities when choosing inputs and outputs for
artifact-based execution. Separate tests check the table's derivatives and asset
authentication. To vendor the package, create `project / "vendor"` and call
`eqiora.vendor_project(project, store, project / "vendor")`; `eqiora.open_project`
then reopens the lock without fetching anything.
