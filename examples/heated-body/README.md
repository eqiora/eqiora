# Steady heated body at a prescribed surface temperature

Run `python examples/heated-body/run.py` from an environment with Eqiora installed.
The script resolves the local `org.example.HeatedBody` package, binds the
geometry it creates, generates a 2 × 2 Q1 mesh, selects the reference linear
solver explicitly, and runs the standard Plan lifecycle. No registry, download,
or provider installation is needed. The model source is [src/main.eqi](src/main.eqi).

The unit square is a two-dimensional cross-section per unit depth. Conductivity
is 1 W/(m K), volumetric heating is 1 W/m³, and all four exterior boundaries are
held at 300 K. Fourier flux is −k grad(T); the steady balance is div(flux) = heating.
The test function vanishes on the complete exterior, while the trial temperature
retains its nonzero prescribed data.

Four h = 1/2 Q1 cells leave one free central hat. Each cell contributes 2/3 to its
stiffness, and its source integral is 1/4 W/m. Thus

```text
(8/3) × (T_center − 300 K) = 1/4 W/m
T_center = 300 + 3/32 K = 300.09375 K
```

The stiffness coefficient has units of W/(m K), and the eight boundary
coefficients are 300 K. This is a hand-derived discrete Q1 heat balance, not the
exact continuum solution, and summing raw boundary-gradient fluxes does not
reproduce the source exactly. The product test checks runs from direct source and
from the locked package, Plan and Result replay, and reopening a moved, vendored
project offline. Package provenance stays distinct from direct-source provenance.

`HeatedBody` itself is steady: it has no time variable, initial condition, or
heat storage. A full three-dimensional heated cube, bundled standard thermal
Components, and combining curated thermal packages with lower-level ones are not
part of this example.

## Heat storage and initial conditions

The same package also exposes `TransientHeatedBody`. Supply the same Geometry
bindings and positive `capacity` in J/(m³ K). Its temperature State starts at
300 K, and all four boundary traces remain 300 K. Resolve with Q1 and
`eqiora.time.BackwardEuler(step_s=1/24)`, initialize with `eqiora.State.initial(plan)`,
then run with `steps=3, output_steps=(1, 2, 3)`. The temperature after each step
is in `result.trajectory.states`, as each State's temperature Field snapshot.
Plan, State, and Result bytes use the standard replay APIs, and
`State.from_result` restarts from any accepted time.

For unit conductivity, capacity and heating on the four-cell unit square, the
single interior Q1 coefficient has mass M=1/9, stiffness K=8/3 and load F=1/4.
Writing θ=T−300, BackwardEuler gives
`(M/dt + K) θ[n] = F + (M/dt) θ[n−1]`. With dt=1/24 s and θ[0]=0,
`T[n] = 300 + (3/32)*(1 - 2**(-n))` K. The installed test checks every step's
mass-plus-conduction balance, all eight prescribed boundary coefficients,
changed capacity/heating/step, restart and moved vendor-only execution.

The transient model has one scalar Q1 field with constant capacity on one
Cartesian Region. It does not give the exact continuum temperature profile,
recover the physical boundary flux, or support nonlinear or state-dependent
capacity, interfaces, transient sensitivities, or a certificate linking the
authored transient law to its weak form. The steady `HeatedBody` keeps its own
authored-form evidence.
