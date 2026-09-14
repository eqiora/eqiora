# Steady heated body at a prescribed surface temperature

Run `python examples/heated-body/run.py` from an environment with Eqiora installed.
The script resolves the local `org.example.HeatedBody` package, binds caller-owned
Geometry, generates a 2 × 2 Q1 mesh, selects an explicit reference linear solver,
and runs the ordinary Plan lifecycle. No registry, implicit fetch, or provider
installation is needed. The maintained source is [src/main.eqi](src/main.eqi).

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

The stiffness coefficient carries W/(m K). Eight boundary coefficients are
300 K. This is an independently manufactured discrete Q1 heat balance, not an
exact continuum solution or a claim that raw boundary-gradient flux sums equal
the source. The product test checks both direct source and locked-package runs,
Plan and Result replay, then moves a vendored project and reopens it offline.
Package provenance remains distinct from direct-source provenance.

This steady specimen replaces the previous speculative heated-body example.
There is no time variable, initialization or thermal storage evolution in this
Model. Transient spatial thermal execution, a full three-dimensional heated
cube, bundled standard thermal Components, and curated-versus-lower-level
thermal package composition remain outside this specimen.

## Heat storage and initial conditions

The same package also exposes `TransientHeatedBody`. Supply the same Geometry
bindings and positive `capacity` in J/(m³ K). Its temperature State starts at
300 K, and all four boundary traces remain 300 K. Resolve with Q1 and
`eqiora.time.BackwardEuler(step_s=1/24)`, initialize with `eqiora.State.initial(plan)`,
then run with `steps=3, output_steps=(1, 2, 3)`. The accepted temperatures are
available from `result.trajectory.states`, through each State's exact temperature
Field snapshot. Plan, State and Result bytes use the same common replay APIs;
`State.from_result` selects a restart at an accepted time.

For unit conductivity, capacity and heating on the four-cell unit square, the
single interior Q1 coefficient has mass M=1/9, stiffness K=8/3 and load F=1/4.
Writing θ=T−300, BackwardEuler gives
`(M/dt + K) θ[n] = F + (M/dt) θ[n−1]`. With dt=1/24 s and θ[0]=0,
`T[n] = 300 + (3/32)*(1 - 2**(-n))` K. The installed test checks every step's
mass-plus-conduction balance, all eight prescribed boundary coefficients,
changed capacity/heating/step, restart and moved vendor-only execution.

This is a constant-capacity, single scalar Q1 Field on one Cartesian Region.
It does not claim a continuum-exact temperature profile, recovered physical
boundary flux, nonlinear or state-dependent capacity, interfaces, transient
sensitivity, or an authored transient Law-to-Form correspondence certificate.
The steady `HeatedBody` entry retains its separate authored-form evidence.
