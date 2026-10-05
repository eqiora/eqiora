# Common complex and real implicit midpoint

The same compiled Model, first-order lowering, ODE Plan, State, Run request and
Trajectory evolve two complex channels and a real two-coordinate oscillator.
The executable evidence is `crates/eqiora/tests/complex_midpoint_time.rs`.

This case proves the bounded phase, amplitude and convergence statements derived
in [expected/README.md](expected/README.md), for identity and dense constant mass.
Plan/State replay, accepted-state restart, identical internal history under changed
output cadence, stale State rejection and invalid initial-domain rejection belong
to the same ordinary lifecycle. An overflowing imaginary State component and
unsupported midpoint event controls must reject.

The claim does not cover complex DAEs, event localization, nonlinear convergence,
all mass matrices, all precisions or optional backends. Model array size is not an
implementation limit; the two-channel fixture defines this scientific claim.
