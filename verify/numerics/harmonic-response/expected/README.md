# Expected semantic projections

The case checks numerical values and typed lineage, not a whole-source byte digest.

| Profile | Voltage / wave amplitude | Current amplitude |
|---|---|---|
| RC, omega*R*C=1 | 0.5+0.5i V | 0.0005-0.0005i A |
| RLC at omega²*L*C=1 | i V | 0.001 A |
| Constant scalar wave | 2+i at every Q1 vertex | — |

RC real voltages at phases 0, pi/2, pi are 0.5, 0.5, -0.5 V; corresponding
currents are 0.0005, -0.0005, -0.0005 A. Wave coefficients at the same phases
are 2, 1, -2. RC period-average resistor loss is 0.00025 W.

The cyclic-frequency and transient comparisons use the analytic formulas and
error bounds in [the derivation](../references/README.md). Initial values 0 and
1 V produce different original/Plan identities, the same settled phasor, and
oppositely signed decaying RC transients. Neither initial value is 0.5 V.

Removed wave boundary Relations must name complete noninitial Relation coverage;
removed body/boundary excitations must name the missing excitation. Nonlinear,
time-varying, mixed-DC and nonpositive-frequency probes must name their corresponding
admission gates. Wrong original Field identity, absent block and nonfinite or
dimensionally invalid reconstruction time must reject. Canonical Plan and Result
replay is checked separately from numerical comparisons.
