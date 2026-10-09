# Harmonic response restriction

This case derives finite RC/RLC and one-dimensional scalar-wave amplitude problems
from original real, fixed-domain linear time-invariant Relations. It uses the common
complex algebraic/Q1 solver, retains the original Model and initial conditions,
and replays the distinct reduced Plan and accepted Result. It is a restriction to
responses proportional to `exp(-i*omega*t)`, not equivalence to every initial-value solution.

The [independent derivation](references/README.md) fixes signs, peak normalization,
angular-frequency conversion, expected values and tolerances before execution.
Negative probes remove source/boundary coverage or introduce nonlinear, time-varying
and mixed-DC mathematics; each must reach its intended admission diagnostic after
the ordinary positive path succeeds. Phase samples expose a Fourier-sign flip;
period-average resistor loss exposes peak/RMS confusion; separate original initial
conditions have distinct identities but the same settled response.

The RC comparison uses the existing host implicit-midpoint time backend for an
independently eliminated differential equation, with a bounded `sin(omega*t+pi/2)`
drive representing the same cosine. It does not claim the time owner accepts every
source operator or all forced circuit DAEs. There is no claim about arbitrary
spatial profiles, moving domains, universal LTI proof, mixed DC/harmonic decomposition,
optional backends, response derivatives, Python/package authoring or an arbitrary
initial condition being satisfied by the reconstructed sinusoid.

Run the repository gate with `--case numerics.harmonic-response`.
