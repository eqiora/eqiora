# Reference provenance

The continuum solution and source are derived in
[`models/problem.md`](../models/problem.md). The convergence table is generated
from the independent tensor sine-mode formulas shared with the 2D case.

See the [shared reference derivation and generator](../../cartesian-poisson-fem-fvm/references/README.md)
for the exact discrete amplitudes, analytic continuous norms, regeneration command,
and solver/roundoff comparison allowance. No native-backend output is used as
an expected value. The three-dimensional test additionally checks monotonicity,
order greater than 1.9, and relative global balance below `2e-11` directly.
