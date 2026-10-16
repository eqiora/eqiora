# Coupled Ito path refinement

4096 fixed sample identities, seed `[83;32]`, channel `sde-strong`, and 8,
32 and 128 steps are fixed before candidate execution. This channel differs
from the [distribution campaign](../ito-distribution-moments/README.md).
The same identity and nominal Wiener channel drive all refinements of a member.
Every eighth-horizon Brownian output must agree across grids up to accumulation
roundoff. Exact retained bridge composition is independently covered by
`time.retained-wiener-bridge`.

The [ordinary source](models/geometric-brownian.eqi) evolves GBM and a Brownian
State using the same noise. Both go through the common time Plan/Run/Result
path. The Brownian State makes the realized driver observable without deriving
it from GBM values or accessing a mutable RNG.

## Independent path reference

For initial value 1, duration 1, `a=1/4` and `b=1/2`, Ito's formula gives

`X_exact = exp((a-b^2/2) + b W_1)`.

For `n` Euler–Maruyama steps, write `h=1/n` and independent `Z_i ~ N(0,h)`.
The numerical endpoint is `X_h = product_i (1+a*h+b*Z_i)`. Exponential tilting
of each Gaussian gives the independently derived joint moments

`J(k,l) = E[X_h^k X_exact^l]`

`= exp(l*a + (l^2-l)*b^2/2) E[(1+a*h+l*b^2*h+b*Z)^k]^n`.

Consequently the exact finite-grid mean squared error is
`M2 = J(2,0)-2 J(1,1)+J(0,2)`, and its fourth error moment is
`M4 = sum_(k=0..4) binomial(4,k) (-1)^(4-k) J(k,4-k)`.
The sample mean squared error must lie within
`6 sqrt((M4-M2^2)/4096)` of `M2`, at each fixed grid. The finest observed MSE
must also be below one quarter of the coarsest observed MSE. This is a bounded
refinement experiment, not an inferred universal order theorem.

The shared independent [reference script](../ito-distribution-moments/references/analytic.py)
evaluates these polynomial/exponential formulas at 70-digit precision to avoid
cancellation in the fourth error moment. Its [fixed expected values](../ito-distribution-moments/expected/campaign.json)
are not measured candidate outputs.

## Falsifier and boundary

Replacing the exact target with `exp(a+b W_1)` omits the Ito correction.
Its measured squared error must fail the same numerical predicate at every
declared grid. It reaches the scientific error criterion, rather than relying
on a source-calculus or Model-identity rejection.

The driver State sums separately rounded increments; the 4096-epsilon
coarse/fine comparison is an arithmetic guard for the at-most-128-step
projection, not the strong-error acceptance criterion. No positivity guarantee
for Euler–Maruyama, adaptive error controller, general stiff SDE/SPDE solver,
trajectory derivative, or cross-device stream identity is claimed.
