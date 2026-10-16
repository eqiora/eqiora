# Fixed-grid Ito distribution moments

The ordinary source components `OrnsteinUhlenbeck` and
`GeometricBrownianMotion` execute through common Plan, initial State, sampled
Run and accepted Result owners. `MomentEstimator` computes their typed mean
and unbiased sample variance. There is no evidence-only integrator or reducer.

The campaign was fixed before candidate execution: 4096 sample identities
`u32::to_le_bytes(0..4096)`, master seed `[83; 32]`, channel `sde-weak`, horizon
1 second, and 8, 32 and 128 Euler–Maruyama steps. Each component declares its
own nominal independent Wiener channel. Independence and finite moments are
the statistical assumptions of this bounded experiment; distinct IDs alone
do not prove them. Sampler/environment restrictions are those of
[retained Wiener paths](../retained-wiener-bridge/README.md).

## Independent reference

For OU, `dX = lambda (mu-X) dt + sigma dW`, with initial value 1,
`lambda=2`, `mu=1/2`, `sigma=1/4`, and `h=1/n`, let `r=1-lambda*h`.
The affine recurrence has exact finite-grid mean

`m_h = mu + (1-mu) r^n`

and variance

`v_h = sigma^2 h (1-r^(2n))/(1-r^2)`.

It is Gaussian, so its fourth central moment is `3 v_h^2`. The continuous
law instead has mean `mu+(1-mu) exp(-lambda)` and variance
`sigma^2 (1-exp(-2 lambda))/(2 lambda)`.

For GBM, `dX = a X dt + b X dW`, with initial value 1, `a=1/4`, `b=1/2`,
the finite-grid solution is the product of independent factors
`1+a*h+b*Z`, where `Z ~ N(0,h)`. Its raw moment of order `k` is

`E[X_h^k] = E[(1+a*h+b*Z)^k]^n`.

Odd Gaussian moments vanish; `E[Z^(2j)] = (2j-1)!! h^j`. Expanding this
polynomial for `k=1..4` gives the mean, variance and fourth central moment.
The continuous mean is `exp(a)` and variance is
`exp(2a) (exp(b^2)-1)`.

[The reference script](references/analytic.py) evaluates only these formulas
using 70-digit Decimal arithmetic. It imports no Eqiora code and consumes no
candidate output. Its [fixed numbers](expected/campaign.json) also contain the
mixed moments for the separate [path experiment](../ito-path-refinement/README.md).
To reproduce the numbers, run `python3 references/analytic.py` from this case
directory and compare its JSON output with the expected file.

## Acceptance and falsifiers

For `N=4096`, mean sampling uncertainty is `sqrt(v_h/N)`. For an unbiased
sample variance it is

`sqrt((mu4_h - ((N-3)/(N-1)) v_h^2)/N)`.

Each observed statistic must lie within six of these analytically derived
standard errors of its **finite-grid** reference. Separately, its distance
from the continuum statistic must be at most that sampling bound plus the
exact finite-grid bias. The reference biases must decrease at each declared
refinement. No tolerance is fitted to a measured ensemble.

The numerical falsifiers reach this same moment predicate:

- OU is affine in volatility. Scaling its centered population by `sqrt(h)`
  represents the erroneous `h*Z` noise scaling in place of `sqrt(h)*Z`.
  The resulting sample variance must fail the original bound.
- Repeating one accepted output for all occurrences represents accidental
  stream sharing despite distinct requested replicas. Its zero variance must
  fail the original bound.

These are mutations of the numerical population at the scientific predicate,
not claims that a foreign Model was accepted by the artifact boundary.
Separate product tests cover sampling-identity admission and failure handling.

This finite declared campaign is not a universal weak-order theorem, a
confidence interval, an adaptive SDE method, an interacting-particle model,
or a cross-environment random-stream guarantee.
