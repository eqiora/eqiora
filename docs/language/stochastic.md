# Explicit stochastic process laws

Source and native Models retain finite Wiener and counting-process laws, their exact channel
identities, typed coefficients and written calculus. The ordinary Model artifact reopens this
meaning without a sampler. Numerical execution currently rejects with an explicit stochastic
execution diagnostic; no stochastic Plan, sampled trajectory or restart is implemented here.
The example below authors an Ornstein–Uhlenbeck process from its coefficients.

```eqiora
model Relaxation(
  parameter rate: 1 / s = 2 [1 / s],
  parameter mean: V = 1 [V],
  parameter sigma: V / s^(1/2) = 0.5 [V / s^(1/2)],
  parameter initial_value: V = 2 [V]
) {
  noise driving: wiener(covariance = 1);
  state value: V;

  initial {
    value = initial_value;
  }
  law relaxation for value {
    calculus ito;
    drift rate * (mean - value);
    diffusion driving = sigma;
  }
}
```

The mathematical equation is `dX = rate*(mean-X) dt + sigma dW`. Rate is positive and sigma
nonnegative. Fresh time is zero and the initial value is deterministic. The process is lumped,
with no spatial support, boundary, periodic clock, or hidden reflecting/absorbing condition.
Its value ranges over the real line; “bounded profile” means finite admitted model structure,
not a claim that every sample path stays within a fixed voltage interval.

## Closed source rules

`noise name: wiener(covariance = 1);` declares one nominal real noise channel with standard
Wiener covariance: increments on disjoint time intervals are independent, mean zero, and
have variance equal to interval duration. The covariance coefficient is dimensionless.
A one-channel covariance must be nonnegative; the written unit value selects standard scaling.
This is process identity in the Model, not a mutable random-number stream.

For Wiener evolution, `law name for state { ... }` selects exactly one
`calculus ito;` or `calculus stratonovich;`, one drift expression, and explicitly named
diffusion-channel bindings. The target must be an owned real continuous state. A channel
appears at most once. Drift has state units divided by time, and diffusion has state units
divided by square root of time. These typed children are distinct from a conservation Law's
storage/flux/source terms; mixing the two child sets rejects.

A Wiener group accepts either scalar covariance or a square array such as
`[[1, 0.5], [0.5, 1]]`. The latter declares two correlated channels; its diffusion coefficient
has an outer two-element array axis, each entry having the State's type divided by square root
of time. A scalar and a one-by-one array retain different shapes. Covariance is a constant or
Parameter-only expression. It must be exactly symmetric and positive semidefinite in its
stored finite real values; no tolerance, clipping or implicit repair changes the process.
The shared exact-rational arithmetic budget bounds admission and can reject an expensive proof.

Distinct groups require explicit `independent = true` on every group in the Model. Otherwise
their joint law is unspecified and admission rejects. Correlated Wiener channels belong in one
covariance group. Reusing the same exact group in two laws deliberately shares its increments.
Correlation is never inferred from names, seeds, matching units or equal diffusion coefficients.

`noise` permits notation after its name but no `on` or `at` in this lumped profile. It cannot
appear as a pure expression yielding a fresh random draw. A stochastic Law retains ordinary
state ownership and initialization rather than creating a second execution lifecycle.

## Independent moments and dimension checks

Let `a=rate`, `mu=mean`, `s=sigma`, and `x0=initial_value`. Multiplying the equation for
`X-mu` by `exp(a*t)` gives:

```text
X(t) = mu + (x0-mu)*exp(-a*t) + s*integral_0^t exp(-a*(t-u)) dW(u)
E[X(t)] = mu + (x0-mu)*exp(-a*t)
Var[X(t)] = s^2*(1-exp(-2*a*t))/(2*a)
```

The stochastic integral has zero mean. Its variance follows by integrating
`s^2*exp(-2*a*(t-u))` over u. With the written parameters, the mean is
`(1 + exp(-(2/s)*t)) V` and the variance is
`0.0625*(1-exp(-(4/s)*t)) V^2`. The initial variance is zero; the stationary variance is
0.0625 V^2. Squaring diffusion gives V^2/s, and integrating over time gives V^2.

These are distribution statements, not predictions for one realized trajectory. A later
ensemble check must fix sample membership, uncertainty criteria, and time-discretization bias
before evaluating results. A pleasing plot or one path near the mean proves neither moment.

## Multiplicative noise and calculus

The same source owner can specify geometric Brownian motion by changing only the Law and
parameter dimensions:

```eqiora
model GeometricGrowth(
  parameter growth: 1 / s,
  parameter volatility: 1 / s^(1/2),
  parameter initial_value: V
) {
  noise driving: wiener(covariance = 1);
  state value: V;
  initial { value = initial_value; }
  law growth_law for value {
    calculus ito;
    drift growth * value;
    diffusion driving = volatility * value;
  }
}
```

For positive initial value its exact Ito path is
`X(t)=x0*exp((growth-volatility^2/2)*t + volatility*W(t))`. Mean and variance are
`x0*exp(growth*t)` and `x0^2*exp(2*growth*t)*(exp(volatility^2*t)-1)`.
Changing the source to Stratonovich with the same written drift changes the process:
the Ito conversion adds `volatility^2*X/2` to the drift. That term has units V/s.
The additive-noise relaxation example has zero correction, which alone cannot test whether
an implementation handles this distinction correctly.

Conversion must be explicit and checked through the common scalar derivative owner, or the
execution request rejects. A future fixed-step Euler–Maruyama profile would not establish
positivity for the geometric process, adaptive stepping, or arbitrary stochastic derivatives.

## Run, evaluation, and restart

Plan selects the admitted solver and sampler. Run records the realized path identity, generator
identity, and continuation state. A seed by itself does not define the same path across
different generators, channel assignments, or interval refinements. The source includes none
of these numerical choices.

Residual evaluation, differentiation of drift/diffusion, diagnostics, and observations must
not advance a random stream. Rejected numerical trials commit no state or path progress.
Refining an interval consumes the accepted path refinement contract, not unrelated replacement
increments. Exact restart retains the realized state and required path/generator continuation;
it neither resamples initialization nor replays an accepted increment.

Reject negative/non-PSD covariance, a foreign noise binding, diffusion with V/s rather than
V/s^(1/2), conflicting state owners, and an implicit calculus change. A Wiener path is not
classically time-differentiable: `derivative(value)` cannot silently replace its stochastic
Law. Differentiating smooth coefficients does not establish pathwise, weak, or distributional
sensitivities of the complete process.

## Finite birth and death channels

Counting processes use nonnegative inverse-time intensities and State-valued jump sizes;
they do not use square-root-time coefficients or Ito/Stratonovich conversion. A scalar intensity
declares one channel and an array declares ordered conditionally independent channels. A pure
jump Law contains only `jump` bindings. A Law containing Wiener and counting channels retains
both sets of bindings, with calculus and drift for its continuous part.

```eqiora
model Population(
  parameter birth: 1/s = 2[1/s],
  parameter death: 1/s = 3[1/s]
) {
  state count: integer;
  initial { count = 0; }
  noise events: poisson(
    intensity = [birth, if count > 0 then death * to_real(count) else 0[1/s]]
  );
  law population for count {
    jump events = [1, -1];
  }
}
```

The first channel increases the exact integer count by one at rate `birth`. The second decreases
it by one at rate `death*count` when the count is positive, and is disabled at zero. These
mathematical intensities have units 1/s; the jump sizes are dimensionless integers. No random
draw occurs when either coefficient is evaluated. Real-valued jump States are also admitted.

Constant negative intensities reject at Model admission. A State-dependent intensity retains
the nonnegative runtime precondition; admission does not prove it for all reachable States.
The future stochastic executor must check evaluated intensities before sampling. Current
execution rejects the whole stochastic request, including requests with dynamic intensities.
Model replay and pure coefficient typing do not establish a population distribution, likelihood,
path derivative, weak derivative or ensemble statistic.
