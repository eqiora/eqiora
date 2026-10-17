# Explicit stochastic process laws

Source and native Models retain finite Wiener and counting-process laws, their exact channel
identities, typed coefficients and written calculus. The ordinary Model artifact reopens this
meaning without a sampler. Native time Plans admit serial Ito Euler–Maruyama on a fixed dyadic
grid and pure Poisson events with Model-constant intensity, including sampled trajectories and
checkpoint restart. Other calculus/method combinations reject explicitly. The example below
authors an Ornstein–Uhlenbeck process from its coefficients.

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
execution request rejects. The fixed-step Euler–Maruyama profile does not establish positivity
for the geometric process, adaptive stepping, or arbitrary stochastic derivatives.

## Run, evaluation, and restart

Plan selects the admitted solver and sampler. Run records the realized path identity, generator
identity, and continuation state. A seed by itself does not define the same path across
different generators, channel assignments, or interval refinements. The source includes none
of these numerical choices.

Residual evaluation, differentiation of drift/diffusion, diagnostics, and observations must
not advance a random stream. Rejected numerical trials commit no state or path progress.
Refining an interval consumes the accepted path refinement contract, not unrelated replacement
increments. Exact restart retains the original request and physical/path continuation. Its
checkpoint decoder regenerates the addressed path and revalidates the accepted partition before
resuming; validation does not commit additional progress or resample initialization.

Saved Trajectories and Results instead retain the accepted step States and exact driving inputs.
Reopening checks the original equations against stored predecessors without drawing a path or
advancing a Run. Wiener stencils retain independent increments; the Model-owned covariance factor
still determines their channel action. Poisson stencils retain coincident arrivals and their
post-jump States. The same physical-time, output-boundary and coefficient owners serve execution
and reopening. Sampler and arithmetic-environment metadata retain their compatibility checks.
Recorded sample identity is provenance, not authentication that an untrusted producer ran the
named generator. A saved Trajectory is not a restart checkpoint.

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


## Probability laws and population observations

A distribution has no activation and draws no samples. Its normalized mass or density is a
mathematical law. `random_variable` binds an ordinary typed expression to that exact law;
`expectation`, `variance` and `covariance` reduce named random variables under their shared law.
An `outcome(law)` is a member of the law's exact Enum, available only in that law's random-variable
context. It is not an ambient random draw.

```eqiora
enum Outcome {Low, High}
model Population(support position: interval(m)) {
  variable anchor: 1;
  relation fixed {anchor = 1;}
  distribution finite: normalize_mass(Outcome, weights=[1, 3]);
  observable x: m = random_variable(finite,
    if outcome(finite) == Outcome.Low then 0[m] else 2[m]);
  observable mean: m = expectation(x);
  observable spread: m^2 = variance(x);

  distribution uniform: normalize_density(measure(position), density=2[1/m]);
  coordinate coordinate_x: m on position from position[0];
  observable u: m = random_variable(uniform, coordinate_x);
  observable v: s = random_variable(uniform, 3[s] - 1[s/m]*u);
  observable uniform_mean: m = expectation(u);
  observable uniform_variance: m^2 = variance(u);
  observable paired: m*s = covariance(u, v);
}
```

Supply a bounded coordinate interval for `position`, for example `[0 m, 3 m]`. Mass weights
must be dimensionless, finite and nonnegative, with positive total mass. `[1, 3]` denotes the
same probabilities as `[1/4, 3/4]`. Density has inverse-measure units; its normalization is
explicit in the constructor. The current density profile admits positive constants on finite
products of bounded coordinate intervals with their volume measure. `spherical_measure` is
retained by syntax but is outside this density admission profile. Law coefficients may use
immutable Parameters, and admission checks the selected Model revision's values.

For the finite law, `E[x] = (0 + 3*2)/4 = 3/2 m` and
`Var(x) = (0 + 3*4)/4 - (3/2)^2 = 3/4 m^2`. On `[0,L]`, normalizing a positive constant
produces density `1/L`; integrating `x/L` and `x^2/L` gives `L/2` and `L^2/3`, hence variance
`L^2/12`. At `L = 3 m`, the uniform mean and variance match the finite example, and
`Cov(u, 3 s - (s/m)u) = -3/4 m*s`. Equal moments do not make the two laws identical.

Finite population observations enumerate all named outcomes using the canonical expression
evaluator. Weighted arithmetic accumulates binary64 inputs as exact rationals within a work
bound and rounds once on output. Continuous observations use the existing Result-owned factor
quadrature with an explicit rule for the law's exact Domain. This is a numerical approximation
to a population moment; it is not a Monte Carlo estimate or an error bound. The two-point
Gauss rule integrates the example's degree-two polynomial moments exactly apart from floating
coordinate and weight representation. Result identity and quadrature provenance remain attached.

The current moment profile uses real invariant scalar quantities. Expectation preserves input
units; variance squares them; covariance multiplies the two inputs' units. Mixing laws, using
an unbound random variable as an ordinary value, negative mass, zero normalization and wrong
density units reject. A realized stochastic State, including dependencies through Observables,
implicit equations, Port connections and event guards, cannot be relabeled as a population moment.
Separate conditions and Port quantities retain their own dependency paths. This check does not
prove statistical independence or symbolic cancellation.

Point evaluation and pullback retain the complete coordinate context of each random variable.
The numerical regularity projection currently supports pullbacks but not `evaluate` inside a
quadrature integrand; native point evaluation can still inspect that expression. Conditional
expectations, interchange of derivatives with probability integrals, general variable densities,
complex/tensor moments and stochastic path execution require separate admission. [Finite-sample estimator conventions](../evaluation/stable-sampling.md#bound-finite-sample-moment-conventions)
bind complete accepted EvaluationMaps, explicit weights and sample identity. Their numerical
reduction and sampling-error reporting remain with the ensemble execution consumer; a single
trajectory is never an exact expectation.
