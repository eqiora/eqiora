# Stable sampled Parameter inputs

`ParameterSampler` draws real input values for an existing `DifferentiableProgram` and freezes
them. You supply a master stream, a full sample ID, and a channel role, then pass the resulting
`point()` directly to `program.evaluate(...)`, or the sampled records to
`EvaluationMapPlan::from_samples`. Sampling
never changes the Program identity and never happens inside evaluation, JVP, VJP, or
diagnostics.

```rust,ignore
let fixed = DifferentiableValues::from_components(&[], &[])?;
let sampler = ParameterSampler::new(
    program.clone(), [42; 32], &fixed, &[[1.0, 2.0]],
)?;
let request = SamplingIdentity::new(
    b"replicate-17", b"diffusion", &[batch_number, request_position],
    None,
)?;
let frozen = sampler.sample(&request)?;
let accepted = program.evaluate(frozen.point())?;
```

The `stable_sampling` integration test runs this path with the typed Poisson Program and also
passes sampled points to a mapped evaluation. A draw only has to be finite to be accepted as an
input; it can still fail the Program's usual physical and numerical checks.

The typed `fixed` product selects complete input leaves and freezes their values. Bounds
cover the remaining input coordinates in original Program order. Empty `fixed` samples every
input; a nonempty product lets a study hold design values fixed while drawing only latent
inputs. The sampler and each frozen record retain `fixed_values()`. Changing these values is
an explicit new sampling request. IID estimator admission rejects mixing different fixed
values or selections, even if intervals and stream identities otherwise match.

Input sharing in `EvaluationMapPlan` remains a separate numerical declaration: common-random-
number requests can share a generated coordinate without making it a fixed design input.
Sampling provenance records this distinction. Absolute Program coordinate indices own draw
streams, so fixing one leaf does not shift the draws of another. No evaluation or retry can
change these frozen values.

## Identity and replay

A request's position in a batch (its occurrence lineage) is recorded as metadata but never used
as key material. Reordering requests, changing chunk positions, retrying work, or changing the
number of workers therefore leaves each sample's draws unchanged. Two requests with equal
numerical points are still distinct; values never determine identity.

Independent-replicate requests derive their stream from the full sample ID.
The explicit `Some(coupling_id)` argument puts a shared coupling ID in the stream key instead, while
each request keeps its own sample ID. Requests with the same coupling ID and channel under the
same master share draws by design. Distinct IDs do not by themselves establish statistical
independence.

## SHA-256 uniform V1

This version's stream digest is SHA-256 of the following concatenation:

1. ASCII `eqiora.parameter-uniform.sha256.v1` and a zero byte;
2. all 32 master bytes;
3. the channel's byte length as unsigned 16-bit big-endian, then all its bytes;
4. byte `0` for independent replication or `1` for common random numbers;
5. the effective sample/coupling ID length as unsigned 16-bit big-endian, then
   all its bytes.

IDs are nonempty byte strings of at most 256 bytes, and occurrence lineage holds at most 32
unsigned 64-bit entries. Every request field and all 256 bits of the stream digest are kept;
identity is never reduced to floating-point precision. SHA-256 makes collisions practically
infeasible, not mathematically impossible.

Coordinate `i` hashes ASCII `eqiora.parameter-uniform.draw.v1`, a zero byte,
the full stream digest and unsigned 64-bit big-endian `i`. Interpret the first
eight digest bytes as big-endian `u64`, shift right by 11, and multiply the
exact resulting integer by `2^-53` to obtain `u` in `[0, 1)`.

Each of the 1–256 ordered intervals needs finite `lower < upper` and a finite `upper - lower`.
The value is computed with separate binary64 subtraction, multiplication, and addition as
`lower + ((upper - lower) * u)`; if rounding reaches `upper`, the result is
`upper.next_down()`. The result therefore lies in `[lower, upper)` even for adjacent or
subnormal endpoints. The draws lie on a finite grid and do not define a continuous probability
density. Reordering the Program's input coordinates changes which draw each Parameter receives.

A private fixed-ID test checks this encoding against references computed independently with
Python's `hashlib`, and also checks that different identities give different streams and that
rounding never reaches `upper`. Results are not guaranteed to match across generator versions
or devices. This Parameter sampler does not define a time-indexed process;
[stochastic time Plans](../language/stochastic.md#run-evaluation-and-restart) own Wiener and jump
execution.


## Bound finite-sample moment conventions

`MomentEstimator::evaluate(&complete, reduction, outputs, weights)` computes a typed
finite statistic over every accepted occurrence of one `CompleteEvaluationMap`. `outputs` are
exact Model-owned references already selected by its Program, in covariance-pair order.
`weights` contains one finite nonnegative entry per occurrence and has positive total mass.
The estimator borrows both the complete map and its weights; it retains their original sample
identity and acceptance evidence without a second registry or any new draws.

`MomentReduction` selects the operation and interpretation together:

- `Sum` returns the unnormalized weighted sum of one real scalar output.
- `Empirical(ProbabilityMoment::Expectation)` returns its weighted arithmetic mean.
- `Empirical(ProbabilityMoment::Variance)` returns its normalized weighted squared deviation.
- `Empirical(ProbabilityMoment::Covariance)` returns the paired centered product.
- `UnbiasedIid(moment)` explicitly assumes independent, identically distributed outputs with
  the required finite population moments. It requires equal positive weights. Variance and
  covariance additionally need at least two occurrences and divide by n-1.

Sum and mean preserve the output units, variance squares them, and covariance multiplies the
paired units. The reduced axis is the complete ordered occurrence axis, including duplicates
and zero-weight members. Selected scalar outputs are quantities, not additional sample axes.
Incomplete collections cannot construct this estimator; there is no success-only filtering,
NaN masking or denominator adjustment. Empty collections and all-zero weights are rejected.

Equal values and repeated sample IDs retain their occurrence multiplicity for empirical
reductions. This implies no IID, importance-sampling, unbiasedness or population-expectation
claim. Deterministic maps can use empirical reductions. IID admission checks necessary metadata
conditions: distinct sample IDs and stream digests, no common-random-number coupling, and one
generator version, master stream, channel and ordered interval design. Frozen samples retain
those intervals, even when different intervals produce identical numeric inputs. These checks
cannot prove independence or finite population moments; the IID choice is a caller assumption
about the declared finite-grid input design, distinct from an ideal continuous distribution.

`standard_error()` accepts only an explicitly IID mean with at least two occurrences and a
finite population second-moment assumption. It returns `sqrt(sample variance / n)` in the
original units. This is an estimated standard error, not a confidence interval or a bound on
the realized error. Weighted/empirical uncertainty and standard errors of other statistics are
rejected; no effective sample size is invented.

`vjp(numerical_bytes_limit)` differentiates the selected statistic with fixed weights and frozen
sample design. It returns a `MomentEstimator<T>` retaining the exact map, reduction,
weights, value and gradient convention; `derivative()` exposes the typed mapped action. It uses the existing mapped VJP owner: occurrence-local input covectors remain
ordered and shared-input covectors are summed. Coordinates have statistic-unit/input-unit;
the covectors' base signatures identify the original input spaces, while `value_type()` owns
the objective's units. The default finite-objective convention does not interchange
differentiation and population expectation.
The storage limit has the mapped-product numerical-buffer scope plus one output seed buffer;
it excludes metadata and transient solver scratch.

Finite distribution moments and collection statistics share bounded exact rational arithmetic.
Binary64 inputs are accumulated exactly within the common 8192-bit numerator/denominator bound;
the final statistic and each row derivative round once. Standard errors take a correctly
rounded square root of the exact radicand, without converting a potentially overflowing or
underflowing squared quantity to binary64. Range errors and exhausted arithmetic bounds fail
closed. Exact statistic values are unchanged by chunk size, workers or retention, provided
accepted members agree. Recompute retention reaccepts frozen points against their receipts.
Shared VJP accumulation retains the existing ordered binary64 overflow policy; arbitrary
reordering or regrouping of gradient sums need not be bit-identical.

These operations reduce accepted outputs. They do not turn output cadence into a path time
integral or stochastic integral. Confidence intervals remain unsupported; exhaustive finite-law score execution is described below.

## Fixed probability realizations and native AD

An explicit `evaluate(x, at=(z=draw))` binds every coordinate of a continuous-law
random variable `x` and produces one deterministic realization. A named real scalar
Observable containing that realization can enter native finite Newton residuals.
For example, `x=random_variable(base,location+scale*z)` and
`loss=evaluate((x-1)^2,at=(z=draw))` retain the location and scale Parameters through
ordinary accepted evaluation and mapped VJPs. Coordinate expressions are evaluated
in their outer scope; two point bindings of the same variable remain distinct.

The sampling request fixes the design Parameters and draws the remaining inputs.
A value/gradient pair consumes those same frozen points. Every point must lie in
its exact declared continuous support; a failed member cannot disappear from the
objective. Typed domain evaluation and active-boundary derivative rules are shared
between scalar and component programs. One-sided reconstruction, named discrete
outcomes and derivatives of population moments remain outside this finite path.

For each fixed draw `z`, `(location+scale*z-1)^2` has derivatives
`2*(location+scale*z-1)` and `2*z*(location+scale*z-1)`. Averaging these derivatives
computes the gradient of that exact finite objective. It does not establish a
population derivative interchange or a likelihood/score estimator without separate admission.

The real scalar and component AD owners share sine, cosine, exponential,
logarithm and square-root actions. Typed mathematical evaluation owns their
domains; invalid logarithms and active square-root derivatives at zero reject.
The native Gaussian reference test composes a standard-normal transform
`sqrt(-2*log(u))*cos(2*pi*v)` from the existing uniform law on a product support.
It checks the exact frozen objective and its derivatives separately from the
Gaussian population moments, using a fixed sample count and error criterion
declared before drawing. A singular endpoint fails rather than redrawing.

At the scalar IR boundary, an explicitly frozen nominal outcome can select a
real expression while continuous Parameters remain differentiable. Frozen values
retain their exact types and are removed from the tangent space; active discrete
inputs and mismatched nominal types reject. A two-outcome reference checks
`dL + L*dlog(p)` with the normalization derivative included, using the Model's
declared proportional weights and random-variable expression. Finite population
observation resolves both weights and losses at the same supplied Parameter
point; declaration defaults cannot silently replace the supplied design.
The law-bound score operation below retains this design point and complete nominal membership.

## Explicit location-scale pathwise gradients

`pathwise_location_scale(variable, location, scale)` admits a pathwise mean interpretation
before `vjp`. It retains `GradientEstimator::PathwiseLocationScale` with the exact base-law,
random-variable and design Parameter IDs; the owning estimator retains their exact Model.
The default remains `GradientEstimator::FixedSamples`. This choice is separate from the
arithmetic reduction and from assumptions about IID samples.

The caller assumes a local integrable bound on the loss's design derivatives and the
required finite moments. Structural admission checks the selected variable's
`location + scale * noise` expression, positive scale, design-independent base noise and
a fixed uniform base law on coordinate intervals. The selected named loss Observable must
explicitly bind every base coordinate to its corresponding sampled Parameter. Its dependency
path must be smooth real arithmetic or an admitted pure operator. The two design inputs are
exactly the map's shared and sampler-fixed inputs; sampling intervals must match the law's
support. Equal positive weights give the mean. Coupled comparisons retain their original
sampling identities and do not acquire an IID claim.

The resulting gradient uses ordinary accepted native VJPs and preserves every occurrence.
Its shared covector is the location/scale design gradient; mapped covectors remain
conditional sample-coordinate derivatives.
No draw occurs during value, differentiation or recomputation. Singular samples fail the
whole request. The finite-grid generator approximates the declared continuous base law;
passing the Gaussian reference is not a general unbiasedness or limit-interchange proof.
Conditional/discrete paths, moving support, trajectory sensitivities, higher derivatives,
and control-variate conventions remain unsupported by this pathwise operation.

Native mapped actions now share `EvaluationMapResult<DifferentiableTangent>` or
`EvaluationMapResult<DifferentiableCotangent>`. The role is retained in the type; only reverse
results expose shared sums and mapped covectors. The former native `EvaluationMapJvp` and
`EvaluationMapVjp` types are removed without compatibility aliases. Python's role-specific
result projections consume this same native owner.


## Exhaustive finite-law score gradients

`DifferentiableValues::from_parameters(document, parameters)` binds a design point to the
exact Model types without constructing an unrelated solver Plan.
`MomentEstimator::enumerated_score(document, variable, point, numerical_bytes_limit)`
computes the expectation and its first derivative over every declared nominal outcome.
The selected random variable supplies the loss, law, support and proportional masses;
other Parameters remain frozen at their Model defaults. This bounded reference accepts
real scalar design Parameters and losses composed from Parameters, the law's own nominal
outcome and existing pure operators. Inter-Observable dependencies and trajectory
sensitivities remain outside this profile.

Native typed AD supplies `dL` and `dlog(w)`. The shared weighted-moment reduction computes
`dlog(Z) = E[dlog(w)]` and then `E[dL + L*(dlog(w)-dlog(Z))]`. Every mass must be finite
and strictly positive at the design point; zero-mass support boundaries reject. Native AD
checks local regularity. No sampling, baseline, control variate or general interchange
theorem is involved. The numerical byte limit covers construction and retained binary64
buffers; Model/signature metadata, nominal tags, exact arithmetic and native AD scratch
are excluded. A separate structural expression-product bound limits enumeration.

The result retains `MomentReduction::Population(Expectation)` and
`GradientEstimator::EnumeratedScore`, including exact law/variable identity and selected
Parameters appearing in the weights. `point()`, `outcomes()`, `outcome_values()`, `weights()`
and `scores()` expose the frozen design, complete declaration-order membership, losses,
proportional masses and normalized score covectors. `derivative()` has expectation-unit
per input-unit; score coordinates have reciprocal input-unit. Their base signatures retain
the input spaces. There is no second derivative or estimated IID standard error.

The two-outcome reference uses masses `[3,t]` and losses `[2+t,6+2t]`. Polynomial division
independently gives `E=2t+3-3/(t+3)` and `E'=2+3/(t+3)^2`. Tests cover two design points,
the normalized scores, common-mass-scale cancellation in either parameter order, invalid
support boundaries, exact Model binding and exhausted numerical resources.

Sample-map and finite-law statistics now share `MomentEstimator` and `MomentReduction`.
The former native `EvaluationMapEstimator` and `EnsembleReduction` names are removed.
Map-only accessors return `Some` for sample results; finite-law accessors return `Some`
for exhaustive results. A sample map cannot be relabeled as a population reduction.
