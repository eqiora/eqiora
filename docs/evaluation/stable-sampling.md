# Stable sampled Parameter inputs

`ParameterSampler` draws real input values for an existing `DifferentiableProgram` and freezes
them. You supply a master stream, a full sample ID, and a channel role, then pass the resulting
`point()` directly to `program.evaluate(...)`, or the sampled records to
`EvaluationMapPlan::from_samples`. Sampling
never changes the Program identity and never happens inside evaluation, JVP, VJP, or
diagnostics.

```rust,ignore
let sampler = ParameterSampler::new(
    program.clone(), [42; 32], &[[1.0, 2.0]], // coherent-SI bounds in the exact input order
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
or devices, and no time-indexed Wiener or jump process is provided.


## Bound finite-sample moment conventions

`EvaluationMapEstimator::evaluate(&complete, reduction, outputs, weights)` computes a typed
finite statistic over every accepted occurrence of one `CompleteEvaluationMap`. `outputs` are
exact Model-owned references already selected by its Program, in covariance-pair order.
`weights` contains one finite nonnegative entry per occurrence and has positive total mass.
The estimator borrows both the complete map and its weights; it retains their original sample
identity and acceptance evidence without a second registry or any new draws.

`EnsembleReduction` selects the operation and interpretation together:

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

`vjp(numerical_bytes_limit)` differentiates the finite statistic with fixed weights and frozen
sample design. It uses the existing mapped VJP owner: occurrence-local input covectors remain
ordered and shared-input covectors are summed. Coordinates have statistic-unit/input-unit;
the covectors' base signatures identify the original input spaces, while `value_type()` owns
the objective's units. This does not interchange differentiation and population expectation.
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
integral or stochastic integral. Population expectation-gradient estimators and confidence
intervals remain separate capabilities.
