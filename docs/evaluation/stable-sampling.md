# Stable sampled Parameter inputs

`ParameterSampler` draws real input values for an existing `DifferentiableProgram` and freezes
them. You supply a master stream, a full sample ID, and a channel role, then pass the resulting
`point().values()` yourself to `program.evaluate(...)` or to an `EvaluationMapPlan`. Sampling
never changes the Program identity and never happens inside evaluation, JVP, VJP, or
diagnostics.

```rust,ignore
let sampler = ParameterSampler::new(
    program.clone(), SamplingGenerator::Sha256UniformV1,
    [42; 32], &[[1.0, 2.0]], // coherent-SI bounds in the exact input order
)?;
let request = SamplingIdentity::new(
    b"replicate-17", b"diffusion", &[batch_number, request_position],
    SamplingCoupling::IndependentReplicate,
)?;
let frozen = sampler.sample(&request)?;
let accepted = program.evaluate(frozen.point().values())?;
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
`CommonRandomNumbers(coupling_id)` puts a shared coupling ID in the stream key instead, while
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
