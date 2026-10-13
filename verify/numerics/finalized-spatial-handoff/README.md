# Finalized spatial handoff verification

One canonical two-dimensional Poisson model is independently realized as Q1
FEM and orthogonal TPFA FVM. Each path stops after deterministic assembly and
exposes the same opaque boundary: finalized CSR and right-hand side, asserted
operator properties, the exact `SolverPlan`, method identity, and assembly
evidence. A separately selected reference CPU backend returns an accepted
`LinearSolution`; only then does the opaque state reconstruct the method-native
field and balance evidence.

The test proves exact equality with the one-call CPU results. Its two
different methods deliberately produce distinct 9-by-9 systems, so a
same-shaped FEM solution cross-wired into the FVM handoff is rejected by an
independent residual over the receiving finalized system. Both plans use zero
relative tolerance and the same absolute tolerance, ensuring this falsifies
the residual rather than merely a target mismatch. It also rejects an
accepted solution carrying a different iteration limit and a valid solution
whose producer evidence claims a CUDA topology for a host Realization.

The same finalizer also consumes a canonically decoded portable graph directly
for both Q1 and TPFA. Valid graph mutations of Model, semantic revision, Domain,
and Field identities must fail at the equation-aware finalizer; wrong runtime
placement must fail before assembly. Parameter-point execution retains that
same graph for subsequent linearization. A separately supplied runtime target
keeps device ordinals out of portable numerical identity while retaining the
existing exact-target solution checks.

This is independent numerical reacceptance, not durable problem identity. A
vector that satisfies two finalized systems is admissible to both. Proving
which system produced a vector is deferred to artifact persistence.

This evidence is limited to generated Cartesian scalar Poisson, Q1 FEM and
orthogonal TPFA, replicated `f64`, and the reference CPU backend. It does not
claim CUDA execution, accelerator assembly, general PDE finalization, imported
meshes, distributed vectors, or artifact-wire persistence of the handoff.

Run it with:

```text
cargo run -p eqiora-verify -- run --case numerics.finalized-spatial-handoff
```
