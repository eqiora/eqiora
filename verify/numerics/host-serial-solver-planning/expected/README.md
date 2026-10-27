# Frozen expected policy

`policy-v8.json` is the exact checked-in output of the independent derivation.
It freezes the rational system and solution, canonical CSR and diagonal
inventory, common IEEE-754 control bits, all candidate/provider/evidence/plan
identities, all three decisions, every observable admitted-subset reranking and
ordered reason trace, the no-admitted diagnostic, manual-versus-decision
equality rule, componentwise bound, complete rejection zero-work ledgers,
per-objective success/failure one-attempt and exact-problem/operator-identity
ledgers, the isolated direct apply/diagonal self-control and reset, successful
true-residual acceptance's exact two total actual-operator applications, zero
operator-diagonal calls, and the complete observable falsifier inventory.
That inventory includes the complete three-objective reranking traces for the
simultaneous stale-evidence/stale-provider mutation, whose only rejection is
`catalog.evidence-mismatch`.

The Rust tests consume these values; the implementation writer may wire them
but must not change them. Run `derive_policy_v8.py` to compare the checked-in
JSON byte-for-byte with a fresh exact derivation.

The `complex_profiles` projection independently binds genuine General,
complex-symmetric, Hermitian and HPD matrices, forward/adjoint right-hand sides
and exact solutions, all six typed tuple descriptors, and every objective trace.
