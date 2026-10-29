# Acceptance

At one, two, and four ranks, both Q1 FEM and cell-centered TPFA with each
explicit Reproducible/Fast policy must satisfy all of the following:

- all ranks reproduce identical canonical Model, Realization, Run, complete
  system, partition, and derived-layout artifact digests;
- decoded Realization policy freshly finalizes to the exact recorded system
  bytes and digest;
- rotated-cyclic ownership derives a valid content-linked layout and the same
  plan-inclusive admission fingerprint on every rank;
- CG/Jacobi retains the exact requested `Reproducible` or `Fast` Plan and
  receipt policy, MPI backend and distributed producer topology, plus
  independent one-worker host verification;
- complete system, partition and derived layout content is identical across
  the policies while their Realization and Run identities differ;
- method-native continuous-L2 error is below `2e-3` and relative reaction or
  flux balance is below `2e-11`;
- every finished algebraic and balance scalar agrees with the separately
  resolved serial reference within `2e-12 + 2e-12 * abs(reference)`; and
- the parent timeout is not exceeded.

The changed-RHS falsification must pass exact content-DAG linkage after all
dependent artifact identities are rebuilt, but must fail exact semantic
derivation replay against the unchanged Model and Realization.
