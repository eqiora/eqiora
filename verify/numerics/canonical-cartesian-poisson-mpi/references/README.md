# Reference provenance

The exact solution and source are derived in
[`models/problem.md`](../models/problem.md); no external numerical table is
used. Each explicit Reproducible/Fast MPI candidate is independently compared with the repository's
one-worker serial reference solver after resolving the same decoded Semantic
Model into the corresponding replicated Q1 FEM or TPFA Realization.

MPI implementation/version, provided thread support, MPI-standard version,
mpi-rs version, adapter/backend identity, rank count, worker count, and
reduction policy are recorded in the typed Run artifact before its digest is
compared across ranks. This is execution provenance, not hardware attestation
or physical-node evidence.
