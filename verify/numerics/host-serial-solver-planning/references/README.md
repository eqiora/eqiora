# Independent reference

`derive_policy_v8.py` is the independent authority for the exact rational
system and deterministic selection table. It uses Python `Fraction` arithmetic
to derive `[1, 2]` and the zero residual, locates both diagonal entries directly
from the captured CSR offsets and column indices, freezes the decimal control
values and their IEEE-754 bit patterns, and ranks a literal five-record
catalog without importing or executing Rust.

The derivation enumerates all 120 input permutations and every nonempty
admitted subset, checking the complete ID-ordered reason trace and exact
reranking for every objective. Independent mutants reverse each precedence
ordering whose effect is observable in the frozen catalog and replace the
final candidate-ID tie-break with input enumeration. Swapping Fast's reduction
and algorithm axes is explicitly not a falsifier because it changes no
selection for the full catalog or any admitted subset. The script also freezes
every rejection and execution falsifier consumed by the Rust oracle. An
isolated test-only ledger on the exact owned canonical view first observes and
resets one direct `problem.operator()` apply and diagonal self-control. It then
freezes the two total exact-operator applications required to recompute initial
and final true residuals on successful fake-backend acceptance, and rejects
one extra direct apply or diagonal.

The derivation separately freezes the complete Robust, Fast, and LowMemory
traces when faer BiCGSTAB simultaneously has stale evidence and provider
identity. The observed first and only rejection is
`catalog.evidence-mismatch`; substituting provider-first precedence changes
each trace.

The faer descriptor's actual dependency inventory has exactly one member, so
the oracle requires exact ordered equality of that inventory and rejects a
changed, missing, or extra dependency. It creates no fake second dependency
and makes no same-members reorder claim.

## Exact complex fixtures

The same independent script now derives the four complex profiles with pairs of
`Fraction` components. It solves each 2x2 system by exact Cramer determinants,
checks the genuine mathematical class, and solves its conjugate transpose for a
second, separately frozen right-hand side. The nonreal Hermitian off-diagonal
entries cannot be replaced by a real symmetric witness. HPD leading principal
minors are 4 and 11; the separate Hermitian matrix `[[1,2i],[-2i,1]]` has
positive diagonal entries, determinant -3 and eigenvalues -1 and 3.

The literal complex catalogs contain 14 Reference and 14 Faer tuples:
Identity/Jacobi Reproducible Reference BiCGSTAB on all four classes, HPD
Identity/Jacobi Reproducible Reference CG, Hermitian/HPD Identity/Jacobi
Reproducible Reference MINRES, Identity/Jacobi Fast Faer BiCGSTAB on all four
classes, HPD Identity/Jacobi Fast Faer CG, and Identity/Fast Faer SparseLU on
each class. Normal and conjugate-transposed manual execution of every tuple
is the independent numerical authority for these catalog evidence links.
Known diagonal presence admits Jacobi; an unproved diagonal rejects it before
ranking. The literal table retains the existing specialized CG/MINRES preference
when shape is unknown. Version6 applies that specialization to Fast's iterative
fallback after its reduction and sparse-LU priorities as well, including when
the optional faer provider is absent. The independently frozen absence traces
therefore keep CG on HPD rather than selecting BiCGSTAB by ID spelling.
MINRES uses the existing realification on Hermitian/HPD. Its Jacobi tuples
require finite strictly positive real diagonal entries and symmetric correction
coordinates with the original residual target, physical predicate and iteration
budget. Positive diagonal entries do not promote the indefinite fixture to HPD.
The exact normal/adjoint solutions and componentwise `2^-40` bound are unchanged.
This adds no ordinary HPD recognition claim.

## Provider source identity

The v8 table binds the inspected Cargo release `0.1.4` for solver and execution
provider descriptors. The v2 table retained `0.1.0-alpha.3` while its public test
checked the actual Cargo version separately. The v8 test requires equality among
the actual provider, the independent oracle, and Cargo source metadata. The exact
rational solution and numerical tolerance derivations are unchanged.

## Historical v2 residual audit

Before the original v2 evidence was committed, its three accepted live backends were run
at protected base `f36f6f029e7cdc59b81163355ff07ec1cdb9c78e` on the frozen
fixture and common controls. Independent CSR replay observed:

| Candidate | Values | True residual norm | Maximum component error |
| --- | --- | --- | --- |
| reference BiCGSTAB | `[1.0000000000000002, 1.9999999999999996]` | `2^-49` | `2^-51` |
| faer BiCGSTAB | `[1.0000000000000002, 1.9999999999999998]` | `2^-50` | `2^-52` |
| faer SparseLU | `[1, 2]` | `0` | `0` |

Each is strictly inside the precommitted componentwise `2^-40` bound and also
passes the unchanged backend-independent true-residual target. These one-host
values validate the bound at the frozen host-serial `f64` execution boundary;
they are not performance, portability, or bitwise cross-environment claims.
