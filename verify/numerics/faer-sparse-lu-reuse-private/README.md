# Faer prepared sparse-LU private falsifiers

This companion case runs the production-private prepared Faer session tests. They require both
complete assembly identity equality and actual CSR topology equality for symbolic reuse, require
numeric refactorization when coefficients change, and prove a singular candidate cannot replace
committed factors. Explicit action tests also retain those factors after successful
tentative coefficient updates followed by failure or caller rejection, then retry
against the retained committed state. They reject idle/nested/failed operation
sequences before mutation, poison active actions on early solve errors, and admit
zero-solve commits. Expectations and existing factorization-count falsifiers remain
independent of implementation outputs; counts do not establish a speedup or memory
bound. The case runs the complete bounded `prepared_sparse_lu::tests::` owner.

```console
cargo run --locked -p eqiora-verify -- run \
  --case numerics.faer-sparse-lu-reuse \
  --case numerics.faer-sparse-lu-reuse-private
```
