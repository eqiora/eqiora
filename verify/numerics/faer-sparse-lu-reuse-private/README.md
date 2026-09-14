# Faer prepared sparse-LU private falsifiers

This companion case runs the production-private prepared Faer session tests. They require both
complete assembly identity equality and actual CSR topology equality for symbolic reuse, require
numeric refactorization when coefficients change, and prove a singular candidate cannot replace
accepted factors.

```console
cargo run --locked -p eqiora-verify -- run \
  --case numerics.faer-sparse-lu-reuse \
  --case numerics.faer-sparse-lu-reuse-private
```
