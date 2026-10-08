# Expected semantic projections

Complex nodal solution: `(1+3i, 7, 13-3i)`.
Reflected complex nodal solution: `(13-3i, 7, 1+3i)`.
Real nodal solution: `(1,7,13)`.

After eliminating the prescribed left coordinate:

```
A = (1+i) [[6, -1.5], [-1.5, 3]]
b = (18+27i, 37.5+19.5i)
```

For `z=(2-i,-1+3i)`:

```
A z   = (24+3i, -16.5+4.5i)
A^T z = (24+3i, -16.5+4.5i)
A^H z = (3-24i, 4.5+16.5i)
diag(A) = (6+6i, 3+3i)
```

The non-real diagonal and different adjoint action exclude a Hermitian
interpretation. Expectations bind these semantic values, not whole files or
generated artifacts. Exact bytes are compared only for canonical Plan/Result replay.
