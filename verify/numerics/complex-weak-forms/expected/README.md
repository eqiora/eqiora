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
interpretation. Conjugate gradient therefore cannot acquire a Plan for this
form: the requested complex/CG/general capability tuple must reject, after
the identical source succeeds with BiCGStab. Expectations bind these semantic values, not whole files or
generated artifacts. Exact bytes are compared only for canonical Plan/Result replay.

For the Helmholtz specialization, `u=1+3i+(2-i)x` has `u''=0`,
`(-1+i)u=-4-2i+(-1+3i)x` and `6u'=12-6i` at the right boundary.
The two-cell reduced matrix is `[[2+2i,-2.5+0.5i],[-2.5+0.5i,1+i]]`;
its determinant is `-6+6.5i`, so the manufactured solution is unique.
For real pure diffusion, the reduced matrix is `[[4,-2],[-2,2]]`,
with determinant `4`; `u=1+2x` has zero volume source and right flux `12`.
These independent equations justify the nodal expectations without reading solver output.
