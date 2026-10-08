# Expected semantic projections

Both real and complex eigenvalues are `(1,3)`, with units J for the dimensioned
Hamiltonian and unit 1 otherwise.

```
Complex P_low  = 1/2 [[1, i], [-i, 1]]
Complex P_high = 1/2 [[1,-i], [ i, 1]]
Real P_low     = 1/2 [[1,-1], [-1, 1]]
Real P_high    = 1/2 [[1, 1], [ 1, 1]]
```

Conjugating the complex source matrix conjugates the projectors without changing
the spectrum. Projectors are dimensionless 2 by 2 maps. Their entries are compared
in declared basis order, without fixing the eigensolver's vector phase.

Non-Hermitian source rejection must name the Hermitian gate. Modified weak
coefficients must name finite weak correspondence. Exact bytes are checked only
for canonical Plan/Result replay, not as a substitute for these numerical values.
