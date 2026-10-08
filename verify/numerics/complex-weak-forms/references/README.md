# Independent derivation

Take `a=6+6i m²`, `q=1+i`, `u(x)=1+3i+(2-i)x/m`. Then `u''=0`,
`q*u=-2+4i+(3+i)x/m`, and the right outward flux is
`a*u'=18+6i m`. Thus the source equation and both boundary conditions hold
pointwise. The conjugate-first test convention gives `conj(v')*a*u'`
and `conj(v)*q*u`; the trial and coefficients are not conjugated.

For a length-three cell, directly integrating the two linear shape functions gives
`K=a/3 [[1,-1],[-1,1]]`, `M=q/2 [[2,1],[1,2]]` and affine-load entries
`(2*f_left+f_right)/2`, `(f_left+2*f_right)/2`.
The load values at the three vertices are `-2+4i`, `7+7i`, `16+10i`.
Assemble the two element contributions, eliminate the left value `1+3i`, and
add the right flux. This gives the matrix and load in the expected-value file.

For the real specialization, take `a=6 m²`, `q=1`, `u=1+2x/m`,
`f=1+2x/m`, and right flux `12 m`.
For reflection, substitute `x -> 6m-x` into the complex solution and load.
The fixed value is then at `x=6m`; the natural boundary is at `x=0`.
Its derivative is `-2+i` per metre, and multiplying by normal `-1` preserves
the outward flux `18+6i m`.

No matrix, coefficient, solution or tolerance is generated from the implementation
under test. Two-point Gauss quadrature is exact for these degree-two integrands.
Absolute action tolerance `1e-12` covers binary64 evaluation and the fixed small
sums at magnitudes below 50. Solution tolerance `1e-10` accommodates the explicit
`1e-12` relative / `1e-14` absolute iterative-solver controls and the small,
well-conditioned two-coordinate system; it is not a refinement error estimate.
