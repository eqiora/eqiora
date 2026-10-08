# Independent derivation

For `H=[[2,-i],[i,2]]`, the characteristic polynomial is
`(2-lambda)^2-1`, so the eigenvalues are 1 and 3. Corresponding vectors are
`(i,1)/sqrt(2)` and `(-i,1)/sqrt(2)`. Their outer products with their conjugate
transposes give the projectors in the expected-value file. Multiplying either
vector by an arbitrary unit complex phase leaves its projector unchanged.

Conjugating H preserves the characteristic polynomial but conjugates both
projectors. Thus the projector comparison detects a transpose or conjugation
mistake that an eigenvalue comparison misses. The real fixture
`[[2,1],[1,2]]` instead has normalized vectors `(1,-1)` and `(1,1)`.

The rectangular factors are
`A=[[2,-i,7],[i,2,9]]` and `B=[[1,0],[0,1],[0,0]]`.
Direct matrix multiplication gives `AB=H`; the third intermediate coordinate
does not alter the spectrum or projectors. Giving H units J gives eigenvalues
1 J and 3 J, while the projectors remain dimensionless. A test-function unit
does not change H or its spectrum; it multiplies both sides of the weak pairing.

The fixed absolute tolerance is `1e-12` for binary64 eigenvalues and projector
components of order one. This is a two-coordinate Hermitian problem with spectral
gap two and an identity metric; no conditioning or large-dimension inference is
made. Residual and normalization defects must also satisfy `1e-12`. No expected
value or tolerance is fitted to numerical output.
