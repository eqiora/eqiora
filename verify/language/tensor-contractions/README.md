# Full-coordinate tensor contraction

This case compiles ordinary source, commits the Model, admits it through
`KernelProgram`, and evaluates the retained Relation operands. No mesh or solve
enters the claim. All values use one model-global Cartesian frame.

The independent references are:

- `[[2,3],[5,7]] * [11,13] = [61,146]`. Unequal off-diagonal entries detect
  transposition. The algebraic trace is 9; explicit componentwise multiplication
  has off-diagonal entry 9, while the vector outer product has entry 143 V².
- `(1+2i)(4-3i) + (3-i)(-2+5i) = (10+5i)+(-1+17i) = 9+22i`.
  Conjugating either operand changes this result. Integer-valued binary64
  arithmetic in these examples is exact, so equality uses zero tolerance.
- A full 2D stiffness tensor has normal coefficients 10 and 20 Pa, coupling
  coefficients 3 Pa, and all four shear coefficients 4 Pa. The source compares
  contraction with explicit component expansions. For strain off-diagonals
  `(0.01,0.01)`, both stresses are 0.08 Pa and half the full contraction is
  0.0008 Pa. For normal strains `(0.01,0.02)`, stresses are `(0.16,0.43)` Pa and
  energy is 0.0051 Pa. Dropping a symmetric off-diagonal slot changes stress and
  energy. The absolute tolerance of 1e-15 Pa bounds binary64 rounding in these
  short sums of terms below 1 Pa; it is not solver or discretization tolerance.
- Symmetric, skew and deviatoric projections are authored as ordinary additions,
  subtractions, transpose, trace and scalar products. A rank-four trace map is
  the outer product of two explicitly constructed Kronecker matrices. Their
  selected components and trace are compared with independent exact constants.

The rejection probes pair a successful same-support field compilation with a
foreign equal-size support, and reject nominal finite coordinates, implicit
array-to-vector conversion, repeated axes, and out-of-range components.

The tested boundary is uniform real/complex full-coordinate constitutive
algebra, with exact physical dimensions. It proves no material symmetry from
sample values, positivity, Hermitian inner-product convention, compressed
Voigt/Mandel mapping, field discretization, or finite-strain solver.

Run `cargo run -p eqiora-verify -- run --case language.tensor-contractions`.
