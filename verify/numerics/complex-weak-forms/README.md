# Real and complex scalar weak forms

The source fixture solves a stationary scalar reaction-diffusion equation on
`[0,6]` with two Q1 cells, one prescribed value and one prescribed outward flux.
Its complex affine solution belongs exactly to Q1. The same compiler, quadrature,
assembly and Plan/Result owners execute the real specialization.

The case first admits the authored weak residual against the original Relation.
It compares the assembled matrix and packet operator with independent element
integrals, including the complex essential-value elimination and natural load.
Normal, transpose and conjugate-transpose actions have separately fixed expected
values. Ordinary Plan execution, Plan replay and Result replay retain the solution.
Reflecting the Geometry support bindings checks a lower outward normal as well as
the upper one. Wrong test conjugation, reaction phase, flux sign and imaginary
volume load fail authored correspondence after the valid path has succeeded.

This is evidence for a bounded stationary scalar path, not arbitrary weak forms,
all spaces, refinement convergence, optional backends, finite Hermitian forms or
Python/package authoring. Packet actions use the internal common owner; this case
does not claim a public matrix-free Plan selection.

Run through the repository gate with `--case numerics.complex-weak-forms`.
The [derivation](references/README.md) and [expected values](expected/README.md)
are independent of implementation output.
