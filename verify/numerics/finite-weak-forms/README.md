# Finite weak forms and spectral phase

The source fixture declares `inner(eta,H*u)=inner(eta,lambda*u)` on an exact
two-coordinate basis. It executes through the existing Model, authored
Formulation, eigen Plan and Result owners. The case compares real and complex
spectra and spectral projectors against independent analytic values; eigenvalues
alone cannot distinguish the complex matrix from its transpose/conjugate.

Positive paths include real specialization, a Hamiltonian measured in joules,
a length-valued test and a rectangular map composition with a three-coordinate
intermediate basis. Canonical Plan/Result replay retains the authored identity;
an automatic Plan is distinct. Wrong weak coefficients and phases reject at
correspondence, while a non-Hermitian source rejects at Hermitian admission.
Unit-only scaling and a zero-multiplied foreign-basis application also reject,
after their corresponding valid paths have succeeded.

Run through the repository gate with `--case numerics.finite-weak-forms`.
The [derivation](references/README.md) and [expected projections](expected/README.md)
do not use implementation output. This case does not establish arbitrary tensor
algebra, generalized metrics, degenerate eigenspaces, Python/package authoring
or time evolution.
