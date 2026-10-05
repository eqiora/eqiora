# Dimensioned finite Hamiltonians through shared eigen and time owners

The executable case is `crates/eqiora/tests/finite_hamiltonian_lifecycle.rs`.
It constructs a two-level Hamiltonian and a 2×3 tensor-factor Hamiltonian,
including an equivalent explicit 6×6 matrix. Shared eigen Plans, time Plans,
initial States, implicit midpoint and typed Observables execute the models.
The Hamiltonian has units J, hbar has units J s, and amplitudes are dimensionless.

[Independent expectations](expected/README.md) derive every spectrum and final
amplitude without calling the implementation to generate reference values.
The case also checks exact Plan replay, initial normalization, complete raw
Hermiticity (including a tiny asymmetry erased by evolution scaling), and
structural generator admission. Nonlinear forcing that vanishes at basis probes
must reject, as must external forcing, time dependence, missing hbar units and
wrong basis/factor/coordinate roles.

The two finite extents define this scientific evidence, not an implementation
size limit. This case does not establish arbitrary Hamiltonian convergence,
nonlinear or time-dependent norm invariants, event/reset behavior, sensitivity
conditions, arbitrary precision, or efficient many-body simulation. Python phase
and explicit basis-change invariance, immutable policy controls and foreign-State
rejection have separate focused product tests; they are not this case's runner.
