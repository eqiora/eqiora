# Two-dimensional Poisson comparison protocol

This protocol is declared before numerical or timing observations. It scopes
one part of the external comparison in #1739. It establishes no result or
performance claim yet. Three-dimensional Poisson, elasticity, cylinder flow,
MPI and GPU comparisons remain outside this slice.

## Same mathematical problem

On the unit square, with numeric coordinates in metres, solve
`-Delta(u) = 2*pi^2*sin(pi*x)*sin(pi*y)`, with zero value on the whole boundary.
The potential is dimensionless. The unique exact solution is
`sin(pi*x)*sin(pi*y)`.

Use uniform square cells, continuous tensor-product Q1 elements, real binary64
coefficients, and tensor two-point Gauss integration for stiffness and load.
FEniCSx must explicitly create quadrilateral cells; its default triangles would
be a different discretization. Its degree-three quadrature rule must have the
same points and weights as the declared two-point tensor rule. The driver
checks the actual four Basix points and weights against that closed-form rule,
with a fixed eight-ULP engineering conformance budget for coordinate/weight
rounding. This is not a rigorous Basix roundoff proof or a PDE error tolerance.

Correctness levels are 4, 8, 16 and 32 cells per axis. Larger levels 64, 128
and 256 extend the same mathematical problem only after passing its analytic
checks. Export coordinates and nodal coefficients and compare by coordinates,
never by library-specific numbering. In FEniCSx, use the explicit topological
Q1 vertex-to-dof and vertex-to-Geometry maps to export authoritative mesh
vertices. `tabulate_dof_coordinates()` evaluates a floating Geometry map and
need not preserve even a dyadic vertex bit for bit. Do not snap its values or
introduce a proximity-matching tolerance.

Eqiora uses the ordinary installed Python compile, mesh, resolve and run APIs,
with imported `Eqiora.Scalar` components. FEniCSx uses its ordinary Python mesh,
function-space, form, assembly and PETSc interfaces. Neither implementation
supplies a private replacement assembly or solver kernel.

## Solver contract

Use one MPI rank and one CPU worker, a fixed single-core affinity, and explicit
one-thread OpenMP, BLAS, MKL and Rayon settings. Eqiora's resolved execution
must report host-serial and one worker. Use BiCGStab (PETSc `bcgs`), identity/no preconditioner,
zero initial guess, relative residual tolerance `1e-12`, absolute tolerance
`1e-14`, and at most 10000 iterations. PETSc must explicitly use the
unpreconditioned Euclidean residual norm. Recompute the true residual and
report the solver's iterations, convergence reason and residual target.

Eqiora eliminates constrained coefficients. Standard FEniCSx assembly retains
identity boundary rows. Homogeneous data and a zero initial guess give the
same effective free operator, but not the same allocated matrix dimensions.
Report total, free and stored algebraic sizes rather than equating them.

The original protocol proposed CG. The first Eqiora smoke attempt stopped at
resolution with `EQ0807`, before a solve or FEniCSx observation: ordinary 2D
Geometry-backed Q1 Plans currently carry `General` operator properties. The
stronger Q1 SPD proof in that owner is limited to one-dimensional intervals.
This does not mean the mathematical 2D Poisson operator is nonsymmetric or
indefinite. Before any comparison solve, both tools were changed to BiCGStab
with the same residual controls. No private legacy path or weakened proof is
used. The analytic residual-to-error bound is independent of this algorithm
choice.

The sine load is a single discrete eigenmode on these grids. This is useful
for reproducible assembly and accuracy checks, but does not represent general
Krylov iteration complexity.

## Independent accuracy reference

Reuse the existing [80-digit sine-mode derivation and generator](
../../../verify/numerics/cartesian-poisson-fem-fvm/references/README.md) and its
continuous L2 error table. Compute each result's continuous Q1 reconstruction
error with eight-point Gauss integration per axis, outside the timed region.
Use the existing a priori residual, local roundoff and integration allowance
in `crates/eqiora-numerics/tests/support/cartesian_poisson.rs::error_budget`.
Require decreasing errors and observed order greater than 1.9. A tool-to-tool
match alone is not a scientific oracle. No allowance is fitted to observed
results or substituted for the product's residual acceptance.

## Measurement boundaries

Record interpreter and library imports separately from workflow measurement.
For each tool report wall time and process CPU time for these declared scopes:

- Mesh: Eqiora GeometryGraph construction and mesh resolve/generate; FEniCSx
  unit-square mesh creation. Explain the additional geometry checks explicitly.
- Preparation: Eqiora compile and resolve; FEniCSx function space, boundary
  data, forms and JIT compilation. Preserve a first-process/fresh-JIT-cache
  observation separately from reuse of compiled forms.
- Execution: fresh numerical assembly, zero-start solve and primary field
  recovery, with existing mesh and model/forms. No matrix, preconditioner,
  factorization or previous Result is supplied for reuse.

Exclude error integration, output inspection, protocol-conformance assertions,
Result destruction and file writes. Boundary-row/column checks and effective
solver-policy assertions run after the execution clock stops. FEniCSx's solve
bracket includes effective-policy getters and accepted-solve summary creation
before native KSP cleanup; their included query cost is not hidden. Follow the
existing [repeated-execution measurement owner](../repeated_execution/run.py):
retain a cold call, two warmups, seven raw samples, median, range and median
absolute deviation. Keep the process CPU readings alongside wall samples.

Collect Eqiora `profile=True` in a separate diagnostic execution. Its inclusive
wall spans and exclusive self times are different quantities; neither is a
partition of the unprofiled median. The current scalar assembly-finalization
span includes numeric preparation, representability checks and canonical
system construction, but excludes some earlier mapping work. The reference solver
currently has no separate solve span. A reviewed addition at the existing
execution owner is needed before a separate Eqiora solve duration can be
reported. Never infer it by subtracting other timings from Run wall time.

FEniCSx assembly and KSPSolve brackets must state their exact included calls.
These are tool-native operation boundaries, not identical internal functions;
do not publish phase speed ratios from differently instrumented scopes.
Compare unprofiled complete execution only when its stated scope matches.

## Environment and memory

Use an isolated FEniCSx 0.11 environment installed through its documented
conda-forge distribution, with an exact solved package/build lock. Run it and
Eqiora in separate processes to avoid mixing MPI library ABIs. Record the
installed Eqiora wheel digest, source revision and Rust build profile; Python,
DOLFINx, Basix, UFL, FFCx, PETSc, MPI and NumPy versions; BLAS configuration;
JIT flags/cache state; CPU, kernel, affinity, thread environment and host load;
and protocol, model and driver digests.

For each mesh level use a fresh child process and report Linux process peak
RSS (`ru_maxrss`, KiB), including imports and preparation. Also record baseline
RSS and stage snapshots. Record any JIT child peak separately. These process
high-water marks are not a simultaneous aggregate process-tree peak and must
not be described as operator-only memory or subtracted into such a claim.

The current shared host is approved only for installation and correctness
smoke tests. Their reports must identify themselves as non-reference runs.
A quiet, explicitly identified environment is required before publishing a
stable timing or speed comparison. Slower results are retained and explained.

## External sources

- [FEniCSx download and supported binary installation](https://fenicsproject.org/download/)
- [DOLFINx 0.11 Poisson demonstration](https://docs.fenicsproject.org/dolfinx/v0.11.0/python/demos/demo_poisson.html)
- [DOLFINx 0.11 mesh constructors](https://docs.fenicsproject.org/dolfinx/v0.11.0/python/generated/dolfinx.mesh.html)
- [Basix 0.11 quadrature API](https://docs.fenicsproject.org/basix/v0.11.0/python/_autosummary/basix.quadrature.html)
- [PETSc residual tolerances](https://petsc.org/release/manualpages/KSP/KSPSetTolerances/)
- [PETSc residual norm selection](https://petsc.org/release/manualpages/KSP/KSPSetNormType/)
