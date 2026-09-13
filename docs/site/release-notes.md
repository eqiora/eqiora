# Release notes

Eqiora `0.1.0a10` is the current public alpha. APIs and saved-file formats may
change before 1.0; review the changes below when upgrading.

## 0.1.0a10 — exact imported nominal types

Python `eqiora.Module` consumers can reference public Record, Enum, and finite
Space declarations through `ModuleRef.record`, `ModuleRef.enum`, and
`ModuleRef.space`. Fields, Parameters, record constructors, enum members, and
finite-space values emit the explicit qualified import path and leave the
provider declaration in its owning module.

These handles use the existing exact Module graph. Private, unknown, transitive,
and equal-shaped foreign nominal declarations remain invalid, and the compiler
continues to own visibility, identity, member typing, source locations, and
locked-package resolution. This alpha adds no loader, schema migration, or
compatibility alias.

## 0.1.0a9 — structured execution profiling

Python runs accept `profile=True` and return an immutable, hierarchical timing
profile with the result. The profile separates run, setup, solve, time-step,
assembly, nonlinear-iteration, linear-solve, backend, and post-processing work,
and retains structured nonlinear convergence observations. The transient
cylinder example prints a compact summary from the same data.

Profiling is disabled by default. Disabled runs install no collector and retain
the same numerical result bits. Timings describe the current process only;
profiles are not serialized with Results and do not claim distributed or global
timing coverage. The common flow paths, ODE execution, Newton iteration, and
Faer factorization/backsolve boundaries are covered in this alpha.

The Rust quick start now uses the current `eqiora::compiler` facade and canonical
source syntax.

## 0.1.0a8 — unified authoring and explicit numerical execution

Eqiora 0.1.0a8 expands source and Python modeling through `eqiora.Module`,
with typed arrays, records, enums, explicit derivatives, events, and clocks.
Projects gain exact local/Git dependency locks and offline vendoring, and the
language-server preview provides diagnostics, formatting, and navigation.

Coupled scalar Q1 equations and fixed-reference FSI share equation-derived
region assembly with exact Field ownership. Solver requests explicitly choose
an objective or a complete algorithm, preconditioner, reduction, and provider.
Typed observables, parameter batches, and bounded JAX `vmap` composition extend
analysis. Smooth ODE functionals add terminal evaluation and accepted-step
Simpson integration independent of output cadence. Steady scalar Laws retain
outward flux and source through Model replay.

These are bounded alpha capabilities. Laws exclude storage and moving volumes;
trajectory functionals exclude events, resets, derivatives, and spatial-time
composition. Convex-polyhedral Geometry supports correspondence to supplied
tetrahedral meshes, without adding a tetrahedral mesher or new PDE/FSI execution.
General equation-driven numerical admission and arbitrary multi-region
composition remain work for the subsequent 0.1.0 release.

When upgrading, replace displaced Python `Source`/builder paths with
`eqiora.Module`, update Model/Component signatures, and use explicit symbolic
`equation(lhs, rhs)` calls. Recompile models and regenerate saved execution
artifacts and package locks; obsolete pre-1.0 decoders and aliases are removed.

## 0.1.0a7

Scalar conservation supports Cartesian problems in 1D–3D. Continuum models
share material and kinematic definitions across elasticity, transient flow,
and fluid–structure interaction.

Runs now reuse prepared structure and Faer factorizations. Python `Source`
adds natural equation authoring, model-local aliases infer dimensions, and
cancellation exposes the last accepted State. Mesh generation uses `MeshPlan`.

Material compositions can combine multiple property releases in a Component
Law from Eqiora source, Python, or a package. `Eqiora.Solid@0.2.0` provides
a composition for Young's modulus and Poisson's ratio.

When upgrading, replace removed legacy dimension aliases with the current
names. Colab is the maintained hosted-notebook example; duplicate Marimo and
Jupyter examples have been removed.

## 0.1.0a6

The language adds scalar property bindings, affine coefficients, scalar primal
formulations, directional Stokes correspondence, `math.pi`, typed model-local
`let`, and derived-dimension aliases. Equivalent additive equation orientations
compile to the same residual.

Plans, Results, and Trajectories can be saved and reopened as local files.
Rust adds a common transient `RunRequest`. Cylinder meshes and startup media
have been refined for clearer presentation.

## 0.1.0a5

Resolved Plans, restartable States, Trajectories, and Results can be serialized
and reopened from Python. Reloading checks that the model and other required
inputs match.

Static scalar, elasticity, and Stokes runs return a Result directly. The
optional `eqiora.View` displays 2D geometry, meshes, selections, and scalar
fields in Python notebooks.

## 0.1.0a4

Python can compile Eqiora equations with geometry, choose numerical settings,
and run steady and transient flow. The release adds a GitHub-backed Colab
walkthrough of the first ten cylinder-flow startup steps. Its short run
illustrates startup rather than a developed wake or benchmark comparison.

The Studio packaged DC-drive example remains executable in this release.

## 0.1.0a3

The exact-cylinder Python example uses Gmsh 4.15.2 to generate its mesh and
plot the resulting pressure field.

## 0.1.0a1

See the [original release](https://github.com/nkiyohara/eqiora/releases/tag/v0.1.0a1)
for its source, package artifacts, and supported platforms.

The repository [changelog](https://github.com/nkiyohara/eqiora/blob/main/CHANGELOG.md)
records additions, fixes, and migrations. Browse the
[capability matrix](capabilities.md) for available functionality and the
[evidence guide](evidence/index.md) for numerical checks.
