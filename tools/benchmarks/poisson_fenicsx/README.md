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

Only levels 4, 8, 16 and 32 cells per axis are implemented, tested and approved.
The driver rejects larger levels. Extending to 64, 128 or 256 needs new
independently generated references and review before observations. Export coordinates and nodal coefficients and compare by coordinates,
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
Reuse the existing analytically motivated regression/conformance allowance
in `crates/eqiora-numerics/tests/support/cartesian_poisson.rs::error_budget`.
It models residual, local roundoff and integration error; it is not a certified
floating-point error bound for FEniCSx.
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

After the first ordinary execution and validation, each tool performs exactly
one additional diagnostic execution, followed by two uninstrumented warmups
and seven uninstrumented timed samples in benchmark mode. FEniCSx enables its
phase clocks only in that extra diagnostic call; it never reuses the cold
call's clocks. Eqiora enables `profile=True` only in its extra diagnostic call.
These diagnostic calls are separate from the timed campaign, within the same
child process. They occur before the ordinary warmups and can warm caches;
this ordering is recorded rather than described as an untouched process.

Collect Eqiora `profile=True` in that separate diagnostic execution. Its inclusive
wall spans and exclusive self times are different quantities; neither is a
partition of the unprofiled median. The current scalar assembly-finalization
span includes numeric preparation, representability checks and canonical
system construction, but excludes some earlier mapping work. The base wheel used when declaring this protocol lacked a separate reference
solver span. The tested development profiling wheel adds that span at the
existing execution owner. Older wheels without it remain unsupported in
benchmark mode. Require the actual phase inventory before reporting a separate
solve duration; never infer it by subtraction from Run wall time.

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

## Reproduce a correctness smoke

Use a [development wheel](../../../docs/site/src/content/docs/get-started/index.mdx#development-build)
containing Scalar and, for phase inspection, the reference-solve telemetry.
Published Eqiora 0.1.5 does not contain this Scalar workflow. Record the wheel's
actual producer revision and build profile; these are not the harness revision.
The declared revision argument is not by itself a build-provenance proof.

Pixi 0.84.0 was used to install the checked-in Linux environment lock. Keep all
environments, JIT caches and output under home-backed scratch:

Set `EQIORA_WHEEL`, `EQIORA_WHEEL_REVISION`, `EQIORA_BUILD_PROFILE` from the
retained build record and `COMPARE_CPU` to an allocated CPU. The output directory
must be new. The driver pins both children to that CPU and sets one-thread
controls before importing either library. It clears foreign loader, MPI,
PETSc and compiler overrides for the isolated FEniCSx child.


```bash
scratch="$HOME/.cache/eqiora/poisson-fenicsx"
mkdir -p "$scratch/environment" "$scratch/tmp"
cp tools/benchmarks/poisson_fenicsx/pixi.toml tools/benchmarks/poisson_fenicsx/pixi.lock "$scratch/environment/"
TMPDIR="$scratch/tmp" PIXI_CACHE_DIR="$scratch/pixi-cache" \
  pixi install --locked --manifest-path "$scratch/environment/pixi.toml"
uv venv --python 3.13.14 "$scratch/eqiora"
uv pip install --python "$scratch/eqiora/bin/python" "$EQIORA_WHEEL" 'numpy==2.5.4'
```


```bash
TMPDIR="$scratch/tmp" "$scratch/eqiora/bin/python" \
  tools/benchmarks/poisson_fenicsx/run.py "$scratch/correctness-run" \
  --eqiora-python "$scratch/eqiora/bin/python" \
  --fenicsx-python "$scratch/environment/.pixi/envs/default/bin/python" \
  --eqiora-wheel "$EQIORA_WHEEL" --eqiora-revision "$EQIORA_WHEEL_REVISION" \
  --eqiora-build-profile "$EQIORA_BUILD_PROFILE" \
  --mode correctness-only --levels 4 8 16 32 --cpu "$COMPARE_CPU"
```

Each child writes its actual environment, effective solver controls, coordinate
and field data, analytic check, phase inventory and memory snapshots. Candidate
field hex files are retained before the accuracy gate, including rejected
candidates. `summary.json` is written only after both tools pass every requested
level and convergence. Correctness-only mode publishes no execution timing table.

The six standalone analytic/admission checks need NumPy but neither native tool:

```bash
TMPDIR="$scratch/tmp" "$scratch/eqiora/bin/python" -m unittest discover \
  -s tools/benchmarks/poisson_fenicsx -p test_protocol.py -v
```

`--mode benchmark` additionally requires actual assembly and linear-solve
profile spans. Older Eqiora wheels fail explicitly before the warm timing loop;
there is no inferred or substituted solve duration. Its raw timing collection is
unqualified: a quiet reference environment and optimized, authenticated builds
remain necessary for publishing a performance comparison. Process CPU readings
exclude compiler child CPU, even when preparation wall time includes JIT.
Use the pre-profile `memory_after_execution_before_accuracy` high-water mark
immediately after cold field return, before output/protocol inspection or
accuracy/profiling. Final whole-child memory also includes untimed
accuracy checks, reporting and diagnostic profiling.

## Matched revision follow-up

This follow-up is declared before implementation or candidate timing observations.
It compares already installed baseline and candidate wheels with the same existing
worker scripts, mathematical problem, solver controls and measurement boundaries.
Wheel payloads must match the installed packages. Retain the actual source/build
records and complete installed-family verification separately; a supplied revision
or build-profile string does not authenticate a build.

Run one exploratory diagnostic campaign per revision first, excluded from the
paired summary. Then run exactly eight matched pairs, alternating baseline then
candidate and candidate then baseline, four pairs in each order. Each campaign
runs both Eqiora and FEniCSx at all four declared levels, with a fresh child and
fresh JIT cache per tool and level. Each retains its cold execution, one extra
diagnostic execution, two warmups and seven raw timed samples. Reuse no numerical
matrix or result. Retain every attempt, failure, command, log and raw sample;
do not stop for a favorable result, select a fastest attempt, or replace failures
with retries. A failed campaign ends the attempt as incomplete.

Use one fixed CPU for all workers, a separate CPU for the supervisor and one
thread per library. Record the selected CPU and its SMT siblings, their load
throughout each campaign, governors, boost controls and host load. Read controls
without changing global settings or acting on other users' processes. Readiness
requires valid declared inputs, matching software, effective affinity/thread
controls and an operating monitor; CPU utilization is descriptive and has no
pass cutoff. The earlier requirement for three exactly 100%-idle observations
was an operational veto, not a mathematical or #727 requirement. It is removed
before these observations, rather than replaced by a tuned utilization threshold.
The two earlier readiness-stopped attempts remain retained and are not pairs in
this campaign.

The independent observational units are the eight pairs, not the seven calls
within a process. Report each candidate/baseline median ratio, their median and
range, each campaign's within-process spread, wall and process-CPU timing,
differences by launch order and chronological drift. The exploratory campaigns
remain separate. This is descriptive evidence on its stated host: it establishes
no stable reference environment, #802 5% non-regression result, 2x cylinder result,
general solver ranking or cross-tool internal-phase speed ratio.

`run_matched.py` uses existing installed environments; it does not install or
build either revision. Supply each retained build record as a file, alongside
the actual wheel digest, source revision and build profile. The runner verifies
installed package payloads against the supplied wheels and matches the two
Python/dependency environments. It retains build records verbatim but does not
authenticate their declarations or replace complete installed-family checks.
The allocated supervisor CPU must be on a different physical core from the
worker and its SMT siblings. The output must be new, under home-backed scratch.

```bash
TMPDIR="$scratch/tmp" python3 tools/benchmarks/poisson_fenicsx/run_matched.py \
  "$scratch/matched-attempt" --cpu "$COMPARE_CPU" --supervisor-cpu "$SUPERVISOR_CPU" \
  --fenicsx-python "$scratch/environment/.pixi/envs/default/bin/python" \
  --baseline-python "$BASELINE_PYTHON" --baseline-wheel "$BASELINE_WHEEL" \
  --baseline-sha256 "$BASELINE_SHA256" --baseline-revision "$BASELINE_REVISION" \
  --baseline-build-profile "$BASELINE_PROFILE" --baseline-build-record "$BASELINE_BUILD_RECORD" \
  --candidate-python "$CANDIDATE_PYTHON" --candidate-wheel "$CANDIDATE_WHEEL" \
  --candidate-sha256 "$CANDIDATE_SHA256" --candidate-revision "$CANDIDATE_REVISION" \
  --candidate-build-profile "$CANDIDATE_PROFILE" --candidate-build-record "$CANDIDATE_BUILD_RECORD"
```

The plan is saved before preflight and campaigns. Each campaign retains its
command, exit status, worker logs, CPU counter samples, control snapshots and
ordinary harness output. `stages.json` preserves completed campaigns, including
exploratory results. `summary.json` is written only after all eighteen campaigns
complete; it reports eight pairs, both launch orders, chronological drift and
process CPU alongside wall time. On interruption or failure, owned campaign
processes are stopped and waited for, and `failure.json` identifies the incomplete
attempt. Keep that directory; a later attempt must use a new one.

Driver tests use synthetic timings and package payloads without native tools:

```bash
TMPDIR="$scratch/tmp" python3 -m unittest discover \
  -s tools/benchmarks/poisson_fenicsx -p test_matched.py -v
```

## External sources

- [FEniCSx download and supported binary installation](https://fenicsproject.org/download/)
- [DOLFINx 0.11 Poisson demonstration](https://docs.fenicsproject.org/dolfinx/v0.11.0/python/demos/demo_poisson.html)
- [DOLFINx 0.11 mesh constructors](https://docs.fenicsproject.org/dolfinx/v0.11.0/python/generated/dolfinx.mesh.html)
- [Basix 0.11 quadrature API](https://docs.fenicsproject.org/basix/v0.11.0/python/_autosummary/basix.quadrature.html)
- [PETSc residual tolerances](https://petsc.org/release/manualpages/KSP/KSPSetTolerances/)
- [PETSc residual norm selection](https://petsc.org/release/manualpages/KSP/KSPSetNormType/)

- [PETSc BiCGStab (`bcgs`)](https://petsc.org/release/manualpages/KSP/KSPBCGS/)
