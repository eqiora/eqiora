# Ordinary affine Q2 scalar diffusion

This case connects full Q2 quadrilateral/hexahedral fields to the ordinary Plan,
region compiler, assembly map, solver and Result codec. The implementation and
protocol are under review. No new Q2 manufactured solve has run yet. Reference
and pure protocol tests are not candidate scientific results.

The initial ordinary scope is static linear scalar diffusion, with real or complex
coefficients, on affine Cartesian quads/hexes in 2D/3D. P2 triangle/tetrahedron
reference bases and traces share the kernel; ordinary P2, vector quadratic spaces,
Taylor–Hood, nonaffine/curved geometry and prisms remain separate work. This case
adds no public scalar force or flux observation API. Unsupported samplers and
derived observations must reject complete quadratic fields rather than keep only
the vertices. CPU host-serial evidence here does not establish MPI/GPU behavior
or performance.

## Sine convergence protocol

The independent [reference and comparator](reference/README.md) own the mathematics,
fixed budgets, continuous physical L2 and H1 **seminorm**, finite-grid reference
rates and falsifiers. The boxes are [0,2]×[0,3] m and [0,2]×[0,3]×[0,4] m, with
N=2,4,8 cells per axis and homogeneous essential data on every face. The
manufactured dimensionless field is
`u=product_i sin(pi*x_i/L_i)`, and `f=pi²*sum_i(1/L_i²)*u` has units 1/m².
The theoretical limiting orders, L2 three and H1 seminorm two, are distinct from
the independently enclosed finite rates at these levels.

The failed three-point load audit is retained in the reference directory. The
production choice is explicitly four Gauss points per axis. Plan v28 must retain
that effective rule and replay it unchanged. It is not sufficient for the exported
coefficient file to declare four. The 1% load perturbation and 1e-3 combined
uncertainty budgets are unchanged; the comparator integrates the complete
candidate field with certified rational/analytic formulas. It does not project
onto sine amplitudes or use sampled field errors.

`check.py` imports an installed package and performs one ordinary call to
`eqiora.run`, on a reopened ordinary Plan. It uses the reference provider's
BiCGStab, identity preconditioning, reproducible reduction, relative tolerance
1e-13, absolute tolerance 1e-14 and maximum 100000 iterations. These solver inputs
do not replace the independent residual bound. The authored pi and forcing
parameters are ordinary binary64 inputs, recorded in the execution receipt; their
errors also enter the comparator's independently assembled residual.

Each invocation requires one caller-pinned CPU, one thread in OMP/OpenBLAS/MKL/Rayon,
and `CUDA_VISIBLE_DEVICES=''`. It creates a fresh output directory and retains
source, model, mesh, Plan, Result, candidate coefficients and producer metadata.
There is one Plan reopen and one Result reopen; neither is a second PDE solve.
There are no profile calls, warmups, timing samples or solution reuse. The fresh
Result is the candidate. Reopening must preserve every coefficient and exact
canonical bytes; the reopened Result cannot replace a failed original.

`production.py` checks the actual Plan schema/policy, actual full Cartesian vertex
and cell inventory, exact caller/output mesh identity, Result/Plan/Model linkage,
scalar type and units, canonical coefficient entities and all association counts.
Q2 has one nodal value per vertex, edge, face where present, and cell. Its nodal
points are affine entity centroids, computed from actual public mesh connectivity.
No field interpolation is involved. Every coefficient is exported as canonical
binary64 hex, and coordinates as exact decimal strings. Nodal cell-center values
are `cell`, never `cell-bubble`. The producer receipt binds the mathematical input
to the Plan/Result/mesh artifacts by SHA-256; the independent comparator consumes
only the mathematical input.

After review and execution authorization, run one fresh invocation for each
(dimension,N), in order (2,2),(2,4),(2,8),(3,2),(3,4),(3,8), stopping on failure.
The launcher must retain actual command, source/artifact/input identities,
terminal status and controls. Do not overwrite, retry a failed numerical outcome,
rebuild the candidate or change thresholds automatically. An example invocation
for a preselected CPU is:

```sh
taskset -c "$cpu" "$installed_python" check.py --dimension 2 --cells 2 --cpu "$cpu" --output "$fresh_output"
python3 reference/compare.py "$fresh_output/candidate.json" --output "$fresh_output/comparison.json"
```

The caller supplies the declared thread/GPU environment above. `check.py` exit zero
means **export completed**, not scientific acceptance. The comparator must exit
zero separately. After all three accepted levels in a dimension, assemble their
unmodified candidate objects into a JSON array and use `reference/compare.py
--levels` to check and report the complete finite-rate set. The ordinary export
script contains no private residual, norm or acceptance implementation.

## Polynomial patch and boundary reaction

A separate N=2 witness will use the same physical boxes and
`u=product_i(1+(x_i/L_i)²)`. Its source is
`f=-sum_i(2/L_i²)*product_j!=i(1+(x_j/L_j)²)`, with units 1/m². The upper first-axis
face has essential data; all other faces have the analytic normal derivative.
This exercises nonzero boundary midpoint/face data and every nodal family. The
native existing `plan.linear_reactions` owner must retain all constrained physical
weak-dual coordinates. The upper sum is +4 in 2D and +64/3 m in 3D. These scalar
weak actions are not mechanical forces.

The independent rational patch oracle and fixed residual/coefficient/reaction
budgets are being prepared separately. Native export and execution remain on hold
until that oracle and the producer protocol are frozen and reviewed. It will use
solver relative and absolute tolerances 1e-14, retain all actual field coordinates
and upper constrained reaction coordinates, and bind the mathematical input to
separate Plan/Result/mesh producer receipts. Neither expectations nor allowances
may be inferred from the native outputs.

## Checks without candidate solves

```sh
python3 -B -m unittest discover -s . -p 'test_production.py'
python3 -B -m unittest discover -s reference -p 'test_*.py'
```

The first command uses synthetic public-protocol objects and imports no Eqiora.
It rejects missing cell coefficients, missing faces, aliased edges, incorrect
strata, a wrong effective rule, altered mesh inventory, wrong Result/Plan/Field
linkage, wrong units, nonfinite values and a bubble-labelled nodal cell. The second
command exercises the independent reference/comparator, including arbitrary
non-sine fields and controlled full-vector perturbations. Core tests separately
cover quadratic monomials/gradients, parent traces, topology-preserving cube
permutations, quadrature minima, effective-policy replay and stale prepared facet
geometry/order/quadrature. These checks do not establish an installed Q2 solve.
