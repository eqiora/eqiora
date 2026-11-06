# Ordinary steady-Stokes cylinder comparison (preparation)

This comparison is being prepared for the actual finite channel and cylinder
in the shipped `steady-flow-past-cylinder.eqi` example. It uses the public
`Eqiora.Fluid` steady-Stokes components, GeometryGraph, Gmsh mesh and MINI
P1-plus-cell-bubble velocity/P1 pressure lifecycle. The channel is 2.2 m by
0.41 m with a circular hole of radius 0.05 m centred at (0.2 m, 0.2 m).
Dynamic viscosity is 0.001 Pa s; the inlet parabola has maximum speed 0.3 m/s.
Walls and cylinder have zero velocity. The outlet has zero full Cauchy
traction; no zero-mean pressure constraint is added.

The preparation will transfer the public Eqiora mesh coordinates, triangle
connectivity and named boundary entities to FEniCSx 0.11, preserving the same
polygonal mesh and boundary partition. No private model/result JSON or
alternative physics implementation supplies Eqiora's side. FEniCSx uses its
independent standard finite-element assembly of the same Stokes problem.

Geometry, mesh, public compilation, function-space and field-export
preparation are implemented. The reviewed specification below fixes the force
reference and empirical budget algorithm before numerical observations. Its
reference batch and every candidate solve remain unexecuted; no timing
campaign is authorised.
DFG/Navier–Stokes benchmark numbers are not Stokes reference values.

The existing shared measurement and child-process owners will run any later
reviewed comparison. This directory does not own another measurement loop.

## Reviewed mathematical specification

The primary mathematical review accepts a discrete comparison against the
independently assembled FEniCSx MINI solution on the same affine mesh. This
supports agreement on that polygonal domain, not analytic continuum cylinder
drag. The cylinder centre is below the channel midpoint; zero lift is not an
acceptance condition.

Use sigma = 2 mu epsilon(u) - p I, with no body force, and

```
a((u,p),(v,q)) = integral(2 mu epsilon(u):epsilon(v)
                         - p div(v) - q div(u))
epsilon(u) = (grad(u) + transpose(grad(u))) / 2
```

Both pressure couplings have the same negative sign. The outlet has zero full
Cauchy traction, including its tangential component. There is no pressure
point constraint, mean subtraction, multiplier or constant-pressure nullspace.
A vector-Laplacian do-nothing outlet would describe a different problem.

Transfer public mesh coordinates, triangle connectivity and labelled boundary
edge vertex pairs. Map DOLFINx-reordered vertices, cells and facets by original
input identities. Check vertex/cell bijections, positive cell areas, complete
boundary coverage, edge incidence, fluid area and clockwise hole orientation.
Do not infer cylinder tags from midpoint distance to the exact circle.

The velocity span is P1 plus b = 27 lambda1 lambda2 lambda3, independently in
each component; pressure is continuous P1. There are 3 Nv + 2 Nc full scalar
unknowns and 3 Nv + 2 Nc - 2 ND free unknowns, where ND counts distinct
Dirichlet velocity vertices. Eqiora coefficients are hierarchical. Basix's
installed enriched basis is nodal at the three vertices and centroid, so the
conversion is

```
beta_cell = u(centroid) - (u(vertex1)+u(vertex2)+u(vertex3))/3.
```

Verify the reconstructed polynomial at additional interior points; raw Basix
coefficient equality is not the comparison. Store both components of the
public vertex velocity and cell-bubble coefficients and the pressure at every
vertex, in original mesh order. Shared serialization must retain every one of
these associations before any accuracy rejection.

Eqiora's assembly uses three-point Gauss on each Duffy axis. The stiffness is
degree at most four and pressure coupling at most three. Verify any external
assembly rule against reference-triangle monomials a! b! / (a+b+2)! through
total degree four. A different independently exact rule is allowed only with
its distinction and roundoff allowance recorded. Use degree-six integration
for the velocity L2 difference, since the bubble is cubic. Symmetrically
eliminate prescribed velocity values with lifting; restrict a full identity-row
FEniCSx system to the same free unknowns before system comparison. Preserve
wall values at outlet corners.

## Cylinder force and flux

For each component, construct a continuous P1 virtual velocity w which is the
component unit vector at cylinder vertices and zero at all other vertices.
Its hierarchical bubbles vanish. In the nodal Basix basis its centroid values
must equal the mean of its three vertex values. Independently assemble

```
R_i = integral(2 mu epsilon(u):epsilon(w_i) - p div(w_i)).
```

R is the force on the fluid; -R is the force on the cylinder, in N/m. Compare
them to the public `boundary_force(...).on_domain` and `.on_selection`,
respectively. This volume virtual work corresponds to Eqiora's sum of original
constrained momentum residuals. Raw boundary stress integration is a separate
diagnostic, not a solver-tolerance oracle for this discrete reaction. Changing
the virtual lift by a function zero on essential boundaries must change force
only by the corresponding free momentum residual.

For inlet segment lengths h_j, the exact flux magnitude of the nodal P1 trace is

```
Q_in,h = 2 U H / 3 - (2 U / (3 H^2)) sum(h_j^3),
U = 0.3 m/s, H = 0.41 m.
```

Signed inlet flux is negative. The continuum parabola's 0.082 m²/s is not the
algebraic flux oracle. Test weak continuity and global flux balance; MINI does
not have pointwise zero divergence. Sum global essential reactions over the
union of constrained vertices once, avoiding inlet/wall corner double counts.

## Reference acceptance meaning

Use independent direct solvers (Eqiora SparseLU/Faer and an available locked
PETSc factorization), float64, recording the actual backend and independently
computed free residuals. A direct solver's nominal relative tolerance does not
establish a forward-error bound. Fix scales from inputs: velocity U, length H,
pressure mu U/H and force per depth mu U. The primary-reviewed algorithm below fixes the
external-only engineering budget before any solved output is read.
Reference conditioning, adjoint residuals, assembly/basis conversion and force
reduction roundoff must be accounted for separately. No numerical drag, lift,
field or force tolerance is selected by this preparatory implementation.

The first campaign uses the explicit empirical algorithm below, without a
claimed certified inverse bound or an additional adjoint solve.

Required falsifiers include reversed force sign, omitted bubbles, symmetric
gradient replaced by vector Laplacian, an added pressure gauge or pressure
shift, missing inlet lifting, wrong cylinder tags and reordered DOFs/cells.
Reject an engineering allowance broad enough to admit these changes. No timing
or candidate solve runs until the primary explicitly authorises the frozen
reference calibration and candidate accuracy stages.

## Frozen first-reference-batch algorithm

The primary review has now selected the following empirical engineering gate.
It is an agreement-resolution requirement, not a certified forward-error or
continuum-error bound. Numerical budget values must come exclusively from the
external batch before Eqiora runs. The initial mesh is the ordinary example
mesh above; later actual-cylinder mesh levels remain required for a broader
campaign.

Run exactly three external solutions in this order, with one rank/thread and
float64, no pressure gauge and the same nodal inlet/no-slip lifting:

| Label | Direct implementation | Assembly quadrature |
| --- | --- | --- |
| A, designated reference | PETSc preonly + LU + MUMPS | positive nine-point Duffy, three Gauss points per axis |
| B | PETSc preonly + LU + serial SuperLU | the same Duffy rule |
| C | PETSc preonly + LU + MUMPS | positive Basix degree-six Gauss–Jacobi, sixteen points |

Both distinct implementations are present in the unchanged locked environment;
SciPy is not installed or required. No fourth reference is planned. Keep the
factorization defaults and record actual PETSc type, options, implementation,
precision and convergence reason. Do not select a different reference after
seeing a candidate. All physical norms and volume virtual-work forces use the
same independent degree-six observation route.

The common coefficient order is original vertex velocity components, original
cell hierarchical bubble components, then original pressure vertices. Apply
the explicitly constructed enrichment basis congruence before essential
elimination. With Dv diagonal (U for both velocity associations, P = mu U/H
for pressure), form

```
xhat = inverse(Dv) x
Ahat = transpose(Dv) Aphysical Dv / (mu U^2)
bhat = transpose(Dv) bphysical / (mu U^2).
```

Let eps be float64 epsilon, m be the maximum nonzero row count plus two,
gamma = m eps/(1-m eps), and
D = ||Ahat||inf ||xhat_A||inf + ||bhat||inf. Reject if m eps >= 1 or D is
nonpositive/nonfinite. Freeze r_cap = 100 gamma D. Every A/B/C residual and
the later candidate residual must satisfy this same infinity-norm ceiling on
the designated-A operator. Export its CSR arrays, RHS, free indices and scales
in a standard NumPy archive. Read with `allow_pickle=False`, verify file hash,
all dimensions, integer bounds, free-index bijection and declared scaling.
The Eqiora environment evaluates this external operator with NumPy; it does
not need PETSc or another PDE assembly.

For each of four metrics independently (complete velocity RMS, pressure RMS
without mean removal, cylinder x force, cylinder y force), set

```
s_Q = U, P, mu U, mu U respectively
S_Q = max(s_Q, reference amplitude of A)
delta_Q = maximum physical difference over all A/B/C pairs
tau_Q = max(sqrt(eps) S_Q, 10 delta_Q)
broad_cap_Q = 1e-4 S_Q.
```

Field RMS is sqrt(integral(|field|²)/fluid area), using degree-six integration
of the complete MINI polynomial. Each force uses its own amplitude and floor;
lift never borrows drag's magnitude. All comparisons are absolute in their
physical units. The candidate must satisfy d_Q(E,A) <= tau_Q.

Before freezing, compute material-mutant distances from A. Velocity uses the
smaller RMS distance to zero velocity and to removal of all hierarchical
bubbles. Pressure uses the smaller distance to zero pressure and addition of
constant P. Drag uses min(abs(Fx,A), mu U). Lift uses mu U. Require

```
tau_Q <= min(broad_cap_Q, material_mutant_distance_Q / 20).
```

Run zero-velocity, omitted-bubble, zero-pressure, pressure-shift, zero-drag,
reversed-drag, one-force-scale drag/lift shifts through the actual agreement
gate and require rejection. Also reject zero/reversed lift when abs(Fy,A) is
at least mu U; otherwise do not claim its sign is numerically resolved. A
failed cap is inconclusive. It cannot increase the margin or remove a
falsifier. The factors 100, 10, 1e-4 and 20 and sqrt(eps) floor were fixed by
the primary mathematical review before any solved output.

Check nodal boundary traces with a declared 256 eps U arithmetic margin and
analytic discrete inlet flux with 256 eps U H times the inlet edge count.
Summing all free pressure equations gives a net-flux allowance
Nv U H r_cap plus that input-only arithmetic margin. These are separate
trace/continuity controls, not the four field/force tolerances.

`calibrate.py --mesh MESH_JSON --output NEW_DIRECTORY` implements this batch.
It records each reference observation, original mesh identities, quadrature,
source and installed environment, common-operator files and their hashes.
Only a completely accepted batch writes `reference-manifest.json`; failure
retains attempts but produces no frozen candidate budget. Candidate collection
must verify this manifest and cannot recalibrate it. The command has not been
run: explicit primary authorisation is still required.
