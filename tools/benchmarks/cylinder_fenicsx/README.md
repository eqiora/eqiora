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

Only geometry, mesh, public compilation/resolution, function-space, boundary
mapping and field-export preparation are authorised at this stage. No solve,
accuracy threshold, force oracle or timing campaign is declared here. The
primary mathematical review must settle the force-reaction versus stress
comparison and scientific acceptance criteria before numerical observations.
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

## Pending external-only acceptance calibration

Use independent direct solvers (Eqiora SparseLU/Faer and an available locked
PETSc factorization), float64, recording the actual backend and independently
computed free residuals. A direct solver's nominal relative tolerance does not
establish a forward-error bound. Fix scales from inputs: velocity U, length H,
pressure mu U/H and force per depth mu U. The primary review must freeze the
external-only engineering-budget algorithm before any solved output is read.
Reference conditioning, adjoint residuals, assembly/basis conversion and force
reduction roundoff must be accounted for separately. No numerical drag, lift,
field or force tolerance is selected by this preparatory implementation.

The force adjoint relation uses J(x) = ell^T x + c and A^T z = ell. In a common
residual norm the force difference is bounded by the dual norm of z times the
sum of the two residual budgets, with separately justified assembly/evaluation
and computed-adjoint allowances. Report empirical calibration as empirical;
do not claim a certified inverse bound from observed residuals alone.

Required falsifiers include reversed force sign, omitted bubbles, symmetric
gradient replaced by vector Laplacian, an added pressure gauge or pressure
shift, missing inlet lifting, wrong cylinder tags and reordered DOFs/cells.
Reject an engineering allowance broad enough to admit these changes. No timing
or candidate solve runs until the primary explicitly authorises the frozen
reference calibration and candidate accuracy stages.
