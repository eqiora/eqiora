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
