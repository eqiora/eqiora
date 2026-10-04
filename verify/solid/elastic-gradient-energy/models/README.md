# Executable model

The source, exact rectangle Geometry, 16-by-16 Q1 mesh and solver request live in
[`common_elasticity_mesh_output.rs`](../../../../crates/eqiora/tests/common_elasticity_mesh_output.rs).
The energy test adds typed body-energy and right-face work Observables to the
ordinary mixed-boundary elasticity source. It adds no solve unknown or equation.

The authored stationarity profiles use the same source and Geometry with a 2-by-2
mesh. One sets lambda=2 and prescribes zero displacement on all four sides; another
keeps lambda=0 and the original mixed boundaries. The energy includes both Lamé
terms and the work of the exact conservative-load definition. The direction is
length-valued and names precisely the essential boundary inventory.

The constant-traction profile and its sign, support and nominal-data falsifiers
live in [elastic_surface_energy.rs](../../../../crates/eqiora/tests/support/elastic_surface_energy.rs).
A selected component input binds a real spatial-vector Parameter, retaining its
identity independently of its numerical value. The surface energy and boundary Law
reference that same Parameter; a same-valued independent input does not substitute
for it in correspondence.
