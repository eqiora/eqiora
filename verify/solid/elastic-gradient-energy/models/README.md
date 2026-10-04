# Executable model

The source, exact rectangle Geometry, 16-by-16 Q1 mesh and solver request live in
[`common_elasticity_mesh_output.rs`](../../../../crates/eqiora/tests/common_elasticity_mesh_output.rs).
The energy test adds typed body-energy and right-face work Observables to the
ordinary mixed-boundary elasticity source. It adds no solve unknown or equation.
