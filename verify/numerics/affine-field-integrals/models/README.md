# Model inputs

[`spatial_integrals.rs`](../../../../crates/eqiora/tests/spatial_integrals.rs) owns the
ordinary source Model, 3–4–5 triangle geometry and three-cell authenticated mesh.
The center coefficient is solved from the harmonic equation, with affine temperature
prescribed only on the three exterior edges. Cyclic cell orders and a foreign mesh
with a different interior vertex exercise exact geometry ownership.
