# Model source

The [integration test](../../../../crates/eqiora/tests/structured_upper_diagonal_schur.rs)
owns the packaged component source and the exact physical similarity substitutions.
It compiles the public Fluid package with retained geometry supports, then uses the
existing affine-triangle rectangle production policy. Its Schur groups use the
compiled velocity and pressure Field identities and the explicit pressure gauge.
No copied mesh, matrix or solver output is an input fixture.
