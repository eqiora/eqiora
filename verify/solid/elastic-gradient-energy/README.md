# Elastic gradient energy on the ordinary Result

The ordinary Cartesian elasticity Plan solves the existing mixed-boundary problem
with mu=3 Pa, lambda=0 Pa and conservative load potential q=6x Pa on the unit square.
Displacement vanishes on x=0; all other sides have zero prescribed traction.
The exact displacement is u=(x-x²/2,0) m in coherent SI coordinates.

The retained Observable integrates mu*epsilon(u):epsilon(u) over the body.
A second Observable pairs the recovered outward stress with trace(u) on x=1.
Both consume the accepted two-component Q1 coefficients and explicit quadrature,
including after Result serialization and replay. The existing pure-operator owner
projects symmetric gradients and contractions; the ordinary scalar IR applies
State JVPs. The positive executable is
[`common_elasticity_mesh_output`](../../../crates/eqiora/tests/common_elasticity_mesh_output.rs).

The independent discrete values are energy 1023/1024 N and boundary pairing 3/32 N.
For the fixed State direction eta=(x,y) m, the products are respectively 3 N and
51/16 N. These two-dimensional integrals have units of energy per unit thickness.
See [the derivation](expected/README.md). The case also rejects an incomplete
vector coefficient direction and a volume quadrature rule supplied to the boundary
Observable. Its existing artifact probes reject noncanonical Result and mesh bytes.

The direction is an arbitrary State perturbation. It is not an admissible
stationarity test: eta does not vanish on the clamped side. This case makes no
claim about authored elastic weak-law admission, Hessian providers, minimization,
reduced-solve sensitivities, arbitrary meshes, or moving Geometry.

Run `cargo run -p eqiora-verify -- run --case solid.elastic-gradient-energy`.
