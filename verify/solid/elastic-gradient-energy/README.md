# Elastic energy through authored variation and ordinary Result

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

The direction above is an arbitrary State perturbation, not an admissible
stationarity test: eta does not vanish on the clamped side.

A second profile uses four Q1 cells, mu=3 Pa, lambda=2 Pa, q=6x Pa and zero
prescribed displacement on every side. One retained functional

```text
F[u] = integral (mu*epsilon(u):epsilon(u) + lambda*div(u)^2/2 - grad(q).u) dA
```

supplies the authored first variation and Result observation. The displacement
direction has units m and holds mu, lambda and q fixed. Admission regenerates the
variation from the live Observable and compares its exact component polynomial
with the independent strong-law stress and body force. Essential sides discharge
by zero test trace; homogeneous natural sides use their admitted zero-flux Laws.
The four-cell stiffness gives central displacement (9/88,0) m and energy −27/352 N.
The ordinary Plan serializes and resolves again with its authored provenance.
The central hat has zero State-direction energy derivative, whereas a constant
translation violating essential restrictions has derivative −6 N. Changing the
volumetric coefficient, load sign, essential test restriction or Parameter identity
must reject, including replacing mu by numerically equal lambda.

A third four-cell profile restores the original left-essential/other-sides-zero-flux
boundary laws with lambda=0. The first variation solves nodal u_x=x−x²/2, u_y=0
and observes total energy −15/16 N. It does not invent zero test traces on the
natural sides. The exact conservative-load definition is evaluated from its
admitted expression and remains fixed under State directions.

The compiler-generated second density is also sampled through ordinary Result.
The central hat pairs with itself to give `44/3 N` and with the transverse central
hat to give zero, matching the independent stiffness above. A constant translation
has zero second product. Parameters, Geometry and the conservative load remain fixed.

This case makes no claim about nonzero surface work, implicit-solve Hessians, minimization,
reduced-solve sensitivities, arbitrary meshes, or moving Geometry.

Run `cargo run -p eqiora-verify -- run --case solid.elastic-gradient-energy`.
