# Cartesian Hex elasticity: pre-candidate reference

This directory prepares the independent mathematics for issue #445. It does not
register evidence, execute Eqiora, or establish a supported capability. Two
independent oracle reviews must accept this protocol before candidate execution.

The box is [0,2] × [0,3] × [0,4] m, divided into 2 × 2 × 2 affine Hex cells.
Coordinates are (i,3j/2,2k), i,j,k in {0,1,2}. Nodes run x fastest, then y,
then z; components run x,y,z within each node. Local corners use the same
binary convention x + 2y + 4z, independently of any product storage ordering.
A candidate adapter must map by exact physical coordinates and component name.
No positional agreement with a product array is assumed.

Use μ=3 Pa, λ=2 Pa and u=Ax with
A=((1,2,0),(0,3,1),(1,0,2))/16. All nine components of A are dimensionless.
Prescribe all displacement components only on x=0 (27 constrained, 54 free).
On the other five faces apply t=σn, with outward unit normal n, zero body force.
The independent constitutive identity is σ=μ(A+Aᵀ)+λ tr(A)I, giving
σ=((9/8,3/8,3/16),(3/8,15/8,3/16),(3/16,3/16,3/2)) Pa.
The strain energy is V σ:sym(A)/2 = 261/32 J. The integrated left-face
stress traction is (-27/2,-9/2,-9/4) N. This is a public stress functional,
not a measurement of recovered weak Dirichlet reactions.

`reference.py` integrates tensor products of linear polynomials with rational
arithmetic. For node a, component i, and node b, component j,
K_ai,bj = ∫[μ δij ∇Na·∇Nb + μ ∂jNa ∂iNb + λ ∂iNa ∂jNb]dV.
Loads integrate Na σn over the five natural faces, with exact physical areas.
Linear shape integrals and derivative products factor by Cartesian axis.
The constrained residual is deliberately retained; only free rows must vanish.
The exact matrix has units N/m, load N, coefficients m, energy J.
Two-point Gauss per axis exactly integrates these polynomials: stiffness degree
at most two per axis, face load degree one, energy and stress observations at
most degree two. Reduced integration is outside this case.

Run `python3 reference.py` to emit reproducible rational expectations and fixed
budgets; run `python3 -m unittest discover -s . -p 'test_*.py'` here for pure
reference tests. Neither command imports or runs Eqiora. No candidate values
have been used. The inverse infinity norm is calculated exactly by rational
Gauss-Jordan elimination and the emitted C is ||K_FF^-1||∞ in m/N.

Fixed acceptance budgets are coefficient absolute error 1e-8 m, free reference
residual 1e-10 N, constrained coefficient absolute error 1e-12 m, energy error
1e-6 J, and each integrated left stress-traction component error 1e-6 N.
The reference residual means K* u_observed - b*, using the independent K*, b*;
it does not require exporting a private native matrix. With exact constrained
data, coefficient error is bounded by C times residual. With constrained error
δD, add C ||K_FD||∞ δD. These are rigorous algebraic implications, not a
floating-point operation-count theorem. The coefficient budget also independently
checks all 81 values. Arithmetic allowance for computing residuals is 1e-12 N;
for coefficient comparisons 1e-14 m; for functional comparisons 1e-10 in their
physical units. These fixed small allowances are conservative engineering
allocations for this 81-variable binary64 calculation, subject to review before
any candidate is seen; failure requires investigation, never retuning to pass.
Public functionals must also pass their own budgets; a small nodal error does
not certify a separately evaluated public functional.

The later observation adapter must use a strict JSON schema: complete keysets,
no extra fields, unique node coordinate/component pairs, all 81 coefficients,
finite canonical binary64 values (reject NaN, infinity and negative zero),
physical units m/J/N, dimension 3, Q1, Hex, affine Cartesian geometry, and
2×2×2 tensor Gauss identity. Include complete Mesh/Model/Result lineage, the
exact authored cell counts and coefficient coordinate bijection. Primary and
replay observations must independently satisfy the same checks. Missing fields,
wrong dimensions, duplicate keys and unknown component/axis names reject before
numerical comparison. `protocol.py` enforces this observation projection, with canonical float.hex
strings for binary64 quantities. The integration change owns the adapter from
ordinary public results, semantic lineage validation, and primary/replay wiring;
identifier presence alone does not prove lineage.

Falsifiers must reject omitted λ divergence, full-gradient substitution for
symmetric strain, reversed traction, omitted displacement component, wrong axis
or local corner mapping, and a unit-Jacobian assumption on this nonunit mesh.
Pure reference controls exercise their mathematical signatures. Native geometry
rejection tests must separately submit inverted, zero-volume, non-finite,
unsupported distorted/non-affine and over-budget meshes and require a typed
diagnostic before any accepted Result. Positive geometry is limited to these
eight cells, 27 nodes, μ=3, λ=2; changing those inputs needs a fresh review.

No CAD sweeps, distorted or curved Hex, Q2, reduced integration, incompatible
modes, near-incompressible robustness, hourglass control, nonlinear behavior,
dynamics, FSI, Prism, Pyramid, mixed cells, AMR, GPU, MPI, or commercial Hex
robustness is claimed. Stop on silent Tet conversion, a universal cell registry,
a broad public enum without a second consumer, weakened oracle, or an unreviewed
architecture-ceiling increase.

For coefficient error bounded by e in every component, left-traction error is
bounded by (248e,120e,108e) N/m using |∂d δu| ≤ 2e/Hd on each adjacent
cell and the 12 m² face area. Energy perturbation is bounded by
||K U||₁ e + (Σij |Kij|) e²/2, directly from the quadratic energy identity.
These bounds describe the independent Q1 functionals. They do not substitute
for checking that public observations actually use the same recovered field.
The residual-derived e (including prescribed-value error) is emitted by the
report, and is substantially smaller than the standalone coefficient cap.
