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

All essential displacement values must equal their exact dyadic expectations.
There is no boundary-value tolerance. Define U0=max(1 m, ||u*||∞)=1 m,
S=||K*||∞ U0+||b*||∞=117.25 N, η=2^-40, and σ=2^-46. The fixed
aggregate residual ceiling is τ=(η+σ)S (approximately 1.08305e-10 N).
These dimensionless precision factors declare an engineering accuracy ceiling;
they are not a floating-point operation-count theorem. No candidate values
were consulted to select them. Rational evaluation introduces no computational
roundoff allowance. Every reference residual is evaluated exactly on the
exported binary64 coefficients, converted to rational values.

Before using an inverse norm, the generator verifies K*_FF times its computed
rational inverse equals the identity in every entry. The tests verify both
product orders. Then C=||K*_FF^-1||∞ (approximately 2.76668 m/N) and
E=Cτ (approximately 2.99644e-10 m). Require every free reference residual
component ≤τ and every one of the 81 coefficient errors ≤E. Exact essential
data make the residual-to-coefficient implication rigorous. This observation
route requires no inaccessible private native matrix or load vector.

Construct three exact rows t_i from Q1 boundary derivatives for integrated
left-face stress traction. Their 1-norms are (220,96,90) N/m. Put V=U0+E.
Each exported public traction T_i must satisfy both
|T_i−t_i·v| ≤ η ||t_i||₁ V and
|T_i−t_i·u*| ≤ ||t_i||₁ E + η ||t_i||₁ V,
where v is the complete exported displacement vector. The first test binds the
public functional to the exported field; the second binds it to the analytic
solution. These quantities do not measure private weak reaction recovery.

For B=Σij |K*ij|, W(v)=vᵀK*v/2 and
w=η B V²/2, exported public energy must satisfy both
|W_native−W(v)| ≤ w and
|W_native−261/32 J| ≤ ||K*u*||₁ E + B E²/2 + w.
These follow from the exact quadratic energy perturbation identity. No extra
allowance is added to any check. Failures require investigation, never tuning
the thresholds to pass. The same decoder and checks apply independently to
primary and replay observations, supplied as two separate documents.

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

The traction rows integrate the constitutive law directly on x=0. For each
adjacent cell, ∫face ∂x N_a=(2a_x−1) H_y H_z/(4H_x),
∫face ∂y N_a=1[a_x=0] · (2a_y−1)H_z/2, and
∫face ∂z N_a=1[a_x=0] · (2a_z−1)H_y/2. Assemble
σ_i0=μ(∂x v_i+∂i v_x)+λ δ_i0 div(v) and negate for the outward
normal. Summing shared-node contributions before taking absolute values gives
the exact row norms above. Norms from per-cell triangle inequalities would be
larger and are not used. Pure falsifiers deliberately produce functionals that
pass analytic-target budgets while failing the exported-field comparison.
