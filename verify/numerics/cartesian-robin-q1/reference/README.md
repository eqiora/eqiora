# Preparatory Cartesian Q1 Robin reference

This standalone mathematical reference prepares four steady affine heat cases.
It does **not** establish that Eqiora implements Robin boundaries. It imports no
Eqiora code, uses no product output to generate expectations, and runs no native
candidate. Production integration and any use with candidate data require source
and mathematical review first.

## Fixed cases and analytic solution

Use a uniform two-by-two Cartesian Q1 mesh on [0,L] x [0,H]. Temperatures are in K,
L and H in m, conductivity k in W/(m K), and transfer coefficient h in W/(m² K).
The four fixed cases are:

| Case | L | H | k | h | Tc | Tinf | Slope a, K/m |
|---|---:|---:|---:|---:|---:|---:|---:|
| insulated | 2 | 1 | 3 | 0 | 280 | 320 | 0 |
| robin | 2 | 1 | 3 | 2 | 280 | 320 | 80/7 |
| changed_h | 2 | 1 | 3 | 5 | 280 | 320 | 200/13 |
| changed_geometry_material_cooling | 3 | 2 | 2 | 4 | 310 | 280 | -60/7 |

The left face has essential temperature Tc, the top and bottom are insulated, and
the right face has physical outward heat flux

```
-k dT/dn = h (T-Tinf).
```

The exact solution is T=Tc+a x, with a=h(Tinf-Tc)/(k+hL). This follows directly
from k a+h(Tc+aL)=h Tinf. Q1 represents the affine solution exactly, so there is
no spatial truncation allowance.

## Independent full physical operator

On an interval element of length ell, direct integration of the linear basis
(1-t,t) gives

```
M_local = ell/6 [[2,1],[1,2]]
K_local = 1/ell [[1,-1],[-1,1]]
q_local = ell/2 [1,1].
```

Assemble these over two intervals in each axis. Order all nine physical nodes
with x index first and y index varying fastest: (i,j) for i=0,1,2 and j=0,1,2.
With eR=(0,0,1), the complete uneliminated matrices are

```
A* = k (Kx tensor My + Mx tensor Ky) + h (eR eR^T) tensor My
b* = h Tinf eR tensor qy.
```

A has units W/(m K); b and nodal weak reactions have units W/m per out-of-plane
thickness. All entries are exact rational numbers. The left three nodes are
essential; the remaining six are free. The code computes the six-by-six inverse
A*_FF^-1 exactly and verifies its product with A*_FF is the identity. Its row-sum
norm C is an exact rational reference quantity, not a measured solver condition.

The exact left weak reactions, in increasing y order, are

```
-k a [H/4, H/2, H/4],  with sum -k a H.
```

Thus physical outward integrated heat on the left is +k a H, the **negative** of
the summed left weak reactions. The right outward heat is hH(T(L)-Tinf)=-k a H.
Integrated heat is W/m; pointwise heat flux is W/m². These two quantities and the
weak-reaction sign must not be conflated.

## Fixed perturbation and recovery budgets

All budgets below depend only on the fixed rational case. They are computed
before inspecting a candidate and cannot be enlarged by observed errors.

```
eta = 2^-40, sigma = 2^-46
U = max(1 K, |Tc|, |Tinf|)
SA = ||A*||_infinity
Sb = SA U + ||b*||_infinity
deltaA = eta SA, deltab = eta Sb, tau = sigma Sb
E = C (tau + deltaA U + deltab) / (1-C deltaA).
```

The denominator must be positive, and is verified for all four fixed cases.
The factor eta equals 8192 binary64 unit roundoffs. **It is a declared numerical
accuracy ceiling, not an operation-count theorem or a universal floating-point
error bound.** Exact rational comparisons audit

```
||A-A*||_infinity <= deltaA
||b-b*||_infinity <= deltab
||(A v-b)_F||_infinity <= tau
|v_i-T*_i| <= E for every physical node
||(A* v-b*)_F||_infinity <= tau + deltaA (U+E) + deltab.
```

The essential coefficients must equal Tc exactly. To derive E, let e=v-T* on the
free nodes and use the exact inverse to obtain
||e|| <= C[tau+deltaA(U+||e||)+deltab]. Rearranging gives the fixed expression
above. Full physical operator/load audits therefore cannot be replaced by a
small residual in a candidate's own altered equations.

For native weak reaction recovery, define fixed ceilings

```
R = eta [(SA+deltaA)(U+E) + ||b*||_infinity + deltab]
rho = deltaA (U+E) + deltab + R.
```

As an explicit pre-candidate protocol clarification, every native left reaction
must first agree with the exact exported physical residual (A v-b) at its node
within the already-declared R. It must also agree with (A* v-b*) within rho, with its exact individual analytic reaction within SA E+rho, and with the
analytic reaction sum within 3(SA E+rho). Every check is reported independently.
No comparison uses a candidate-derived scale or tolerance.

## Strict mathematical input

The JSON object has exactly these keys:

```text
schema: 1
case: one fixed case name above
order: nine [x,y] exact decimal coordinate-string pairs, x-major/y-fast
units: exactly the dictionary below
A_binary64_hex: complete 9x9 array of canonical finite float.hex() strings
b_binary64_hex: complete nine-entry canonical finite float.hex() array
coefficients: nine coordinate/value records in the same physical order
left_reactions: three coordinate/value records in increasing left-face y order
heat_convention: "physical_outward_heat=-weak_reaction"
```

A coordinate/value record has exactly
`{"coordinates_m":["x","y"],"value_binary64_hex":"canonical hex"}`. The units
object is exactly:

```json
{
  "coordinates": "m",
  "A": "W/(m*K)",
  "b": "W/m",
  "coefficients": "K",
  "left_reactions": "W/m",
  "integrated_heat": "W/m",
  "pointwise_heat_flux": "W/m^2"
}
```

All coordinates must equal their exact dyadic grid values; no association
rounding tolerance is used. Values are decoded with `float.fromhex` followed by
`as_integer_ratio`, so the exported binary64 value is captured exactly. Decimal
approximations do not replace binary64 values. The full uneliminated nine-node
A/b are required, including essential rows and columns. Six-node eliminated
operators, missing nodes or entries, duplicate/reordered coordinates, wrong-face
reactions, extra/missing keys, duplicate JSON keys, wrong units, nonfinite values,
noncanonical hex values, and incorrect essential coefficients are rejected.

A separate future producer receipt must authenticate the actual Model, Plan,
mesh, DOF associations, complete boundary coverage, quadrature, physical
unscaling and raw artifact hashes. Declaring coordinates, ordering, units or a
case name in this mathematical input does not prove those production facts.
This reference does not parse Eqiora Plans or claim a native heat API is checked.

The comparator derives two physical outward heat reports: left heat is minus
the native weak-reaction sum; right heat is the exact rational boundary integral
h sum_j qy_j (v_right,j-Tinf). They are **derived reports**, not validation of an
independently exported native heat value. The explicit heat convention is checked
without inventing another native-heat tolerance.

## Running the independent checks

```sh
python3 robin.py reference robin --output rational-reference.json
python3 robin.py compare mathematical-input.json --output comparison.json
python3 -m unittest discover -s . -p 'test_*.py'
```

Reference output contains the full rational operator/load, six-free inverse,
analytic values and fixed budgets. Comparator output contains all observations,
budgets, individual checks, complete counts, derived physical heat, and SHA-256
of the exact raw input bytes. Rational numbers are serialized as numerator and
denominator strings. Exit status is zero for acceptance/reference generation,
one for a valid comparison failing a numerical check, and two for malformed or
unsupported input. No candidate data or generated numeric table is committed.

Tests use only independently constructed mathematical exports. They check direct
polynomial integration, continuum affine balance, exact inverse multiplication,
individual weak reactions and heat signs for all four cases. Falsifiers omit the
Robin matrix, reverse the boundary sign, change h in only matrix or load, move the
Robin face, eliminate the right test trace, omit endpoint facet weights, perturb
coefficients/reactions, or reverse the weak-reaction-to-heat convention. A recovery
falsifier fits the looser reference allowance rho but violates the direct
exported-operator recovery ceiling R. Robin
matrix falsifiers apply to h>0; omitting a zero matrix is not called a falsifier.
No transient, nonconstant, curved, MPI, GPU or native product claim follows.
