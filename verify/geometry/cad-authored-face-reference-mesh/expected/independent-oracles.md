# Independent pre-implementation oracles

The implementer did not author, tune, or relax these values. Two independent
provider lanes received only the frozen public claim, non-claims, orientation,
indexing, sizing predicate, quality definition already owned by
`SimplicialMesh`, and resource limits.

## Analytic route — Opus 5

The analytic route derived the structured-grid counts and invariants directly:

- `nu = 4`, `nv = 3`, `V = (nu + 1)(nv + 1) = 20`, and
  `T = 2 nu nv = 24`;
- boundary edges `2nu + 2nv = 14`, interior edges `29`, total edges `43`, and
  `V - E + T = 1`;
- per-cell spacings `5/4 m` and `1 m`, per-triangle area `5/8 m²`, total area
  `15 m²`, and perimeter `16 m`;
- maximum edge `sqrt(41)/4 m` and minimum cell quality `40/41` under the
  repository's existing triangle-quality definition;
- end-cap frame `origin=(-2,-1,4.5)`, `u=+x`, `v=+y`, `normal=+z`, with corner
  indices `0,4,19,15`;
- start-cap frame `origin=(-2,-1,0.5)`, `u=+y`, `v=+x`, `normal=-z`, with corner
  indices `0,3,19,16`.

It also required validate-then-reveal handle semantics, a classification
tolerance separate from sizing, exact affine lifting, a strictly-below quality
rejection, and checked budget arithmetic before allocation.

## Exact enumeration route — Fable 5

The independent route enumerated all vertices and triangles with exact rational
coordinates, reduced undirected edges to multiplicities, and accumulated
determinant areas. It independently obtained `V/T/E = 20/24/43`, boundary and
interior counts `14/29`, Euler characteristic 1, triangle area `5/8`, total area
15, perimeter 16, the same corner indices, and the exact `40/41` quality value.
It separately enumerated the start-cap frame and confirmed distinct coordinate
and connectivity orderings under the frozen orientation rule.

## Binary64 sizing checks

For `L = 5`, target `0x1.2db2eaabf5c80p+1` selects three intervals; its
immediate predecessor `0x1.2db2eaabf5c7fp+1` selects four. With the other axis
fixed at `L = 3` and two intervals, the resulting counts are `V/T = 12/12` and
`15/16` respectively.

## Post-review realized-coordinate addendum

The first complete-diff review found an oracle gap rather than an implementation
value mismatch: `L/n` can round down while the final generated coordinate is
snapped to the exact endpoint `L`. The maximum realized interval can therefore
exceed the nominal spacing. Opus 5 and Fable 5 independently re-derived the
repair using distinct bit-level and exact-rational/enumeration routes before the
implementation was changed.

Both routes freeze the same per-axis rule. Compute `s = L/n`; generate
`x_i = i*s` for `i<n` and `x_n = L`; measure every generated binary64 gap and
let `D(n)` be their maximum. The least accepted count satisfies
`D(n).hypot(D(n)) <= h`. The old nominal count is a lower bound; under the
50,000-division cap the realized answer is that count or its successor, so at
most two O(n) coordinate scans are required.

For `L = 3` and `h = 0x1.3651a0eb63341p-1`, nominal n=7 uses
`s = 0x1.b6db6db6db6dbp-2`, but endpoint snapping produces
`D(7) = 0x1.b6db6db6db6e0p-2` and realized diagonal
`0x1.3651a0eb63345p-1 > h`. The least repaired count is eight. A 3 m square then
has `V/T/E = 81/128/208`, 32 boundary and 176 interior edges, and maximum edge
`0x1.0f876ccdf6cd9p-1` (0.5303300858899106 m).

The replacement estimate falsifier is `L = 4.875`
(`0x1.3800000000000p+2`), `h = 0x1.f844a57e8134bp-1`.
`ceil(sqrt(2)*L/h)` and `ceil(L/(h/sqrt(2)))` both give seven. At n=7,
nominal spacing is
`0x1.6492492492492p-1`, maximum generated spacing is
`0x1.6492492492498p-1`, and the realized diagonal
`0x1.f844a57e81353p-1` rejects. At n=8 the exact spacing
`0x1.3800000000000p-1` gives diagonal `0x1.b93c10ceb10e1p-1` and accepts.

The primary, start-cap, and length-5 one-ulp witnesses are unchanged. The old
`L = 8.375`, `h = 0x1.b12745f33a78cp+0` witness changes from nominal n=7 to
realized n=8 and is no longer an estimate falsifier; it must remain a regression
check for endpoint-aware correction rather than evidence for n=7.

## Independent evidence follow-up — Fable 5

This lane re-read the amended sections above and adopts their naming and prose
as its own: the frozen per-axis realized-gap rule, the
`endpoint_snapping_witness` and `estimate_mutant_witness` names, and every
numeric value in this document are owned by the independent evidence lane, and
none was changed.

The old `L = 8.375` witness is frozen by name as `retained_regression_witness`:
at `h = 0x1.b12745f33a78cp+0`, nominal n=7 has maximum realized gap
`0x1.324924924924cp+0`, whose diagonal rejects, and the least accepted count is
eight — the count both ceiling estimates also give — so it guards
endpoint-aware correction and is not an estimate falsifier.

### Dual-axis witness derivation

Route: CPython 3.12 binary64 arithmetic applied to the frozen public rule only
— `s = L/n`, `x_i = i*s` for `i<n`, `x_n = L`, `D(n)` the maximum adjacent
realized gap — with `hypot` taken from glibc 2.39 through `ctypes` (the libm
that Rust `f64::hypot` delegates to on x86-64 Linux) and every pinned diagonal
cross-checked to equal the correctly rounded binary64 result by exact integer
arithmetic. The implementation was not consulted. A scripted sweep over dyadic
and decimal side lengths selected the pair below; least-count claims were then
verified exhaustively over every smaller division count. Because the facade
admits only faces whose unit axes normalize exactly, the sweep kept only side
lengths with `L * fl(1/L) = 1` — this excluded a first candidate (`9.1 m`)
whose axis fails exact normalization, and the exclusion is a frame-admission
constraint, not a sizing one.

For the non-square rectangle `u = 17.5 m`, `v = 10.5 m`, both nominal spacings
`17.5/25` and `10.5/15` round to the same binary64 `0x1.6666666666666p-1`
(0.7 m), whose diagonal is exactly the target `h = 0x1.fadaa8f7eed51p-1`.
Endpoint snapping widens the two axes unequally:
`D(25) = 0x1.6666666666680p-1` (26 ulps above nominal, diagonal
`0x1.fadaa8f7eed76p-1`) rejects u at 25, and `D(15) = 0x1.6666666666670p-1`
(10 ulps above nominal, diagonal `0x1.fadaa8f7eed5fp-1`) rejects v at 15.
Every smaller count also rejects, so the least divisions are 26 and 16, again
with unequal maximum realized gaps: `0x1.589d89d89d8a0p-1` on u (two ulps
above its nominal spacing) versus the exact dyadic spacing
`0x1.5000000000000p-1` (0.65625 m) on v, where no snapping widening survives.
The accepted grid has 459 vertices, 832 triangles, and 1290 edges (84
boundary, 1206 interior). The correctly rounded binary64 value of the exact
widest cell diagonal is `0x1.e14e6a48457e0p-1 <= h`. This remains the
mathematical reference: the accepted gaps are exactly
`u = 189454311247557 / 2^48` and `v = 21/32`, so their squared diagonal is
`S = 70013658383089496624645579673 / 2^96`. The midpoint between the
adjacent binary64 candidates ending in `57df` and `57e0` is
`m = 16934460272979903 / 2^54`, and
`S - m² = 7341760852451199 / 2^108 > 0`. Together with the candidates'
squared bounds on `S`, this selects `57e0` by nearest rounding.
It is not a mandated last bit for Rust's
[`hypot`](https://doc.rust-lang.org/std/primitive.f64.html#method.hypot), whose
precision is unspecified. The executable witness checks every actual triangle
edge's absolute u and v projections against the frozen accepted gaps and
requires an actual edge attaining both maxima simultaneously. Thus it proves
the widest diagonal is present, with no tolerance or libm rounding assumption;
the computed maximum edge must also remain at or below the target.
On u the two closed-form ceiling estimates
straddle the answer (26 versus 25), so neither estimate is the selection rule.

### Environment scope

The bit-exact hypot witnesses in this case are x86-64 Linux/glibc evidence:
Rust `f64::hypot` resolves to glibc's libm there. Every frozen diagonal in
this document equals the correctly rounded binary64 result, so any correctly
rounded libm reproduces them. Rust does not guarantee that every libm version
is correctly rounded. The mixed-axis maximum is consequently checked by its
actual edge projections rather than its `hypot` output bits. Cross-platform
mesh-byte identity is not claimed.
