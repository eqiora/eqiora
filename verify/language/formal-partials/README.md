# Explicit formal partials

The case admits real scalar polynomial partials and local operator composition at named
independent bindings, without a mesh or solved-state sensitivity. Compiler tests preserve
Parameter alias dependencies and reject dependent, foreign, conflicting, and unsupported
bindings or rules. Installed Python product tests separately cover direct authoring, emitted
source, and Model artifact replay.

Independent values are derived from the product and chain rules: for `f=x*x*y`, at `x=3,y=5`,
`f_x=30`, `f_y=9`, and `(f*f)_x=2700`. A let alias `z=x*x` has `z_x=6` and `z_y=0` even
when the independently declared Parameters happen to have equal values. For
`k=k0*(1+a*(T-T0)+a*a*(T-T0)*(T-T0))`, with `k0=10`, `a=0.01`, and `T-T0=20 K`,
`k_T=0.14 W/(m*K^2)`. The test tolerance is `1e-15` in these coherent units for the
short binary64 polynomial, independent of implementation output. Temperature subtraction
uses ordinary scalar Kelvin values; affine absolute-temperature typing is not claimed.

For `f=x²y+y³`, independent differentiation gives `∇f=(2xy,x²+3y²)` and
`H=[[2y,2x],[2x,6y]]`. At `(3,5)`, direction `(2,-1)` gives `Jv=-24` and
`Hv=(14,-18)`. Output dual seed `7` gives `Jᵀ7=(210,588)`; both dual pairings
are `-168`. No numerical differencing supplies these expectations. Direct, aliased and
composed third partials of `x³` all give `6`, independently of the evaluation point.
The repeated polynomial transform has no separate derivative-order ceiling; its existing
node and depth budgets still bound construction and replay. `x*abs(x)` is not admitted
as a C2 expression even at a currently positive evaluation point.

The heterogeneous specimen `f=x²v`, with `x=3 m`, `v=5 m/s` and direction
`(2 m,-1 m/s)`, gives `Jv=51 m³/s`. Its Hessian-action blocks are `14 m²/s`
and `12 m²`, not a homogeneous vector. A separate `f=xv` specimen uses dual seed
`7 s/m²`: input dual blocks are `35/m` and `21 s/m`, and their direction pairing
is `49`, equal to the output pairing. Wrong direction/seed units and repeated input
selections fail. Values in these added integer specimens are exact in binary64.

Local operator formals remain independent before substitution: for `g(x,y)=xy²`,
`g_x=y²` and `g_y=2xy`. Binding both inputs to `p` gives `p²` and `2p²`, whose
outer derivatives are `2p` and `4p`. At `p=3` these are `6` and `12`.

The spatial specimen binds the differentiated operator's two length inputs to actual
`coordinate(0)` and `coordinate(1)` on an authenticated unit-square Geometry. Ordinary
Q1 Result Observables and Model replay integrate `f_xy=f_yx=2x`: both integrals are
`1 m³`. Weighting by `x` gives `2/3 m⁴`; reversing the argument-axis binding gives
`1/2 m⁴`. These values follow from integrating monomials over `[0,1]²`, independently
of the solver state. Two-point tensor Gaussian quadrature is polynomial-exact here;
`1e-12` in coherent units covers binary64 accumulation. The weighted/reversed pair
falsifies axis substitution and the unweighted pair falsifies doubled mixed entries.
This is analytic operator differentiation before binding to Cartesian coordinates;
it does not admit general `partial(field, wrt=coordinate_factor)` or unknown-field
second derivatives, which remain owned by the coordinate-partial profile.

This case does not establish complex, nonsmooth, general coordinate-factor/unknown-field,
material or implicit-solution derivatives, or arbitrary expression size and derivative order
within a fixed execution budget. The nonzero derivative specimens reach order three.

Run `cargo run -p eqiora-verify -- run --case language.formal-partials`.
