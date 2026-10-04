# Exact coordinate partials

This case proves selected real scalar polynomial derivatives and one Cartesian Q1 Field
path through the existing Model, Formulation, Plan and Result owners. Exact support, leaf
factor and within-factor axis identify each coordinate. The Model retains analytic calculus
requests and unknown Field derivative nodes; the latter also project to Form expressions.
Arbitrary authored weak forms, product-domain PDEs, intrinsic boundary charts and higher
unknown-Field derivatives are outside this claim.

Independent differentiation of `u=Q0*((x/L)^2+3*x*y/L²)` gives
`u_x=Q0*(2*x+3*y)/L²`, `u_y=3*Q0*x/L²`. The test uses `Q0=8 kg`, `L=2 m`,
three asymmetric points and exact binary reciprocals. For `f=F0*x*v/(L*V0)`,
`f_x=F0*v/(L*V0)` and `f_v=F0*x/(L*V0)`. With `F0=6 kg`, `L=2 m`, `V0=4 m/s`,
three asymmetric points distinguish factors and scales. The respective derivative units
are `kg/m` and `kg*s/m`. These short binary64 polynomial values are exact; no tolerance
is needed. A Parameter and a different coordinate's polynomial prove independent zero.

On the unit square, the harmonic bilinear solution `u=x*(3+2*y)` (in coherent units)
is exactly representable by Q1. Its derivatives are `u_x=3+2*y` and `u_y=2*x`;
integrating over the square gives `4 m` and `1 m`. Both agree with the correspondingly
ordered `grad` components. The Field chain rule gives `d(u²)/dx=2*x*(3+2*y)^2`;
its integral is `9+6+4/3=49/3 m`. Two-point tensor Gaussian quadrature integrates
these polynomials exactly. Absolute bounds `1e-11` for first derivatives and `1e-10`
for the quadratic composition cover the small reference linear solve and binary64
accumulation, not discretization error or convergence. No observed solver output defines
an expectation or tolerance.

Model replay preserves these observations. Component forwarding, wrong factors/units/axes,
foreign equal-unit supports, boundary-chart rejection and higher unknown-Field rejection
probe identity and admission. A separately solved TPFA Result reaches the Q1 representation
check and rejects the same derivative observation. Q1 derivatives are classical inside cells
and weak almost everywhere; global gradient continuity is not claimed.

Run `cargo run -p eqiora-verify -- run --case language.coordinate-partials`.
