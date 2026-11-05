# Structured upper diagonal Schur pilot

This case runs the manufactured Stokes oracle from
[the existing MINI case](../../fluid/simplicial-mini-stokes-2d/README.md)
through an ordinary `Eqiora.Fluid` package Model, common Plan admission and
common Run. Its 54-member similarity sweep passes the predeclared analytic,
gauge, equilibrium and independent true-residual checks for every method and
physical member. A separate smallest case covers `n=2`, `L=1 m`, `mu=1 Pa s`.

Focused controls admit equivalent grouped and signed complete-vector boundary
laws while preserving a single maximal prescribed subtree. They reject a
duplicate velocity trace and preserve generic mixed admission for a non-normal
tangential-traction boundary. No private solver or callback-only execution path
substitutes for the ordinary Run.

The comparison has explicit direct faer, manual BiCGSTAB with the
fixed upper diagonal Schur composition, and ranking over that explicitly
requested composition. Groups use the retained velocity Field, then the
retained pressure Field and its gauge multiplier. Construction limits are
explicit. They do not bound factor fill or process RSS.

The pilot uses refinements `2,4,8`, lengths `1,2 m`, and viscosities
`0.5,1,2 Pa s`, with `U=1 m/s` and `P=mu U/L`. Its exact fields are
`u=U((x/L)^2,-2(x/L)(y/L))`, `p=P(x/L-1/2)`, and conservative force potential
`q=-P(x/L-1/2)`. This is a physical similarity, units and geometry sweep.
The normalized equations are identical. It provides no independent
conditioning or viscosity robustness claim.

For this polynomial, `div(u)=0`, `div(2 mu sym(grad(u)))=(2 mu U/L^2,0)`,
and `grad(p)=(P/L,0)`. Substituting `P=mu U/L` gives
`-div(2 mu sym(grad(u))-p I)=(-P/L,0)=grad(q)`. The pressure
has zero mean by symmetry. These identities establish the oracle independently
of every tested solver.

The analytic convergence, gauge and equilibrium limits are predeclared in
`case.toml`; they retain the existing oracle's normalized limits. Solver
agreement alone cannot pass the case. Positive degree-four assembly and
independent degree-six error quadrature remain required. Existing acceptance
criteria must stay unchanged.

Run the complete pilot and its controls with:

```sh
cargo test --locked -p eqiora --test structured_upper_diagonal_schur
cargo run --locked -p eqiora-verify -- run --case numerics.structured-upper-diagonal-schur
```

Canonical Plan replay checks complete composition and provider intent. Result
replay checks canonical retention of gauge metadata. It does not independently
reauthenticate a finite retained Stokes multiplier numerically; the live Run
supplies the numerical gauge evidence.
