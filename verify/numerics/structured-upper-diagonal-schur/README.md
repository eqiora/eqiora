# Structured upper diagonal Schur pilot

This specified case carries the manufactured Stokes oracle from
[the existing MINI case](../../fluid/simplicial-mini-stokes-2d/README.md)
through an ordinary `Eqiora.Fluid` package Model, common Plan admission and
common Run. It is not yet passing evidence. The first test uses `n=2`,
`L=1 m`, and `mu=1 Pa s` to expose integration gaps before the full pilot.

The proposed comparison has explicit direct faer, manual BiCGSTAB with the
fixed upper diagonal Schur composition, and ranking over that explicitly
requested composition. Groups use the retained velocity Field, then the
retained pressure Field and its gauge multiplier. Construction limits are
explicit. They do not bound factor fill or process RSS.

The full pilot uses refinements `2,4,8`, lengths `1,2 m`, and viscosities
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

Run the initial integration test with:

```sh
cargo test --locked -p eqiora --test structured_upper_diagonal_schur
```
