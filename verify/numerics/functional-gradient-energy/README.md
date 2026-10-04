# Retained gradient energy through first-variation execution

This case uses the same retained Observables for first variations and ordinary
Result energy observations. Volume-only fixtures use one Observable; the prescribed
flux fixture composes a volume-energy variation with a separate boundary-work variation. The ordinary Python API compiles source or a Module,
resolves the generated form, serializes and reads its Plan, and runs Cartesian Q1.
The executable target is the existing
[`python_authored_scalar_form`](../../../crates/eqiora-python/tests/python_authored_scalar_form.rs)
integration test. It loads the public Python package over the current Rust
extension; this case does not claim an installed-wheel environment.

The source is a unit square with four equal square cells and homogeneous essential
conditions on all four sides, followed by a second fixture with only the left
side essential and the other three carrying zero natural flux. The functional is

```text
F[u] = integral (k*grad(u).grad(u)/2 - f*u) dA
DF[u;w] = integral (k*grad(u).grad(w) - f*w) dA.
```

The independently authored Law is `-div(k*grad(u))=f`. Integration by parts
produces the boundary term `k*(normal.grad(u))*w`; each essential side uses its explicit
zero-trace test restriction, and each natural side uses its exact homogeneous
constitutive-flux Law. No zero test trace is added on a natural side. The compiler derives the
variation from the retained energy. Numerical admission first regenerates that
body from the live Observable, then compares it with the Law's separately derived
weak residual using bounded exact polynomial arithmetic. The test never uses a
finite difference or a sampled residual match as the correspondence oracle.

For k=f=1 in the declared SI coordinates, one unconstrained central Q1 hat remains.
The independent integrals give K=8/3, b=1/4, u_center=b/K=3/32 and
F[u]=u_center²*K/2-b*u_center=-3/256. Both the dimensionless profile and the
length-valued Field with joule-valued functional must give these numerical values.
The direct Python Module and its emitted source follow the same ordinary path.
The mixed-boundary fixture has nodal values 0, 3/8, 1/2 at x=0, 1/2, 1
and discrete energy −5/32, with no y dependence.
A third fixture applies `normal(k*grad(u))=f*x` on the right side. Its retained
boundary Observable is S[u]=−integral_right f*x*trace(u) ds, and its authored form
is DF[u;w]+DS[u;w]=0. With k=f=1 its nodal values are 0, 7/8, 3/2;
F=11/32 and S=−3/2, so F+S=−37/32. The actual boundary measure and trace are
retained; they are not replaced by a volume expression or a sampled load value.
Sums replay each live functional before exact comparison. The test also admits
DF+DF−DF and rejects the doubled derivative DF+DF against the original Law.
See [the derivation and tolerances](expected/README.md).

Falsifiers change only the energy diffusion coefficient, load sign, or load
Parameter identity while keeping the Law; all must fail the weak-residual
correspondence check. A dimensionless direction for the length-valued Field is
rejected. Multiplying the energy by a numerically unit-valued but dimensional
`1[J]` must fail the strong-law test-pairing dimension check even when its numerical
polynomial coefficients match. The target also checks that missing or incomplete essential restrictions
fail before execution. The mixed-boundary path also rejects extra zero traces on
natural sides, missing essential zero traces and nonzero natural loads omitted
from the functional. The prescribed-flux fixture rejects missing, sign-reversed and misplaced surface
work. Positive solves and Plan replay precede these denials.

This is a bounded scalar first-variation claim. It proves no single composite
volume-plus-surface Observable, elastic surface-load solve, executable second variation, moving-domain
variation, continuum minimizer error estimate, stability, or energy decay.

Run `cargo run -p eqiora-verify -- run --case numerics.functional-gradient-energy`.
