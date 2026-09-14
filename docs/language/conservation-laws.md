# Fixed-domain conservation Laws

A `law` retains physical flux and source as mathematical terms. Numerical methods
consume these terms through the same Model as ordinary Relations.

```eqiora
model HeatedInterval() {
  domain body = box(0, 1);
  domain left = boundary(body, axis = 0, side = lower);
  domain right = boundary(body, axis = 0, side = upper);
  variable temperature: K on body;
  parameter conductivity: kg * m / s^3 / K = 2;
  parameter heating: kg / m / s^3 = 4;
  law heat_balance on body {
    flux -conductivity * grad(temperature);
    source heating;
  }
  relation left_temperature on left { trace(temperature) = 0; }
  relation right_temperature on right { trace(temperature) = 0; }
}
```

A Law specifies the steady balance `div(flux) = source`. Flux is the
physical outward flux; diffusion therefore uses `-conductivity * grad(temperature)`.
The compiler requires exactly one flux and one source, with compatible dimensions
and the Law's exact volume support. Write `source 0;` for no production.
Boundary and interface conditions remain ordinary Relations on their exact supports.

Python Module authoring uses `component.law("heat_balance", on=body,
flux=-conductivity * eqiora.lang.grad(temperature), source=heating)`. It produces the
same source AST and retains the same physical terms in Model artifacts and package
composition. Changing a physical term changes Model meaning even when an equation
could otherwise be rewritten to an equivalent residual.

The admitted boundary is real scalar steady conservation on a fixed volume. Existing
scalar diffusion realizations impose their own coefficient, Geometry, boundary and
method restrictions. The parser rejects storage terms; no stored quantity is admitted
until accumulation correspondence has a checked implementation. This does not implement
transient thermal execution, moving-domain transport,
arbitrary vector Laws, or general authored Law-to-form correspondence.

A steady real scalar Law on an exact one-dimensional Geometry support can also
carry an authored mathematical conservation form:

```eqi
form conservative for balance {
  interval segment(a, b) on body;
  outward_flux(segment, a, -k * grad(T))
    + outward_flux(segment, b, -k * grad(T)) = integrate(segment, s);
}
```

Here `balance` retains `flux -k * grad(T); source s;`, and `k` and `s` are
explicit Component parameter bindings. The binder quantifies every ordered
interval `(a,b)` contained in `body`. Its endpoints are mathematical variables;
they are not mesh cells or the exterior endpoints of the parent support.
`outward_flux` applies normal `-1` at `a` and `+1` at `b`.

Fresh compilation checks dimensions and exact physical terms. The separate
projection checker reads the live Law, Field support and authenticated Geometry
again, without generating an expected form. The conditional implication assumes
a fixed one-dimensional domain and classical divergence and boundary traces;
it does not establish these regularity assumptions or the reverse implication.

Native AST construction, source formatting, Python inspection and mathematical
rendering retain the interval. The scalar projection uses one tagged v3 wire for
weak-test and interval binders; the old v2 projection decoder is removed. Forms
remain outside Model identity. The ordinary single-region scalar TPFA path admits
this form on a fixed 1D Geometry with the existing positive diffusion and supported boundary conditions.
Automatic and exact integral-conservative requests derive mathematical content
and pass the same checker; only authored requests retain authored source identity.
Plan and Result replay preserve this distinction and rerun admission. Storage,
multidimensional, mixed and complex forms remain unavailable.
