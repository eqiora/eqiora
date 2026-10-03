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
rendering retain the interval. The projection uses one tagged v5 wire for plural Relation-owned equations,
weak-test inventories and interval binders; displaced projection decoders are removed. Forms
remain outside Model identity. The ordinary single-region scalar TPFA path admits
this form on a fixed 1D Geometry with the existing positive diffusion and supported boundary conditions.
Automatic and exact integral-conservative requests derive mathematical content
and pass the same checker; only authored requests retain authored source identity.
Plan and Result replay preserve this distinction and rerun admission. Storage,
multidimensional, mixed and complex forms remain unavailable.

An interval form can retain an explicit constant scalar gauge after its binder:

```eqi
gauge potential {
  reference integrate(body, potential) = 0;
  compatibility integrate(body, source_value) + lower_load + upper_load = 0;
}
```

The reference and compatibility equalities are separately dimension-checked and
bound to the exact trial Field and parent support. Here the endpoint loads denote
`n k grad(potential)`, the negative of physical outward flux, so their sum enters
compatibility with the source integral using a plus sign. The numerical admission
checker matches this condition to the original source and both natural boundary
relations; a written condition alone does not establish that their values balance.
The ordinary fixed-1D TPFA path admits the explicit zero-integral reference with
two natural endpoints. Plan resolution retains the existing `ZeroIntegral`
constraint and requires a solver for the resulting symmetric-indefinite bordered
system. Before solving, execution checks the constant vector against the actual
original matrix and checks the assembled load balance without projection. Result
acceptance and replay reassemble the original equations and reference, retaining
separate original-equation, compatibility and gauge residuals and the numerical
multiplier. The scalar Field contains no multiplier entry. Python exposes these
as `scalar_original_residual_norm`, `scalar_compatibility_residual`,
`scalar_gauge_residual` and `scalar_gauge_multiplier`; the solve report describes
the bordered system. Constrained differentiation is not admitted.

The native P1 and TPFA paths also support an explicit spatial mean. General
nullspaces, automatic mode discovery and finite floating-network Formulations
remain outside this interval profile.

A bounded real steady 2D Stokes system may retain two weak equations in one form.
`form weak for momentum, continuity` pairs each equation with its declared Relation;
velocity and pressure have distinct named tests. The velocity test carries the exact
complete homogeneous essential boundary, while the pressure test has no boundary
restriction. `frobenius` contracts equal real rank-two shapes and remains distinct
from vector `dot`. Compilation through the public Model API checks every produced
term against the live Stokes certificate before exposing Python inspection or
rendering. This profile supports inspection only; authored mixed numerical
execution and stability claims remain unavailable.
